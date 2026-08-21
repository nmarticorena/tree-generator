"""Export ForSym tree URDFs to simulation-ready MJCF assets."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from math import cos, sin
from pathlib import Path
from random import Random

import mujoco

from forsym.dynamics import rud_deflection_param
from forsym.tree.postprocess import (
    _destination_path,
    _fixed_child,
    _source_path,
    _validate_dynamics,
    branch_level,
)


def export_tree_mjcf(
    source: str | Path,
    destination: str | Path | None = None,
    prune_fixed: bool = True,
    base_kp: float = 400.0,
    noise_std: float = 1.0,
    friction: float = 0.01,
    armature: float = 0.01,
    effort: float = 100.0,
    contype: int = 0,
    conaffinity: int = 0,
    collision_geom_group: int = 1,
    rng: Random | None = None,
) -> Path:
    """Convert a generated tree URDF into a simulation-ready MJCF asset.

    Unlike :func:`forsym.tree.postprocess.postprocess_tree_urdf`, this writes
    spring stiffness, joint armature, and collision contact masks, none of
    which URDF can represent.

    Parameters
    ----------
    source : str or pathlib.Path
        Generated tree URDF to convert.
    destination : str or pathlib.Path, optional
        Output path. By default, the file is written beside ``source`` with
        an ``.xml`` suffix.
    prune_fixed : bool, default=True
        Disable contacts on collision geoms belonging to non-trunk links
        joined by fixed joints, mirroring
        :func:`~forsym.dynamics.rud_deflection_param`; halved per
    base_kp : float, default=400.0
        L1 joint stiffness passed to
        :func:`~forsym.tree.postprocess.rud_deflection_param`; halved per
        subsequent branch level.
    noise_std : float, default=1.0
        Std. dev. of Gaussian noise added to each joint's stiffness.
    friction : float, default=0.01
        Friction loss written to flexible joints.
    armature : float, default=0.01
        Rotor inertia written to flexible joints.
    contype, conaffinity : int, default=0
        Contact bitmasks written to collision geoms that are *not* pruned.
        Contacts are off by default because generated branches initially
        overlap; callers can enable them when using a contact-aware setup.
    collision_geom_group : int, default=1
        Group assigned to collision geoms by the URDF-to-MJCF converter.
    rng : random.Random, optional
        Source of the stiffness noise. Defaults to a fresh, unseeded
        ``Random()`` instance if omitted.

    Returns
    -------
    pathlib.Path
        Absolute path to the exported MJCF.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If the URDF structure or a dynamics value is invalid.
    """
    source = _source_path(source)
    destination = _destination_path(source, destination)
    if destination.suffix != ".xml":
        destination = destination.with_suffix(".xml")
    _validate_dynamics(base_kp, noise_std, friction)
    rng = rng if rng is not None else Random()

    fixed_children = _fixed_child_links(source) if prune_fixed else set()

    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_mjcf_from_urdf(source, destination, contype, conaffinity)

    tree = ET.parse(destination)
    root = tree.getroot()
    # MuJoCo 3.10+ reserves ``world`` for its implicit world body.
    if (worldbody := root.find("worldbody")) is not None:
        root_body = worldbody.find("body")
        if root_body is not None and root_body.get("name") == "world":
            root_body.set("name", "tree_root")
    _write_joint_dynamics(root, base_kp, noise_std, friction, armature, effort, rng)
    _write_contacts(root, fixed_children, contype, conaffinity, collision_geom_group)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(destination))
    return destination


def _fixed_child_links(source: Path) -> set[str]:
    joints = ET.parse(source).getroot().findall("joint")
    return {link for joint in joints if (link := _fixed_child(joint))}


def _write_mjcf_from_urdf(source: Path, destination: Path, contype: int, conaffinity: int) -> None:
    """Convert the generated primitive URDF without discarding visuals."""
    urdf = ET.parse(source).getroot()
    root = ET.Element("mujoco", {"model": urdf.get("name", "forsym_tree")})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    asset = ET.SubElement(root, "asset")
    materials = {}
    for link in urdf.findall("link"):
        for visual in link.findall("visual"):
            material = visual.find("material")
            color = None if material is None else material.find("color")
            if material is not None and color is not None:
                materials[material.get("name", "default_material")] = color.get("rgba", "0.7 0.7 0.7 1")
    for name, rgba in materials.items():
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    defaults = ET.SubElement(root, "default")
    visual_default = ET.SubElement(defaults, "default", {"class": "visual"})
    ET.SubElement(visual_default, "geom", {"contype": "0", "conaffinity": "0", "group": "2"})
    collision_default = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(collision_default, "geom", {"contype": str(contype), "conaffinity": str(conaffinity), "group": "1"})

    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {_link_ref(joint, "child"): joint for joint in urdf.findall("joint")}
    joints_by_parent = {}
    for joint in urdf.findall("joint"):
        joints_by_parent.setdefault(_link_ref(joint, "parent"), []).append(joint)
    root_name = next(name for name in links if name not in children)
    worldbody = ET.SubElement(root, "worldbody")
    root_body = _add_body(worldbody, root_name, links, joints_by_parent, children, materials)
    root_body.set("name", "tree_root" if root_name == "world" else root_name)

    actuators = ET.SubElement(root, "actuator")
    for joint in urdf.findall("joint"):
        if joint.get("type") in {"revolute", "continuous", "prismatic"}:
            name = joint.get("name")
            ET.SubElement(actuators, "motor", {"name": f"{name}_ctrl", "joint": name})
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def _add_body(parent, name, links, joints_by_parent, children, materials):
    joint = children.get(name)
    origin = joint.find("origin") if joint is not None else None
    attrib = {"name": name, "pos": (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0")}
    if origin is not None and origin.get("rpy") != "0 0 0":
        attrib["quat"] = _rpy_to_quat(origin.get("rpy"))
    body = ET.SubElement(parent, "body", attrib)
    if joint is not None and joint.get("type") in {"revolute", "continuous", "prismatic"}:
        j = {"name": joint.get("name"), "type": "hinge" if joint.get("type") != "prismatic" else "slide"}
        axis = joint.find("axis")
        limit = joint.find("limit")
        if axis is not None:
            j["axis"] = axis.get("xyz", "0 0 1")
        if limit is not None and limit.get("lower") is not None:
            j["range"] = f"{limit.get('lower')} {limit.get('upper')}"
        ET.SubElement(body, "joint", j)
    _add_inertial(body, links[name].find("inertial"))
    for kind, cls in (("collision", "collision"), ("visual", "visual")):
        for element in links[name].findall(kind):
            geom = _geometry(element.find("geometry"), element.find("origin"), cls, element.find("material"))
            if geom is not None:
                body.append(geom)
    for child_joint in joints_by_parent.get(name, []):
        _add_body(body, _link_ref(child_joint, "child"), links, joints_by_parent, children, materials)
    return body


def _link_ref(joint, element):
    link = joint.find(element)
    return link.get("link") if link is not None else None


def _geometry(geometry, origin, cls, material):
    if geometry is None:
        return None
    shape = next(iter(geometry), None)
    if shape is None:
        return None
    tag, values = shape.tag, shape.attrib
    attrib = {"class": cls, "type": tag}
    if tag == "cylinder":
        attrib["size"] = f"{values.get('radius', '0')} {float(values.get('length', 0)) / 2}"
    elif tag == "sphere":
        attrib["size"] = values.get("radius", "0")
    elif tag == "box":
        attrib["size"] = " ".join(str(float(v) / 2) for v in values.get("size", "0 0 0").split())
    else:
        return None
    if origin is not None:
        attrib["pos"] = origin.get("xyz", "0 0 0")
        if origin.get("rpy") != "0 0 0":
            attrib["quat"] = _rpy_to_quat(origin.get("rpy"))
    if cls == "visual" and material is not None:
        attrib["material"] = material.get("name", "default_material")
    return ET.Element("geom", attrib)


def _add_inertial(body, inertial):
    if inertial is None:
        return
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        return
    attrib = {
        "mass": mass.get("value", "0"),
        "diaginertia": " ".join(inertia.get(k, "0") for k in ("ixx", "iyy", "izz")),
    }
    if origin is not None:
        attrib["pos"] = origin.get("xyz", "0 0 0")
    ET.SubElement(body, "inertial", attrib)


def _rpy_to_quat(rpy):
    roll, pitch, yaw = (float(value) for value in rpy.split())
    cr, sr, cp, sp, cy, sy = cos(roll / 2), sin(roll / 2), cos(pitch / 2), sin(pitch / 2), cos(yaw / 2), sin(yaw / 2)
    return f"{cr * cp * cy + sr * sp * sy} {sr * cp * cy - cr * sp * sy} {cr * sp * cy + sr * cp * sy} {cr * cp * sy - sr * sp * cy}"


def _write_joint_dynamics(
    root: ET.Element, base_kp: float, noise_std: float, friction: float, armature: float, effort: float, rng: Random
) -> int:
    count = 0
    worldbody = root.find("worldbody")
    if worldbody is None:
        return count
    for joint in worldbody.iter("joint"):
        if joint.get("type") not in {"hinge", "slide", None}:  # None == MJCF default (hinge)
            continue
        name = joint.get("name", "")
        try:
            level = branch_level(name)
        except ValueError:
            continue
        kp, kd = rud_deflection_param(level, base_kp, noise_std)
        joint.set("stiffness", str(kp))
        joint.set("damping", str(kd))
        joint.set("frictionloss", str(friction))
        joint.set("armature", str(armature))
        joint.set("actuatorfrcrange", f"{-effort} {effort}")
        count += 1
    return count


def _write_contacts(
    root: ET.Element,
    fixed_children: set[str],
    contype: int,
    conaffinity: int,
    collision_geom_group: int,
) -> int:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return 0
    count = 0
    for body in worldbody.iter("body"):
        name = body.get("name", "")
        for geom in body.findall("geom"):
            if geom.get("class") != "collision" and geom.get("group") != str(collision_geom_group):
                continue  # visual-only geom, leave untouched
            if name in fixed_children:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            else:
                geom.set("contype", str(contype))
                geom.set("conaffinity", str(conaffinity))
            count += 1
    return count
