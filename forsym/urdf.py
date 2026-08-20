"""Portable post-processing for generated ForSym URDF trees."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

_LEVEL = re.compile(r"to-branch-[^-]*?L(?P<level>\d+)P")


def postprocess_tree_urdf(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    prune_fixed: bool = True,
    stiffness: float = 400.0,
    damping: float = 0.2,
    friction: float = 0.01,
) -> Path:
    """Write a simulation-ready URDF using the ForSym tree policy.

    Parameters
    ----------
    source : str or pathlib.Path
        Generated tree URDF to process.
    destination : str or pathlib.Path, optional
        Output path. By default, the file is written beside ``source`` with a
        ``_processed.urdf`` suffix.
    prune_fixed : bool, default=True
        Remove collision geometry from non-trunk links joined by fixed joints.
    stiffness : float, default=400.0
        Base spring stiffness used to derive level-dependent damping.
    damping : float, default=0.2
        Scale applied to each joint's derived stiffness.
    friction : float, default=0.01
        Friction loss written to flexible URDF joints.

    Returns
    -------
    pathlib.Path
        Absolute path to the processed URDF.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If the URDF structure or a dynamics value is invalid.

    Notes
    -----
    URDF cannot represent spring stiffness, armature, or MuJoCo collision
    masks. Use :func:`forsym.mujoco.export_tree_mjcf` when those settings must
    be serialized.
    """
    source = _source_path(source)
    destination = _destination_path(source, destination)
    _validate_dynamics(stiffness, damping, friction)
    tree = ET.parse(source)
    root = tree.getroot()
    _validate_robot(root)
    if prune_fixed:
        _remove_fixed_collisions(root)
    _write_joint_dynamics(root, stiffness, damping, friction)
    ET.indent(tree, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def fixed_child_links(path: str | Path) -> set[str]:
    """Return non-trunk links attached through fixed joints."""
    joints = ET.parse(path).getroot().findall("joint")
    return {link for joint in joints if (link := _fixed_child(joint))}


def tree_joint_dynamics(name: str, stiffness: float = 400.0, damping: float = 0.2) -> tuple[float, float] | None:
    """Return spring and damper values inferred from a generated joint name.

    Parameters
    ----------
    name : str
        Generated branch or fruit joint name.
    stiffness : float, default=400.0
        Spring stiffness assigned to level-one branches.
    damping : float, default=0.2
        Scale applied to branch stiffness when deriving damping.

    Returns
    -------
    tuple of float or None
        Spring and damping values, or ``None`` for an unrelated joint.
    """
    if "to-fruit-" in name:
        return 5.0, 5.0
    match = _LEVEL.search(name)
    if match is None:
        return None
    spring = max(stiffness / 2 ** (int(match.group("level")) - 1), 2.0)
    return spring, max(spring * damping, 2.0)


def _source_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tree URDF does not exist: {path}")
    return path


def _destination_path(source: Path, destination: str | Path | None) -> Path:
    if destination is None:
        return source.with_name(f"{source.stem}_processed.urdf")
    return Path(destination).expanduser().resolve()


def _validate_robot(root: ET.Element) -> None:
    if root.tag != "robot":
        raise ValueError("URDF root element must be <robot>")
    links = {link.get("name") for link in root.findall("link")}
    if None in links or len(links) != len(root.findall("link")):
        raise ValueError("URDF links must have unique names")
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None or parent.get("link") not in links or child.get("link") not in links:
            raise ValueError(f"Joint {joint.get('name')!r} references a missing link")


def _remove_fixed_collisions(root: ET.Element) -> int:
    links = {link.get("name"): link for link in root.findall("link")}
    children = {link for joint in root.findall("joint") if (link := _fixed_child(joint))}
    collisions = []
    for name in children:
        link = links.get(name)
        if link is not None:
            collisions.extend((link, collision) for collision in link.findall("collision"))
    for link, collision in collisions:
        link.remove(collision)
    return len(collisions)


def _fixed_child(joint: ET.Element) -> str | None:
    if joint.get("type") != "fixed" or "trunk" in joint.get("name", ""):
        return None
    child = joint.find("child")
    return child.get("link") if child is not None else None


def _write_joint_dynamics(root: ET.Element, stiffness: float, damping: float, friction: float) -> int:
    count = 0
    for joint in root.findall("joint"):
        values = tree_joint_dynamics(joint.get("name", ""), stiffness, damping)
        if values is None or joint.get("type") not in {"revolute", "prismatic"}:
            continue
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", str(values[1]))
        dynamics.set("friction", str(friction))
        count += 1
    return count


def _validate_dynamics(stiffness: float, damping: float, friction: float) -> None:
    values = (("stiffness", stiffness), ("damping", damping), ("friction", friction))
    for name, value in values:
        if value < 0 or (name == "stiffness" and value == 0):
            raise ValueError(f"{name} must be {'positive' if name == 'stiffness' else 'non-negative'}")
