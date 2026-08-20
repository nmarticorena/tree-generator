"""Serialize physical tree nodes to URDF."""

import copy
import xml.etree.ElementTree as ET
from collections import deque
from math import pi
from pathlib import Path
from types import SimpleNamespace

from forsym.tree.domain import Fruit, JointType, TreeBranch

def should_prune_collision(node: TreeBranch) -> bool:
    """Return whether a fixed branch should be visual-only."""
    return (
        node.joint_type == JointType.fixed
        and node.parent is not None
        and node.parent.parent is not None
    )

def gen_urdf(graph, output_path, prune_collision: bool = True):
    """Write a branch-and-fruit hierarchy as a URDF tree.

    Parameters
    ----------
    graph : forsym.tree.domain.TreeBranch
        Root of the physical tree hierarchy.
    output_path : str or path-like
        Destination URDF path.
    """

    outline, templates = gen_outline()
    queue = deque([graph])
    while queue:
        node = queue.popleft()
        add_child(node, outline, templates,prune_collision)
        queue.extend(node.children)

    _write_urdf(outline, output_path)


def add_child(node, outline, templates, prune_collision: bool = True):
    """Append one branch or fruit to a partially assembled URDF.

    Parameters
    ----------
    node : TreeBranch or Fruit
        Physical component to serialize.
    outline : xml.etree.ElementTree.Element
        URDF robot element updated in place.
    templates : types.SimpleNamespace
        Branch, spherical-joint, and fruit templates.
    prune_collision : bool, default=False
        Remove collision.

    Raises
    ------
    TypeError
        If ``node`` is not a supported physical component.
    """

    if isinstance(node, TreeBranch):
        joints = add_joint(node, templates.joint, templates.spherical)
        link = add_link(node, templates.link)
        if (
            prune_collision 
            and should_prune_collision(node)
        ):
            _remove_element(link, "./collision")
    elif isinstance(node, Fruit):
        joints = add_fruit_joint(node, templates.fruit_joint)
        link = add_fruit_link(node, templates.fruit_link)
    else:
        raise TypeError(f"Unsupported tree component: {type(node).__name__}")

    for joint in joints:
        outline.append(joint)

    outline.append(link)


def add_fruit_joint(node, fruit_joint_template):
    """Create the two rotational joints and breakaway joint for a fruit.

    Parameters
    ----------
    node : Fruit
        Fruit providing its parent, pose, and unique identifier.
    fruit_joint_template : xml.etree.ElementTree.Element
        Template containing the fruit joint chain and dummy links.

    Returns
    -------
    list of xml.etree.ElementTree.Element
        Fruit joints and their uniquely named dummy links.
    """
    joint_set = copy.deepcopy(fruit_joint_template)
    x_link_name = f"fr-x-dummy-link-{node.idx}"
    y_link_name = f"fr-y-dummy-link-{node.idx}"
    for joint in joint_set.iter("joint"):
        default_name = joint.get("name")
        joint.set("name", f"{default_name}-{node.parent.name}-to-{node.name}")
        if "-x-" in default_name:
            joint.set("type", JointType.revolute.name)
            _set_attribute(joint, "./parent", "link", f"link-{node.parent.name}")
            origin = " ".join(str(value) for value in node.fruit_origin_xyz)
            angle = " ".join(str(value) for value in node.fruit_angle_rpy)
            _set_attribute(joint, "./origin", "xyz", origin)
            _set_attribute(joint, "./origin", "rpy", angle)
            _set_attribute(joint, "./child", "link", x_link_name)
        elif "-y-" in default_name:
            joint.set("type", JointType.revolute.name)
            _set_attribute(joint, "./parent", "link", x_link_name)
            _set_attribute(joint, "./child", "link", y_link_name)
        elif "-prismatic-" in default_name:
            joint.set("type", JointType.prismatic.name)
            _set_attribute(joint, "./parent", "link", y_link_name)
            _set_attribute(joint, "./child", "link", f"link-{node.name}")

    for link in joint_set.iter("link"):
        if "-x-" in link.get("name"):
            link.set("name", x_link_name)
        elif "-y-" in link.get("name"):
            link.set("name", y_link_name)
    return list(joint_set)


def add_joint(node, joint_template, spherical_template):
    """Create the URDF joint elements for one branch.

    Parameters
    ----------
    node : forsym.tree.domain.TreeBranch
        Branch connected by the generated joint.
    joint_template : xml.etree.ElementTree.Element
        Template for fixed and revolute joints.
    spherical_template : xml.etree.ElementTree.Element
        Template for the supported two-axis spherical approximation.

    Returns
    -------
    list of xml.etree.ElementTree.Element
        Joint elements and any required dummy link.
    """
    if node.parent is None:
        return [_trunk_joint(node, joint_template)]
    if node.joint_type == JointType.spherical:
        return _spherical_joints(node, spherical_template)
    return [_branch_joint(node, joint_template)]


def _trunk_joint(node, template):
    joint = copy.deepcopy(template)
    joint.set("name", f"joint-world-{node.name}")
    joint.set("type", node.joint_type.name)
    _remove_element(joint, "./axis")
    _remove_element(joint, "./limit")
    _remove_element(joint, "dynamics")
    _set_attribute(joint, "parent", "link", "world")
    _set_attribute(joint, "child", "link", f"link-{node.name}")
    return joint


def _spherical_joints(node, template):
    joint_set = copy.deepcopy(template)
    x_link_name = f"sp-x-dummy-link-{node.idx}"
    origin, angle, _ = calc_joint_origin(node)
    for joint in joint_set.iter("joint"):
        default_name = joint.get("name")
        joint.set("name", f"{default_name}-{node.parent.name}-to-{node.name}")
        joint.set("type", JointType.revolute.name)
        if "-x-" in default_name:
            _set_attribute(joint, "./parent", "link", f"link-{node.parent.name}")
            _set_attribute(joint, "./child", "link", x_link_name)
            _set_attribute(joint, "./origin", "xyz", origin)
            _set_attribute(joint, "./origin", "rpy", angle)
        else:
            _set_attribute(joint, "./parent", "link", x_link_name)
            _set_attribute(joint, "./child", "link", f"link-{node.name}")
    joint_set.find("link").set("name", x_link_name)
    return list(joint_set)


def _branch_joint(node, template):
    joint = copy.deepcopy(template)
    joint.set("name", f"joint-{node.parent.name}-to-{node.name}")
    joint.set("type", node.joint_type.name)
    _set_attribute(joint, "./parent", "link", f"link-{node.parent.name}")
    _set_attribute(joint, "./child", "link", f"link-{node.name}")
    origin, angle, axis = calc_joint_origin(node)
    _set_attribute(joint, "./origin", "xyz", origin)
    _set_attribute(joint, "./origin", "rpy", angle)
    _set_attribute(joint, "./axis", "xyz", axis)
    limit = round(pi / 9, 3)
    _set_attribute(joint, "./limit", "lower", str(-limit))
    _set_attribute(joint, "./limit", "upper", str(limit))
    return joint


def calc_joint_origin(node):
    """Format a branch joint's origin and rotation axis for URDF.

    Parameters
    ----------
    node : forsym.tree.domain.TreeBranch
        Child branch whose parent determines the joint position.

    Returns
    -------
    origin_xyz : str
        Space-separated joint position.
    origin_rpy : str
        Space-separated roll, pitch, and yaw angles.
    rotation_axis : str
        Space-separated revolute axis.
    """
    joint_origin_z = round((node.parent.length), 5)

    branch_angle_rpy = node.branch_angle_rpy
    branch_rotation_axis = node.branch_rotation_axis

    joint_origin_xyz = f"0.0 0.0 {joint_origin_z}"
    joint_origin_rpy = f"{branch_angle_rpy[0]} {branch_angle_rpy[1]} {branch_angle_rpy[2]}"
    joint_rotation_axis = f"{branch_rotation_axis[0]} {branch_rotation_axis[1]} {branch_rotation_axis[2]}"

    return joint_origin_xyz, joint_origin_rpy, joint_rotation_axis


def add_fruit_link(node, fruit_link_template):
    """Create the spherical link for one fruit.

    Parameters
    ----------
    node : Fruit
        Fruit providing its name and radius.
    fruit_link_template : xml.etree.ElementTree.Element
        Link template copied and populated for the fruit.

    Returns
    -------
    xml.etree.ElementTree.Element
        Populated fruit link.
    """
    link = copy.deepcopy(fruit_link_template)
    link.set("name", f"link-{node.name}")
    _set_attribute(link, "./visual/geometry/sphere", "radius", str(node.radius))
    _set_attribute(link, "./collision/geometry/sphere", "radius", str(node.radius))
    return link


def add_link(node, link_template):
    """Create the URDF link element for a physical branch.

    Parameters
    ----------
    node : forsym.tree.domain.TreeBranch
        Branch providing geometry and material values.
    link_template : xml.etree.ElementTree.Element
        Link element copied and populated for the branch.

    Returns
    -------
    xml.etree.ElementTree.Element
        Populated link element.
    """
    link = copy.deepcopy(link_template)
    center = f"0 0 {round(node.length / 2, 3)}"
    link.set("name", f"link-{node.name}")
    _set_attribute(link, "./visual/geometry/cylinder", "length", str(node.length))
    _set_attribute(link, "./visual/geometry/cylinder", "radius", str(node.radius))
    _set_attribute(link, "./visual/origin", "xyz", center)

    material_name, material_rgba = _material(node)
    _set_attribute(link, "./visual/material", "name", material_name)
    _set_attribute(link, "./visual/material/color", "rgba", material_rgba)

    _set_attribute(link, "./collision/geometry/cylinder", "length", str(node.length))
    _set_attribute(link, "./collision/geometry/cylinder", "radius", str(node.radius))
    _set_attribute(link, "./collision/origin", "xyz", center)

    return link


def _material(node):
    return node.material_name, " ".join(map(str, node.material_rgba))


def _remove_element(parent, elem_name):
    element = parent.find(elem_name)
    if element is not None:
        parent.remove(element)


def _set_attribute(parent, elem_name, elem_key, elem_val):
    parent.find(elem_name).set(elem_key, elem_val)


def gen_outline():
    """Load the fixed templates used by the tree URDF writer.

    Returns
    -------
    outline : xml.etree.ElementTree.Element
        Base robot element.
    templates : tuple
        Branch, two-axis spherical-joint, and fruit templates.
    """
    outline, joint_template, link_template = gen_basic_outline()
    fruit_joint_template, fruit_link_template = gen_fruit_outline()
    templates = SimpleNamespace(
        joint=joint_template,
        link=link_template,
        spherical=gen_spherical_outline(),
        fruit_joint=fruit_joint_template,
        fruit_link=fruit_link_template,
    )
    return outline, templates


def gen_basic_outline():
    """Extract reusable elements from the base tree template.

    Returns
    -------
    root : xml.etree.ElementTree.Element
        Robot element with placeholder components removed.
    joint : xml.etree.ElementTree.Element
        Basic joint template.
    link : xml.etree.ElementTree.Element
        Cylindrical branch link template.
    """
    template = ET.parse(_template_path("tree.urdf"))
    root = template.getroot()

    link = root.find("link[@name='default_link_name']")
    joint = root.find("joint[@name='default_joint_name']")
    link_template = copy.deepcopy(link)
    joint_template = copy.deepcopy(joint)
    root.remove(link)
    root.remove(joint)

    return root, joint_template, link_template


def gen_spherical_outline():
    """Extract the supported two-axis spherical joint template.

    Returns
    -------
    xml.etree.ElementTree.Element
        Spherical joint set containing two revolute joints and a dummy link.
    """
    root = ET.parse(_template_path("spherical_joint.urdf")).getroot()
    return copy.deepcopy(root.find("spherical_joint_set[@name='spherical_joint']"))


def gen_fruit_outline():
    """Extract reusable fruit joint and link templates.

    Returns
    -------
    joint_set : xml.etree.ElementTree.Element
        Two rotational joints, one prismatic joint, and dummy links.
    link : xml.etree.ElementTree.Element
        Spherical fruit link.
    """
    root = ET.parse(_template_path("fruit.urdf")).getroot()
    joint_set = copy.deepcopy(root.find("fruit_joint_set[@name='fruit_joint']"))
    link = copy.deepcopy(root.find("link[@name='default_fruit_link_name']"))
    return joint_set, link


def _template_path(name):
    return Path(__file__).resolve().parent / "templates" / name


def _write_urdf(element, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(element)
    ET.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
