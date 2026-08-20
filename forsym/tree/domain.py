"""Physical branch and fruit nodes."""

from enum import Enum

from anytree import NodeMixin


class JointType(str, Enum):
    revolute = "revolute"
    fixed = "fixed"
    spherical = "spherical"
    prismatic = "prismatic"


class TreeBranch(NodeMixin):
    def __init__(
        self,
        idx,
        length,
        radius,
        branch_angle_rpy,
        branch_rotation_axis,
        joint_type: JointType,
        parent=None,
    ):
        super().__init__()

        self.length = length
        self.radius = radius

        self.parent = parent

        # if the joint from this branch to parent is fixed/revolute. Ignored for trunk.
        self.joint_type = joint_type

        if self.joint_type == JointType.fixed:
            self.material_rgba, self.material_name = [0.1, 0.2, 0.2, 1], "brown"
        else:
            self.material_rgba, self.material_name = [0.1, 0.8, 0.2, 1], "green"

        self.branch_angle_rpy = branch_angle_rpy
        self.branch_rotation_axis = branch_rotation_axis

        self.idx = idx
        self.level = self.parent.level + 1 if parent is not None else 0
        self.name = (
            f"branch-B{self.idx}L{self.level}P{parent.idx}" if parent is not None else f"trunk-B{self.idx}L{self.level}"
        )


class Fruit(NodeMixin):
    """A fruit attached to one generated branch."""

    def __init__(self, idx, origin, angle, parent):
        super().__init__()
        self.idx = idx
        self.fruit_origin_xyz = origin
        self.fruit_angle_rpy = angle
        self.radius = 0.02
        self.parent = parent
        self.level = parent.level + 1
        self.name = f"fruit-F{idx}T{self.level}P{parent.idx}"
