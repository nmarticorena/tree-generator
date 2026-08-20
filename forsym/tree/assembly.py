"""Convert interpreted turtle geometry into physical tree nodes."""

from typing_extensions import Generator
from random import Random
import numpy as np
from forsym.tree.config import TreeConfig, LSystemConfig

import copy

from forsym.fractal.turtle import TurtleBranch
from forsym.tree.domain import Fruit, JointType, TreeBranch
from forsym.utils import rotation


def get_joint_type(tree_branch_parent: TreeBranch, t_config: TreeConfig, child_cnt: int) -> JointType:
    """Choose the joint connecting a branch to its parent.

    Parameters
    ----------
    tree_branch_parent : TreeBranch
        Parent branch in the assembled tree.
    t_config : forsym.tree.config.TreeConfig
        Configured revolute and spherical subtree roots.
    child_cnt : int
        Number of child branches below the new node.

    Returns
    -------
    JointType
        Spherical, revolute, or fixed joint type for the new branch.
    """

    if child_cnt and (
        tree_branch_parent.name == t_config.flex_root or tree_branch_parent.joint_type == JointType.spherical
    ):
        return JointType.spherical

    if (
        tree_branch_parent.name == t_config.dof_root
        or tree_branch_parent.joint_type == JointType.revolute
        or tree_branch_parent.joint_type == JointType.spherical
    ):
        return JointType.revolute

    return JointType.fixed


def add_fruits(root: TreeBranch, t_config: TreeConfig, rng: Random):
    """Attach configured fruit nodes to selected branches.

    Parameters
    ----------
    root : TreeBranch
        Root of the physical branch hierarchy.
    t_config : forsym.tree.config.TreeConfig
        Fruit count and selected branch name.
    rng : random.Random
        Local random-number generator used to place fruit.

    Returns
    -------
    int
        Number of fruit nodes added.
    """
    fruit_index = 0
    for branch in [root, *root.descendants]:
        if branch.name != t_config.fruit_branch:
            continue
        locations = sample_fruit_locations(round(branch.length - 0.01, 3), t_config.fruit_count, rng)
        for location in locations:
            x, y = _random_xy((0.03, 0.01), rng)
            Fruit(
                idx=fruit_index,
                origin=(x, y, location),
                angle=(0.0, 0.0, 0.0),
                parent=branch,
            )
            fruit_index += 1
    return fruit_index


def sample_fruit_locations(length: float, count: int, rng: Random) -> list[float]:
    """Sample fruit positions from the outer thirds of a branch.

    Parameters
    ----------
    length : float
        Available branch length.
    count : int
        Number of positions to sample.
    rng : random.Random
        Local random-number generator.

    Returns
    -------
    list of float
        Distances from the branch origin.

    Raises
    ------
    ValueError
        If ``length`` or ``count`` is not positive.
    """
    if length <= 0 or count <= 0:
        raise ValueError(f"Fruit length and count must be positive, got {length} and {count}")

    third = length / 3
    return [rng.uniform(0, third) if rng.choice((0, 2)) == 0 else rng.uniform(2 * third, length) for _ in range(count)]


def _random_xy(options, rng):
    x, y = rng.sample(options, 2)
    x *= rng.choice((-1, 1))
    y *= rng.choice((-1, 1))
    return (x, y) if rng.choice((True, False)) else (y, x)


def gen_branch_graph(t_root: TurtleBranch, t_config: TreeConfig) -> TreeBranch:
    """Convert turtle geometry into a physical tree hierarchy.

    Parameters
    ----------
    t_root : forsym.fractal.turtle.TurtleBranch
        Root of the interpreted L-system geometry.
    t_config : forsym.tree.config.TreeConfig
        Length, radius, and joint configuration.

    Returns
    -------
    TreeBranch
        Root of the physical branch hierarchy.
    """

    def _length(node):
        start = node.turtle_line.start
        end = node.turtle_line.end
        return start.distance_to(end)

    def _radius(node):
        return node.turtle_line.width / 2.0

    trunk_length = _length(t_root) * t_config.length_scale
    trunk_radius = _radius(t_root) * t_config.radius_scale

    trunk = TreeBranch(
        idx=0,
        length=trunk_length,
        radius=trunk_radius,
        branch_angle_rpy=(0, 0, 0),
        branch_rotation_axis=(0, 0, 0),
        parent=None,
        joint_type=JointType.fixed,
    )

    branch_store = {t_root: trunk}

    for index, turtle_branch in enumerate(t_root.descendants, start=1):
        parent_turtle = turtle_branch.parent
        angle = rotation.calculate_rpy(
            l1_start=parent_turtle.turtle_line.start,
            l1_end=parent_turtle.turtle_line.end,
            l2_start=turtle_branch.turtle_line.start,
            l2_end=turtle_branch.turtle_line.end,
        )
        parent_branch = branch_store[parent_turtle]
        branch = TreeBranch(
            idx=index,
            length=_length(turtle_branch) * t_config.length_scale,
            radius=_radius(turtle_branch) * t_config.radius_scale,
            branch_angle_rpy=angle,
            branch_rotation_axis=(0, 1, 0),
            parent=parent_branch,
            joint_type=get_joint_type(parent_branch, t_config, len(turtle_branch.children)),
        )

        branch_store[turtle_branch] = branch

    return trunk


def iter_lsystem_configs(l_config: LSystemConfig, max_trees: int, rng: np.random.Generator) -> Generator[LSystemConfig]:
    """Yield the configured L-system parameter sets.

    Parameters
    ----------
    l_config : forsym.tree.config.LSystemConfig
        Base L-system settings, output count, and relative sampling spread.
    rng : numpy.random.Generator
        Local NumPy random-number generator.

    Yields
    ------
    LSystemConfig
        Base configuration when one tree is requested, otherwise an
        independently sampled copy.

    Raises
    ------
    ValueError
        If the configured sample count is not positive.
    """

    def _sample(value):
        spread = l_config.relative_std * abs(value)
        return round(value + rng.normal(0, spread), 2)

    for _ in range(max_trees):
        l_clone = copy.deepcopy(l_config)
        l_clone.free_params = {name: _sample(value) for name, value in l_config.free_params.items()}
        l_clone.bending = _sample(l_config.bending)
        l_clone.angle_std = _sample(l_config.angle_std)
        l_clone.initial_angle = _sample(l_config.initial_angle)
        yield l_clone
