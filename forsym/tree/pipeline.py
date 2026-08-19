"""Assemble expanded L-systems into tree URDFs."""

import copy
from pathlib import Path

from forsym.fractal import turtle
from forsym.tree import assembly, writer


def select_flexible_root(t_root, tree_config, rng):
    """Select the flexible subtree and its fruit-bearing branch.

    Parameters
    ----------
    t_root : forsym.fractal.turtle.TurtleBranch
        Root of the generated turtle hierarchy.
    tree_config : forsym.tree.config.TreeConfig
        Tree settings used for the provisional assembly.
    rng : random.Random
        Local random-number generator used for branch selection.

    Returns
    -------
    flexible_root : str
        Name of the branch at the root of the flexible subtree.
    fruit_branch : str
        Name of the branch selected to carry the configured fruits.

    Raises
    ------
    ValueError
        If no suitable subtree exists or it does not have 198 degrees of
        freedom.
    """
    mock_trunk = assembly.gen_branch_graph(t_root, tree_config)
    branches = [mock_trunk, *mock_trunk.descendants]
    candidates = [(node, len(node.descendants)) for node in branches if 40 <= len(node.descendants) <= 120]
    if not candidates:
        raise ValueError("The generated tree has no suitable flexible subtree")

    flexible_root, branch_count = rng.choice(candidates)
    branching_descendants = sum(bool(node.children) for node in flexible_root.descendants)
    total_dofs = branch_count + 2 * branching_descendants
    if total_dofs != 198:
        raise ValueError(f"The selected flexible subtree has {total_dofs} DOFs; expected 198")

    fruit_branch = rng.choice([flexible_root, *flexible_root.descendants])
    return flexible_root.name, fruit_branch.name


def generate_tree_urdf(index, l_string, l_config, tree_config, output_pattern, output_root, rng):
    """Assemble one expanded L-system and write its tree URDF.

    Parameters
    ----------
    index : int
        Index used to format the YAML output pattern.
    l_string : str
        Expanded L-system to assemble.
    l_config : forsym.tree.config.LSystemConfig
        L-system settings used to interpret the string.
    tree_config : forsym.tree.config.TreeConfig
        Base tree settings copied for this asset.
    output_pattern : str
        Relative output pattern from the YAML configuration.
    output_root : str or pathlib.Path
        Root directory for generated files.
    rng : random.Random
        Local random-number generator used for geometry and fruit placement.

    Returns
    -------
    pathlib.Path
        Path to the generated URDF.
    """
    tree_config = copy.deepcopy(tree_config)
    output_path = Path(output_root) / output_pattern.format(index=index)

    turtle_lines = turtle.l_string_to_turtle_lines(l_string, l_config, rng)
    t_root = turtle.turtle_lines_to_branches(turtle_lines)

    flexible_root, fruit_branch = select_flexible_root(t_root, tree_config, rng)
    tree_config.dof_root = flexible_root
    tree_config.flex_root = flexible_root
    tree_config.fruit_branch = fruit_branch

    trunk = assembly.gen_branch_graph(t_root, tree_config)
    assembly.add_fruits(trunk, tree_config, rng)
    writer.gen_urdf(trunk, output_path)
    return output_path
