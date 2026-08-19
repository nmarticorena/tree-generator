"""Public MuJoCo integration for generated ForSym trees."""

import math
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from .visualisation.tree_scene import TREE, add_ground, bounds, load_tree


def build_forest(
    paths: Sequence[str | Path],
    *,
    columns: int | None = None,
    spacing: float | None = None,
    ground: bool = True,
    stiffness: float = 400.0,
    damping: float = 0.2,
    armature: float = 0.01,
    friction: float = 0.01,
    prune_fixed: bool = True,
    self_contacts: bool = False,
) -> tuple[mujoco.MjModel, list[tuple[Path, np.ndarray, np.ndarray]]]:
    """Build a compiled MuJoCo forest from generated tree URDFs.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        Generated tree URDFs to place in the scene.
    columns : int, optional
        Number of grid columns. By default, a near-square grid is used.
    spacing : float, optional
        Distance between tree origins. By default, the largest tree sets it.
    ground : bool, default=True
        Add a ground plane to the scene.
    stiffness : float, default=400.0
        Base stiffness for flexible tree branches.
    damping : float, default=0.2
        Scale used to derive branch damping from stiffness.
    armature : float, default=0.01
        Armature assigned to flexible tree joints.
    friction : float, default=0.01
        Friction loss assigned to flexible tree joints.
    prune_fixed : bool, default=True
        Disable collisions on non-trunk fixed child links.
    self_contacts : bool, default=False
        Preserve collisions between geometries in the same tree.

    Returns
    -------
    model : mujoco.MjModel
        Compiled forest model.
    entities : list of tuple
        Source path, world-space center, and size for each tree.

    Raises
    ------
    ValueError
        If no paths are supplied or a layout or dynamics value is invalid.
    FileNotFoundError
        If a source path does not exist.
    """
    if not paths:
        raise ValueError("At least one tree is required")
    trees = _load_trees(paths, prune_fixed, self_contacts, stiffness, damping, armature, friction)
    columns, spacing = _layout(trees, columns, spacing)
    spec = _forest_spec(ground, self_contacts)
    entities = _attach_trees(spec, trees, _grid(len(trees), columns, spacing))
    return spec.compile(), entities


def _load_trees(paths, prune_fixed, self_contacts, stiffness, damping, armature, friction):
    contacts = None if self_contacts else (TREE, 2)
    trees = []
    for path in paths:
        path, spec = load_tree(
            path,
            contacts=contacts,
            prune_fixed=prune_fixed,
            stiffness=stiffness,
            damping=damping,
            armature=armature,
            friction=friction,
        )
        lower, upper = bounds(spec.compile())
        trees.append((path, spec, (lower + upper) / 2.0, np.maximum(upper - lower, 0.05)))
    return trees


def _layout(trees, columns, spacing) -> tuple[int, float]:
    columns = math.ceil(math.sqrt(len(trees))) if columns is None else columns
    if columns < 1:
        raise ValueError("columns must be at least 1")
    span = max(float(np.max(size[:2])) for _, _, _, size in trees)
    spacing = max(1.25 * span, 0.25) if spacing is None else spacing
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return columns, spacing


def _forest_spec(ground: bool, self_contacts: bool) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.compiler.discardvisual = False
    spec.modelname = "forsym_forest"
    if ground:
        mask = 1 if self_contacts else TREE
        add_ground(spec, name="forest_ground", contype=2 if not self_contacts else 1, conaffinity=mask)
    return spec


def _grid(count: int, columns: int, spacing: float) -> np.ndarray:
    rows = math.ceil(count / columns)
    origins = np.zeros((count, 3))
    for index in range(count):
        row, column = divmod(index, columns)
        origins[index, :2] = (column - (columns - 1) / 2, (rows - 1) / 2 - row)
    return origins * spacing


def _attach_trees(spec, trees, origins):
    entities = []
    for index, ((path, tree, center, size), origin) in enumerate(zip(trees, origins, strict=True)):
        prefix = f"tree_{index}_"
        frame = spec.worldbody.add_frame(name=f"{prefix}frame", pos=origin)
        spec.attach(tree, frame=frame, prefix=prefix)
        entities.append((path, center + origin, size * 1.05))
    return entities


def export_tree_mjcf(
    source: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Export a generated tree with its MuJoCo settings encoded as MJCF.

    Parameters
    ----------
    source : str or pathlib.Path
        Generated tree URDF to load.
    destination : str or pathlib.Path, optional
        Output MJCF path. By default, the file is written beside ``source``
        with a ``_processed.xml`` suffix.

    Returns
    -------
    pathlib.Path
        Absolute path to the exported MJCF file.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    """
    source_path, spec = load_tree(source)
    if destination is None:
        destination = source_path.with_name(f"{source_path.stem}_processed.xml")
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spec.to_xml())
    return destination


__all__ = ["build_forest", "export_tree_mjcf", "load_tree"]
