from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


def _urdf_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Return a stable, non-empty list of URDF paths.

    A directory is searched recursively because generated variants may be
    grouped below several dataset/configuration directories.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    result = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            result.extend(path.rglob("*.urdf"))
        elif path.suffix == ".urdf":
            result.append(path)
        else:
            raise ValueError(f"Expected a URDF file or directory, got: {path}")

    result = sorted({path.resolve() for path in result})
    if not result:
        raise FileNotFoundError("No URDF files found")
    return result


def _load_model(path: Path) -> mujoco.MjModel:
    """Load one URDF; kept top-level so it is easy to profile or replace."""
    return mujoco.MjModel.from_xml_path(str(path))


def _make_batched_trees_from_models(models, *, nconmax=None):
    """Create one MJWarp batch from models with identical topology."""
    base = models[0]

    # Verify that every tree has identical topology.
    counts = ("nq", "nv", "nbody", "njnt", "ngeom", "nu")
    structure = (
        "body_parentid",
        "body_jntnum",
        "body_jntadr",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "dof_jntid",
        "dof_parentid",
        "geom_type",
        "geom_bodyid",
    )

    for model in models[1:]:
        for name in counts:
            if getattr(model, name) != getattr(base, name):
                raise ValueError(f"Tree topology differs: {name}")

        for name in structure:
            if not np.array_equal(getattr(model, name), getattr(base, name)):
                raise ValueError(f"Tree topology differs: {name}")

    # Warp's primitive narrowphase is unstable for the sphere contacts in
    # these generated trees on some CUDA/driver combinations.  An ellipsoid
    # with three equal radii is geometrically identical to a sphere, and its
    # contacts use Warp's convex path instead.
    for item in models:
        sphere = item.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
        item.geom_type[sphere] = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        item.geom_size[sphere] = np.repeat(item.geom_size[sphere, :1], 3, axis=1)

    # MuJoCo Warp does not support multicontact for cylinder-cylinder pairs,
    # which occur in the generated branch geometry.  Leaving MULTICCD enabled
    # makes Warp enter the unsupported primitive narrowphase kernel and can
    # result in an illegal CUDA memory access.
    base.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    model = mjw.put_model(base)
    data_kwargs = {} if nconmax is None else {"nconmax": nconmax}
    data = mjw.make_data(base, nworld=len(models), **data_kwargs)

    def stack(name):
        return np.stack([getattr(item, name) for item in models])

    # Default configuration.
    model.qpos0 = wp.array(stack("qpos0"), dtype=float)
    model.qpos_spring = wp.array(stack("qpos_spring"), dtype=float)

    # Branch transforms and inertial properties.
    model.body_pos = wp.array(stack("body_pos"), dtype=wp.vec3)
    model.body_quat = wp.array(stack("body_quat"), dtype=wp.quat)
    model.body_ipos = wp.array(stack("body_ipos"), dtype=wp.vec3)
    model.body_iquat = wp.array(stack("body_iquat"), dtype=wp.quat)
    model.body_mass = wp.array(stack("body_mass"), dtype=float)
    model.body_subtreemass = wp.array(stack("body_subtreemass"), dtype=float)
    model.body_inertia = wp.array(stack("body_inertia"), dtype=wp.vec3)
    model.body_invweight0 = wp.array(stack("body_invweight0"), dtype=wp.vec2)

    # Joint placement.
    model.jnt_pos = wp.array(stack("jnt_pos"), dtype=wp.vec3)
    model.jnt_axis = wp.array(stack("jnt_axis"), dtype=wp.vec3)
    model.dof_invweight0 = wp.array(stack("dof_invweight0"), dtype=float)

    # Branch geometry.
    model.geom_size = wp.array(stack("geom_size"), dtype=wp.vec3)
    model.geom_pos = wp.array(stack("geom_pos"), dtype=wp.vec3)
    model.geom_quat = wp.array(stack("geom_quat"), dtype=wp.quat)
    model.geom_rbound = wp.array(stack("geom_rbound"), dtype=float)
    model.geom_rgba = wp.array(stack("geom_rgba"), dtype=wp.vec4)

    geom_aabb = np.stack(
        [item.geom_aabb.reshape(base.ngeom, 2, 3) for item in models]
    )
    model.geom_aabb = wp.array(geom_aabb, dtype=wp.vec3)

    # make_data() initially used only the base tree, so reset it after batching.
    mjw.reset_data(model, data)
    mjw.forward(model, data)

    return base, model, data


_TOPOLOGY_FIELDS = (
    "body_parentid",
    "body_jntnum",
    "body_jntadr",
    "jnt_type",
    "jnt_bodyid",
    "jnt_qposadr",
    "jnt_dofadr",
    "dof_jntid",
    "dof_parentid",
    "geom_type",
    "geom_bodyid",
)


def _topology_key(model):
    counts = ("nq", "nv", "nbody", "njnt", "ngeom", "nu")
    return tuple(
        [getattr(model, name) for name in counts]
        + [getattr(model, name).tobytes() for name in _TOPOLOGY_FIELDS]
    )


def make_batched_tree_groups(
    paths: str | Path | Iterable[str | Path],
    *,
    max_workers: int | None = None,
    nconmax: int | None = None,
):
    """Load all URDFs concurrently and batch each compatible topology group.

    MuJoCo Warp requires one structural layout per batch. Generated datasets
    can contain more than one layout, so this function returns one
    ``(model_cpu, model, data)`` tuple per layout while still loading every
    URDF concurrently.
    """
    paths = _urdf_paths(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        models = list(executor.map(_load_model, paths))

    groups = {}
    for model in models:
        groups.setdefault(_topology_key(model), []).append(model)
    return [
        _make_batched_trees_from_models(group, nconmax=nconmax)
        for group in groups.values()
    ]


def make_batched_trees(
    paths: str | Path | Iterable[str | Path],
    *,
    max_workers: int | None = None,
    nconmax: int | None = None,
):
    """Create one MJWarp batch from supplied URDFs with one topology.

    For a directory containing multiple topology families, use
    :func:`make_batched_tree_groups` instead.
    """
    groups = make_batched_tree_groups(
        paths,
        max_workers=max_workers,
        nconmax=nconmax,
    )
    if len(groups) != 1:
        raise ValueError(
            f"Found {len(groups)} incompatible tree topologies; "
            "use make_batched_tree_groups() to load them all"
        )
    return groups[0]


if __name__ == "__main__":
    # Warp 1.16's CUDA convex-CCD kernel currently fails for these dense tree
    # collision sets.  Keep the loader runnable while leaving contacts
    # available to callers via make_batched_tree_groups(..., nconmax=...).
    batches = make_batched_tree_groups("generated", nconmax=0)

    # Advances every generated tree together.
    while True:
        for _, model, data in batches:
            mjw.step(model, data)
