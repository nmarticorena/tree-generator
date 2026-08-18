"""Small MuJoCo helpers shared by the interactive viewers."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

TREE = 1
ROBOT = 2
GROUND = 4

_LEVEL = re.compile(r"to-branch-[^-]*?L(?P<level>\d+)P")


def find_urdfs(source: Path, pattern: str = "*.urdf") -> list[Path]:
    """Find one URDF or every matching URDF below a directory."""
    source = source.expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Tree path does not exist: {source}")
    paths = [path for path in sorted(source.rglob(pattern)) if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No {pattern!r} files below {source}")
    return paths


def bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper world-space geometry bounds."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.ngeom == 0:
        return _empty_bounds(model)
    centers, half_sizes = _geom_bounds(model, data)
    return np.min(centers - half_sizes, axis=0), np.max(centers + half_sizes, axis=0)


def _empty_bounds(model):
    radius = max(float(model.stat.extent), 0.025)
    return model.stat.center - radius, model.stat.center + radius


def _geom_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    rotations = data.geom_xmat.reshape(-1, 3, 3)
    centers = data.geom_xpos + np.einsum("nij,nj->ni", rotations, model.geom_aabb[:, :3])
    half_sizes = np.einsum("nij,nj->ni", np.abs(rotations), model.geom_aabb[:, 3:])
    return centers, half_sizes


def load_tree(
    path: Path,
    *,
    contacts: tuple[int, int] | None = None,
    prune_fixed: bool = False,
) -> tuple[Path, mujoco.MjSpec, np.ndarray, np.ndarray]:
    """Load a tree and return its path, spec, center, and size."""
    path = _tree_path(path)
    spec = _tree_spec(path, contacts, prune_fixed)
    lower, upper = bounds(spec.compile())
    return path, spec, (lower + upper) / 2.0, np.maximum(upper - lower, 0.05)


def _tree_path(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tree URDF does not exist: {path}")
    return path


def _tree_spec(path, contacts, prune_fixed):
    spec = mujoco.MjSpec.from_file(str(path))
    spec.compiler.discardvisual = False
    if prune_fixed:
        disable_collisions(spec, fixed_child_links(path))
    if contacts is not None:
        set_contacts(spec, *contacts)
    return spec


def set_contacts(spec: mujoco.MjSpec, contype: int, conaffinity: int) -> int:
    """Assign a collision class to every active geometry."""
    geoms = [geom for geom in spec.geoms if geom.contype or geom.conaffinity]
    for geom in geoms:
        geom.contype, geom.conaffinity = contype, conaffinity
    return len(geoms)


def fixed_child_links(path: Path) -> set[str]:
    """Return non-trunk links attached through fixed joints."""
    joints = ET.parse(path).getroot().findall("joint")
    return {link for joint in joints if (link := _fixed_child(joint))}


def _fixed_child(joint):
    if joint.get("type") != "fixed" or "trunk" in joint.get("name", ""):
        return None
    child = joint.find("child")
    return child.get("link") if child is not None else None


def disable_collisions(spec: mujoco.MjSpec, links: set[str]) -> int:
    """Disable collision geometries belonging to the given links."""
    geoms = [geom for geom in spec.geoms if geom.parent.name in links and (geom.contype or geom.conaffinity)]
    for geom in geoms:
        geom.contype = geom.conaffinity = 0
    return len(geoms)


def tree_joint_dynamics(name: str, stiffness: float = 400.0, damping: float = 0.2) -> tuple[float, float] | None:
    """Return spring and damper values inferred from a generated joint name."""
    if "to-fruit-" in name or "to-leaf-" in name:
        return 5.0, 5.0
    match = _LEVEL.search(name)
    if match is None:
        return None
    spring = max(stiffness / 2 ** (int(match.group("level")) - 1), 2.0)
    return spring, max(spring * damping, 2.0)


def tune_tree(
    model: mujoco.MjModel,
    *,
    prefix: str = "",
    stiffness: float = 400.0,
    damping: float = 0.2,
    armature: float = 0.01,
    friction: float = 0.01,
) -> int:
    """Apply stable passive dynamics to generated tree joints."""
    _validate_dynamics(stiffness, damping, armature, friction)
    joints = list(_tree_joints(model, prefix, stiffness, damping))
    for joint_id, spring, damper in joints:
        _tune_joint(model, joint_id, spring, damper, armature, friction)
    model.qpos_spring[:] = model.qpos0
    return len(joints)


def _tree_joints(model, prefix, stiffness, damping):
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        values = tree_joint_dynamics(name, stiffness, damping) if name.startswith(prefix) else None
        if values is not None:
            yield joint_id, *values


def _validate_dynamics(stiffness: float, damping: float, armature: float, friction: float) -> None:
    values = (("stiffness", stiffness), ("damping", damping), ("armature", armature), ("friction", friction))
    for name, value in values:
        if value < 0 or (name == "stiffness" and value == 0):
            raise ValueError(f"{name} must be {'positive' if name == 'stiffness' else 'non-negative'}")


def _tune_joint(model, joint_id, stiffness, damping, armature, friction) -> None:
    model.jnt_stiffness[joint_id] = stiffness
    start = int(model.jnt_dofadr[joint_id])
    end = int(model.jnt_dofadr[joint_id + 1]) if joint_id + 1 < model.njnt else model.nv
    model.dof_damping[start:end] = damping
    model.dof_armature[start:end] = armature
    model.dof_frictionloss[start:end] = friction


def add_ground(
    spec: mujoco.MjSpec,
    *,
    name: str = "ground",
    contype: int = 1,
    conaffinity: int = 1,
    color=(0.25, 0.3, 0.25, 1.0),
) -> None:
    """Add a simple ground plane."""
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.05],
        rgba=color,
        contype=contype,
        conaffinity=conaffinity,
    )
