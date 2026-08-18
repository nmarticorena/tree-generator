"""View generated tree URDFs together with MuJoCo Warp and mjviser."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    import mujoco
    import mujoco_warp as mjwarp
    import numpy as np
    import viser
    import warp as wp
    from mjviser import Viewer

    from .tree_scene import TREE, add_ground, find_urdfs, load_tree, tune_tree
except ImportError:
    print('Install the forest viewer with: pip install "forsym[mujoco-warp]"', file=sys.stderr)
    raise SystemExit(1) from None


def build_forest(
    paths: Sequence[Path],
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
    """Build one grid scene and return its model and selectable trees."""
    if not paths:
        raise ValueError("At least one tree is required")
    trees = _load_trees(paths, prune_fixed, self_contacts)
    columns, spacing = _layout(trees, columns, spacing)
    spec = _forest_spec(ground, self_contacts)
    entities = _attach_trees(spec, trees, _grid(len(trees), columns, spacing))
    return _compile_forest(spec, entities, stiffness, damping, armature, friction)


def _load_trees(paths, prune_fixed, self_contacts):
    contacts = None if self_contacts else (TREE, 2)
    return [load_tree(Path(path), contacts=contacts, prune_fixed=prune_fixed) for path in paths]


def _compile_forest(spec, entities, stiffness, damping, armature, friction):
    model = spec.compile()
    count = tune_tree(model, stiffness=stiffness, damping=damping, armature=armature, friction=friction)
    print(f"Loaded {len(entities)} trees and configured {count} flexible joints.")
    return model, entities


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


def warp_callbacks(model, data, device=None):
    """Create mjviser-compatible MuJoCo Warp step and reset callbacks."""
    wp.init()
    if device:
        wp.set_device(device)
    warp_model = mjwarp.put_model(model)
    warp_data = mjwarp.put_data(model, data, nconmax=_capacity(data.ncon), njmax=_capacity(data.nefc))

    def step(_model, host_data):
        _copy_to_warp(warp_data, host_data)
        mjwarp.step(warp_model, warp_data)
        mjwarp.get_data_into(host_data, model, warp_data)

    def reset(_model, host_data):
        mjwarp.reset_data(warp_model, warp_data)
        mjwarp.forward(warp_model, warp_data)
        mjwarp.get_data_into(host_data, model, warp_data)

    return step, reset


def _capacity(current: int) -> int:
    return math.ceil(max(math.ceil(current * 1.25), 256) / 256) * 256


def _batch(values):
    return wp.array(values.astype(np.float32, copy=False)[None])


def _copy_to_warp(target, source) -> None:
    wp.copy(target.ctrl, _batch(source.ctrl))
    wp.copy(target.act, _batch(source.act))
    forces = source.xfrc_applied.astype(np.float32, copy=False)[None]
    wp.copy(target.xfrc_applied, wp.array(forces, dtype=wp.spatial_vector))
    wp.copy(target.qpos, _batch(source.qpos))
    wp.copy(target.qvel, _batch(source.qvel))
    wp.copy(target.time, wp.array([source.time], dtype=wp.float32))


def show_forest(model, entities, *, host="127.0.0.1", port=8080, device=None) -> None:
    """Open a forest model in mjviser."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    step, reset = warp_callbacks(model, data, device)
    server = _forest_server(host, port, entities)
    Viewer(model, data, step_fn=step, reset_fn=reset, server=server).run()


def _forest_server(host, port, entities):
    server = viser.ViserServer(host=host, port=port, label="ForSym forest")
    _set_camera(server, entities)
    _add_selector(server, entities)
    return server


def _forest_frame(entities) -> tuple[np.ndarray, float]:
    centers = np.stack([center for _, center, _ in entities])
    lower = np.min([center - size / 2 for _, center, size in entities], axis=0)
    upper = np.max([center + size / 2 for _, center, size in entities], axis=0)
    return np.mean(centers, axis=0), max(float(np.max(upper - lower)), 1.0)


def _set_camera(server, entities) -> None:
    center, distance = _forest_frame(entities)
    server.initial_camera.look_at = center
    server.initial_camera.position = center + np.array([distance, -distance, 0.7 * distance])


def _add_selector(server, entities) -> None:
    box, label = _selection_marker(server, entities[0])
    with server.gui.add_folder("Forest"):
        slider = server.gui.add_slider(
            "Tree", min=0, max=len(entities) - 1, step=1, initial_value=0, disabled=len(entities) == 1
        )
        info = server.gui.add_html("")

    @slider.on_update
    def _(_) -> None:
        _select(server, entities[int(slider.value)], box, label, info)

    _select(server, entities[0], box, label, info, focus=False)


def _selection_marker(server, entity):
    path, center, size = entity
    box = server.scene.add_box("/forest/selected", color=(255, 210, 0), dimensions=size, wireframe=True)
    label = server.scene.add_label("/forest/selected/label", path.name, position=_label_position(center, size))
    return box, label


def _select(server, entity, box, label, info, focus=True) -> None:
    path, center, size = entity
    box.position, box.dimensions = center, size
    label.position, label.text = _label_position(center, size), path.name
    info.content = str(path)
    if focus:
        _focus_clients(server, center, size)


def _label_position(center, size):
    return center + np.array([0.0, 0.0, size[2] / 2])


def _focus_clients(server, center, size) -> None:
    for client in server.get_clients().values():
        offset = np.asarray(client.camera.position) - np.asarray(client.camera.look_at)
        if np.linalg.norm(offset) < 1e-6:
            distance = max(2.5 * float(np.max(size)), 1.0)
            offset = np.array([distance, -distance, 0.6 * distance])
        client.camera.position, client.camera.look_at = center + offset, center


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="View generated trees together with MuJoCo Warp.")
    result.add_argument("source", type=Path, help="A URDF or directory of URDFs.")
    _add_layout_args(result)
    _add_dynamics_args(result)
    _add_viewer_args(result)
    return result


def _add_layout_args(parser) -> None:
    parser.add_argument("--pattern", default="*.urdf")
    parser.add_argument("--columns", type=int)
    parser.add_argument("--spacing", type=float)
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--keep-fixed-collisions", action="store_true")
    parser.add_argument("--tree-self-contacts", "--keep-tree-self-collisions", action="store_true")


def _add_dynamics_args(parser) -> None:
    parser.add_argument("--stiffness", "--base-stiffness", type=float, default=400.0)
    parser.add_argument("--damping", "--damping-scale", type=float, default=0.2)
    parser.add_argument("--armature", type=float, default=0.01)
    parser.add_argument("--friction", "--joint-friction", type=float, default=0.01)


def _add_viewer_args(parser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", help="Warp device, for example cuda:0 or cpu.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        model, entities = _scene_from_args(args)
    except (FileNotFoundError, ValueError) as error:
        parser().error(str(error))
    show_forest(model, entities, host=args.host, port=args.port, device=args.device)


def _scene_from_args(args):
    return build_forest(
        find_urdfs(args.source, args.pattern),
        columns=args.columns,
        spacing=args.spacing,
        ground=not args.no_ground,
        stiffness=args.stiffness,
        damping=args.damping,
        armature=args.armature,
        friction=args.friction,
        prune_fixed=not args.keep_fixed_collisions,
        self_contacts=args.tree_self_contacts,
    )


if __name__ == "__main__":
    main()
