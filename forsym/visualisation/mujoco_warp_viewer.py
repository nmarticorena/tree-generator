"""Minimal mjviser viewer for the batched MuJoCo Warp tree loader."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
from mjviser import Viewer

from forsym.mujoco_loader import make_batched_tree_groups


def main() -> None:
    """Open mjviser and simulate every generated tree batch."""
    batches = make_batched_tree_groups("generated", nconmax=0)
    first_cpu_model, _, first_data = batches[0]
    first_data_worlds = first_data.nworld

    # mjviser uses a host MuJoCo model for geometry and GUI metadata. The
    # actual simulation state remains in Warp; this data object is only the
    # viewer's host-side control/render buffer.
    viewer_data = mujoco.MjData(first_cpu_model)

    def step_all(_model, _data) -> None:
        for _, warp_model, warp_data in batches:
            mjw.step(warp_model, warp_data)

    def render(scene) -> None:
        body_xpos = first_data.xpos.numpy()
        body_xmat = first_data.xmat.numpy().reshape(first_data_worlds, first_cpu_model.nbody, 3, 3)
        scene.update_from_arrays(
            body_xpos=body_xpos,
            body_xmat=body_xmat,
            qpos=first_data.qpos.numpy(),
            qvel=first_data.qvel.numpy(),
        )

    def reset_all(_model, _data) -> None:
        for _, warp_model, warp_data in batches:
            mjw.reset_data(warp_model, warp_data)
            mjw.forward(warp_model, warp_data)

    print(f"Loaded {sum(data.nworld for _, _, data in batches)} trees in {len(batches)} Warp batches.")
    print("Open the mjviser URL printed below in a browser.")

    Viewer(
        first_cpu_model,
        viewer_data,
        step_fn=step_all,
        render_fn=render,
        reset_fn=reset_all,
        num_envs=first_data_worlds,
    ).run()


if __name__ == "__main__":
    main()
