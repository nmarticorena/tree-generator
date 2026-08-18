"""Teleoperate a Franka Panda in mjviser with Mink inverse kinematics."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

try:
    import mink
    import mujoco
    import numpy as np
    import viser
    from mjviser import Viewer
    from robot_descriptions import panda_mj_description

    from .tree_scene import GROUND, ROBOT, TREE, add_ground, load_tree, set_contacts, tune_tree
except ImportError:
    print(
        'Use Python 3.10--3.13 and install: pip install "forsym[teleoperation]"',
        file=sys.stderr,
    )
    raise SystemExit(1) from None


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
END_EFFECTOR = "attachment_site"
TARGET = "teleop_target"


def build_franka(
    tree: Path | None = None,
    *,
    position=(1.15, 0.0, 0.0),
    yaw=0.0,
    stiffness=400.0,
    damping=0.2,
) -> tuple[mujoco.MjModel, mujoco.MjModel]:
    """Return the full contact model and a small Panda-only IK model."""
    spec = _panda_spec()
    ik_model = spec.compile()
    _complete_scene(spec, tree, position, yaw)
    model = spec.compile()
    _configure_tree(model, tree, stiffness, damping)
    return model, ik_model


def _complete_scene(spec, tree, position, yaw) -> None:
    add_ground(spec, contype=GROUND, conaffinity=ROBOT, color=(0.2, 0.24, 0.28, 1.0))
    _add_target(spec)
    if tree:
        _attach_tree(spec, tree, position, yaw)


def _configure_tree(model, tree, stiffness, damping) -> None:
    if tree:
        count = tune_tree(model, prefix="tree_", stiffness=stiffness, damping=damping)
        print(f"Loaded {Path(tree).name} and configured {count} flexible joints.")


def _panda_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    spec.compiler.discardvisual = False
    spec.modelname = "forsym_franka"
    set_contacts(spec, ROBOT, TREE | ROBOT | GROUND)
    spec.body("hand").add_site(name=END_EFFECTOR, pos=[0, 0, 0.1034], size=[0.012])
    return spec


def _add_target(spec) -> None:
    target = spec.worldbody.add_body(name=TARGET, mocap=True)
    colors = ([1, 0.1, 0.1, 0.8], [0.1, 1, 0.1, 0.8], [0.1, 0.3, 1, 0.8])
    for axis, color in enumerate(colors):
        end = np.zeros(3)
        end[axis] = 0.1
        target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, *end],
            size=[0.004],
            rgba=color,
            contype=0,
            conaffinity=0,
        )


def _attach_tree(spec, path, position, yaw) -> None:
    _, tree, center, size = load_tree(Path(path), contacts=(TREE, ROBOT))
    position = np.asarray(position, dtype=float).copy()
    position[2] -= center[2] - size[2] / 2
    angle = math.radians(yaw) / 2
    frame = spec.worldbody.add_frame(name="tree_frame", pos=position, quat=[math.cos(angle), 0, 0, math.sin(angle)])
    spec.attach(tree, frame=frame, prefix="tree_")


class _PandaController:
    """A small Mink controller shaped like an mjviser callback."""

    def __init__(self, model, ik_model, gizmo, gripper, report, max_velocity):
        self.model, self.gizmo, self.gripper = model, gizmo, gripper
        self.configuration, self.end_task, self.posture_task = _ik_tasks(ik_model)
        self.limits = _ik_limits(ik_model, max_velocity)
        self.arm_qpos = _joint_addresses(model, ARM_JOINTS)
        self.arm_actuators = _actuator_ids(model, ARM_ACTUATORS)
        self.gripper_actuator = model.actuator("actuator8").id
        self.report = report
        self.snap_requested = False

    def reset(self, _model, data) -> None:
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("home").id)
        self.configuration.update([*data.qpos[self.arm_qpos], 0.04, 0.04])
        self.posture_task.set_target_from_configuration(self.configuration)
        mujoco.mj_forward(self.model, data)
        self.gripper.value = True
        self.snap(data)

    def step(self, _model, data) -> None:
        if self.snap_requested:
            self.snap(data)
        self._follow_target(data)
        self._apply_controls(data)
        mujoco.mj_step(self.model, data)
        self.report(data)

    def snap(self, data) -> None:
        position, wxyz = _site_pose(data, END_EFFECTOR)
        data.mocap_pos[0], data.mocap_quat[0] = position, wxyz
        self.gizmo.position, self.gizmo.wxyz = position, wxyz
        self.snap_requested = False

    def _follow_target(self, data) -> None:
        data.mocap_pos[0], data.mocap_quat[0] = self.gizmo.position, self.gizmo.wxyz
        self.end_task.set_target(mink.SE3.from_mocap_name(self.model, data, TARGET))
        velocity = mink.solve_ik(
            self.configuration,
            [self.end_task, self.posture_task],
            self.model.opt.timestep,
            "daqp",
            limits=self.limits,
            damping=1e-5,
        )
        self.configuration.integrate_inplace(velocity, self.model.opt.timestep)

    def _apply_controls(self, data) -> None:
        data.ctrl[self.arm_actuators] = self.configuration.q[:7]
        data.ctrl[self.gripper_actuator] = 255.0 if self.gripper.value else 0.0


def _ik_tasks(model):
    configuration = mink.Configuration(model)
    end = mink.FrameTask(END_EFFECTOR, "site", 1.0, 1.0, lm_damping=1.0)
    return configuration, end, mink.PostureTask(model, cost=1e-2)


def _ik_limits(model, velocity):
    speeds = {name: velocity for name in ARM_JOINTS}
    return [mink.ConfigurationLimit(model), mink.VelocityLimit(model, speeds)]


def _joint_addresses(model, names):
    return np.asarray([model.jnt_qposadr[model.joint(name).id] for name in names])


def _actuator_ids(model, names):
    return np.asarray([model.actuator(name).id for name in names])


def _site_pose(data, name):
    site = data.site(name)
    wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(wxyz, site.xmat.reshape(-1))
    return site.xpos.copy(), wxyz


def _contact_reporter(model, output, frequency):
    next_report = 0.0
    had_contacts = False

    def report(data):
        nonlocal next_report, had_contacts
        if frequency <= 0 or (not data.ncon and not had_contacts):
            return
        if not data.ncon:
            message, had_contacts = "No active contacts.", False
        elif time.monotonic() < next_report:
            return
        else:
            message, had_contacts = _contact_message(model, data), True
        next_report = time.monotonic() + 1 / frequency
        output.content = f"<b>Contact monitor</b><br>{message}"
        print(message)

    return report


def _contact_message(model, data) -> str:
    force = np.zeros(6)
    contacts = []
    for index, contact in enumerate(data.contact[: data.ncon]):
        mujoco.mj_contactForce(model, data, index, force)
        contacts.append((abs(float(force[0])), int(contact.geom1), int(contact.geom2)))
    strength, geom1, geom2 = max(contacts)
    return f"{data.ncon} contacts; {_geom_name(model, geom1)} ↔ {_geom_name(model, geom2)}: {strength:.2f} N"


def _geom_name(model, geom_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"#{geom_id}"


def show_franka(
    model,
    ik_model,
    *,
    host="0.0.0.0",
    port=8080,
    contact_hz=5.0,
    max_velocity=1.5,
) -> None:
    """Open a Panda contact scene in mjviser."""
    data = _home_data(model)
    server = viser.ViserServer(host=host, port=port, label="ForSym Franka")
    controls = _controls(server, data)
    controller = _controller(model, ik_model, data, controls, contact_hz, max_velocity)
    viewer = _viewer(model, data, controller, server)
    _finish_viewer_setup(viewer, server)
    viewer.run()


def _home_data(model):
    data = mujoco.MjData(model)
    _home(model, data)
    return data


def _controller(model, ik_model, data, controls, contact_hz, max_velocity):
    gizmo, gripper, snap, output = controls
    report = _contact_reporter(model, output, contact_hz)
    controller = _PandaController(model, ik_model, gizmo, gripper, report, max_velocity)
    controller.reset(model, data)
    _bind_snap(snap, controller)
    return controller


def _viewer(model, data, controller, server):
    return Viewer(
        model,
        data,
        step_fn=controller.step,
        reset_fn=controller.reset,
        server=server,
    )


def _home(model, data) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    mink.move_mocap_to_frame(model, data, TARGET, END_EFFECTOR, "site")


def _controls(server, data):
    gizmo = server.scene.add_transform_controls(
        "/teleoperation/target",
        scale=0.35,
        depth_test=False,
        position=data.mocap_pos[0],
        wxyz=data.mocap_quat[0],
        translation_limits=((-0.5, 2.0), (-1.5, 1.5), (0.02, 2.0)),
    )
    with server.gui.add_folder("Franka teleoperation"):
        server.gui.add_markdown("Drag the RGB gizmo to move or rotate the gripper.")
        gripper = server.gui.add_checkbox("Open gripper", initial_value=True)
        snap = server.gui.add_button("Target current gripper pose")
        output = server.gui.add_html("<b>Contact monitor</b><br>No active contacts.")
    return gizmo, gripper, snap, output


def _bind_snap(button, controller) -> None:
    @button.on_click
    def _(_) -> None:
        controller.snap_requested = True


def _finish_viewer_setup(viewer, server) -> None:
    viewer._MAX_SLIDERS = 50
    viewer.scene.show_contact_points = True
    viewer.scene.show_contact_forces = True
    server.initial_camera.look_at = np.array([0.6, 0, 0.55])
    server.initial_camera.position = np.array([1.75, -1.75, 1.25])
    print(f"Open http://{server.get_host()}:{server.get_port()}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Teleoperate a Franka Panda and inspect contacts.")
    result.add_argument("tree", nargs="?", type=Path)
    _add_scene_args(result)
    _add_viewer_args(result)
    return result


def _add_scene_args(parser) -> None:
    parser.add_argument("--tree-position", nargs=3, type=float, default=(0.7, 0, 0), metavar=("X", "Y", "Z"))
    parser.add_argument("--tree-yaw", type=float, default=0.0)
    parser.add_argument("--stiffness", "--base-stiffness", type=float, default=400.0)
    parser.add_argument("--damping", "--damping-scale", type=float, default=0.2)


def _add_viewer_args(parser) -> None:
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--contact-hz", "--contact-report-hz", type=float, default=5.0)
    parser.add_argument("--max-velocity", "--max-joint-velocity", type=float, default=1.5)


def main() -> None:
    args = parser().parse_args()
    _validate(args)
    model, ik_model = build_franka(
        args.tree,
        position=args.tree_position,
        yaw=args.tree_yaw,
        stiffness=args.stiffness,
        damping=args.damping,
    )
    show_franka(
        model,
        ik_model,
        host=args.host,
        port=args.port,
        contact_hz=args.contact_hz,
        max_velocity=args.max_velocity,
    )


def _validate(args) -> None:
    if args.stiffness <= 0 or args.max_velocity <= 0:
        raise SystemExit("stiffness and max velocity must be positive")
    if args.damping < 0 or args.contact_hz < 0:
        raise SystemExit("damping and contact frequency must be non-negative")


if __name__ == "__main__":
    main()
