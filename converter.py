from pathlib import Path

from isaaclab.app import AppLauncher

# Must launch Isaac Sim before importing pxr/omni-dependent modules.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import PhysxSchema, Usd, UsdPhysics

import re
import numpy as np

from pxr import PhysxSchema, Usd, UsdPhysics


def branch_level(joint_name: str) -> int:
    """Extract the level from the child branch in a tree DOF name."""
    match = re.search(r"to-branch-[^-]*L(?P<level>\d+)P", joint_name)
    if match is None:
        raise ValueError(f"Cannot determine child branch level from: {joint_name}")
    level = int(match.group("level"))
    if not 0 < level <= 7:
        raise ValueError(f"Branch level must be in 1..7, got {level}: {joint_name}")
    return level


def rud_deflection_param(
    branch_level: int,
    base_kp: float = 400.0,
    noise_std: float = 1.0,
):
    if branch_level > 5:
        base_kp *= 2 ** (branch_level - 5)

    if base_kp <= 100:
        assert 0 < branch_level < 6

    kp = base_kp / (2 ** (branch_level - 1))

    if noise_std > 0.0:
        kp += np.random.normal(0.0, noise_std)

    kd = kp / 5.0

    return round(max(kp, 2.0), 2), round(max(kd, 2.0), 2)


def configure_tree_joints(
    usd_path: str,
    *,
    base_kp: float = 400.0,
    noise_std: float = 1.0,
    friction: float = 0.01,
    effort_limit: float = 100.0,
    armature: float = 0.01,
):
    stage = Usd.Stage.Open(usd_path)

    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue

        if prim.IsA(UsdPhysics.FixedJoint):
            continue

        name = prim.GetName()
        level = branch_level(name)

        kp, kd = rud_deflection_param(
            branch_level=level,
            base_kp=base_kp,
            noise_std=noise_std,
        )

        # --------------------------------------------------------------
        # Position drive
        # --------------------------------------------------------------
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(kp)
        drive.CreateDampingAttr(kd)
        drive.CreateMaxForceAttr(effort_limit)

        # --------------------------------------------------------------
        # PhysX-specific joint properties
        # --------------------------------------------------------------
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(prim)

        physx_joint.CreateArmatureAttr().Set(armature)
        physx_joint.CreateJointFrictionAttr().Set(friction)

        print(f"{name}: level={level}, kp={kp}, kd={kd}, friction={friction}, effort={effort_limit}")

    stage.GetRootLayer().Save()


def urdf_to_usd(
    urdf_path: str,
    usd_path: str,
    *,
    fix_base: bool = True,
    merge_fixed_joints: bool = True,
    armature: float = 0.01,
) -> str:
    urdf_path = Path(urdf_path).resolve()
    usd_path = Path(usd_path).resolve()

    usd_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. URDF -> USD
    # ------------------------------------------------------------------
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(usd_path.parent),
        usd_file_name=usd_path.name,
        fix_base=fix_base,
        merge_fixed_joints=merge_fixed_joints,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    )

    converter = UrdfConverter(cfg)

    generated_usd_path = Path(converter.usd_path)

    # ------------------------------------------------------------------
    # 2. Add armature to movable joints
    # ------------------------------------------------------------------
    stage = Usd.Stage.Open(str(generated_usd_path))

    if stage is None:
        raise RuntimeError(f"Failed to open generated USD: {generated_usd_path}")

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue

        # Fixed joints don't have a movable DoF.
        if prim.IsA(UsdPhysics.FixedJoint):
            continue

        physx_joint = PhysxSchema.PhysxJointAPI.Apply(prim)
        physx_joint.CreateArmatureAttr().Set(armature)

    stage.GetRootLayer().Save()

    return str(generated_usd_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python converter.py <input.urdf> <output.usd>")
        sys.exit(1)
    usd_path = urdf_to_usd(
        sys.argv[1],
        sys.argv[2],
        fix_base=True,
        merge_fixed_joints=True,
        armature=0.01,
    )
    configure_tree_joints(
        usd_path,
        base_kp=400.0,
        noise_std=1.0,
        friction=0.01,
        effort_limit=100.0,
        armature=0.01,
    )

    print(f"Generated: {usd_path}")

    simulation_app.close()
