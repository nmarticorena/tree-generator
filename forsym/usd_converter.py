# treeforge/usd_postprocess.py
from pxr import Usd, Sdf, UsdPhysics

_JOINT_TYPES = {
    "PhysicsRevoluteJoint", "PhysicsPrismaticJoint",
    "PhysicsFixedJoint", "PhysicsSphericalJoint", "PhysicsJoint",
}

def apply_armature(stage: Usd.Stage, armature: float) -> None:
    """Equivalent of AssetOptions.armature, without needing PhysxSchema."""
    for prim in stage.Traverse():
        print (f"Processing prim: {prim.GetPath()} of type {prim.GetTypeName()}")
        if prim.GetTypeName() not in _JOINT_TYPES:
            continue
        # Register the applied API schema so Isaac Sim/PhysX recognizes it
        prim.AddAppliedSchema("PhysxJointAPI")
        attr = prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float, custom=False)
        attr.Set(armature)

def override_mass_properties(stage: Usd.Stage, com=None, diagonal_inertia=None) -> None:
    """Equivalent of override_com / override_inertia (UsdPhysics is part of usd-core)."""
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.MassAPI) and not prim.IsA(UsdPhysics.RigidBodyAPI):
            continue
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if com is not None:
            mass_api.CreateCenterOfMassAttr(com)
        if diagonal_inertia is not None:
            mass_api.CreateDiagonalInertiaAttr(diagonal_inertia)

def prepare_asset(usd_path: str, armature: float = 0.01) -> str:
    stage = Usd.Stage.Open(usd_path)
    apply_armature(stage, armature)
    stage.Save()
    return usd_path
