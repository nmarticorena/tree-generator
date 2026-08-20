import urdf_usd_converter
from pxr import Sdf
from forsym import generate_trees
from forsym.usd_converter import prepare_asset

converter = urdf_usd_converter.Converter()  # pass ros_packages=[...] if your URDFs use package:// URIs

def main() -> None:
    """Generate, compile, and display the YAML-configured forest."""
    urdfs = list(generate_trees(1, seed=42))

    gt_tree_usd_paths = []

    for urdf in urdfs:
        print(f"Generated tree URDF: {urdf}")
        asset: Sdf.AssetPath = converter.convert(
            urdf.__str__(),
            f"generated_tree/{urdf.stem}.usd",
        )
        gt_tree_usd_paths.append(asset.path)
    
    for usd_path in gt_tree_usd_paths:
        prepare_asset(usd_path, armature=0.01)
        print(f"Prepared USD asset: {usd_path}")

if __name__ == "__main__":
    main()
