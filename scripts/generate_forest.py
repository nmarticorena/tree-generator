"""Generate the default trees and open them together as a MuJoCo forest."""

from forsym import generate_trees


def main() -> None:
    """Generate, compile, and display the YAML-configured forest."""
    urdfs = list(generate_trees(100, seed=42))  # Generate 100 trees with a fixed seed for reproducibility

    # You can also lazy-load the trees; here we reuse the already-generated list.
    for tree in urdfs:
        print(tree)


if __name__ == "__main__":
    main()
