"""Public MuJoCo integration for generated ForSym trees."""

from pathlib import Path

from .visualisation.tree_scene import load_tree


def export_tree_mjcf(
    source: str | Path,
    destination: str | Path | None = None,
    **load_options,
) -> Path:
    """Export a tree with all MuJoCo post-processing encoded as MJCF."""
    source_path, spec = load_tree(source, **load_options)
    if destination is None:
        destination = source_path.with_name(f"{source_path.stem}_processed.xml")
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spec.to_xml())
    return destination


__all__ = ["export_tree_mjcf", "load_tree"]
