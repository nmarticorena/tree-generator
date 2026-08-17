from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass
class TreeGenerationConfig:
    """Configuration for generating a PCAP-style tree asset."""

    source_asset: Path
    output_dir: Path
    seed: int = 0
    output_name: str | None = None

    @property
    def output_path(self) -> Path:
        name = self.output_name or f"pcap_tree_seed_{self.seed}.urdf"
        return self.output_dir / name


def generate_tree_asset(config: TreeGenerationConfig) -> Path:
    """Generate a deterministic tree asset and return its path."""

    source = config.source_asset.expanduser().resolve()
    output = config.output_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Tree source asset does not exist: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return output


if __name__ == "__main__":
    # Example usage
    import tyro
    tyro.cli(generate_tree_asset)
