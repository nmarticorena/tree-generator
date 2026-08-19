from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forsym.lsystems2.assemblers import pipeline_runner, tree_generator
from forsym.lsystems2.conf import yaml_parser

BUILTIN_CONFIG_ROOT = Path(__file__).resolve().parent / "lsystems2" / "conf"


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


def generate_trees(
    tree_type: str | Path,
    output_root: str | Path,
    *,
    count: int | None = None,
    seed: int = 42,
) -> list[Path]:
    """Generate URDF trees selected by a built-in name or YAML configuration path."""
    config_path = _tree_config_path(tree_type)
    output_root = Path(output_root).expanduser().resolve()
    if count is not None and count < 1:
        raise ValueError("count must be at least 1")

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        lsystem_config = yaml_parser.yaml_to_lsystem(config_path)
        if count is not None:
            lsystem_config.randomise = count > 1
            lsystem_config.randomise_cnt = count
        tree_config = yaml_parser.yaml_to_tree_config(config_path)
        output_pattern = yaml_parser.out_file(config_path)
        l_strings, l_configs = tree_generator.yaml_to_l_string(lsystem_config)
        _validate_output_pattern(output_pattern, len(l_strings))
        return [
            pipeline_runner.par_processor(
                index,
                l_string,
                tree_config,
                output_pattern,
                l_configs,
                generated_root=output_root,
            ).resolve()
            for index, l_string in enumerate(l_strings)
        ]
    finally:
        tree_generator.TreeBranch.unique = -1
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _tree_config_path(tree_type: str | Path) -> Path:
    requested = Path(tree_type).expanduser()
    if requested.is_file():
        return requested.resolve()
    if requested.parent != Path("."):
        raise FileNotFoundError(f"Tree configuration does not exist: {requested}")
    name = requested.name if requested.suffix else f"{requested.name}.yaml"
    built_in = BUILTIN_CONFIG_ROOT / name
    if not built_in.is_file():
        available = ", ".join(path.stem for path in sorted(BUILTIN_CONFIG_ROOT.glob("*.yaml")))
        raise ValueError(f"Unknown tree type {requested.stem!r}. Available built-ins: {available}")
    return built_in.resolve()


def _validate_output_pattern(pattern: str, count: int) -> None:
    path = Path(pattern)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("The YAML outfile must be relative to output_root")
    if count > 1 and "{r_idx" not in pattern:
        raise ValueError("The YAML outfile must contain an {r_idx} placeholder when generating multiple trees")


if __name__ == "__main__":
    # Example usage
    import tyro
    tyro.cli(generate_tree_asset)
