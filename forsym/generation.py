"""Public entry point for lazy, YAML-driven tree generation."""

from __future__ import annotations
from forsym.mujoco_tools import export_tree_mjcf

import random
from collections.abc import Iterator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tyro

from forsym.fractal import rewriter
from forsym.tree import assembly, pipeline
from forsym.tree.config import TreeGenerationConfig


def generate_tree(args) -> Path:
    """Generate one tree inside a worker process."""
    (
        index,
        varied_config,
        tree_config,
        output_pattern,
        output_root,
        seed,
    ) = args

    l_string = rewriter.expand_lsystem(varied_config)

    return pipeline.generate_tree_urdf(
        index=index,
        l_string=l_string,
        l_config=varied_config,
        tree_config=tree_config,
        output_pattern=output_pattern,
        output_root=output_root,
        rng=random.Random(seed + index),
    ).resolve()

def generate_mujoco_mjcf(args) -> Path:
    """Generate one tree inside a worker process."""
    (
        index,
        varied_config,
        tree_config,
        output_pattern,
        output_root,
        seed,
    ) = args

    l_string = rewriter.expand_lsystem(varied_config)

    urdf = pipeline.generate_tree_urdf(
        index=index,
        l_string=l_string,
        l_config=varied_config,
        tree_config=tree_config,
        output_pattern=output_pattern,
        output_root=output_root,
        rng=random.Random(seed + index),
    ).resolve()
    return export_tree_mjcf(
        source=urdf,
        destination=None,
    )

def generate_mujoco(
    n_trees: int = 100,
    config: TreeGenerationConfig = TreeGenerationConfig.default(),
    output_root: str | Path = "generated",
    seed: int = 42,
    workers: int | None = None,
) -> list[Path]:
    numpy_rng = np.random.default_rng(seed)
    
    jobs = [
        (
            index,
            varied_config,
            config.tree,
            config.output_pattern,
            output_root,
            seed,
        )
        for index, varied_config in enumerate(
            assembly.iter_lsystem_configs(
                config.lsystem,
                n_trees,
                numpy_rng,
            )
        )
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(generate_mujoco_mjcf, jobs))


    
    
def generate_all_trees(
    n_trees: int = 100,
    config: TreeGenerationConfig = TreeGenerationConfig.default(),
    output_root: str | Path = "generated",
    seed: int = 42,
    workers: int | None = None,
) -> list[Path]:
    """Generate all trees in parallel."""
    print(f"Generating {n_trees} trees with {workers} workers...")
    output_root = Path(output_root).expanduser().resolve()
    _validate_output_pattern(config.output_pattern)

    numpy_rng = np.random.default_rng(seed)

    jobs = [
        (
            index,
            varied_config,
            config.tree,
            config.output_pattern,
            output_root,
            seed,
        )
        for index, varied_config in enumerate(
            assembly.iter_lsystem_configs(
                config.lsystem,
                n_trees,
                numpy_rng,
            )
        )
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(generate_tree, jobs))

def generate_trees(
    n_trees: int = 100,
    config: TreeGenerationConfig = TreeGenerationConfig.default(),
    output_root: str | Path = "generated",
    seed: int = 42,
) -> Iterator[Path]:
    """Yield simulation-ready tree URDFs described by a YAML file.

    Generation is lazy: each call to :func:`next` expands, assembles, and
    the param tree_max setts a limit

    Parameters
    ----------
    n_trees : int, default=100
        Number of trees to generate
    config : TreeGenerationConfig
        Config of the ternary tree to generate. Can be a :class:`TreeGenerationConfig` object, or a path to a YAML file.
    output_root : str or pathlib.Path, default="generated"
        Directory below which the YAML output pattern is created.
    seed : int, default=42
        Seed for local Python and NumPy random-number generators.

    Yields
    ------
    pathlib.Path
        Absolute path to each generated URDF.
    """
    output_root = Path(output_root).expanduser().resolve()
    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)

    lsystem_config, tree_config = config.lsystem, config.tree
    output_pattern = config.output_pattern
    _validate_output_pattern(output_pattern)

    for index, varied_config in enumerate(assembly.iter_lsystem_configs(lsystem_config, n_trees, numpy_rng)):
        l_string = rewriter.expand_lsystem(varied_config)
        yield pipeline.generate_tree_urdf(
            index=index,
            l_string=l_string,
            l_config=varied_config,
            tree_config=tree_config,
            output_pattern=output_pattern,
            output_root=output_root,
            rng=python_rng,
        ).resolve()


def main():
    import time

    ti = time.perf_counter()
    tyro.cli(generate_all_trees)
    print(f"Finished generating trees in {time.perf_counter() - ti:.2f} seconds")

def main_mujoco():
    import time

    ti = time.perf_counter()
    tyro.cli(generate_mujoco)
    print(f"Finished generating MJCF trees in {time.perf_counter() - ti:.2f} seconds")


def _validate_output_pattern(pattern: str) -> None:
    path = Path(pattern)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("The outfile must stay below output_root")
    try:
        formatted = pattern.format(index=0)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid outfile pattern: {pattern!r}") from error
    if not Path(formatted).name:
        raise ValueError("The outfile must name a URDF file")


if __name__ == "__main__":
    for p in tyro.cli(generate_trees):
        print(p)
