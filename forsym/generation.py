"""Public entry point for lazy, YAML-driven tree generation."""

from __future__ import annotations

import argparse
import itertools
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from forsym.fractal import rewriter
from forsym.tree import assembly, pipeline
from forsym.tree.config import TreeGenerationConfig


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

    for index, varied_config in enumerate(assembly.iter_lsystem_configs(lsystem_config, n_trees ,numpy_rng)):
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


def main(argv: Sequence[str] | None = None) -> int:
    """Generate every tree configured by a YAML file from the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse instead of :data:`sys.argv`.

    Returns
    -------
    int
        Zero after every configured tree has been generated.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("generated"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    for path in generate_trees(args.config, output_root=args.output_root, seed=args.seed):
        print(path)
    return 0


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


__all__ = ["generate_trees"]


if __name__ == "__main__":
    raise SystemExit(main())
