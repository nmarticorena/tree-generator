"""Load the canonical tree-generation YAML schema."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class _Rule:
    pred: str
    succ: str


@dataclass
class LSystemConfig:
    """Values required to expand and interpret the tree L-system."""

    axiom: str
    generations: int
    tropism: list[float]
    bending: float
    initial_angle: float
    angle_std: float
    rules: list[_Rule]
    free_params: dict[str, float]
    tree_count: int
    relative_std: float


@dataclass
class TreeConfig:
    """Values required to assemble branches and fruits."""

    length_scale: float
    radius_scale: float
    fruit_count: int
    dof_root: str = ""
    flex_root: str = ""
    fruit_branch: str = ""


def load_config(path):
    """Load the complete tree-generation configuration.

    Parameters
    ----------
    path : str or path-like
        YAML configuration file.

    Returns
    -------
    lsystem_config : LSystemConfig
        Rewrite, geometry, output-count, and sampling settings.
    tree_config : TreeConfig
        Scale and fruit settings used during assembly.
    output_pattern : str
        Relative URDF output pattern.
    """
    tree = yaml.safe_load(Path(path).read_text())["tree"]
    return _lsystem_config(tree["lsystem"]), _tree_config(tree), tree["outfile"]


def _lsystem_config(values):
    return LSystemConfig(
        axiom=values["axiom"],
        generations=values["generations"],
        tropism=values["tropism"],
        bending=values["bending"],
        initial_angle=values["initial_angle"],
        angle_std=values["angle_std"],
        rules=[_Rule(rule["pred"], rule["succ"]) for rule in values["rules"]],
        free_params={name: value for item in values["free_params"] for name, value in item.items()},
        tree_count=values["tree_count"],
        relative_std=values["relative_std"],
    )


def _tree_config(tree):
    return TreeConfig(
        length_scale=tree["scale"]["length"],
        radius_scale=tree["scale"]["radius"],
        fruit_count=tree["fruits"]["count"],
    )
