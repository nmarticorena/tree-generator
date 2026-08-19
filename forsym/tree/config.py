"""Load the canonical tree-generation YAML schema."""
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Rule:
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
    rules: list[Rule]
    free_params: dict[str, float]
    relative_std: float

    @classmethod
    def from_dict(cls, values:dict) -> "LSystemConfig":
        return cls(
            axiom=values["axiom"],
            generations=values["generations"],
            tropism=values["tropism"],
            bending=values["bending"],
            initial_angle=values["initial_angle"],
            angle_std=values["angle_std"],
            rules=[Rule(rule["pred"], rule["succ"]) for rule in values["rules"]],
            free_params={name: value for item in values["free_params"] for name, value in item.items()},
            relative_std=values["relative_std"],
        )


@dataclass
class TreeConfig:
    """Values required to assemble branches and fruits."""

    length_scale: float
    radius_scale: float
    fruit_count: int
    dof_root: str = ""
    flex_root: str = ""
    fruit_branch: str = ""

    @classmethod
    def from_dict(cls, values:dict) -> "TreeConfig":
        return cls(
            length_scale=values["length_scale"],
            radius_scale=values["radius_scale"],
            fruit_count=values["fruit_count"],
            dof_root=values.get("dof_root", ""),
            flex_root=values.get("flex_root", ""),
            fruit_branch=values.get("fruit_branch", ""),
        )


def load_config(path: str | Path) -> tuple[LSystemConfig, TreeConfig, str]:
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
    return LSystemConfig.from_dict(tree["lsystem"]), TreeConfig.from_dict(tree), tree["outfile"]

@dataclass
class TreeGenerationConfig:
    """Everything needed to run the pipeline: a plain, freely-constructable object.
 
    YAML is just one way to build one (`from_yaml`); you can equally build one
    by hand, or take an existing one and tweak a few fields with `replace`.
    """
 
    lsystem: LSystemConfig
    tree: TreeConfig
    output_pattern: str
 
    @classmethod
    def from_yaml(cls, path: str | Path) -> "TreeGenerationConfig":
        lsystem, tree, output_pattern = load_config(path)
        return cls(lsystem=lsystem, tree=tree, output_pattern=output_pattern)
 
    @classmethod
    def default(cls) -> "TreeGenerationConfig":
        """The shipped ternary_tree.yaml, as a config object."""
        default_path = Path(__file__).resolve().parent / "ternary_tree.yaml"
        return cls.from_yaml(default_path)
 
    def replace(self, **overrides: object) -> "TreeGenerationConfig":
        """Override any leaf field by name, regardless of which sub-config it lives on.
 
        >>> cfg = TreeGenerationConfig.default().replace(generations=6, fruit_count=2)
        """
        lsystem_fields = {f.name for f in dataclasses.fields(LSystemConfig)}
        tree_fields = {f.name for f in dataclasses.fields(TreeConfig)}
        unknown = overrides.keys() - lsystem_fields - tree_fields - {"output_pattern"}
        if unknown:
            raise TypeError(f"Unknown config field(s): {sorted(unknown)}")
 
        lsystem_overrides = {k: v for k, v in overrides.items() if k in lsystem_fields}
        tree_overrides = {k: v for k, v in overrides.items() if k in tree_fields}
        return dataclasses.replace(
            self,
            lsystem=dataclasses.replace(self.lsystem, **lsystem_overrides),
            tree=dataclasses.replace(self.tree, **tree_overrides),
            output_pattern=overrides.get("output_pattern", self.output_pattern),
        )
 
 

