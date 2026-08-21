# ForSym tree generator

Generate fruit-bearing URDF tree models and simulate them in MuJoCo, including forest and Franka Panda scenes.

## Install

Install [Pixi](https://pixi.prefix.dev/latest/#installation),
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Global install
then install the project "globally" you can run the following:

```bash
pixi global install --path .
```

This will compile the library and install it under a `conda` package, this will create a virtual environment that will just install this tool independently.

### Local install
If you are using the package locally you can install it as
```bash
pixi install
```

Then every time you use it you need to active it:
```bash
pixi shell # Roughly the same as conda activate
```

### As a library
You can also add this library as a python dependency just need to point to this repo

## Generate trees

### CLI

If the package was installed globally you just need to run
```bash
forsym-generate-trees
```
And will generate 100 urdf trees under the `generated/`
For more info you can run
```bash
forsym-generate-trees --help
```
to check all the arguments.

### Python
The packaged ternary-tree generator is also available as a Python API:

```python
from forsym import generate_trees
from forsym.tree.config import TreeGenerationConfig
config = TreeGenerationConfig.default() # Create instance with the default values from pcap
# Config is heavily typed so you can explore the base config dataclass or use your auto complete to check configs
urdfs = generate_trees(n_trees= 100, config=config, output_root="generated", seed=42)
first_urdf = next(urdfs) # Generate and return the path to the urdf file
```

### Mujoco
To test the generated trees you can export the `.xml` mujoco ready files that consider the parameters of the joints

To generate:
```bash
pixi run forsym-generate-mjcf
```
Its the same as the generate-trees but will create both the `raw` `.urdf` file and the post-process ready to simulate in mujoco `.xml`

Then you can simply simulate it running:
```bash
pixi run mujoco-simulate generated/trees/tree_{idx}_processed.xml
```

