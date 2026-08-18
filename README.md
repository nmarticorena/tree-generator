# ForSym tree generator

Generate URDF tree models and visualise them with a IK Franka Panda robot.

## Install

Install [Pixi](https://pixi.prefix.dev/latest/#installation), then install the project dependencies:

```bash
pixi install
```

## Generate trees

```bash
pixi run forsym-generate-trees
```

This reads the tree settings from `configs/ternary_a.yaml` and generates URDF trees under `generated/`.

## Mink visualisation

After generating the trees, open one with the Franka Panda visualisation with allows you to check the contact of the robot against the tree.

```bash
pixi run forsym-teleop-franka generated/tree/gen/pliable04/ta/raw_train/ta_0000/ternary_a.urdf
```

