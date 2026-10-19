# Autoresearch workbench

TOML files in this directory are LLM-generated mutable search inputs and are
ignored by Git. Reuse one workbench manifest across block rotations; completed
`run.json` files retain the resolved axes, robustness radii, execution inputs,
and local replay inputs.

The exceptions are tracked: the three report_lean 128 x 128 keeper grids
written by `scripts/yb_lean_study.py` (`top100-…`, `nevergrad-top13-…`,
`clean-2x2-…`) and the two slice files it reads (`top100_slices_20261018.json`,
`nevergrad_top13_20261018.json`). Their runs replay from this directory, the
evaluator's working directory.

Move only a human-reviewed, durable manifest into `configs/experiments/`.
