# Design Document

## Overview

The multi-task extension is implemented as a configurable branch inside the
existing LightGBM training stack. It does not wrap independent models. The
training pass accepts two gradient views:

- Joint gradients and Hessians, used to learn the tree structure.
- Per-task gradients and Hessians, used to recalculate leaf values for each
  task.

The design is single-machine and intentionally uses the `serial2` tree learner.

## Architecture

### Configuration

`include/LightGBM/config.h` defines `num_labels`. Configuration parsing accepts
the new parameter and the Python layer checks that it matches the label matrix.

### Python API

`python-package/lightgbm/basic.py`:

- validates `num_labels` against the training label shape;
- requires `tree_learner=serial2` for multi-task training;
- passes joint and per-task derivatives to the C API;
- exposes `Booster.save_model2()` for saving a selected task.

### C API

`include/LightGBM/c_api.h` adds the multi-task update entry point and task-aware
model-export entry point. The C layer is a transport boundary; it must not
duplicate objective logic.

### Boosting

`src/boosting/gbdt.cpp` implements the training iteration:

1. Apply the existing bagging and split gating rules.
2. Learn a tree structure from the joint gradients and Hessians.
3. Recalculate the tree's leaf outputs with the per-task derivatives.
4. Shrink and add the tree to the corresponding task score stream.
5. Continue until the standard stopping conditions are met.

### Tree learner

`src/treelearner/serial_mt_tree_learner.*` owns:

- the ordinary tree-structure learning path;
- `Train_serial2()`, which recalculates leaf values from task-specific
  gradients and Hessians;
- bagging-aware indexing when the training subset differs from the full data.

### Score and storage

`src/boosting/score_updater.hpp` and the model containers allocate
`num_data * num_tree_per_iteration * num_labels` scores. Task models are stored
separately and can be exported by index.

## Data flow

```text
multi-column label matrix
        |
        v
Python validation: shape, num_labels, serial2
        |
        v
C API: joint + per-task gradient buffers
        |
        v
GBDT: joint gradients learn splits
        |
        v
serial2: per-task gradients set leaf outputs
        |
        v
score updater + task model collection
        |
        v
save_model2(task_index)
```

## Correctness constraints

- Joint and per-task arrays must be aligned by row.
- Per-task arrays must be stored contiguously.
- When `num_labels=1`, the standard LightGBM path remains authoritative.
- A task's leaf output must use that task's gradients and Hessians, not the
  joint derivative.
- C ABI changes require checking every binding that compiles against the header.

## Testing strategy

- Single-label smoke test: standard binary/regression training and prediction.
- Multi-label smoke test: two or more labels, non-identical targets, training
  completes and produces distinct task outputs.
- API validation: wrong label count and non-matching label shape fail clearly.
- Serialization: save and load a selected task model.
- Examples: rerun a small deterministic subset before quoting notebook results.

## Known limitations

- The multi-task implementation is limited to the single-machine `serial2`
  learner.
- Existing notebooks are research demonstrations and are not automated tests.
- The current exported-tree self-check lives in `demo/train_export.py`; a small
  pytest wrapper for CI is still desirable.
