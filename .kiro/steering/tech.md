---
inclusion: always
---

# Technology Stack: MT-GBM

## Core stack

- Base project: LightGBM 3.3.2.99.
- Core implementation: C++ with CMake.
- Python package: `python-package/lightgbm`, using NumPy, pandas, SciPy, and
  scikit-learn in examples and checks.
- R package: `R-package`.
- External interfaces: C API, CLI, Python, and R.
- Static demo: plain HTML, CSS, and JavaScript with browser-side tree inference.
- Experimental notebooks: Jupyter, pandas, NumPy, scikit-learn, XGBoost, Keras,
  and TensorFlow where explicitly imported by the example.

## Multi-task extension points

- Configuration: `include/LightGBM/config.h` and generated configuration code.
- Dataset and score layout: `include/LightGBM/dataset.h`,
  `src/boosting/score_updater.hpp`.
- Objective interface: `include/LightGBM/objective_function.h`.
- Boosting path: `src/boosting/gbdt.cpp`.
- Tree learner: `src/treelearner/serial_mt_tree_learner.*`.
- C API: `include/LightGBM/c_api.h` and `src/c_api.cpp`.
- Python API: `python-package/lightgbm/basic.py`.
- Demo training/export: `demo/train_export.py`.

## Build and test tools

- Configure C++ builds with CMake.
- Build Python bindings through the repository's standard LightGBM build or
  package workflow.
- Use `pytest` for Python tests.
- Use `ctest` for C++ tests where the local build exposes them.

## Technical constraints

- Multi-task training currently relies on `tree_learner=serial2`.
- Multi-task label arrays must have shape compatible with `num_labels`.
- Custom multi-task objectives must return joint gradients/Hessians and
  per-label gradients/Hessians.
- The extension should not silently alter standard single-label behavior.
- Public APIs and model serialization need compatibility checks because the
  fork changes both the C ABI and Python `Booster` behavior.

## Dependency policy

- Do not add a dependency for functionality already available in the repository
  or the Python standard library.
- Disclose any pre-existing implementation reused from LightGBM or another
  project when describing the MT-GBM contribution.
