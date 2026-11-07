---
inclusion: fileMatch
fileMatchPattern: "**/{tests,src,python-package,examples}/**"
---

# Testing Standards: MT-GBM

## Required validation

1. Compile changed C++ translation units with the repository build system.
2. Run the smallest available Python or C++ test that exercises the changed
   path.
3. Run a single-label smoke test to confirm upstream behavior is unchanged.
4. Run a multi-label smoke test when `num_labels` or the custom-gradient path
   changes.
5. Check model save and load when serialization code changes.
6. Regenerate the fintech demo when a public task model changes and confirm its
   exported-tree check passes.

## Useful commands

Run from the repository root and adjust paths for the local build:

```bash
python3 -m compileall -q python-package/lightgbm
python3 -m pytest tests/python_package_test -q
PYTHONPATH=python-package python3 demo/train_export.py
cmake --build build
ctest --test-dir build --output-on-failure
```

## Multi-task checks

- Use at least two labels with visibly different targets.
- Confirm that the shared tree structure is produced before task-specific leaf
  outputs are recalculated.
- Confirm that each exported model corresponds to the requested task index.
- Confirm that invalid `num_labels` and label-shape mismatches raise errors.
- Inspect the first few training iterations when changing gradient layout.

## Evidence

Record the command, dataset identifier, split, seed, and result for any benchmark
claim. Notebook output alone is not proof that the current source tree still
reproduces the result.
