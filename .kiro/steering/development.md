---
inclusion: always
---

# Development Rules: MT-GBM

## Correctness

- Read the full call path before editing: config -> Python API -> C API -> GBDT ->
  tree learner -> score updater -> model save/load.
- Preserve single-label behavior when `num_labels=1`.
- Check array shape, dtype, and C-contiguity at every Python/C++ boundary that
  accepts gradients or Hessians.
- Keep joint gradients and per-task gradients semantically distinct.
- Never change a metric or split in a way that hides a regression.

## Experiment integrity

- Fraud, time-series, and entity-level examples must use leakage-safe folds or
  temporal splits.
- Do not calculate target-derived features across the evaluation future.
- Report the exact task, metric, fold count, seed, and baseline when citing a
  result.
- Do not present an illustrative target or an unrun notebook as a measured
  result.

## Compatibility

- Treat C ABI and model serialization changes as breaking changes unless proven
  otherwise.
- Check Python documentation and type/shape validation after API changes.
- Keep examples runnable from their documented input layout.

## Minimal implementation

- Reuse existing LightGBM abstractions and helpers.
- Add no production dependency unless it is strictly required.
- Add comments only for non-obvious invariants, such as why joint gradients are
  used for structure while per-task gradients are used for leaf values.
- If a known limitation must remain, write a `shortcut:` comment that names the
  limitation and the upgrade condition.
