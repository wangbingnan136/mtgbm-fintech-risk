---
inclusion: always
---

# Product Overview: MT-GBM

## Purpose

MT-GBM is a specialized LightGBM branch for multi-label and multi-task learning
on tabular data. The main use case is financial risk modeling, including fraud
classification where a main fraud label can be learned jointly with related
labels such as repeated-attack or entity-level risk signals.

The project changes gradient-boosted tree training rather than wrapping several
independent LightGBM models. It accepts multiple labels per row, builds a shared
tree structure from the combined objective, and recalculates leaf outputs for
each task-specific objective.

## Target users

- Machine learning engineers working on tabular financial risk models.
- Researchers experimenting with multi-task gradient boosting.
- Developers maintaining the Python, R, C API, and command-line surfaces of
  this LightGBM branch.

## Key capabilities

- Multi-label training through the `num_labels` parameter.
- Custom joint and per-task gradient/Hessian inputs.
- The `serial2` tree learner for single-machine multi-task training.
- Task-specific leaf-output recalculation after a shared tree is constructed.
- Per-task model export through `Booster.save_model2()`.
- Example notebooks for time-series amount prediction and IEEE-CIS fraud data.

## Scope boundaries

- The repository is not a production fraud decisioning platform.
- The repository includes a static browser demo that runs exported task models
  client-side.
- The demo uses synthetic data and must not be presented as production evidence.
- Benchmark claims must be tied to a named dataset, split, metric, and command.
- Upstream LightGBM code is inherited under the repository's MIT license and is
  not represented as newly written MT-GBM work.
- Real customer data, credentials, and personally identifiable information must
  never be committed to this repository.

## Quality goals

- Preserve standard LightGBM behavior when `num_labels=1`.
- Keep the multi-task path explicit and testable across the C++, C API, Python,
  and command-line boundaries.
- Prefer reproducible evaluation over undocumented leaderboard claims.
- Treat label leakage and time-based leakage as first-class correctness risks.
