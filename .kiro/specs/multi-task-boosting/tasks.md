# Implementation Plan

## 1. Add the multi-label training contract

- [x] 1.1 Add the `num_labels` configuration parameter.
- [x] 1.2 Validate `num_labels` against the training label matrix in Python.
- [x] 1.3 Require the `serial2` tree learner for multi-task training.

## 2. Add the dual-gradient API path

- [x] 2.1 Add the multi-task C API entry point.
- [x] 2.2 Validate gradient and Hessian shape and contiguity in Python.
- [x] 2.3 Preserve the single-label custom-objective path when `num_labels=1`.

## 3. Implement shared-structure training

- [x] 3.1 Use joint gradients and Hessians to learn tree splits.
- [x] 3.2 Recalculate task-specific leaf outputs with per-task derivatives.
- [x] 3.3 Add the bagging-aware `serial2` training path.
- [x] 3.4 Update task-specific score streams after each iteration.

## 4. Add task-aware model export

- [x] 4.1 Add the C API entry point for task-specific model export.
- [x] 4.2 Expose `Booster.save_model2()` with a task index.
- [x] 4.3 Preserve standard model save and load behavior.

## 5. Add research examples

- [x] 5.1 Add a two-label time-series amount example.
- [x] 5.2 Add an IEEE-CIS fraud example with main and auxiliary labels.
- [x] 5.3 Add a deterministic synthetic multi-task training and export self-check.

## 6. Add Kiro project context

- [x] 6.1 Add product, technology, structure, development, and testing steering.
- [x] 6.2 Add the `mtgbm-engineer` custom agent.
- [x] 6.3 Add Python compile and multi-task review hooks.
- [x] 6.4 Document requirements, design, and implementation status.

## 7. Add the fintech application demo

- [x] 7.1 Generate deterministic synthetic fraud and account-takeover data.
- [x] 7.2 Train single-task baselines and a shared MT-GBM model.
- [x] 7.3 Export task-specific trees and verify browser scoring against Python.
- [x] 7.4 Build a responsive static browser interface.
- [x] 7.5 Verify desktop and mobile layouts with screenshots.
