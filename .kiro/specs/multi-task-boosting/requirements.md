# Requirements Document

## Introduction

MT-GBM extends LightGBM 3.3.2.99 with a multi-task training path for tabular
models. A training row can contain multiple labels. The implementation learns a
shared tree structure and then recalculates leaf outputs for task-specific
objectives.

## Requirements

### Requirement 1

**User Story:** As a tabular ML engineer, I want to train one gradient-boosted
model against multiple labels so that related tasks can share tree structure.

#### Acceptance Criteria

1. WHEN a training dataset contains multiple label columns THEN THE SYSTEM SHALL
   expose the number of labels through the `num_labels` parameter.
2. WHEN `num_labels` does not match the label array shape THEN THE SYSTEM SHALL
   fail with a clear error.
3. WHEN multi-task training is requested THEN THE SYSTEM SHALL require the
   `serial2` tree-learner path.

### Requirement 2

**User Story:** As a model developer, I want the shared tree structure and
task-specific leaf outputs to be controlled separately so that one task's
gradient does not overwrite the others.

#### Acceptance Criteria

1. WHEN a boosting iteration starts THEN THE SYSTEM SHALL accept joint
   gradients/Hessians and per-task gradients/Hessians.
2. WHEN a split structure has been learned from the joint objective THEN THE
   SYSTEM SHALL recalculate leaf outputs using the selected task's gradients and
   Hessians.
3. WHEN task models are stored THEN EACH task SHALL remain addressable by its
   label index.

### Requirement 3

**User Story:** As a maintainer, I want the Python, C API, and C++ layers to
agree on tensor layout so that multi-task training is deterministic and safe.

#### Acceptance Criteria

1. WHEN gradients cross the Python/C API boundary THEN THE SYSTEM SHALL validate
   shape, length, and C-contiguous memory.
2. WHEN `num_labels=1` THEN THE SYSTEM SHALL preserve the standard LightGBM
   custom-objective path.
3. WHEN model saving is requested for a task THEN THE SYSTEM SHALL use the
   task-specific export path and preserve load compatibility.

### Requirement 4

**User Story:** As an ML researcher, I want reproducible examples so that I can
compare multi-task and single-task behavior without target leakage.

#### Acceptance Criteria

1. WHEN an example evaluates fraud or time-series labels THEN IT SHALL document
   its dataset, split, metric, and random seed.
2. WHEN target-derived features are created THEN THE FEATURE PIPELINE SHALL
   respect the evaluation time boundary.
3. WHEN a result is reported THEN THE RESULT SHALL distinguish measured output
   from illustrative or expected behavior.

### Requirement 5

**User Story:** As a project contributor, I want Kiro context and validation
automation so that changes follow the architecture and do not regress
single-task behavior.

#### Acceptance Criteria

1. WHEN a contributor opens the repository THEN Kiro SHALL load the project
   steering and the `mtgbm-engineer` agent configuration.
2. WHEN a multi-task core file is saved THEN Kiro SHALL prompt for the required
   regression checks.
3. WHEN the Python package changes THEN Kiro SHALL compile the package.
