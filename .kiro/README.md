# Kiro Project Context for MT-GBM

This directory contains the project-level Kiro configuration for the MT-GBM
repository. It follows the current Kiro project scope layout:

- `.kiro/steering/`: persistent product, architecture, and development context.
- `.kiro/agents/`: reusable project-specific custom agents.
- `.kiro/hooks/`: event-driven validation and review automation.
- `.kiro/specs/`: requirements, design, and implementation tracking for features.

The directory is intentionally tracked by Git. Do not add `.kiro` or any of its
subdirectories to `.gitignore`.

## Recommended workflow

1. Select the `mtgbm-engineer` agent in Kiro.
2. Reference `#spec:multi-task-boosting` when changing the multi-task path.
3. Keep visible LightGBM behavior unchanged when `num_labels=1`.
4. Run the validation commands documented in
   `.kiro/steering/testing.md` before claiming a change is complete.
5. Update the spec and tasks when behavior or architecture changes.

These files contain project instructions only. Do not place credentials,
private dataset paths, customer data, or model artifacts in this directory.
