---
inclusion: always
---

# Project Structure: MT-GBM

## Repository layout

```text
include/LightGBM/       Public C++ headers and the C API.
src/                    C++ implementation, objectives, boosting, and tree learners.
python-package/lightgbm Python bindings and high-level training API.
R-package/              R bindings and tests.
tests/                  C++ and Python test suites.
examples/multitask/     Multi-task demonstrations and notebooks.
demo/                   Reproducible fintech demo and static browser UI.
docs/                   User and developer documentation.
.kiro/                  Kiro steering, agents, hooks, and specs.
```

## Multi-task ownership

- Configuration and parameter validation belong near the existing LightGBM
  configuration code.
- Gradient layout and score storage belong in the boosting and score-update
  layers.
- Tree structure and leaf-value behavior belong in the multi-task tree learner.
- Python must only translate data into the expected API shape; it should not
  duplicate C++ training logic.
- Examples must remain separate from production code and must not be required
  to build or test the library.
- The static demo may consume only its generated `model.json`; browser code must
  not contain a second hand-written model.

## Naming conventions

- Preserve existing LightGBM names and style in inherited code.
- Use `num_labels`, `serial2`, and `Booster.save_model2()` when referring to the
  public multi-task extension.
- Use `mtgbm` only for MT-GBM-specific helpers or documentation; otherwise keep
  upstream `LightGBM` naming for compatibility.

## Change boundaries

- Make the smallest change that reaches every required layer.
- Update the relevant example or spec when changing public parameter behavior.
- Do not rewrite inherited modules merely to change formatting or naming.
