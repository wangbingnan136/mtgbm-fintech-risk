# MT-GBM Fintech Risk Lab

This demo turns the MT-GBM multi-task path into a small, testable application.
It trains one shared LightGBM tree structure for two related fintech risks:

- Card-not-present transaction fraud.
- Account-takeover risk.

The generated static site runs entirely in the browser. Model trees produced by
`Booster.save_model2()` are exported to JSON and evaluated by a small JavaScript
tree scorer, so the public demo does not need a Python server.

## Build the model

From the repository root:

```bash
PYTHONPATH=python-package python3 demo/train_export.py
```

The script writes `demo/web/model.json`. It also verifies the exported tree
evaluator against the current MT-GBM Python predictions before reporting success.

## Run locally

```bash
python3 -m http.server 8000 --directory demo/web
```

Open `http://localhost:8000`.

## Data

All rows are synthetic and generated deterministically by the training script.
No customer data or third-party transaction records are included.

The demo is a model-behavior demonstration, not a production fraud decision
system. A production deployment would require identity controls, audit logging,
monitoring, fairness review, analyst workflows, and model-risk governance.
