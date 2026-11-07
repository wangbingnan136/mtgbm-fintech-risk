"""Train and export the MT-GBM fintech risk demo."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python-package"))

import lightgbm as lgb  # noqa: E402


FEATURES = [
    {
        "id": "amount_usd",
        "label": "Transaction amount",
        "hint": "Payment amount in US dollars.",
        "min": 5,
        "max": 5000,
        "step": 5,
        "value": 420,
        "unit": "USD",
    },
    {
        "id": "hour_of_day",
        "label": "Hour of day",
        "hint": "Local time when the transaction was attempted.",
        "min": 0,
        "max": 23,
        "step": 1,
        "value": 2,
        "unit": "h",
    },
    {
        "id": "tx_velocity_1h",
        "label": "Transactions in last hour",
        "hint": "Recent payment velocity for the account or card.",
        "min": 0,
        "max": 20,
        "step": 1,
        "value": 7,
        "unit": "tx",
    },
    {
        "id": "distance_from_home_km",
        "label": "Distance from home",
        "hint": "Distance between the transaction and the usual home region.",
        "min": 0,
        "max": 1200,
        "step": 10,
        "value": 480,
        "unit": "km",
    },
    {
        "id": "device_trust",
        "label": "Device trust",
        "hint": "Trust score based on device history and fingerprint consistency.",
        "min": 0,
        "max": 1,
        "step": 0.01,
        "value": 0.28,
        "unit": "",
    },
    {
        "id": "email_age_days",
        "label": "Email age",
        "hint": "Age of the account email address.",
        "min": 1,
        "max": 3000,
        "step": 1,
        "value": 16,
        "unit": "days",
    },
    {
        "id": "failed_logins_24h",
        "label": "Failed logins",
        "hint": "Failed account login attempts in the last 24 hours.",
        "min": 0,
        "max": 10,
        "step": 1,
        "value": 5,
        "unit": "",
    },
    {
        "id": "merchant_risk",
        "label": "Merchant risk",
        "hint": "Merchant category and chargeback-history risk score.",
        "min": 0,
        "max": 1,
        "step": 0.01,
        "value": 0.82,
        "unit": "",
    },
    {
        "id": "card_present",
        "label": "Card present",
        "hint": "1 when the physical card was present, otherwise 0.",
        "min": 0,
        "max": 1,
        "step": 1,
        "value": 0,
        "unit": "",
    },
    {
        "id": "cards_per_device_7d",
        "label": "Cards per device",
        "hint": "Distinct cards used on this device in the last seven days.",
        "min": 1,
        "max": 20,
        "step": 1,
        "value": 8,
        "unit": "cards",
    },
    {
        "id": "geo_mismatch",
        "label": "Geo mismatch",
        "hint": "1 when browser, IP, and billing country signals disagree.",
        "min": 0,
        "max": 1,
        "step": 1,
        "value": 1,
        "unit": "",
    },
    {
        "id": "account_age_days",
        "label": "Account age",
        "hint": "Age of the customer account.",
        "min": 1,
        "max": 4000,
        "step": 1,
        "value": 45,
        "unit": "days",
    },
]

FEATURE_NAMES = [feature["id"] for feature in FEATURES]
TASK_NAMES = ["card_fraud", "account_takeover"]
NUM_LABELS = len(TASK_NAMES)
TASK_WEIGHTS = np.array([1.0, 0.35], dtype=np.float32)
ROUNDS = 140
SEED = 20261108


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def standardize(values: np.ndarray) -> np.ndarray:
    return (values - np.mean(values)) / (np.std(values) + 1e-9)


def make_dataset(size: int = 6500) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic synthetic fintech transactions."""
    rng = np.random.default_rng(SEED)

    amount = np.clip(np.exp(rng.normal(4.5, 1.15, size)), 5, 5000)
    hour = rng.integers(0, 24, size)
    velocity = np.clip(rng.poisson(2.1, size), 0, 20)
    distance = np.clip(np.exp(rng.normal(3.0, 1.35, size)), 0, 1200)
    device_trust = np.clip(rng.beta(5.0, 2.2, size), 0, 1)
    email_age = np.clip(np.exp(rng.normal(5.1, 1.45, size)), 1, 3000)
    failed_logins = np.clip(rng.poisson(0.35, size), 0, 10)
    merchant_risk = np.clip(rng.beta(2.0, 5.0, size), 0, 1)
    card_present = rng.binomial(1, 0.68, size)
    cards_per_device = np.clip(1 + rng.poisson(2.2, size), 1, 20)

    geo_base = 0.06 + 0.22 * (1.0 - device_trust) + 0.08 * (distance > 400)
    geo_mismatch = rng.binomial(1, np.clip(geo_base, 0, 0.75), size)
    account_age = np.clip(np.exp(rng.normal(5.8, 1.35, size)), 1, 4000)

    late_hour = ((hour <= 3) | (hour >= 23)).astype(float)
    log_amount = standardize(np.log1p(amount))
    log_distance = standardize(np.log1p(distance))
    log_email_age = standardize(np.log1p(email_age))
    log_account_age = standardize(np.log1p(account_age))

    fraud_logit = (
        -3.7
        + 0.72 * log_amount
        + 0.48 * log_distance
        + 0.34 * velocity
        + 1.45 * merchant_risk
        + 0.78 * (1.0 - card_present)
        + 0.52 * geo_mismatch
        + 0.19 * cards_per_device
        + 0.35 * late_hour
        - 0.62 * device_trust
        + rng.normal(0, 0.55, size)
    )
    ato_logit = (
        -4.1
        + 0.91 * failed_logins
        + 1.65 * (1.0 - device_trust)
        + 0.72 * geo_mismatch
        + 0.58 * (1.0 - card_present)
        - 0.48 * log_email_age
        - 0.28 * log_account_age
        + 0.27 * cards_per_device
        + 0.38 * late_hour
        + rng.normal(0, 0.65, size)
    )

    probs = np.column_stack([sigmoid(fraud_logit), sigmoid(ato_logit)])
    labels = (rng.random((size, 2)) < probs).astype(np.float32)

    features = np.column_stack(
        [
            amount,
            hour,
            velocity,
            distance,
            device_trust,
            email_age,
            failed_logins,
            merchant_risk,
            card_present,
            cards_per_device,
            geo_mismatch,
            account_age,
        ]
    ).astype(np.float32)
    return features, labels


def multi_task_objective(preds, train_data, _epoch):
    labels = train_data.get_label().reshape((NUM_LABELS, -1)).T
    scores = preds.reshape((NUM_LABELS, -1)).T
    probs = sigmoid(scores)
    task_grad = probs - labels
    task_hess = probs * (1.0 - probs)

    joint_grad = np.sum(task_grad * TASK_WEIGHTS, axis=1)
    joint_hess = np.sum(task_hess * TASK_WEIGHTS, axis=1)
    return (
        joint_grad.astype(np.float32),
        joint_hess.astype(np.float32),
        task_grad.T.reshape(-1).astype(np.float32),
        task_hess.T.reshape(-1).astype(np.float32),
    )


def train_models(features: np.ndarray, labels: np.ndarray):
    indexes = np.arange(len(features))
    train_idx, rest_idx = train_test_split(
        indexes, test_size=0.30, random_state=SEED, stratify=labels[:, 0]
    )
    valid_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, random_state=SEED, stratify=labels[rest_idx, 0]
    )

    params = {
        "verbosity": -1,
        "num_threads": 4,
        "num_leaves": 15,
        "max_depth": 6,
        "learning_rate": 0.04,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 0.2,
        "seed": SEED,
    }

    baseline_metrics = {}
    for task_index, task_name in enumerate(TASK_NAMES):
        train_set = lgb.Dataset(
            features[train_idx],
            label=labels[train_idx, task_index],
            feature_name=FEATURE_NAMES,
        )
        valid_set = lgb.Dataset(
            features[valid_idx],
            label=labels[valid_idx, task_index],
            reference=train_set,
        )
        model = lgb.train(
            {
                **params,
                "objective": "binary",
                "metric": "auc",
            },
            train_set,
            num_boost_round=ROUNDS,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        model.set_num_labels(1)
        probs = model.predict(features[test_idx])
        baseline_metrics[task_name] = {
            "auc": float(roc_auc_score(labels[test_idx, task_index], probs)),
            "log_loss": float(log_loss(labels[test_idx, task_index], probs)),
        }

    train_set = lgb.Dataset(
        features[train_idx],
        label=labels[train_idx],
        feature_name=FEATURE_NAMES,
    )
    valid_set = lgb.Dataset(
        features[valid_idx],
        label=labels[valid_idx],
        reference=train_set,
    )
    multi_model = lgb.train(
        {
            **params,
            "objective": "custom",
            "num_labels": NUM_LABELS,
            "tree_learner": "serial2",
            "metric": "None",
        },
        train_set,
        num_boost_round=ROUNDS,
        valid_sets=[valid_set],
        fobj=multi_task_objective,
    )
    multi_model.set_num_labels(NUM_LABELS)
    raw_scores = np.asarray(multi_model.predict(features[test_idx]))
    multi_probs = sigmoid(raw_scores)
    multi_metrics = {
        task_name: {
            "auc": float(roc_auc_score(labels[test_idx, task_index], multi_probs[:, task_index])),
            "log_loss": float(log_loss(labels[test_idx, task_index], multi_probs[:, task_index])),
        }
        for task_index, task_name in enumerate(TASK_NAMES)
    }

    return multi_model, {
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
        "positive_rates": labels.mean(axis=0).round(4).tolist(),
        "single_task": baseline_metrics,
        "multi_task": multi_metrics,
    }, features[test_idx], labels[test_idx]


def compact_tree(node: dict) -> dict:
    if "leaf_value" in node:
        return {"v": round(float(node["leaf_value"]), 8)}
    if node.get("decision_type") != "<=":
        raise ValueError(f"Unsupported decision type: {node.get('decision_type')}")
    return {
        "f": int(node["split_feature"]),
        "t": round(float(node["threshold"]), 8),
        "l": compact_tree(node["left_child"]),
        "r": compact_tree(node["right_child"]),
    }


def score_tree(node: dict, values: np.ndarray) -> float:
    while "f" in node:
        node = node["l"] if values[node["f"]] <= node["t"] else node["r"]
    return float(node["v"])


def export_task_model(model, task_index: int, test_features: np.ndarray) -> dict:
    task_name = TASK_NAMES[task_index]
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / f"{task_name}.txt"
        model.save_model2(str(model_path), num_label=task_index)
        task_model = lgb.Booster(model_file=str(model_path))
        task_model.set_num_labels(1)
        dumped = task_model.dump_model()
        raw_preds = np.asarray(
            task_model.predict(test_features[:100], raw_score=True)
        )

    trees = [
        compact_tree(tree_info["tree_structure"])
        for tree_info in dumped["tree_info"]
    ]
    exported_preds = np.array(
        [sum(score_tree(tree, row) for tree in trees) for row in test_features[:100]]
    )
    max_error = float(np.max(np.abs(raw_preds - exported_preds)))
    if max_error > 1e-5:
        raise AssertionError(f"Tree export mismatch for {task_name}: {max_error}")

    return {
        "name": task_name,
        "tree_count": len(trees),
        "trees": trees,
        "export_check_max_error": max_error,
    }


def main():
    features, labels = make_dataset()
    multi_model, metrics, test_features, _ = train_models(features, labels)

    tasks = [
        export_task_model(multi_model, task_index, test_features)
        for task_index in range(NUM_LABELS)
    ]
    for task in tasks:
        task.update(metrics["multi_task"][task["name"]])
        task["single_task"] = metrics["single_task"][task["name"]]

    output = {
        "title": "MT-GBM Fintech Risk Lab",
        "model_version": "LightGBM 3.3.2.99 / MT-GBM",
        "features": FEATURES,
        "tasks": tasks,
        "training": {
            "train_rows": metrics["train_rows"],
            "validation_rows": metrics["validation_rows"],
            "test_rows": metrics["test_rows"],
            "boosting_rounds": ROUNDS,
            "task_weights": TASK_WEIGHTS.tolist(),
            "positive_rates": metrics["positive_rates"],
            "seed": SEED,
        },
    }

    output_path = ROOT / "demo" / "web" / "model.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, separators=(",", ":")))

    print(f"wrote {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    for task in tasks:
        print(
            f"{task['name']}: MT-GBM AUC={task['auc']:.4f} "
            f"single-task AUC={task['single_task']['auc']:.4f} "
            f"trees={task['tree_count']} export_error={task['export_check_max_error']:.2e}"
        )


if __name__ == "__main__":
    main()
