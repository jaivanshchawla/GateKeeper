#!/usr/bin/env python3
"""
T.7: Retrain with parity-verified features (35 features, email-based author identity).
Evaluate under cross-repo LORO with 1000 bootstrap CIs.
Log to MLflow but do NOT promote — the fixed gate handles promotion.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import skops.io as sio

sys.path.insert(0, ".")
from ml.train import load_config

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
config = load_config("ml/config.yaml")
FEATURE_COLS = config["feature_columns"]
assert len(FEATURE_COLS) == 35, f"Expected 35 features, got {len(FEATURE_COLS)}"

# Load data
df = pd.read_csv(DATA_DIR / "commit_features.csv")
print(f"Loaded {len(df)} rows, {len(FEATURE_COLS)} features")
print(f"Repos: {df['source_repo'].value_counts().to_dict()}")
print(f"Risky rate: {df['risky'].mean():.4f}")

X = df[FEATURE_COLS].fillna(0)
y = df["risky"]
repos = df["source_repo"].values


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, seed=42):
    """Bootstrap CI for ROC-AUC, resampling ROWS."""
    rng = np.random.RandomState(seed)
    samples = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        samples.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.mean(samples)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def train_evaluate(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=100,
        random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "y_prob": y_prob,
        "model": model,
    }


# Cross-repo LORO
print("\n" + "=" * 60)
print("CROSS-REPO LORO EVALUATION")
print("=" * 60)

all_repos = sorted(set(repos))
repo_metrics = {}
for held in all_repos:
    mask_train = repos != held
    mask_test = repos == held
    Xtr, ytr = X.loc[mask_train, FEATURE_COLS].values, y[mask_train].values
    Xte, yte = X.loc[mask_test, FEATURE_COLS].values, y[mask_test].values
    m = train_evaluate(Xtr, ytr, Xte, yte)
    repo_metrics[held] = m
    print(f"  {held:<15} ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")

# Pooled predictions for CI
all_yt = np.concatenate([y[repos == r].values for r in all_repos if r in repo_metrics])
all_yp = np.concatenate([repo_metrics[r]["y_prob"] for r in all_repos if r in repo_metrics])
roc_mean, roc_lo, roc_hi = bootstrap_auc_ci(all_yt, all_yp)

# Per-repo CIs
print(f"\n{'Repo':<15} {'ROC-AUC':>10} {'95% CI':>25}")
print("-" * 50)
for r in all_repos:
    if r not in repo_metrics:
        continue
    m = repo_metrics[r]
    yt = y[repos == r].values
    yp = m["y_prob"]
    rm, rl, rh = bootstrap_auc_ci(yt, yp)
    print(f"  {r:<15} {rm:.4f} [{rl:.4f}, {rh:.4f}]")

mean_roc = np.mean([repo_metrics[r]["roc_auc"] for r in all_repos if r in repo_metrics])
mean_pr = np.mean([repo_metrics[r]["pr_auc"] for r in all_repos if r in repo_metrics])
mean_f1 = np.mean([repo_metrics[r]["f1"] for r in all_repos if r in repo_metrics])
mean_mcc = np.mean([repo_metrics[r]["mcc"] for r in all_repos if r in repo_metrics])

print(f"\n  MEAN ROC-AUC: {mean_roc:.4f} (pooled CI: [{roc_lo:.4f}, {roc_hi:.4f}])")
print(f"  MEAN PR-AUC: {mean_pr:.4f}")
print(f"  MEAN F1: {mean_f1:.4f}")
print(f"  MEAN MCC: {mean_mcc:.4f}")

# Train final model on all data for deployment
print("\n" + "=" * 60)
print("TRAINING FINAL MODEL")
print("=" * 60)

final_model = lgb.LGBMClassifier(
    num_leaves=31, learning_rate=0.05, n_estimators=100,
    random_state=42, verbose=-1,
)
final_model.fit(X.values, y.values)
print(f"Model type: {type(final_model).__name__}")

# Save model
MODEL_DIR.mkdir(exist_ok=True)
model_path = MODEL_DIR / "gatekeeper_risk_model.skops"
sio.dump(final_model, model_path)
print(f"Saved model to {model_path}")

# Log to MLflow
print("\n" + "=" * 60)
print("LOGGING TO MLFLOW")
print("=" * 60)

tracking_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("gatekeeper")

with mlflow.start_run() as run:
    mlflow.log_params({
        "model_type": type(final_model).__name__,
        "n_features": len(FEATURE_COLS),
        "n_rows": len(df),
        "n_repos": len(all_repos),
        "eval_protocol": "cross_repo_loro",
        "eval_comparison": "roc_auc_cross_repo",
        "source": "t7_retrain_parity_verified",
        "promoted": "pending_gate",  # gate decides
    })

    # Log cross-repo metrics
    mlflow.log_metrics({
        "roc_auc": float(mean_roc),
        "roc_auc_cross_repo": float(mean_roc),
        "roc_auc_pooled": float(roc_mean),
        "pr_auc": float(mean_pr),
        "f1": float(mean_f1),
        "mcc": float(mean_mcc),
    })

    # Log per-repo metrics
    for r in all_repos:
        if r in repo_metrics:
            m = repo_metrics[r]
            mlflow.log_metrics({
                f"roc_auc_{r}": m["roc_auc"],
                f"pr_auc_{r}": m["pr_auc"],
                f"f1_{r}": m["f1"],
            })

    mlflow.sklearn.log_model(
        final_model,
        "model",
        serialization_format="skops",
        registered_model_name="GatekeeperRiskPredictor",
        skops_trusted_types=[
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier",
            "numpy.dtype",
            "numpy.ndarray",
            "pandas.core.frame.DataFrame",
            "pandas.core.series.Series",
        ],
    )
    print(f"Logged to MLflow run: {run.info.run_id}")
    print(f"roc_auc_cross_repo = {mean_roc:.4f}")

print("\nDone. Model registered as GatekeeperRiskPredictor.")
print(f"Honest headline: ROC-AUC {mean_roc:.4f} (cross-repo LORO)")
