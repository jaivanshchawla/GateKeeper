#!/usr/bin/env python3
"""
Leakage-aware evaluation for Gatekeeper.

Implements:
  (a) Pooled random 80/20 (leaky baseline, for contrast)
  (b) Purged time-ordered split
  (c) Leave-one-repo-out
  (d) GroupKFold by repo
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, train_test_split

warnings.filterwarnings("ignore")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "ml" / "config.yaml"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "commit_features.csv"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

FEATURE_COLS = config["feature_columns"]
PARAMS = config["lightgbm_params"]


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def purged_time_ordered(df, feature_cols, params, purge_days=7):
    """Purged time-ordered split: train on oldest 80%, purge boundary, test on rest."""
    df = df.sort_values("parsed_date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    split_date = df.iloc[split_idx]["parsed_date"]
    purge_start = split_date - pd.Timedelta(days=purge_days)
    purge_end = split_date + pd.Timedelta(days=purge_days)

    train = df[df["parsed_date"] < purge_start]
    purged = df[(df["parsed_date"] >= purge_start) & (df["parsed_date"] <= purge_end)]
    test = df[df["parsed_date"] > purge_end]

    if len(test) == 0:
        return None

    model = lgb.LGBMClassifier(**params)
    model.fit(train[feature_cols], train["risky"])
    y_pred = model.predict(test[feature_cols])
    return evaluate(test["risky"], y_pred), len(train), len(purged), len(test)


def main():
    df = pd.read_csv(DATA_PATH)
    df["parsed_date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("parsed_date").reset_index(drop=True)

    print("=" * 80)
    print("LEAKAGE-AWARE EVALUATION")
    print("=" * 80)
    print(f"Dataset: {len(df)} rows, {df['source_repo'].nunique()} repos")
    print()

    results = {}

    # (a) Pooled random 80/20
    X = df[FEATURE_COLS]
    y = df["risky"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model_a = lgb.LGBMClassifier(**PARAMS)
    model_a.fit(X_train, y_train)
    y_pred_a = model_a.predict(X_test)
    results["(a) Pooled random 80/20"] = evaluate(y_test, y_pred_a)

    # (b) Purged time-ordered
    purged_result = purged_time_ordered(df, FEATURE_COLS, PARAMS)
    if purged_result:
        metrics_b, n_train, n_purged, n_test = purged_result
        results["(b) Purged time-ordered"] = metrics_b
        print(f"(b) Purged: train={n_train}, purged={n_purged}, test={n_test}")

    # (c) Leave-one-repo-out
    repos = sorted(df["source_repo"].unique())
    all_preds_c, all_true_c = [], []
    per_repo_c = {}
    for held_out in repos:
        train_pool = df[df["source_repo"] != held_out]
        test_pool = df[df["source_repo"] == held_out]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(train_pool[FEATURE_COLS], train_pool["risky"])
        p = m.predict(test_pool[FEATURE_COLS])
        all_preds_c.extend(p)
        all_true_c.extend(test_pool["risky"].tolist())
        per_repo_c[held_out] = evaluate(test_pool["risky"], p)
    results["(c) Leave-one-repo-out"] = evaluate(all_true_c, all_preds_c)

    # (d) GroupKFold
    gkf = GroupKFold(n_splits=5)
    fold_f1s = []
    for _, test_idx in gkf.split(df, df["risky"], df["source_repo"]):
        train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(df.iloc[train_idx][FEATURE_COLS], df.iloc[train_idx]["risky"])
        p = m.predict(df.iloc[test_idx][FEATURE_COLS])
        fold_f1s.append(f1_score(df.iloc[test_idx]["risky"], p, zero_division=0))
    results["(d) GroupKFold by repo"] = {
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.mean(fold_f1s),
    }

    # Summary
    print()
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Protocol':<42} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        acc = f"{m['accuracy']:.4f}" if not np.isnan(m.get("accuracy", np.nan)) else "  N/A"
        prec = f"{m['precision']:.4f}" if not np.isnan(m.get("precision", np.nan)) else "  N/A"
        rec = f"{m['recall']:.4f}" if not np.isnan(m.get("recall", np.nan)) else "  N/A"
        f1 = f"{m['f1']:.4f}"
        print(f"{name:<42} {acc:>8} {prec:>8} {rec:>8} {f1:>8}")

    # Per-repo (c)
    print()
    print("Per-repo leave-one-repo-out:")
    for repo, m in per_repo_c.items():
        print(
            f"  {repo:<15} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} f1={m['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
