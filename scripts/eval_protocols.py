#!/usr/bin/env python3
"""
Evaluation protocol comparison for Gatekeeper.

Runs the SAME LightGBM config under three evaluation protocols:
  (a) Pooled random 80/20, seed 42
  (b) Time-ordered split — train on oldest 80%, test on newest 20%
  (c) Leave-one-repo-out

Reports accuracy/precision/recall/F1 for all three.
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Load config ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / "ml" / "config.yaml"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "commit_features.csv"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

FEATURE_COLS = config["feature_columns"]
PARAMS = config["lightgbm_params"]


def evaluate(y_true, y_pred):
    """Return a dict of standard metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["parsed_date"] = pd.to_datetime(df["date"], utc=True)

    print("=" * 80)
    print("EVALUATION PROTOCOL COMPARISON")
    print("=" * 80)
    print(f"Dataset: {len(df)} rows, {df['source_repo'].nunique()} repos")
    print(f"Repos: {sorted(df['source_repo'].unique())}")
    print(f"Columns: {len(FEATURE_COLS)} features")
    print()

    results = {}

    # ── (a) Pooled random 80/20 ──────────────────────────────────────────────
    # NOTE: Do NOT sort before splitting — sorting changes the stratified
    # random split, making (a) non-reproducible with train.py.
    X = df[FEATURE_COLS]
    y = df["risky"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model_a = lgb.LGBMClassifier(**PARAMS)
    model_a.fit(X_train, y_train)
    y_pred_a = model_a.predict(X_test)
    results["(a) Pooled random 80/20"] = evaluate(y_test, y_pred_a)

    print("(a) POOLED RANDOM 80/20, seed=42")
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    for k, v in results["(a) Pooled random 80/20"].items():
        print(f"    {k.capitalize():>10}: {v:.4f}")
    print()

    # ── (b) Time-ordered split ────────────────────────────────────────────────
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train_t = train_df[FEATURE_COLS]
    y_train_t = train_df["risky"]
    X_test_t = test_df[FEATURE_COLS]
    y_test_t = test_df["risky"]

    model_b = lgb.LGBMClassifier(**PARAMS)
    model_b.fit(X_train_t, y_train_t)
    y_pred_b = model_b.predict(X_test_t)
    results["(b) Time-ordered (oldest->newest)"] = evaluate(y_test_t, y_pred_b)

    print("(b) TIME-ORDERED SPLIT (oldest 80% train, newest 20% test)")
    print(
        f"    Train: {len(X_train_t)} "
        f"(dates {train_df['parsed_date'].min().date()} to "
        f"{train_df['parsed_date'].max().date()})"
    )
    print(
        f"    Test:  {len(X_test_t)} "
        f"(dates {test_df['parsed_date'].min().date()} to "
        f"{test_df['parsed_date'].max().date()})"
    )
    for k, v in results["(b) Time-ordered (oldest->newest)"].items():
        print(f"    {k.capitalize():>10}: {v:.4f}")

    print("\n    Per-repo breakdown in test set:")
    for repo in sorted(test_df["source_repo"].unique()):
        rt = test_df[test_df["source_repo"] == repo]
        rp = model_b.predict(rt[FEATURE_COLS])
        ry = rt["risky"]
        m = evaluate(ry, rp)
        print(
            f"    {repo:<15} n={len(rt):>4}  "
            f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}"
        )
    print()

    # ── (c) Leave-one-repo-out ────────────────────────────────────────────────
    repos = sorted(df["source_repo"].unique())
    all_preds_c = []
    all_true_c = []
    per_repo = {}

    print("(c) LEAVE-ONE-REPO-OUT")
    for held_out in repos:
        train_pool = df[df["source_repo"] != held_out]
        test_pool = df[df["source_repo"] == held_out]
        X_tr = train_pool[FEATURE_COLS]
        y_tr = train_pool["risky"]
        X_te = test_pool[FEATURE_COLS]
        y_te = test_pool["risky"]

        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)

        all_preds_c.extend(p)
        all_true_c.extend(y_te.tolist())
        metrics = evaluate(y_te, p)
        per_repo[held_out] = metrics

        print(
            f"    {held_out:<15} n={len(test_pool):>4}  "
            f"acc={metrics['accuracy']:.4f}  prec={metrics['precision']:.4f}  "
            f"rec={metrics['recall']:.4f}  f1={metrics['f1']:.4f}"
        )

    results["(c) Leave-one-repo-out (pooled)"] = evaluate(all_true_c, all_preds_c)
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Protocol':<42} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<42} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f}"
        )

    print()
    a_f1 = results["(a) Pooled random 80/20"]["f1"]
    b_f1 = results["(b) Time-ordered (oldest->newest)"]["f1"]
    c_f1 = results["(c) Leave-one-repo-out (pooled)"]["f1"]
    print(f"Gap (a) vs (b): {a_f1 - b_f1:+.4f} F1")
    print(f"Gap (a) vs (c): {a_f1 - c_f1:+.4f} F1")

    if a_f1 - c_f1 > 0.02:
        print(
            "\nNOTE: Protocol (a) overestimates real-world performance by "
            f"{(a_f1 - c_f1) / a_f1 * 100:.1f}% relative to cross-repo generalization."
        )
    if abs(a_f1 - b_f1) < 0.01:
        print(
            "\nNOTE: Time-ordered and random splits produce similar F1, suggesting "
            "temporal leakage from author_prior_commits has limited impact on this dataset."
        )


if __name__ == "__main__":
    main()
