#!/usr/bin/env python3
"""
P.3+P.4: Cross-repo LORO evaluation with 1000-resample bootstrap CIs.
Verifies the promotion gate and computes honest headline metrics.
"""

import json
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def bootstrap_ci(values, n_resamples=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    alpha = (1 - ci) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return np.mean(means), lo, hi


def train_evaluate(X_train, y_train, X_test, y_test):
    """Train LightGBM and evaluate on test set."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = 0.5

    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except ValueError:
        pr_auc = 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def main():
    with open("ml/config.yaml") as f:
        cfg = yaml.safe_load(f)
    features = cfg["feature_columns"]

    df = pd.read_csv("data/commit_features.csv")
    print(f"Dataset: {len(df)} rows, {df['source_repo'].nunique()} repos")
    print(f"Features: {len(features)}")
    print(f"Positive rate: {df['risky'].mean():.4f}")

    X = df[features].copy()
    y = df["risky"].values
    groups = df["source_repo"].values

    # Ensure no NaN
    X = X.fillna(0)

    # ── Leave-One-Repo-Out ──
    logo = LeaveOneGroupOut()
    all_metrics = []
    per_repo = {}

    print("\n=== Leave-One-Repo-Out (35 features, LGBM) ===")
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        repo = groups[test_idx[0]]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        result = train_evaluate(X_train, y_train, X_test, y_test)

        # Per-row accuracy for bootstrap
        row_correct = (result["y_pred"] == y_test).astype(float)

        # Bootstrap CI on ROC-AUC (resample rows, not folds)
        y_test_np = y_test if isinstance(y_test, np.ndarray) else np.array(y_test)
        y_proba_np = result["y_proba"]

        # Bootstrap on ROC-AUC
        n_boot = 1000
        rng = np.random.RandomState(42)
        roc_samples = []
        for _ in range(n_boot):
            idxs = rng.choice(len(y_test_np), size=len(y_test_np), replace=True)
            try:
                roc_samples.append(roc_auc_score(y_test_np[idxs], y_proba_np[idxs]))
            except ValueError:
                roc_samples.append(0.5)
        roc_samples = np.array(roc_samples)
        roc_mean = np.mean(roc_samples)
        roc_lo = np.percentile(roc_samples, 2.5)
        roc_hi = np.percentile(roc_samples, 97.5)

        # Bootstrap on accuracy (as proxy for MCC)
        acc_mean, acc_lo, acc_hi = bootstrap_ci(row_correct, n_resamples=n_boot)

        print(f"  {repo:15s}: F1={result['f1']:.4f}, ROC-AUC={result['roc_auc']:.4f} "
              f"[{roc_lo:.4f}, {roc_hi:.4f}], ACC={result['accuracy']:.4f} "
              f"[{acc_lo:.4f}, {acc_hi:.4f}], MCC={result['mcc']:.4f}, "
              f"PR-AUC={result['pr_auc']:.4f}, n={len(test_idx)}")

        per_repo[repo] = {
            "f1": result["f1"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "roc_auc": result["roc_auc"],
            "roc_auc_ci_lo": float(roc_lo),
            "roc_auc_ci_hi": float(roc_hi),
            "pr_auc": result["pr_auc"],
            "mcc": result["mcc"],
            "n_test": len(test_idx),
            "positive_rate": float(y_test.mean()),
        }
        all_metrics.append(result)

    # Pooled metrics across all folds
    all_y_true = np.concatenate([y[logo.split(X, y, groups).__next__()[1]] for _ in [1]])
    # Actually need to collect from the loop — let me recompute
    # Pool predictions from all folds
    pooled_y_true = []
    pooled_y_pred = []
    pooled_y_proba = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        result = train_evaluate(X_train, y_train, X_test, y_test)
        pooled_y_true.extend(y_test)
        pooled_y_pred.extend(result["y_pred"])
        pooled_y_proba.extend(result["y_proba"])

    pooled_y_true = np.array(pooled_y_true)
    pooled_y_pred = np.array(pooled_y_pred)
    pooled_y_proba = np.array(pooled_y_proba)

    pooled_acc = accuracy_score(pooled_y_true, pooled_y_pred)
    pooled_f1 = f1_score(pooled_y_true, pooled_y_pred, zero_division=0)
    pooled_prec = precision_score(pooled_y_true, pooled_y_pred, zero_division=0)
    pooled_rec = recall_score(pooled_y_true, pooled_y_pred, zero_division=0)
    pooled_mcc = matthews_corrcoef(pooled_y_true, pooled_y_pred)
    pooled_roc = roc_auc_score(pooled_y_true, pooled_y_proba)
    pooled_pr = average_precision_score(pooled_y_true, pooled_y_proba)

    # Bootstrap CIs on pooled metrics
    row_correct = (pooled_y_pred == pooled_y_true).astype(float)
    acc_mean, acc_lo, acc_hi = bootstrap_ci(row_correct, n_resamples=1000)

    rng = np.random.RandomState(42)
    roc_samples = []
    for _ in range(1000):
        idxs = rng.choice(len(pooled_y_true), size=len(pooled_y_true), replace=True)
        try:
            roc_samples.append(roc_auc_score(pooled_y_true[idxs], pooled_y_proba[idxs]))
        except ValueError:
            roc_samples.append(0.5)
    roc_samples = np.array(roc_samples)

    print(f"\n{'=' * 80}")
    print("POOLED (cross-repo, 5-fold):")
    print(f"  ROC-AUC:  {pooled_roc:.4f} [{np.percentile(roc_samples, 2.5):.4f}, {np.percentile(roc_samples, 97.5):.4f}]")
    print(f"  F1:       {pooled_f1:.4f}")
    print(f"  Accuracy: {pooled_acc:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
    print(f"  Precision:{pooled_prec:.4f}")
    print(f"  Recall:   {pooled_rec:.4f}")
    print(f"  MCC:      {pooled_mcc:.4f}")
    print(f"  PR-AUC:   {pooled_pr:.4f}")

    # ── Per-repo summary table ──
    print(f"\n{'=' * 80}")
    print("PER-REPO SUMMARY:")
    print(f"{'Repo':15s} {'F1':>8s} {'ROC-AUC':>14s} {'PR-AUC':>8s} {'MCC':>8s} {'n':>6s} {'pos%':>8s}")
    print("-" * 80)
    for repo, m in per_repo.items():
        print(f"{repo:15s} {m['f1']:8.4f} {m['roc_auc']:6.4f} [{m['roc_auc_ci_lo']:.4f},{m['roc_auc_ci_hi']:.4f}] "
              f"{m['pr_auc']:8.4f} {m['mcc']:8.4f} {m['n_test']:6d} {m['positive_rate']:8.4f}")

    mean_f1 = np.mean([m["f1"] for m in per_repo.values()])
    mean_roc = np.mean([m["roc_auc"] for m in per_repo.values()])
    mean_mcc = np.mean([m["mcc"] for m in per_repo.values()])
    mean_pr = np.mean([m["pr_auc"] for m in per_repo.values()])
    print(f"{'MEAN':15s} {mean_f1:8.4f} {mean_roc:14.4f} {mean_pr:8.4f} {mean_mcc:8.4f}")

    # ── Compare against v5 baseline ──
    v5_roc_cross = 0.6744
    print(f"\n{'=' * 80}")
    print("PROMOTION GATE COMPARISON:")
    print(f"  v5 baseline (roc_auc_cross_repo): {v5_roc_cross:.4f}")
    print(f"  v7 (this run, mean LORO ROC-AUC):  {mean_roc:.4f}")
    print(f"  Delta:                              {mean_roc - v5_roc_cross:+.4f}")
    print(f"  Decision: {'PROMOTE (v7 wins)' if mean_roc > v5_roc_cross else 'KEEP v5 (v7 does not improve)'}")

    # Save results
    results = {
        "v7_loro_roc_auc": float(mean_roc),
        "v5_baseline_roc_auc": float(v5_roc_cross),
        "delta": float(mean_roc - v5_roc_cross),
        "pooled_roc_auc": float(pooled_roc),
        "pooled_f1": float(pooled_f1),
        "per_repo": per_repo,
    }
    with open("data/p3_p4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to data/p3_p4_results.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
