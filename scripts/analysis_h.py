#!/usr/bin/env python3
"""
Parts H.1 + H.2: Baseline comparison and discrimination analysis under LORO.
Run from gatekeeper/ directory.
"""

import sys
import time

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── Config ─────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
with open("ml/config.yaml") as _f:
    FEATURE_COLS = yaml.safe_load(_f)["feature_columns"]

RANDOM_SEED = 42
N_BOOTSTRAP = 1000


def load_data():
    df = pd.read_csv("data/commit_features.csv")
    df["committer_date"] = pd.to_datetime(df["committer_date"], utc=True)
    return df


def train_lgbm(X_train, y_train, X_test):
    """Train LightGBM, return predictions + probabilities."""
    model = LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=100,
        random_state=RANDOM_SEED, verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def constant_risky_baseline(y_test):
    """Always predict 1 (risky)."""
    return np.ones(len(y_test), dtype=int), np.ones(len(y_test))


def stratified_random_baseline(y_test, p_risky):
    """Random predictions with probability p_risky."""
    rng = np.random.RandomState(RANDOM_SEED)
    y_proba = rng.random(len(y_test))
    y_pred = (y_proba < p_risky).astype(int)
    return y_pred, y_proba


def single_feature_baseline(X_test, threshold):
    """Predict 1 if lines_added > threshold."""
    y_pred = (X_test["lines_added"] > threshold).astype(int)
    y_proba = y_pred.astype(float)  # binary
    return y_pred, y_proba


def compute_metrics(y_true, y_pred, y_proba):
    """Compute all requested metrics."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # ROC-AUC (needs at least 2 classes in y_true)
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = float("nan")

    # PR-AUC
    if len(np.unique(y_true)) >= 2:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        metrics["pr_auc"] = float("nan")

    return metrics


def analytic_constant_f1(p):
    """F1 for always-predict-positive: 2p/(1+p)."""
    return 2 * p / (1 + p)


def bootstrap_roc_auc_ci(y_true, y_proba, n_boot=1000, ci=0.95):
    """Bootstrap CI on ROC-AUC."""
    rng = np.random.RandomState(RANDOM_SEED)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    lo = np.percentile(boot_aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_aucs, (1 + ci) / 2 * 100)
    return float(np.mean(boot_aucs)), float(lo), float(hi)


def main():
    t0 = time.time()
    print("=" * 70)
    print("PARTS H.1 + H.2: BASELINE COMPARISON & DISCRIMINATION ANALYSIS")
    print("=" * 70)

    df = load_data()
    print(f"\nDataset: {len(df)} rows, {df['source_repo'].nunique()} repos")

    # ── H.1: Baseline Comparison under LORO ───────────────────────────
    print(f"\n{'='*70}")
    print("H.1: BASELINE COMPARISON UNDER LEAVE-ONE-REPO-OUT")
    print(f"{'='*70}")

    all_results = {}

    for held_out in df["source_repo"].unique():
        print(f"\n--- Held-out: {held_out} ---")

        train_mask = df["source_repo"] != held_out
        test_mask = df["source_repo"] == held_out

        X_train = df.loc[train_mask, FEATURE_COLS]
        y_train = df.loc[train_mask, "risky"]
        X_test = df.loc[test_mask, FEATURE_COLS]
        y_test = df.loc[test_mask, "risky"]

        p_risky = y_train.mean()  # base rate from training set
        median_lines = X_train["lines_added"].median()

        print(f"  Train: {len(X_train)} rows (risky rate: {p_risky:.3f})")
        print(f"  Test:  {len(X_test)} rows (risky rate: {y_test.mean():.3f})")

        # 1. Trained model
        y_pred, y_proba = train_lgbm(X_train, y_train, X_test)
        m_model = compute_metrics(y_test, y_pred, y_proba)

        # 2. Always-predict-risky
        y_pred_c, y_proba_c = constant_risky_baseline(y_test)
        m_const = compute_metrics(y_test, y_pred_c, y_proba_c)

        # 3. Stratified random guess
        y_pred_r, y_proba_r = stratified_random_baseline(y_test, p_risky)
        m_random = compute_metrics(y_test, y_pred_r, y_proba_r)

        # 4. Single-feature threshold (lines_added > median)
        y_pred_s, y_proba_s = single_feature_baseline(X_test, median_lines)
        m_single = compute_metrics(y_test, y_pred_s, y_proba_s)

        # Analytic constant-classifier F1
        const_f1_analytic = analytic_constant_f1(p_risky)

        # Print results
        header = f"  {'Method':<25} {'F1':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'MCC':>6} {'ROC-AUC':>8} {'PR-AUC':>8}"
        print(header)
        print(f"  {'-'*75}")
        for name, m in [
            ("Model (LightGBM)", m_model),
            ("Always-risky", m_const),
            ("Random (base rate)", m_random),
            (f"lines_added>{median_lines:.0f}", m_single),
        ]:
            print(f"  {name:<25} {m['f1']:>6.4f} {m['accuracy']:>6.4f} "
                  f"{m['precision']:>6.4f} {m['recall']:>6.4f} "
                  f"{m['mcc']:>6.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")

        print(f"  Analytic constant F1 = 2p/(1+p) = {const_f1_analytic:.4f}  "
              f"(p={p_risky:.4f})")

        # Comparison verdict
        if m_const["f1"] > m_model["f1"]:
            print(f"  *** CONSTANT CLASSIFIER BEATS MODEL: {m_const['f1']:.4f} > {m_model['f1']:.4f}")
        if const_f1_analytic > m_model["f1"]:
            print(f"  *** ANALYTIC CONSTANT F1 BEATS MODEL: {const_f1_analytic:.4f} > {m_model['f1']:.4f}")

        all_results[held_out] = {
            "model": m_model, "const": m_const, "random": m_random,
            "single": m_single, "p_risky": p_risky,
            "const_f1_analytic": const_f1_analytic,
        }

    # Pooled
    print("\n--- POOLED (all repos) ---")
    X = df[FEATURE_COLS]
    y = df["risky"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    p_risky = y_train.mean()
    median_lines = X_train["lines_added"].median()

    y_pred, y_proba = train_lgbm(X_train, y_train, X_test)
    m_model = compute_metrics(y_test, y_pred, y_proba)
    y_pred_c, y_proba_c = constant_risky_baseline(y_test)
    m_const = compute_metrics(y_test, y_pred_c, y_proba_c)
    y_pred_r, y_proba_r = stratified_random_baseline(y_test, p_risky)
    m_random = compute_metrics(y_test, y_pred_r, y_proba_r)
    y_pred_s, y_proba_s = single_feature_baseline(X_test, median_lines)
    m_single = compute_metrics(y_test, y_pred_s, y_proba_s)
    const_f1_analytic = analytic_constant_f1(p_risky)

    header = f"  {'Method':<25} {'F1':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'MCC':>6} {'ROC-AUC':>8} {'PR-AUC':>8}"
    print(header)
    print(f"  {'-'*75}")
    for name, m in [
        ("Model (LightGBM)", m_model),
        ("Always-risky", m_const),
        ("Random (base rate)", m_random),
        (f"lines_added>{median_lines:.0f}", m_single),
    ]:
        print(f"  {name:<25} {m['f1']:>6.4f} {m['accuracy']:>6.4f} "
              f"{m['precision']:>6.4f} {m['recall']:>6.4f} "
              f"{m['mcc']:>6.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")
    print(f"  Analytic constant F1 = 2p/(1+p) = {const_f1_analytic:.4f}")

    # Summary comparison
    print(f"\n{'='*70}")
    print("H.1 SUMMARY: Model F1 vs Constant-classifier F1")
    print(f"{'='*70}")
    print(f"  {'Repo':<15} {'Model F1':>9} {'Const F1':>9} {'Analytic':>9} {'Winner':>10}")
    print(f"  {'-'*55}")
    model_wins = 0
    for repo, r in all_results.items():
        mf = r["model"]["f1"]
        cf = r["const"]["f1"]
        ca = r["const_f1_analytic"]
        winner = "MODEL" if mf > ca else "CONSTANT"
        if winner == "MODEL":
            model_wins += 1
        print(f"  {repo:<15} {mf:>9.4f} {cf:>9.4f} {ca:>9.4f} {winner:>10}")
    pooled_mf = m_model["f1"]
    pooled_ca = const_f1_analytic
    pooled_winner = "MODEL" if pooled_mf > pooled_ca else "CONSTANT"
    print(f"  {'POOLED':<15} {pooled_mf:>9.4f} {m_const['f1']:>9.4f} {pooled_ca:>9.4f} {pooled_winner:>10}")
    print(f"\n  Model wins: {model_wins}/5 repos (+ pooled)")

    # ── H.2: Does the model discriminate? ──────────────────────────────
    print(f"\n{'='*70}")
    print("H.2: DOES THE MODEL DISCRIMINATE?")
    print(f"{'='*70}")

    print(f"\n  {'Repo':<15} {'ROC-AUC':>8} {'95% CI':>16} {'PR-AUC':>8} {'PR-AUC lift':>11} {'PR-AUC - p':>11}")
    print(f"  {'-'*72}")

    for held_out in df["source_repo"].unique():
        train_mask = df["source_repo"] != held_out
        test_mask = df["source_repo"] == held_out

        X_train = df.loc[train_mask, FEATURE_COLS]
        y_train = df.loc[train_mask, "risky"]
        X_test = df.loc[test_mask, FEATURE_COLS]
        y_test = df.loc[test_mask, "risky"]

        p_risky = y_test.mean()

        y_pred, y_proba = train_lgbm(X_train, y_train, X_test)

        roc = roc_auc_score(y_test, y_proba)
        pr = average_precision_score(y_test, y_proba)
        pr_lift = pr - p_risky  # lift over base rate

        # Bootstrap CI on ROC-AUC
        mean_roc, lo, hi = bootstrap_roc_auc_ci(y_test.values, y_proba)

        ci_ok = "SIGNAL" if lo > 0.5 else "WEAK/NO SIGNAL"
        print(f"  {held_out:<15} {roc:>8.4f} [{lo:.4f}, {hi:.4f}] "
              f"{pr:>8.4f} {pr_lift:>+10.4f} {pr-p_risky:>+10.4f}  {ci_ok}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("  ROC-AUC > 0.5 = model has signal (better than random)")
    print("  ROC-AUC CI containing 0.5 = no reliable signal")
    print("  ROC-AUC ~0.65+ with F1 losing to constant = real signal,")
    print("    but wrong threshold and wrong headline metric")
    print("  PR-AUC lift > 0 = model ranks risky commits higher than random")

    total_time = time.time() - t0
    print(f"\nTotal analysis time: {total_time:.1f}s")
    print("H.1 + H.2 COMPLETE")


if __name__ == "__main__":
    main()
