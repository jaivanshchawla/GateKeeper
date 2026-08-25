#!/usr/bin/env python3
"""
N.1-N.4: Leakage control with bootstrap CIs and feature group ablations.
Under LORO, evaluate ROC-AUC with bootstrap 95% CIs for:
  (a) 9-feature baseline
  (b) 36 features (full)
  (c) 36 minus file-history group (M.1a)
  (d) 36 minus author-familiarity group (M.1b)
  (e) 36 minus change-shape group (M.1c)
  (f) 36 minus coupling group (M.1d)
  (g-i) Individual suspects: file_prior_risky_max, file_revert_count_max,
        days_since_last_change_max
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier


# ── Feature groups ────────────────────────────────────────────────────

BASE_FEATURES = [
    "lines_added", "lines_deleted", "files_touched", "dirs_touched",
    "author_prior_commits", "hour_of_day", "day_of_week",
    "commit_msg_length", "is_fix_bug_revert",
]

FILE_HISTORY = [
    "file_prior_changes_max", "file_prior_changes_mean",
    "file_prior_risky_max", "file_prior_risky_mean",
    "file_revert_count_max", "file_revert_count_mean",
    "file_age_days_max", "file_age_days_mean",
    "file_authors_count_max", "file_authors_count_mean",
    "days_since_last_change_max", "days_since_last_change_mean",
]

AUTHOR_FAMILIARITY = [
    "author_file_prior_commits_max", "author_file_prior_commits_mean",
    "author_dir_prior_commits_max", "author_dir_prior_commits_mean",
    "is_author_first_touch_file", "is_author_first_touch_dir",
    "author_days_since_last_commit",
]

CHANGE_SHAPE = [
    "churn_ratio", "change_entropy", "max_file_churn",
    "is_test_only", "test_to_code_ratio", "config_touch",
    "is_merge", "files_per_dir_ratio",
]

ALL_FEATURES = BASE_FEATURES + FILE_HISTORY + AUTHOR_FAMILIARITY + CHANGE_SHAPE


def train_evaluate(X_train, y_train, X_test, y_test):
    """Train LightGBM and return predictions."""
    model = LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=100,
        verbose=-1, random_state=42,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {"y_prob": y_prob, "y_pred": y_pred, "model": model}


def bootstrap_ci(values, n_resamples=1000, confidence=0.95):
    """Bootstrap 95% CI on the mean of values."""
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, (1 - confidence) / 2 * 100)
    hi = np.percentile(means, (1 + confidence) / 2 * 100)
    return np.mean(values), lo, hi


def loro_evaluate(df, feature_cols, n_splits=5):
    """Leave-one-repo-out evaluation with per-repo and pooled ROC-AUC."""
    repos = df["source_repo"].unique()
    all_y_true = []
    all_y_prob = []
    per_repo = {}

    for held_out in repos:
        train_df = df[df["source_repo"] != held_out]
        test_df = df[df["source_repo"] == held_out]

        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["risky"].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df["risky"].values

        result = train_evaluate(X_train, y_train, X_test, y_test)
        auc = roc_auc_score(y_test, result["y_prob"])
        per_repo[held_out] = auc
        all_y_true.extend(y_test)
        all_y_prob.extend(result["y_prob"])

    pooled_auc = roc_auc_score(all_y_true, all_y_prob)
    return pooled_auc, per_repo


def bootstrap_loro(df, feature_cols, n_bootstrap=200):
    """Bootstrap LORO ROC-AUC: resample rows within each fold."""
    repos = df["source_repo"].unique()
    rng = np.random.RandomState(42)
    aucs = []

    for b in range(n_bootstrap):
        pooled_aucs = []
        for held_out in repos:
            train_df = df[df["source_repo"] != held_out]
            test_df = df[df["source_repo"] == held_out]

            # Bootstrap: resample rows within train and test
            train_boot = train_df.sample(n=len(train_df), replace=True, random_state=rng)
            test_boot = test_df.sample(n=len(test_df), replace=True, random_state=rng)

            X_train = train_boot[feature_cols].fillna(0).values
            y_train = train_boot["risky"].values
            X_test = test_boot[feature_cols].fillna(0).values
            y_test = test_boot["risky"].values

            if len(np.unique(y_test)) < 2:
                continue

            result = train_evaluate(X_train, y_train, X_test, y_test)
            try:
                auc = roc_auc_score(y_test, result["y_prob"])
                pooled_aucs.append(auc)
            except ValueError:
                continue

        if pooled_aucs:
            aucs.append(np.mean(pooled_aucs))

    mean, lo, hi = bootstrap_ci(np.array(aucs))
    return mean, lo, hi, aucs


def main():
    print("=" * 70)
    print("N.1-N.4: LEAKAGE CONTROL AND FEATURE GROUP ABLATIONS")
    print("=" * 70)

    # Load data
    df = pd.read_csv("data/commit_features.csv")
    print(f"\nDataset: {len(df)} rows, {df['source_repo'].nunique()} repos")
    print(f"Repos: {dict(df['source_repo'].value_counts())}")
    print(f"Label distribution: {dict(df['risky'].value_counts())}")

    # Define ablation configs
    ablations = {
        "(a) 9-feature baseline": BASE_FEATURES,
        "(b) 36 features (full)": ALL_FEATURES,
        "(c) 36 minus file-history": [f for f in ALL_FEATURES if f not in FILE_HISTORY],
        "(d) 36 minus author-familiarity": [f for f in ALL_FEATURES if f not in AUTHOR_FAMILIARITY],
        "(e) 36 minus change-shape": [f for f in ALL_FEATURES if f not in CHANGE_SHAPE],
        "(g) minus file_prior_risky_max": [f for f in ALL_FEATURES if f != "file_prior_risky_max"],
        "(h) minus file_revert_count_max": [f for f in ALL_FEATURES if f != "file_revert_count_max"],
        "(i) minus days_since_last_change_max": [f for f in ALL_FEATURES if f != "days_since_last_change_max"],
    }

    # Run ablations with bootstrap CIs
    print("\n" + "=" * 70)
    print("N.1: LORO ROC-AUC WITH BOOTSTRAP 95% CIs")
    print("=" * 70)

    results = {}
    for name, features in ablations.items():
        print(f"\n--- {name} ({len(features)} features) ---")
        mean, lo, hi, aucs = bootstrap_loro(df, features, n_bootstrap=100)
        per_repo_mean = {}
        # Also get per-repo breakdown for the final run
        pooled, per_repo = loro_evaluate(df, features)
        results[name] = {
            "mean": mean, "lo": lo, "hi": hi,
            "per_repo": per_repo, "pooled": pooled,
            "n_features": len(features),
        }
        print(f"  ROC-AUC: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        print(f"  Per-repo: ", end="")
        for r, v in sorted(per_repo.items()):
            print(f"{r}={v:.4f} ", end="")
        print()

    # Print comparison table
    print("\n" + "=" * 70)
    print("N.1: COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Config':<40} {'Features':>8} {'ROC-AUC':>10} {'95% CI':>18} {'Delta vs baseline':>18}")
    print("-" * 96)
    baseline_mean = results["(a) 9-feature baseline"]["mean"]
    for name, r in results.items():
        delta = r["mean"] - baseline_mean
        ci = f"[{r['lo']:.4f}, {r['hi']:.4f}]"
        delta_str = f"{delta:+.4f}" if "(a)" not in name else "---"
        ci_overlap = "OVERLAPS" if r["lo"] <= baseline_mean <= results["(a) 9-feature baseline"]["hi"] else ""
        print(f"{name:<40} {r['n_features']:>8} {r['mean']:>10.4f} {ci:>18} {delta_str:>18} {ci_overlap}")

    # Cumulative deltas
    print("\n" + "=" * 70)
    print("N.1: CUMULATIVE GROUP DELTAS")
    print("=" * 70)
    group_deltas = {
        "M.1a (file-history)": results["(c) 36 minus file-history"]["mean"],
        "M.1b (author-familiarity)": results["(d) 36 minus author-familiarity"]["mean"],
        "M.1c (change-shape)": results["(e) 36 minus change-shape"]["mean"],
    }
    full_mean = results["(b) 36 features (full)"]["mean"]
    base_mean = results["(a) 9-feature baseline"]["mean"]
    print(f"Baseline (9 features): {base_mean:.4f}")
    print(f"Full (36 features):    {full_mean:.4f} (total delta: {full_mean - base_mean:+.4f})")
    print()
    for group_name, abl_mean in group_deltas.items():
        delta = full_mean - abl_mean
        print(f"  Removing {group_name}: {delta:+.4f} contribution")
        # Check CI overlap with baseline
        abl_result = [r for n, r in results.items() if group_name.split("(")[1].split(")")[0] in n.lower().replace("-", "")]
        print(f"    => Ablated model ROC-AUC {abl_mean:.4f}")

    # CIs overlapping baseline
    print("\nGroups whose CI overlaps baseline:")
    for name, r in results.items():
        if "(a)" in name or "(b)" in name:
            continue
        bl = results["(a) 9-feature baseline"]
        if r["lo"] <= bl["mean"] <= r["hi"] or bl["lo"] <= r["mean"] <= bl["hi"]:
            print(f"  ✓ {name}: [{r['lo']:.4f}, {r['hi']:.4f}] overlaps baseline [{bl['lo']:.4f}, {bl['hi']:.4f}]")

    # Save results
    with open("data/n1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to data/n1_results.json")


if __name__ == "__main__":
    main()
