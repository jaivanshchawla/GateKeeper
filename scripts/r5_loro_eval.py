#!/usr/bin/env python3
"""
R.5: Cross-repo LORO evaluation with correct bootstrap CIs.

Bootstrap MUST resample ROWS, not folds. Q.3's mean CI was narrower than
every per-repo CI because it resampled folds (repos), not rows.
"""
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

# Load config and data
with open("ml/config.yaml") as f:
    config = yaml.safe_load(f)

FEATURE_COLS = config["feature_columns"]
df = pd.read_csv("data/commit_features.csv")
print(f"Loaded {len(df)} rows, {len(FEATURE_COLS)} features")

X = df[FEATURE_COLS].values
y = df["risky"].values
repos = df["source_repo"].values

all_repos = sorted(set(repos))
print(f"Repos: {all_repos}")
print(f"Class balance: {y.mean():.4f} ({y.sum()}/{len(y)})")


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI on ROC-AUC, resampling ROWS."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        samples.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.mean(samples)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_pr_auc_ci(y_true, y_prob, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI on PR-AUC, resampling ROWS."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        samples.append(average_precision_score(y_true[idx], y_prob[idx]))
    return float(np.mean(samples)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ── Leave-one-repo-out ──
print("\n" + "=" * 70)
print("LEAVE-ONE-REPO-OUT")
print("=" * 70)

all_yt = []
all_yp = []
repo_results = {}

for held in all_repos:
    mask_train = repos != held
    mask_test = repos == held
    Xtr, ytr = X[mask_train], y[mask_train]
    Xte, yte = X[mask_test], y[mask_test]

    m = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100,
                            random_state=42, verbose=-1)
    m.fit(Xtr, ytr)
    yp = m.predict_proba(Xte)[:, 1]

    roc = roc_auc_score(yte, yp)
    pr_auc = average_precision_score(yte, yp)

    # Bootstrap CI for this repo (resample rows within the test set)
    roc_mean, roc_lo, roc_hi = bootstrap_auc_ci(yte, yp)
    pr_mean, pr_lo, pr_hi = bootstrap_pr_auc_ci(yte, yp)

    repo_results[held] = {
        "roc_auc": roc, "roc_ci": (roc_mean, roc_lo, roc_hi),
        "pr_auc": pr_auc, "pr_ci": (pr_mean, pr_lo, pr_hi),
        "n_test": len(yte), "base_rate": yte.mean(),
    }

    all_yt.extend(yte.tolist())
    all_yp.extend(yp.tolist())

# ── Per-repo table ──
print(f"\n{'Repo':<15} {'n':>5} {'ROC-AUC':>10} {'95% CI':>22} {'PR-AUC':>10} {'95% CI':>22} {'Base':>6}")
print("-" * 95)
for r in all_repos:
    rr = repo_results[r]
    rm, rl, rh = rr["roc_ci"]
    pm, pl, ph = rr["pr_ci"]
    print(f"{r:<15} {rr['n_test']:>5} {rr['roc_auc']:>10.4f} [{rl:.4f}, {rh:.4f}]  {rr['pr_auc']:>10.4f} [{pl:.4f}, {ph:.4f}]  {rr['base_rate']:>6.4f}")

# ── Pooled LORO (resample ROWS) ──
all_yt_arr = np.array(all_yt)
all_yp_arr = np.array(all_yp)

pool_roc, pool_roc_lo, pool_roc_hi = bootstrap_auc_ci(all_yt_arr, all_yp_arr, n_bootstrap=1000)
pool_pr, pool_pr_lo, pool_pr_hi = bootstrap_pr_auc_ci(all_yt_arr, all_yp_arr, n_bootstrap=1000)
pool_base = all_yt_arr.mean()
pr_lift = pool_pr - pool_base

print(f"\n{'POOLED':<15} {len(all_yt_arr):>5} {pool_roc:>10.4f} [{pool_roc_lo:.4f}, {pool_roc_hi:.4f}]  {pool_pr:>10.4f} [{pool_pr_lo:.4f}, {pool_pr_hi:.4f}]  {pool_base:>6.4f}")
print(f"\nPR-AUC lift over base rate: {pr_lift:.4f}")

# ── Verify CI is row-resampled ──
# If CI were fold-resampled, pooled CI would be narrower than per-repo CIs.
# With row resampling, pooled CI should be comparable.
repo_ci_widths = [repo_results[r]["roc_ci"][2] - repo_results[r]["roc_ci"][1] for r in all_repos]
pool_ci_width = pool_roc_hi - pool_roc_lo
print(f"\nCI width check (should be similar, not pooled much narrower):")
print(f"  Per-repo widths: {[f'{w:.4f}' for w in repo_ci_widths]}")
print(f"  Pooled width:    {pool_ci_width:.4f}")
if pool_ci_width < min(repo_ci_widths) * 0.5:
    print("  WARNING: Pooled CI is much narrower than per-repo — possible fold resampling!")
else:
    print("  OK: Pooled CI width is consistent with row resampling.")

# ── Determinism check ──
print("\n" + "=" * 70)
print("DETERMINISM CHECK")
print("=" * 70)
Xtr2, Xte2, ytr2, yte2 = train_test_split if False else (None, None, None, None)
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
m1 = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
m1.fit(Xtr, ytr)
yp1 = m1.predict_proba(Xte)[:, 1]
f1_1 = roc_auc_score(yte, yp1)

m2 = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
m2.fit(Xtr, ytr)
yp2 = m2.predict_proba(Xte)[:, 1]
f1_2 = roc_auc_score(yte, yp2)

print(f"  Run 1 ROC-AUC: {f1_1:.8f}")
print(f"  Run 2 ROC-AUC: {f1_2:.8f}")
print(f"  Match: {f1_1 == f1_2}")
