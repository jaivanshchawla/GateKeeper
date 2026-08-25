#!/usr/bin/env python3
"""
Q.3: Re-measure headline ROC-AUC with 1,000 bootstrap CIs.
Bootstrap must resample ROWS, not folds.
"""
import sys
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score

# Load config and data
with open("ml/config.yaml") as f:
    config = yaml.safe_load(f)

FEATURE_COLS = config["feature_columns"]
df = pd.read_csv("data/commit_features.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Features: {len(FEATURE_COLS)}")

X = df[FEATURE_COLS].values
y = df["risky"].values
repos = df["source_repo"].values

print(f"Class balance: {y.mean():.4f} positive ({y.sum()}/{len(y)})")

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

# ── Protocol (a): Pooled random 80/20 ──
print("\n" + "=" * 60)
print("PROTOCOL (a): POOLED RANDOM 80/20")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

acc = (y_pred == y_test).mean()
prec = y_pred[y_pred == 1].sum() / max(y_pred.sum(), 1)
rec = y_pred[y_pred == 1 & (y_test == 1)].sum() / max(y_test.sum(), 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-10)
roc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)

roc_mean, roc_lo, roc_hi = bootstrap_auc_ci(y_test, y_prob, n_bootstrap=1000)

print(f"  ROC-AUC: {roc:.4f} (point)")
print(f"  Bootstrap mean: {roc_mean:.4f} [{roc_lo:.4f}, {roc_hi:.4f}]")
print(f"  PR-AUC: {pr_auc:.4f}  F1: {f1:.4f}  MCC: {mcc:.4f}  Acc: {acc:.4f}")

# ── Protocol (c): Leave-one-repo-out ──
print("\n" + "=" * 60)
print("PROTOCOL (c): LEAVE-ONE-REPO-OUT")
print("=" * 60)

all_repos = sorted(set(repos))
repo_results = {}
for held in all_repos:
    mask_train = repos != held
    mask_test = repos == held
    Xtr, ytr = X[mask_train], y[mask_train]
    Xte, yte = X[mask_test], y[mask_test]
    
    m = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
    m.fit(Xtr, ytr)
    yp = m.predict_proba(Xte)[:, 1]
    repo_results[held] = {
        "roc_auc": roc_auc_score(yte, yp),
        "pr_auc": average_precision_score(yte, yp),
        "n_test": len(yte),
    }

# Pooled LORO — need probabilities
all_yp_list = []
all_yt_list = []
for held in all_repos:
    mask_test = repos == held
    Xte, yte = X[mask_test], y[mask_test]
    m = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
    m.fit(X[repos != held], y[repos != held])
    yp = m.predict_proba(Xte)[:, 1]
    all_yp_list.append(yp)
    all_yt_list.append(yte)

all_yt_pooled = np.concatenate(all_yt_list)
all_yp_pooled = np.concatenate(all_yp_list)

loro_roc_mean, loro_roc_lo, loro_roc_hi = bootstrap_auc_ci(all_yt_pooled, all_yp_pooled, n_bootstrap=1000)
loro_pr_auc = average_precision_score(all_yt_pooled, all_yp_pooled)
base_rate = all_yt_pooled.mean()

print(f"\n  Per-repo ROC-AUC:")
for r in all_repos:
    print(f"    {r:<15} ROC-AUC={repo_results[r]['roc_auc']:.4f}  PR-AUC={repo_results[r]['pr_auc']:.4f}  n={repo_results[r]['n_test']}")

print(f"\n  Pooled LORO ROC-AUC: {loro_roc_mean:.4f} [{loro_roc_lo:.4f}, {loro_roc_hi:.4f}]")
print(f"  Pooled LORO PR-AUC: {loro_pr_auc:.4f}")
print(f"  PR-AUC lift: {loro_pr_auc - base_rate:.4f}")
print(f"  Base rate: {base_rate:.4f}")
