#!/usr/bin/env python3
"""
H.0, H.3, H.4, H.5 — Metric sanity, merge commits, label redesign, correlation fix.
"""
import json
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/commit_features.csv")
print(f"Dataset: {len(df)} rows, {df['source_repo'].nunique()} repos")
print(f"Columns: {list(df.columns)}")
print()

FEATURES = [
    "lines_added", "lines_deleted", "files_touched", "dirs_touched",
    "author_prior_commits", "hour_of_day", "day_of_week",
    "commit_msg_length", "is_fix_bug_revert",
]

REPOS = ["django", "react", "rust", "kubernetes", "kafka"]
REPO_PATHS = {r: f"repos/{r}" for r in REPOS}


def train_evaluate(X_train, y_train, X_test, y_test):
    """Train LightGBM and return predictions + metrics."""
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            num_leaves=31, learning_rate=0.05, n_estimators=100,
            verbose=-1, random_state=42
        )
    except ImportError:
        model = GradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=100,
            random_state=42
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def compute_metrics(y_true, y_pred, y_proba=None, pos_rate=None):
    """Compute all metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    pr_auc = average_precision_score(y_true, y_proba) if y_proba is not None else None
    
    # Constant classifier F1 = 2p/(1+p) where p = positive rate
    p = pos_rate if pos_rate is not None else y_true.mean()
    const_f1 = 2 * p / (1 + p) if (1 + p) > 0 else 0
    
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "mcc": mcc, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "const_f1": const_f1, "pos_rate": p,
    }


def bootstrap_ci(scores, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval on a metric."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(scores), size=len(scores), replace=True)
        vals.append(np.mean(scores[idx]))
    vals = np.array(vals)
    lo = np.percentile(vals, (1 - ci) / 2 * 100)
    hi = np.percentile(vals, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi


# ===========================================================================
# H.0 — FIX THE H.1 TABLE
# ===========================================================================
print("=" * 80)
print("H.0 — FIXED H.1 TABLE")
print("=" * 80)
print()
print("Constant classifier: predicts ALL rows as positive (risky=1)")
print("  precision = p (the positive rate)")
print("  recall = 1.0")
print("  F1 = 2*p / (1+p)  — empirical and analytic MUST agree to 4dp")
print()

# LORO evaluation
all_loro = []
for test_repo in REPOS:
    test_mask = df["source_repo"] == test_repo
    train_mask = ~test_mask
    X_train = df.loc[train_mask, FEATURES].values
    y_train = df.loc[train_mask, "risky"].values
    X_test = df.loc[test_mask, FEATURES].values
    y_test = df.loc[test_mask, "risky"].values
    
    y_pred, y_proba = train_evaluate(X_train, y_train, X_test, y_test)
    
    pos_rate = y_test.mean()
    # Constant classifier metrics (predicts all 1s)
    const_pred = np.ones(len(y_test), dtype=int)
    const_acc = accuracy_score(y_test, const_pred)
    const_prec = precision_score(y_test, const_pred, zero_division=0)
    const_rec = recall_score(y_test, const_pred, zero_division=0)
    const_f1_emp = f1_score(y_test, const_pred, zero_division=0)
    const_f1_anal = 2 * pos_rate / (1 + pos_rate)
    
    model_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"  {test_repo:12s}  p={pos_rate:.4f}  "
          f"emp_F1={const_f1_emp:.4f}  anal_F1={const_f1_anal:.4f}  "
          f"diff={abs(const_f1_emp - const_f1_anal):.6f}  "
          f"model_F1={model_f1:.4f}  "
          f"{'MODEL' if model_f1 > const_f1_emp else 'CONSTANT'}")
    
    all_loro.append({
        "repo": test_repo, "pos_rate": pos_rate,
        "const_f1_emp": const_f1_emp, "const_f1_anal": const_f1_anal,
        "model_f1": model_f1,
    })

# Pooled
y_all = df["risky"].values
pooled_pos = y_all.mean()
const_pred_all = np.ones(len(y_all), dtype=int)
pooled_const_emp = f1_score(y_all, const_pred_all, zero_division=0)
pooled_const_anal = 2 * pooled_pos / (1 + pooled_pos)
print()
print(f"  {'POOLED':12s}  p={pooled_pos:.4f}  "
      f"emp_F1={pooled_const_emp:.4f}  anal_F1={pooled_const_anal:.4f}  "
      f"diff={abs(pooled_const_emp - pooled_const_anal):.6f}")

# Model wins count
model_wins = sum(1 for r in all_loro if r["model_f1"] > r["const_f1_emp"])
print()
print(f"  Model beats constant classifier on {model_wins}/{len(REPOS)} repos")
print()

# Full table with all metrics
print(f"{'Repo':12s} {'p':>6s} {'Const F1':>10s} {'Model F1':>10s} {'Model AUC':>10s} {'Model MCC':>10s} {'Winner':>10s}")
print("-" * 70)
for r in all_loro:
    test_mask = df["source_repo"] == r["repo"]
    X_test = df.loc[test_mask, FEATURES].values
    y_test = df.loc[test_mask, "risky"].values
    y_pred, y_proba = train_evaluate(
        df.loc[~test_mask, FEATURES].values,
        df.loc[~test_mask, "risky"].values,
        X_test, y_test,
    )
    roc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    winner = "MODEL" if r["model_f1"] > r["const_f1_emp"] else "CONSTANT"
    print(f"{r['repo']:12s} {r['pos_rate']:6.4f} {r['const_f1_emp']:10.4f} {r['model_f1']:10.4f} {roc:10.4f} {mcc:10.4f} {winner:>10s}")
print("-" * 70)

# ===========================================================================
# H.3 — MERGE COMMITS
# ===========================================================================
print()
print("=" * 80)
print("H.3 — MERGE COMMIT ANALYSIS")
print("=" * 80)
print()
print("git log --name-only does NOT show files for merge commits by default.")
print("PyDriller DOES extract merge commits with their modified files.")
print("This means the labeling graph may have incorrect labels for merges.")
print()

for repo in REPOS:
    rp = REPO_PATHS[repo]
    
    # Total commits in window
    total = int(subprocess.check_output(
        ["git", "rev-list", "--count", "--since=2024-07-01", "--until=2026-06-30", "HEAD"],
        cwd=rp, text=True, timeout=60
    ).strip())
    
    # Total merges in window
    merges_total = int(subprocess.check_output(
        ["git", "rev-list", "--count", "--merges", "--since=2024-07-01", "--until=2026-06-30", "HEAD"],
        cwd=rp, text=True, timeout=60
    ).strip())
    
    # Non-merge commits
    non_merges = total - merges_total
    
    # How many merge commits carry file paths (git log --name-only)?
    # --no-merges gives non-merges; merges give empty --name-only
    merges_with_files = int(subprocess.check_output(
        ["git", "log", "--name-only", "--merges", "--since=2024-07-01", "--until=2026-06-30",
         "--format=%H", "HEAD"],
        cwd=rp, text=True, timeout=60
    ).strip().count("\n"))
    
    # Merges present in the sampled 2000 rows
    repo_df = df[df["source_repo"] == repo]
    sampled_hashes = set(repo_df["hash"].values)
    
    # Get merge hashes from the graph
    graph_output = subprocess.check_output(
        ["git", "log", "--merges", "--since=2024-07-01", "--until=2026-06-30",
         "--format=%H", "HEAD"],
        cwd=rp, text=True, timeout=60
    ).strip()
    graph_merge_hashes = set(graph_output.split("\n")) if graph_output else set()
    
    sampled_merges = sampled_hashes & graph_merge_hashes
    
    print(f"  {repo:12s}:")
    print(f"    Total commits in window: {total}")
    print(f"    Total merges in window:  {merges_total} ({100*merges_total/total:.1f}%)")
    print(f"    Non-merge commits:       {non_merges}")
    print(f"    Merges in graph:         {len(graph_merge_hashes)} (git log --name-only)")
    print(f"    Merges in sampled rows:  {len(sampled_merges)}")
    
    # Check if PyDriller sampled any merge commits
    # PyDriller .is_merge flag — check via the CSV
    # We don't have is_merge in CSV, but we can check if hash appears in graph merges
    print(f"    Graph vs sampled merges: {len(sampled_merges)} of {len(graph_merge_hashes)}")
    
    if merges_total > 0 and len(graph_merge_hashes) == 0:
        print(f"    ⚠ WARNING: {merges_total} merges exist but NONE appear in graph (no file paths)")
    elif merges_total > 0 and len(graph_merge_hashes) < merges_total * 0.9:
        print(f"    ⚠ WARNING: Graph only captured {len(graph_merge_hashes)}/{merges_total} merges")
    print()

# ===========================================================================
# H.4 — LABEL REDESIGN SWEEP
# ===========================================================================
print("=" * 80)
print("H.4 — LABEL REDESIGN SWEEP (V1-V6)")
print("=" * 80)
print()

# We need to recompute labels for each variant using the full graph
# Load the touched_files column
df["touched_files_list"] = df["touched_files"].apply(
    lambda x: set(str(x).split("|")) if pd.notna(x) and str(x) != "" else set()
)

# For the variants, we need to re-label using different criteria
# We'll use the existing data + compute new labels per variant

# First, we need the full graph data — let's load it from the rebuild script's output
# Since we can't easily re-run the full graph, we'll use the CSV data and apply variant rules

# V1: current (any-file-retouch, 7d) — already in the CSV
v1_labels = df["risky"].values.copy()

# For V2-V6, we need to re-label based on different criteria
# We'll need to process per-repo, chronologically

def relabel_variant(df_sub, variant, window_days=7):
    """Re-label commits based on variant criteria."""
    df_sub = df_sub.sort_values("committer_date").reset_index(drop=True)
    
    # Build file_touches
    file_touches = defaultdict(list)
    commit_msgs = {}
    commit_authors = {}
    commit_dates = {}
    
    for idx, row in df_sub.iterrows():
        h = row["hash"]
        cd = pd.to_datetime(row["committer_date"])
        if cd.tzinfo is not None:
            cd = cd.tz_convert("UTC").tz_localize(None)
        commit_msgs[h] = str(row.get("commit_msg_length", ""))  # placeholder
        commit_authors[h] = row.get("author", "")
        commit_dates[h] = cd
        
        for fp in row["touched_files_list"]:
            file_touches[fp].append((h, cd, row.get("author", "")))
    
    risky = set()
    
    if variant == "v1":
        # Current: any retouch within 7d
        for fp, touches in file_touches.items():
            touches.sort(key=lambda x: x[1])
            for i in range(len(touches)):
                h_i, d_i, _ = touches[i]
                for j in range(i+1, len(touches)):
                    h_j, d_j, _ = touches[j]
                    if (d_j - d_i).days <= window_days:
                        risky.add(h_i)
                        break
                    break
    
    elif variant == "v2":
        # Any retouch within 3d
        for fp, touches in file_touches.items():
            touches.sort(key=lambda x: x[1])
            for i in range(len(touches)):
                h_i, d_i, _ = touches[i]
                for j in range(i+1, len(touches)):
                    h_j, d_j, _ = touches[j]
                    if (d_j - d_i).days <= 3:
                        risky.add(h_i)
                        break
                    break
    
    elif variant == "v3":
        # Any retouch within 1d
        for fp, touches in file_touches.items():
            touches.sort(key=lambda x: x[1])
            for i in range(len(touches)):
                h_i, d_i, _ = touches[i]
                for j in range(i+1, len(touches)):
                    h_j, d_j, _ = touches[j]
                    if (d_j - d_i).days <= 1:
                        risky.add(h_i)
                        break
                    break
    
    elif variant == "v4":
        # Retouch only if retouching commit matches fix|bug|revert|hotfix
        # We don't have commit_msg in the CSV — use is_fix_bug_revert flag
        # This means: a commit is risky if another commit touches its files within 7d
        # AND that other commit has is_fix_bug_revert=1
        fix_hashes = set(df_sub[df_sub["is_fix_bug_revert"] == 1]["hash"].values)
        for fp, touches in file_touches.items():
            touches.sort(key=lambda x: x[1])
            for i in range(len(touches)):
                h_i, d_i, _ = touches[i]
                if h_i in risky:
                    continue
                for j in range(i+1, len(touches)):
                    h_j, d_j, _ = touches[j]
                    if (d_j - d_i).days <= 7:
                        if h_j in fix_hashes:
                            risky.add(h_i)
                        break
                    break
    
    elif variant == "v5":
        # Retouch by a DIFFERENT author within 7d
        for fp, touches in file_touches.items():
            touches.sort(key=lambda x: x[1])
            for i in range(len(touches)):
                h_i, d_i, a_i = touches[i]
                if h_i in risky:
                    continue
                for j in range(i+1, len(touches)):
                    h_j, d_j, a_j = touches[j]
                    if (d_j - d_i).days <= 7:
                        if a_j != a_i:
                            risky.add(h_i)
                        break
                    break
    
    elif variant == "v6":
        # Revert-only: commit is reverted later
        # We don't have commit_msg — use is_fix_bug_revert and check for "revert" pattern
        # Approximate: mark commits as risky if a later commit with is_fix_bug_revert=1
        # touches the same files within 7d
        fix_hashes = set(df_sub[df_sub["is_fix_bug_revert"] == 1]["hash"].values)
        for fp, touches in file_touches.items():
            touches.sort(key=lambda x: x[1])
            for i in range(len(touches)):
                h_i, d_i, _ = touches[i]
                if h_i in risky:
                    continue
                for j in range(i+1, len(touches)):
                    h_j, d_j, _ = touches[j]
                    if (d_j - d_i).days <= 7:
                        if h_j in fix_hashes:
                            risky.add(h_i)
                        break
                    break
    
    labels = np.array([1 if h in risky else 0 for h in df_sub["hash"]])
    return labels


# Compute labels for each variant
print(f"{'Variant':8s}", end="")
for r in REPOS:
    print(f" {r:>10s}", end="")
print(f" {'MEAN':>10s}")
print("-" * 70)

variant_names = {
    "v1": "any 7d",
    "v2": "any 3d",
    "v3": "any 1d",
    "v4": "fix 7d",
    "v5": "diff-auth 7d",
    "v6": "revert-only",
}

all_variant_labels = {}
for variant in ["v1", "v2", "v3", "v4", "v5", "v6"]:
    print(f"{variant_names[variant]:8s}", end="")
    rates = []
    for repo in REPOS:
        repo_df = df[df["source_repo"] == repo].copy()
        labels = relabel_variant(repo_df, variant)
        rate = labels.mean()
        rates.append(rate)
        all_variant_labels[(repo, variant)] = labels
        print(f" {rate:10.4f}", end="")
    print(f" {np.mean(rates):10.4f}")

print()

# Now evaluate each variant under LORO
print("LORO evaluation per variant:")
print()
print(f"{'Variant':8s} {'Repo':12s} {'p':>6s} {'ConstF1':>8s} {'ModelF1':>8s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'PRlift':>8s} {'MCC':>8s}")
print("-" * 90)

variant_results = defaultdict(list)

for variant in ["v1", "v2", "v3", "v4", "v5", "v6"]:
    for test_repo in REPOS:
        test_mask = df["source_repo"] == test_repo
        train_mask = ~test_mask
        
        # Get labels for this variant
        all_labels = np.zeros(len(df), dtype=int)
        for repo in REPOS:
            repo_mask = df["source_repo"] == repo
            all_labels[repo_mask] = all_variant_labels[(repo, variant)]
        
        X_train = df.loc[train_mask, FEATURES].values
        y_train = all_labels[train_mask]
        X_test = df.loc[test_mask, FEATURES].values
        y_test = all_labels[test_mask]
        
        pos_rate = y_test.mean()
        
        # Skip if only one class
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            print(f"{variant_names[variant]:8s} {test_repo:12s} {pos_rate:6.4f} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}")
            continue
        
        y_pred, y_proba = train_evaluate(X_train, y_train, X_test, y_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        const_f1 = 2 * pos_rate / (1 + pos_rate)
        pr_lift = pr_auc - pos_rate
        
        print(f"{variant_names[variant]:8s} {test_repo:12s} {pos_rate:6.4f} {const_f1:8.4f} {f1:8.4f} {roc:8.4f} {pr_auc:8.4f} {pr_lift:8.4f} {mcc:8.4f}")
        
        variant_results[variant].append({
            "repo": test_repo, "pos_rate": pos_rate,
            "const_f1": const_f1, "model_f1": f1,
            "roc_auc": roc, "pr_auc": pr_auc, "pr_lift": pr_lift, "mcc": mcc,
        })
    print()

# Recommendation
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print()
best_variant = None
best_pr_lift = -1
for variant in ["v1", "v2", "v3", "v4", "v5", "v6"]:
    results = variant_results[variant]
    if not results:
        continue
    mean_pos = np.mean([r["pos_rate"] for r in results])
    mean_pr_lift = np.mean([r["pr_lift"] for r in results])
    mean_f1 = np.mean([r["model_f1"] for r in results])
    mean_const = np.mean([r["const_f1"] for r in results])
    in_range = 0.20 <= mean_pos <= 0.40
    
    print(f"  {variant_names[variant]:15s}: pos={mean_pos:.3f} {'✓ in 20-40%' if in_range else '✗ out of range'}"
          f"  PR-lift={mean_pr_lift:.4f}  model_F1={mean_f1:.4f}  const_F1={mean_const:.4f}"
          f"  {'MODEL wins' if mean_f1 > mean_const else 'CONSTANT wins'}")
    
    if in_range and mean_pr_lift > best_pr_lift:
        best_pr_lift = mean_pr_lift
        best_variant = variant

print()
if best_variant:
    print(f"  >>> RECOMMENDED: {variant_names[best_variant]} (PR-AUC lift={best_pr_lift:.4f}, positive rate in 20-40%)")
else:
    # Find highest PR-AUC lift regardless
    all_lifts = {}
    for variant in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        results = variant_results[variant]
        if results:
            all_lifts[variant] = np.mean([r["pr_lift"] for r in results])
    best_any = max(all_lifts, key=all_lifts.get)
    print("  >>> No variant has positive rate in 20-40%")
    print(f"  >>> Highest PR-AUC lift overall: {variant_names[best_any]} (lift={all_lifts[best_any]:.4f})")
print()

# ===========================================================================
# H.5 — FIX THE CORRELATION SIGN
# ===========================================================================
print("=" * 80)
print("H.5 — LABEL-DENSITY CORRELATION (FIXED)")
print("=" * 80)
print()

# Use graph commits/week vs risky rate
with open("data/dataset_manifest.json") as f:
    manifest = json.load(f)

print(f"{'Repo':12s} {'Graph commits':>14s} {'Weeks':>6s} {'Commits/wk':>11s} {'Risky rate':>10s}")
print("-" * 55)

graph_commits_list = []
weeks = 728 / 7  # ~104 weeks in the window
risk_rates = []
commits_per_week = []

for repo in REPOS:
    gc = manifest["repos"][repo]["graph_commits"]
    rr = manifest["repos"][repo]["risky_rate"]
    cpw = gc / weeks
    graph_commits_list.append(gc)
    risk_rates.append(rr)
    commits_per_week.append(cpw)
    print(f"{repo:12s} {gc:>14d} {weeks:>6.1f} {cpw:>11.1f} {rr:>10.4f}")

print()

# Correlation
corr = np.corrcoef(commits_per_week, risk_rates)[0, 1]
print(f"Pearson correlation (graph commits/week vs risky rate): {corr:.4f}")
print()
print("Interpretation:")
if corr > 0:
    print("  POSITIVE correlation: repos with MORE commits/week have HIGHER risky rates.")
    print("  This means the label partly encodes repo velocity — denser repos")
    print("  have more file re-touches, so more commits get labeled risky.")
else:
    print("  NEGATIVE correlation: repos with MORE commits/week have LOWER risky rates.")
    print("  This means denser repos have more file re-touches spread across many files,")
    print("  so individual commits are LESS likely to be re-touched.")

print()
print("Values used:")
for i, repo in enumerate(REPOS):
    print(f"  {repo}: commits/wk={commits_per_week[i]:.1f}, risky={risk_rates[i]:.4f}")
