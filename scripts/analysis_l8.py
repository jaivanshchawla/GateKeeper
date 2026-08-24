#!/usr/bin/env python3
"""
L.8: Line-level revert tracking label.

A commit is "risky" if specific LINES it introduced are later modified
by a commit whose message matches fix|bug|revert|hotfix within 7 days.

Uses git log -S to find commits that add/remove specific strings,
then checks if those commits are fix/bug/revert commits.

Does NOT use is_fix_bug_revert as a feature (that's V4 leakage).
"""

import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

# Config
COMMIT_HASHES = {
    "django": None,  # Will be loaded from CSV
    "react": None,
    "rust": None,
    "kubernetes": None,
    "kafka": None,
}
REPO_PATHS = {
    "django": "repos/django",
    "react": "repos/react",
    "rust": "repos/rust",
    "kubernetes": "repos/kubernetes",
    "kafka": "repos/kafka",
}

FIX_PATTERN = re.compile(r"(fix|bug|revert|hotfix)", re.IGNORECASE)
LABEL_WINDOW_DAYS = 7


def get_diff_lines(repo_path: str, commit_hash: str) -> dict[str, list[str]]:
    """Get lines added by a commit, per file.

    Returns {file_path: [added_line_content, ...]}
    """
    try:
        result = subprocess.run(
            ["git", "show", "--format=", "--diff-filter=A", "--unified=0",
             commit_hash],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return {}

        added_lines = defaultdict(list)
        current_file = None
        for line in result.stdout.split("\n"):
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("+") and not line.startswith("+++") and current_file:
                content = line[1:].strip()
                if content and len(content) > 3:  # Skip very short lines
                    added_lines[current_file].append(content)

        return dict(added_lines)
    except Exception:
        return {}


def get_fix_commits_for_files(
    repo_path: str,
    commit_date: str,
    files: list[str],
    window_days: int = LABEL_WINDOW_DAYS,
) -> list[dict]:
    """Find fix/bug/revert commits touching the same files within window.

    Returns list of {hash, date, message, files}.
    """
    try:
        # Use git log to find fix commits after this commit
        result = subprocess.run(
            ["git", "log", f"--since={commit_date}",
             f"--until={commit_date}+{window_days} days",
             "--format=%H|%aI|%s", "--all"],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return []

        fix_commits = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            h, date, msg = parts

            # Check if message matches fix pattern
            if not FIX_PATTERN.search(msg):
                continue

            # Check if this commit touches any of our files
            file_result = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h],
                cwd=repo_path, capture_output=True, text=True, timeout=10
            )
            if file_result.returncode == 0:
                touched = set(file_result.stdout.strip().split("\n"))
                if touched & set(files):
                    fix_commits.append({
                        "hash": h.strip(),
                        "date": date,
                        "message": msg,
                        "files": list(touched),
                    })

        return fix_commits
    except Exception:
        return []


def check_lines_modified(
    repo_path: str,
    added_lines: dict[str, list[str]],
    fix_commit: dict,
) -> bool:
    """Check if a fix commit modifies any of the lines we added."""
    try:
        # Get the diff of the fix commit
        result = subprocess.run(
            ["git", "show", "--format=", "--diff-filter=M", "--unified=0",
             fix_commit["hash"]],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return False

        # Parse the fix commit's diff
        modified_lines = defaultdict(set)
        current_file = None
        for line in result.stdout.split("\n"):
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("-") and not line.startswith("---") and current_file:
                content = line[1:].strip()
                if content and len(content) > 3:
                    modified_lines[current_file].add(content)

        # Check overlap: does the fix modify any line we added?
        for fp, lines in added_lines.items():
            if fp in modified_lines:
                for line in lines:
                    if line in modified_lines[fp]:
                        return True

        return False
    except Exception:
        return False


def compute_v8_label(
    repo_name: str,
    repo_path: str,
    commits: pd.DataFrame,
    window_days: int = LABEL_WINDOW_DAYS,
) -> dict[str, bool]:
    """Compute V8 (line-level revert) labels for all commits.

    Returns {commit_hash: is_risky}
    """
    labels = {}
    total = len(commits)

    for i, (_, row) in enumerate(commits.iterrows()):
        h = row["hash"]
        if (i + 1) % 50 == 0:
            print(f"  [{repo_name}] {i+1}/{total} commits processed...")

        # Get lines added by this commit
        added_lines = get_diff_lines(repo_path, h)
        if not added_lines:
            labels[h] = False
            continue

        files = list(added_lines.keys())
        commit_date = row.get("committer_date", row.get("date", ""))

        # Find fix commits touching same files within window
        fix_commits = get_fix_commits_for_files(repo_path, commit_date, files, window_days)

        is_risky = False
        for fc in fix_commits:
            if check_lines_modified(repo_path, added_lines, fc):
                is_risky = True
                break

        labels[h] = is_risky

    return labels


def loro_eval_v8(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Evaluate V8 label under leave-one-repo-out protocol."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, matthews_corrcoef
    )

    repos = df["source_repo"].unique()
    results = {}

    for held_out in repos:
        train = df[df["source_repo"] != held_out]
        test = df[df["source_repo"] == held_out]

        X_train = train[feature_columns].values
        y_train = train["risky_v8"].values
        X_test = test[feature_columns].values
        y_test = test["risky_v8"].values

        # Skip if no positive examples
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            print(f"  {held_out}: skipped (no positive or all-positive in test)")
            continue

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        mcc = matthews_corrcoef(y_test, (y_proba > 0.5).astype(int))
        pos_rate = y_test.mean()

        # Constant classifier F1 for this base rate
        const_f1 = 2 * pos_rate / (1 + pos_rate) if pos_rate > 0 else 0

        results[held_out] = {
            "pos_rate": pos_rate,
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "pr_lift": pr_auc - pos_rate,
            "mcc": mcc,
            "const_f1": const_f1,
        }

    # Print table
    print("\n=== V8 (Line-Level Revert) — LORO Evaluation ===")
    print(f"{'Repo':<14} {'pos_rate':>10} {'ROC-AUC':>10} {'PR-AUC':>10} "
          f"{'PR lift':>10} {'MCC':>10} {'const F1':>10}")
    print("-" * 84)

    for name, r in results.items():
        print(f"{name:<14} {r['pos_rate']:>10.4f} {r['roc_auc']:>10.4f} "
              f"{r['pr_auc']:>10.4f} {r['pr_lift']:>10.4f} "
              f"{r['mcc']:>10.4f} {r['const_f1']:>10.4f}")

    if results:
        mean_pos = np.mean([r["pos_rate"] for r in results.values()])
        mean_roc = np.mean([r["roc_auc"] for r in results.values()])
        mean_pr = np.mean([r["pr_auc"] for r in results.values()])
        mean_lift = np.mean([r["pr_lift"] for r in results.values()])
        mean_mcc = np.mean([r["mcc"] for r in results.values()])
        mean_const = np.mean([r["const_f1"] for r in results.values()])
        print("-" * 84)
        print(f"{'MEAN':<14} {mean_pos:>10.4f} {mean_roc:>10.4f} "
              f"{mean_pr:>10.4f} {mean_lift:>10.4f} "
              f"{mean_mcc:>10.4f} {mean_const:>10.4f}")


def main():
    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['source_repo'].nunique()} repos")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    feature_columns = config.get("feature_columns", [])
    # Remove is_fix_bug_revert from features (V4 leakage prevention)
    feature_columns = [c for c in feature_columns if c != "is_fix_bug_revert"]
    print(f"Features (without is_fix_bug_revert): {len(feature_columns)}")

    # Sample 200 commits per repo for tractability
    SAMPLE_PER_REPO = 200
    sampled_dfs = []
    for repo_name in df["source_repo"].unique():
        repo_df = df[df["source_repo"] == repo_name]
        if len(repo_df) > SAMPLE_PER_REPO:
            sampled = repo_df.sample(n=SAMPLE_PER_REPO, random_state=42)
        else:
            sampled = repo_df
        sampled_dfs.append(sampled)
    sampled_df = pd.concat(sampled_dfs)
    print(f"Sampled {len(sampled_df)} commits for V8 evaluation")

    # Compute V8 labels per repo on sampled data
    all_labels = {}
    for repo_name in sampled_df["source_repo"].unique():
        repo_path = REPO_PATHS.get(repo_name)
        if not repo_path or not os.path.exists(repo_path):
            print(f"WARNING: {repo_path} not found, skipping {repo_name}")
            continue

        repo_df = sampled_df[sampled_df["source_repo"] == repo_name]
        print(f"\nComputing V8 labels for {repo_name} ({len(repo_df)} commits)...")

        start = time.perf_counter()
        labels = compute_v8_label(repo_name, repo_path, repo_df)
        elapsed = time.perf_counter() - start

        pos_count = sum(labels.values())
        pos_rate = pos_count / len(labels) if labels else 0
        print(f"  {repo_name}: {pos_count}/{len(labels)} risky ({pos_rate:.4f}) "
              f"in {elapsed:.1f}s")

        all_labels.update(labels)

    # Add V8 labels to sampled DataFrame
    sampled_df = sampled_df.copy()
    sampled_df["risky_v8"] = sampled_df["hash"].map(all_labels).fillna(False).astype(bool)

    # Print per-repo positive rates
    print("\n=== V8 Per-Repo Positive Rates ===")
    for repo in sampled_df["source_repo"].unique():
        mask = sampled_df["source_repo"] == repo
        rate = sampled_df.loc[mask, "risky_v8"].mean()
        print(f"  {repo}: {rate:.4f}")

    # Compare with V1
    print("\n=== V1 vs V8 Positive Rates (on sampled data) ===")
    for repo in sampled_df["source_repo"].unique():
        mask = sampled_df["source_repo"] == repo
        v1_rate = sampled_df.loc[mask, "risky"].mean()
        v8_rate = sampled_df.loc[mask, "risky_v8"].mean()
        print(f"  {repo}: V1={v1_rate:.4f}, V8={v8_rate:.4f}, "
              f"delta={v8_rate - v1_rate:+.4f}")

    # Evaluate under LORO
    loro_eval_v8(sampled_df, feature_columns)


if __name__ == "__main__":
    main()
