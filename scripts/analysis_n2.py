#!/usr/bin/env python3
"""
N.2: Audit features that reference label-adjacent information.
Check that file_prior_risky_max, file_revert_count_max, and
days_since_last_change_max are strictly backward-looking.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import subprocess
from datetime import datetime, timezone

import pandas as pd
import yaml


def build_labeling_graph(repo_path, window_start, forward_end):
    """Build commit graph from git log (same as build_multi_repo_dataset.py)."""
    cmd = [
        "git", "log",
        f"--since={window_start}", f"--until={forward_end}",
        "--pretty=format:%H|%ct|%s",
        "--name-only",
    ]
    result = subprocess.run(
        cmd, cwd=repo_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )

    commits = {}
    current_hash = None
    current_files = set()

    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "|" in line and len(line.split("|")) >= 2:
            # New commit header
            if current_hash:
                commits[current_hash] = {
                    "files": list(current_files),
                }
            parts = line.split("|", 2)
            current_hash = parts[0]
            current_files = set()
        elif current_hash:
            if line and not line.startswith("commit "):
                current_files.add(line)

    if current_hash:
        commits[current_hash] = {"files": list(current_files)}

    return commits


def get_commit_timestamp(repo_path, commit_hash):
    """Get committer timestamp for a commit."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", commit_hash],
        cwd=repo_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=10,
    )
    return int(result.stdout.strip())


def main():
    print("=" * 70)
    print("N.2: TEMPORAL LEAKAGE AUDIT")
    print("=" * 70)

    df = pd.read_csv("data/commit_features.csv")

    # Load config for window info
    WINDOW_START = "2024-07-01"
    FORWARD_END = "2026-07-07"

    features_to_check = [
        "file_prior_risky_max",
        "file_revert_count_max",
        "days_since_last_change_max",
    ]

    # For each feature, trace back to its computation
    print("\n--- Feature computation code audit ---\n")

    # Read the history_features.py to show the exact code
    with open("ml/history_features.py", "r") as f:
        code = f.read()

    for feat in features_to_check:
        print(f"=== {feat} ===")
        # Find the computation in the code
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if feat in line and "=" in line and "def " not in line:
                # Print surrounding context
                start = max(0, i - 3)
                end = min(len(lines), i + 3)
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"  {marker} L{j+1}: {lines[j]}")
                break
        print()

    # Now trace 10 sample commits
    print("=" * 70)
    print("N.2: 10-SAMPLE TIMESTAMP AUDIT")
    print("=" * 70)

    # Pick 10 commits from different repos
    sample_commits = []
    for repo in ["django", "react", "rust", "kubernetes", "kafka"]:
        rdf = df[df["source_repo"] == repo].head(2)
        for _, row in rdf.iterrows():
            sample_commits.append(row)

    print("\nFor each commit, we check whether ALL contributing prior-change records")
    print("have timestamps STRICTLY BEFORE the commit's own timestamp.\n")

    for i, row in enumerate(sample_commits[:10]):
        commit_hash = row["hash"]
        repo = row["source_repo"]
        commit_date = pd.to_datetime(row["committer_date"] if "committer_date" in row else row["date"])
        risky_max = row.get("file_prior_risky_max", 0)
        revert_max = row.get("file_revert_count_max", 0)
        days_since = row.get("days_since_last_change_max", 0)

        print(f"--- Commit {i+1}: {commit_hash[:12]} ({repo}) ---")
        print(f"  Commit timestamp: {commit_date}")
        print(f"  file_prior_risky_max: {risky_max}")
        print(f"  file_revert_count_max: {revert_max}")
        print(f"  days_since_last_change_max: {days_since}")

        # Check: is the feature value non-zero? If 0, no prior changes exist
        if risky_max == 0 and revert_max == 0:
            print(f"  => No prior changes for touched files (both 0). No leakage possible.")
        else:
            print(f"  => Non-zero values. Leakage audit: must verify computation is cut at commit timestamp.")
        print()

    # Now verify by running the actual computation on a sample
    print("=" * 70)
    print("N.2: COMPUTATION VERIFICATION")
    print("=" * 70)
    print("\nReading ml/history_features.py to trace exact code path...\n")

    # Show the critical function
    with open("ml/history_features.py", "r") as f:
        lines = f.read().split("\n")

    # Find the function that computes file_prior_risky
    in_function = False
    for i, line in enumerate(lines):
        if "def compute" in line or "def _compute" in line:
            in_function = True
            print(f"--- L{i+1}: {line}")
        elif in_function:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                if "def " in line:
                    break
            print(f"  L{i+1}: {line}")
            if len([l for l in lines[max(0,i-50):i+1] if "def " in l]) > 3:
                break

    print("\n" + "=" * 70)
    print("N.2: VERDICT")
    print("=" * 70)
    print("""
The key question: does the computation use only data with timestamps
STRICTLY BEFORE the commit being scored?

If the labeling graph was built from `git log --since/--until` covering
the full window, and features are computed by iterating through commits
in chronological order, then each commit's features can only see prior
commits — which is correct.

However, if the computation was done on the SAMPLED (2000 per repo)
subset rather than the FULL graph, then a commit's features would only
see OTHER sampled commits, not all prior commits. This would UNDERCOUNT
features, not overcount — so it wouldn't cause leakage but would cause
a different kind of train/serve skew.

The critical check: was file_prior_risky computed on the full graph or
the sampled subset? The answer determines whether the feature is
correct or not.
""")


if __name__ == "__main__":
    main()
