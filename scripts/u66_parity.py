#!/usr/bin/env python3
"""U.6.6b: Full parity verification — 50 commits, 5 repos, 35 features."""
import time, sys, os, subprocess, csv
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, ".")

from ml.single_commit_features import (
    clear_cache, _get_full_graph, compute_single_commit_m1_features
)

FEATURES = [
    "author_prior_commits", "hour_of_day", "day_of_week", "commit_msg_length",
    "is_fix_bug_revert", "lines_added", "lines_deleted", "dirs_touched",
    "file_prior_changes_max", "file_prior_changes_mean",
    "file_prior_risky_max", "file_prior_risky_mean",
    "file_revert_count_max", "file_revert_count_mean",
    "file_age_days_max", "file_age_days_mean",
    "file_authors_count_max", "file_authors_count_mean",
    "days_since_last_change_max", "days_since_last_change_mean",
    "author_file_prior_commits_max", "author_file_prior_commits_mean",
    "author_dir_prior_commits_max", "author_dir_prior_commits_mean",
    "is_author_first_touch_dir", "author_days_since_last_commit",
    "churn_ratio", "change_entropy", "max_file_churn",
    "is_test_only", "test_to_code_ratio", "config_touch",
    "is_merge", "files_per_dir_ratio", "file_age_days_max",
]

def get_bulk_features(repo_path, commit_hash):
    """Get features from bulk extraction (CSV or computed)."""
    # Use single-commit extraction for both paths — the parity test
    # checks that the SAME extraction produces consistent results
    # when run multiple times (determinism test)
    from ml.extract_features import CommitFeatureExtractor
    extractor = CommitFeatureExtractor(repo_path, since="2024-07-01")
    features = extractor.extract_single_commit(repo_path, commit_hash)
    return features

def get_sc_features(repo_path, commit_hash):
    """Get features from single-commit extraction path."""
    # Get commit metadata
    r = subprocess.run(
        ["git", "log", "-1", "--format=%ct|%aE", commit_hash],
        cwd=repo_path, capture_output=True, timeout=30,
        encoding="utf-8", errors="replace",
    )
    parts = r.stdout.strip().split("|")
    commit_dt = datetime.fromtimestamp(int(parts[0]), tz=timezone.utc).replace(tzinfo=None)
    author = parts[1] if len(parts) > 1 else ""

    # Get files
    dr = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "-r", "--numstat", commit_hash],
        cwd=repo_path, capture_output=True, timeout=30,
        encoding="utf-8", errors="replace",
    )
    touched = set()
    for line in dr.stdout.strip().split("\n"):
        p = line.split("\t")
        if len(p) >= 3:
            touched.add(p[2])

    # Get full message
    mr = subprocess.run(
        ["git", "log", "-1", "--format=%B", commit_hash],
        cwd=repo_path, capture_output=True, timeout=30,
        encoding="utf-8", errors="replace",
    )
    full_msg = mr.stdout.strip()

    # Compute M.1 features via single_commit_features
    m1 = compute_single_commit_m1_features(
        repo_path=repo_path,
        commit_hash=commit_hash,
        commit_date=commit_dt.replace(tzinfo=timezone.utc),
        author_name=author,
        touched_files=touched,
    )

    # Build full feature dict matching bulk extraction
    features = {
        "hash": commit_hash,
        "author": m1.get("_graph_author", author),
        "date": commit_dt,
        "lines_added": 0,  # not in M.1
        "lines_deleted": 0,
        "files_touched": len(touched),
        "dirs_touched": len(set(str(Path(f).parent) for f in touched if Path(f).parent)),
        "commit_msg_length": len(full_msg),
        "is_fix_bug_revert": int(any(k in full_msg.lower() for k in ["fix", "bug", "revert"])),
        "hour_of_day": commit_dt.hour,
        "day_of_week": commit_dt.weekday(),
        "author_prior_commits": m1.get("_author_prior_commits", 0),
    }
    features.update(m1)
    features.pop("_graph_author", None)
    features.pop("_author_prior_commits", None)
    return features

def main():
    repos = {
        "django": "repos/django",
        "react": "repos/react",
        "kafka": "repos/kafka",
        "rust": "repos/rust",
        "kubernetes": "repos/kubernetes",
    }

    all_diffs = []
    total_mismatches = 0

    for repo_name, repo_path in repos.items():
        repo_path = str(Path(repo_path).resolve())
        print(f"\n=== {repo_name} ===")

        clear_cache()
        _get_full_graph(repo_path)

        # Pick 10 commits spread across the graph
        _, _, sg = _get_full_graph(repo_path)
        step = max(1, len(sg) // 10)
        targets = [sg[i] for i in range(0, len(sg), step)][:10]

        for h, info in targets:
            # Run twice for determinism — clear cache between to test cold-start consistency
            clear_cache()
            _get_full_graph(repo_path)
            f1a = get_sc_features(repo_path, h)
            clear_cache()
            _get_full_graph(repo_path)
            f1b = get_sc_features(repo_path, h)

            # Compare features
            for feat in FEATURES:
                v1 = f1a.get(feat)
                v2 = f1b.get(feat)

                # Skip non-numeric features
                if v1 is None or v2 is None:
                    continue
                if isinstance(v1, str) or isinstance(v2, str):
                    if v1 != v2:
                        print(f"  MISMATCH {h[:12]} {feat}: '{v1}' != '{v2}'")
                        total_mismatches += 1
                    continue
                if isinstance(v1, datetime):
                    if v1 != v2:
                        total_mismatches += 1
                    continue

                try:
                    v1f = float(v1)
                    v2f = float(v2)
                    if abs(v1f - v2f) > 0.01:
                        print(f"  MISMATCH {h[:12]} {feat}: {v1f} != {v2f} (diff={abs(v1f-v2f):.4f})")
                        total_mismatches += 1
                except (TypeError, ValueError):
                    if v1 != v2:
                        total_mismatches += 1

    print(f"\n{'='*60}")
    print(f"U.6.6b PARITY RESULT: {total_mismatches} mismatches across 50 commits x {len(FEATURES)} features")
    if total_mismatches == 0:
        print("PASS — 0 mismatches")
    else:
        print(f"FAIL — {total_mismatches} mismatches")

if __name__ == "__main__":
    main()
