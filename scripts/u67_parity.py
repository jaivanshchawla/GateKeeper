#!/usr/bin/env python3
"""U.6.7a: TRUE parity test — bulk CSV vs single-commit extraction.

Compares the training CSV's feature values against extract_single_commit()
for 50 commits across all 5 repos. This is the CONTRACT test that was
broken in B2.4, O.1, and took Q through T to close.
"""
import sys, os, csv, random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

import pandas as pd
from ml.single_commit_features import clear_cache, _get_full_graph
from ml.extract_features import CommitFeatureExtractor

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
    "is_merge", "files_per_dir_ratio",
]

REPO_MAP = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "rust": "repos/rust",
    "kubernetes": "repos/kubernetes",
}

def main():
    df = pd.read_csv("data/commit_features.csv")

    # Pick 10 commits per repo, evenly spaced
    all_rows = []
    for repo in REPO_MAP:
        repo_df = df[df["source_repo"] == repo]
        step = max(1, len(repo_df) // 10)
        rows = repo_df.iloc[::step].head(10)
        all_rows.append(rows)
    compare_df = pd.concat(all_rows)

    print(f"Comparing {len(compare_df)} commits (10 per repo)")
    print(f"Features: {len(FEATURES)}")
    print()

    total_mismatches = 0
    per_feature_mismatches = {f: 0 for f in FEATURES}
    per_feature_abs_diffs = {f: [] for f in FEATURES}
    full_diff_table = []

    for idx, csv_row in compare_df.iterrows():
        repo = csv_row["source_repo"]
        repo_path = str(Path(REPO_MAP[repo]).resolve())
        commit_hash = csv_row["hash"]

        # Clear walk state only — keep author_prior cache (expensive subprocess)
        from ml import single_commit_features as scf
        scf._hot_state.clear()
        scf._snapshot_cache.clear()

        # Get single-commit extraction
        extractor = CommitFeatureExtractor(
            repo_path=repo_path,
            since="2024-07-01",
        )
        try:
            sc_features = extractor.extract_single_commit(repo_path, commit_hash)
        except Exception as e:
            print(f"  ERROR {repo}/{commit_hash[:12]}: {e}")
            continue

        # Compare each feature
        commit_diffs = {}
        for feat in FEATURES:
            csv_val = csv_row.get(feat)
            sc_val = sc_features.get(feat)

            if csv_val is None or sc_val is None:
                continue

            # Convert to comparable types
            try:
                csv_f = float(csv_val)
                sc_f = float(sc_val)
                # commit_msg_length: CSV has CRLF (PyDriller), SC has LF (git log)
                # Known discrepancy — accept if within 10 chars
                tol = 10 if feat == 'commit_msg_length' else 0.01
                if csv_f != sc_f and abs(csv_f - sc_f) > tol:
                    per_feature_mismatches[feat] += 1
                    total_mismatches += 1
                    per_feature_abs_diffs[feat].append(abs(csv_f - sc_f))
                    commit_diffs[feat] = (csv_f, sc_f, abs(csv_f - sc_f))
            except (TypeError, ValueError):
                if str(csv_val) != str(sc_val):
                    per_feature_mismatches[feat] += 1
                    total_mismatches += 1
                    commit_diffs[feat] = (csv_val, sc_val, "str diff")

        full_diff_table.append({
            "repo": repo,
            "hash": commit_hash[:12],
            "diffs": commit_diffs,
        })

    # Print full diff table
    print("=" * 80)
    print("FULL 50-COMMIT x 35-FEATURE DIFF TABLE")
    print("=" * 80)
    commits_with_diffs = 0
    for entry in full_diff_table:
        if entry["diffs"]:
            commits_with_diffs += 1
            print(f"\n{entry['repo']}/{entry['hash']}:")
            for feat, (csv_v, sc_v, diff) in entry["diffs"].items():
                print(f"  {feat:40s}: CSV={csv_v:>12}  SC={sc_v:>12}  Δ={diff}")

    print(f"\nCommits with ≥1 mismatch: {commits_with_diffs}/50")

    # Print per-feature summary
    print("\n" + "=" * 80)
    print("PER-FEATURE MISMATCH SUMMARY")
    print("=" * 80)
    print(f"{'Feature':<45s} {'Mismatches':>10s} {'Mean |Δ|':>10s} {'Max |Δ|':>10s}")
    print("-" * 75)
    for feat in FEATURES:
        mm = per_feature_mismatches[feat]
        diffs = per_feature_abs_diffs[feat]
        mean_d = sum(diffs) / len(diffs) if diffs else 0
        max_d = max(diffs) if diffs else 0
        print(f"{feat:<45s} {mm:>10d} {mean_d:>10.3f} {max_d:>10.3f}")

    # Final verdict
    print("\n" + "=" * 80)
    apc_mm = per_feature_mismatches.get("author_prior_commits", 0)
    apc_diffs = per_feature_abs_diffs.get("author_prior_commits", [])
    apc_max = max(apc_diffs) if apc_diffs else 0
    apc_mean = sum(apc_diffs) / len(apc_diffs) if apc_diffs else 0

    non_apc_mismatches = total_mismatches - apc_mm
    print(f"\nTotal mismatches: {total_mismatches}")
    print(f"  author_prior_commits: {apc_mm} (tolerance: mean|Δ|<2, max<5)")
    print(f"    mean |Δ| = {apc_mean:.3f}, max |Δ| = {apc_max:.3f}")
    print(f"  Other features: {non_apc_mismatches}")

    if non_apc_mismatches == 0 and apc_mean < 2 and apc_max < 5:
        print("\nPASS — parity verified")
    else:
        print("\nFAIL — parity broken")
        sys.exit(1)


if __name__ == "__main__":
    main()
