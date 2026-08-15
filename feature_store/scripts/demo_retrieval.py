#!/usr/bin/env python3
"""Demonstrate online feature retrieval from the Feast feature store."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from feast import FeatureStore


def demo_retrieval():
    store = FeatureStore(repo_path=str(Path(__file__).resolve().parent.parent))

    # Load a few commit hashes from the offline data
    parquet_path = Path("data/commit_features.parquet")
    if not parquet_path.exists():
        print("Run prepare_feast_data.py first to create the Parquet file.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    sample_hashes = df["commit_hash"].head(5).tolist()

    print("=== Feast Online Feature Retrieval Demo ===\n")
    print(f"Retrieving features for {len(sample_hashes)} commits...\n")

    # Build entity DataFrame for online retrieval
    entity_df = pd.DataFrame({"commit_hash": sample_hashes})

    # Retrieve from online store
    feature_vector = store.get_online_features(
        features=[
            "commit_features:lines_added",
            "commit_features:lines_deleted",
            "commit_features:files_touched",
            "commit_features:dirs_touched",
            "commit_features:author_prior_commits",
            "commit_features:hour_of_day",
            "commit_features:day_of_week",
            "commit_features:commit_msg_length",
            "commit_features:is_fix_bug_revert",
            "commit_features:risky",
        ],
        entity_rows=entity_df.to_dict(orient="records"),
    ).to_dict()

    # Display results
    for i, h in enumerate(sample_hashes):
        print(f"Commit: {h[:12]}...")
        print(f"  lines_added={feature_vector['lines_added'][i]}, "
              f"lines_deleted={feature_vector['lines_deleted'][i]}, "
              f"files_touched={feature_vector['files_touched'][i]}")
        print(f"  author_prior_commits={feature_vector['author_prior_commits'][i]}, "
              f"hour_of_day={feature_vector['hour_of_day'][i]}, "
              f"is_fix_bug_revert={feature_vector['is_fix_bug_revert'][i]}")
        print(f"  risky={feature_vector['risky'][i]}")
        print()


if __name__ == "__main__":
    demo_retrieval()
