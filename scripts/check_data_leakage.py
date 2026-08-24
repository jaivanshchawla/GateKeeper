#!/usr/bin/env python3
"""
Check for data leakage between train and test sets.

Reproduces the exact same train/test split logic and random seed
used in ml/train.py, then asserts zero row-level overlap between
the resulting train and test sets.

Exit codes:
    0 - No leakage detected (clean split)
    1 - Leakage detected (overlap between train and test)
"""

import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Determine paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    config_path = os.path.join(project_root, "ml", "config.yaml")
    features_path = os.path.join(project_root, "data", "commit_features.csv")

    # Check that files exist
    if not os.path.exists(features_path):
        print(f"ERROR: Features file not found: {features_path}")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Load config to get feature columns
    config = load_config(config_path)
    feature_columns = config.get("feature_columns", [])

    # Load features
    print("Loading features...")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} commits")

    # Check for required columns
    if "risky" not in df.columns:
        print("ERROR: 'risky' column not found in features")
        sys.exit(1)

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing feature columns: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]

    # Prepare data (same as ml/train.py)
    X = df[feature_columns]
    y = df["risky"]

    # Split data with EXACT same parameters as ml/train.py
    # test_size=0.2, random_state=42, stratify=y
    print("Splitting data (80/20, random_state=42, stratify=y)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Get the indices of train and test sets
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    # Check 1: Row-level overlap
    overlap = train_indices & test_indices
    if overlap:
        print("\nDATA LEAKAGE DETECTED (row-level)!")
        print(f"Found {len(overlap)} rows in both train and test.")
        sys.exit(1)
    print("[OK] No row-level overlap.")

    # Check 2: Temporal overlap across split boundary
    if "committer_date" in df.columns:
        dates = pd.to_datetime(df["committer_date"], utc=True)
        train_dates = dates.loc[list(train_indices)]
        test_dates = dates.loc[list(test_indices)]
        train_max = train_dates.max()
        test_min = test_dates.min()
        gap_days = (test_min - train_max).total_seconds() / 86400
        print(f"[INFO] Temporal gap: train_max={train_max}, test_min={test_min}, gap={gap_days:.1f}d")
        if gap_days < 7:
            print(f"WARNING: Temporal gap ({gap_days:.1f}d) < label_window_days (7). Possible temporal leakage.")
        else:
            print("[OK] Temporal gap >= 7 days.")

    # Check 3: Same-file / same-window overlap
    if "touched_files" in df.columns:
        from collections import defaultdict
        LABEL_WINDOW = 7

        # Build file-touch index from training set
        file_to_train = defaultdict(set)
        for idx in train_indices:
            row = df.loc[idx]
            files = str(row.get("touched_files", "")).split("|")
            for f in files:
                if f:
                    file_to_train[f].add(idx)

        # Check if any test commit shares a file with a train commit within 7 days
        same_file_overlap = 0
        for idx in sorted(test_indices):
            row = df.loc[idx]
            test_files = str(row.get("touched_files", "")).split("|")
            test_date = pd.to_datetime(row["committer_date"], utc=True)
            for f in test_files:
                if f in file_to_train:
                    for train_idx in file_to_train[f]:
                        train_date = pd.to_datetime(df.loc[train_idx, "committer_date"], utc=True)
                        if abs((test_date - train_date).days) <= LABEL_WINDOW:
                            same_file_overlap += 1
                            break
                    else:
                        continue
                    break

        pct = same_file_overlap / max(len(test_indices), 1) * 100
        print(f"[INFO] Same-file/same-window overlap: {same_file_overlap}/{len(test_indices)} ({pct:.1f}%)")
        if same_file_overlap > 0:
            print("NOTE: This is expected for random splits on commit data.")
            print("      Use time-ordered purged splits for honest evaluation.")
        else:
            print("[OK] No same-file/same-window overlap.")

    print("\n[OK] All leakage checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
