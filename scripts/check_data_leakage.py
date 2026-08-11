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

    # Check for overlap
    overlap = train_indices & test_indices

    if overlap:
        print("\nDATA LEAKAGE DETECTED!")
        print(f"Found {len(overlap)} rows that appear in both train and test sets.")
        print(f"Overlapping indices: {sorted(list(overlap))[:10]}...")
        sys.exit(1)
    else:
        print("\n[OK] No data leakage detected - train and test sets are disjoint.")
        sys.exit(0)


if __name__ == "__main__":
    main()
