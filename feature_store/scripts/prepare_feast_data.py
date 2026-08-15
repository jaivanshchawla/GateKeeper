#!/usr/bin/env python3
"""Convert commit_features.csv to Parquet format for Feast offline store."""

from pathlib import Path

import pandas as pd


def prepare_data():
    csv_path = Path("data/commit_features.csv")
    parquet_path = Path("data/commit_features.parquet")

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Ensure 'date' column is datetime for Feast timestamp_field
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Drop rows with invalid dates
        before = len(df)
        df = df.dropna(subset=["date"])
        if len(df) < before:
            print(f"Dropped {before - len(df)} rows with invalid dates")

    # Ensure commit_hash column exists for Feast entity
    if "hash" in df.columns and "commit_hash" not in df.columns:
        df["commit_hash"] = df["hash"]

    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} rows to {parquet_path}")


if __name__ == "__main__":
    prepare_data()
