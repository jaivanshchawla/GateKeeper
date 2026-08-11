#!/usr/bin/env python3
"""
Smoke test: Data drift detection using Evidently.

Loads data/commit_features.csv, splits chronologically into reference (older 80%)
and current (recent 20%), runs Evidently's data drift report, and saves the report.

Note: This test does NOT hard-fail based on drift being detected.
The purpose is to prove the drift detection mechanism works.
Phase 9's live deployment can later compare real production traffic against training data.
"""

import os

import pandas as pd


def test_drift_report_generates():
    """Validate that Evidently drift report generates without error."""
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset

    # Load data - try multiple paths for compatibility with different environments
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv"),
        os.path.join(os.getcwd(), "data", "commit_features.csv"),
        "/app/data/commit_features.csv",  # Docker container path
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        pytest.skip(f"Data file not found in any of: {possible_paths}. Skipping drift test.")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from commit_features.csv")

    # Sort by date chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Split into reference (older 80%) and current (recent 20%)
    split_index = int(len(df) * 0.8)
    reference = df.iloc[:split_index].copy()
    current = df.iloc[split_index:].copy()

    print(f"Reference set: {len(reference)} rows (older)")
    print(f"Current set: {len(current)} rows (recent)")

    # Drop non-numeric columns for drift detection
    drop_cols = ["hash", "author", "date", "risky"]
    reference_numeric = reference.drop(columns=[c for c in drop_cols if c in reference.columns])
    current_numeric = current.drop(columns=[c for c in drop_cols if c in current.columns])

    # Create and run the drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_numeric, current_data=current_numeric)

    # Save the report as HTML
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "drift_report.html")
    report.save_html(output_path)

    assert os.path.exists(output_path), f"Drift report not saved: {output_path}"
    file_size = os.path.getsize(output_path)
    print(f"Drift report saved to: {output_path} ({file_size:,} bytes)")

    # Get the drift results
    result = report.as_dict()
    drift_detection = result.get("metrics", [{}])[0].get("result", {})

    # Print per-feature drift status
    dataset_drift = drift_detection.get("dataset_drift", False)
    drift_share = drift_detection.get("drift_share", 0)

    print("\n=== Drift Detection Results ===")
    print(f"Dataset drift detected: {dataset_drift}")
    print(f"Drift share: {drift_share:.2%}")

    # Print per-feature drift if available
    columns_drift = drift_detection.get("drift_by_columns", {})
    if columns_drift:
        print("\nPer-feature drift:")
        for feature_name, feature_result in columns_drift.items():
            is_drifted = feature_result.get("drift_detected", False)
            status = "DRIFTED" if is_drifted else "OK"
            print(f"  {feature_name}: {status}")

    # NOTE: We intentionally do NOT fail based on drift detection.
    # This test proves the mechanism works. Phase 9's live deployment
    # will compare real production traffic against training data.
    if dataset_drift:
        print("\nNOTE: Drift was detected, but this test intentionally does not fail.")
        print("Phase 9 will use this mechanism to monitor real production data.")
    else:
        print("\nNo drift detected between older and recent data subsets.")

    print("\nDrift detection mechanism validated successfully.")
