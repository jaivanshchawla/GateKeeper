#!/usr/bin/env python3
"""
Per-repo drift monitoring using Evidently.

Splits each repo's data chronologically into reference (older 80%)
and current (recent 20%), runs drift detection, and saves results.

Output: data/drift_results.json with per-repo drift status.
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd


def run_drift_per_repo(csv_path: str = "data/commit_features.csv",
                       output_path: str = "data/drift_results.json",
                       retrain_threshold: float = 0.5):
    """Run per-repo drift detection and save results.

    Args:
        csv_path: Path to commit features CSV
        output_path: Path to save drift results JSON
        retrain_threshold: Fraction of repos that must show drift before
            flagging needs_retraining. Default 0.5 (50% of repos).
            E.g. 0.5 on a 5-repo set = flag when 3+ repos drift.
    """
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    date_col = "committer_date" if "committer_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])

    # Feature columns (exclude metadata)
    exclude = {"hash", "author", "date", "author_date", "committer_date",
               "touched_files", "risky", "source_repo", "commit_message",
               "commit_timestamp"}
    feature_cols = [c for c in df.columns if c not in exclude]

    results = {
        "generated_at": datetime.utcnow().isoformat(),
        "repos": {},
    }

    for repo_name in sorted(df["source_repo"].unique()):
        repo_df = df[df["source_repo"] == repo_name].sort_values(date_col).reset_index(drop=True)
        print(f"\n--- {repo_name}: {len(repo_df)} rows ---")

        if len(repo_df) < 20:
            print(f"  Skipping (too few rows)")
            results["repos"][repo_name] = {"status": "skipped", "reason": "too_few_rows"}
            continue

        # Chronological split
        split_idx = int(len(repo_df) * 0.8)
        reference = repo_df.iloc[:split_idx][feature_cols].copy()
        current = repo_df.iloc[split_idx:][feature_cols].copy()

        print(f"  Reference: {len(reference)} rows (older)")
        print(f"  Current: {len(current)} rows (recent)")

        # Run drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        result_dict = report.as_dict()
        drift_detection = result_dict.get("metrics", [{}])[0].get("result", {})

        dataset_drift = drift_detection.get("dataset_drift", False)
        drift_share = drift_detection.get("drift_share", 0)

        columns_drift = drift_detection.get("drift_by_columns", {})
        drifted_features = [f for f, r in columns_drift.items() if r.get("drift_detected", False)]

        # Save HTML report
        html_path = os.path.join(os.path.dirname(output_path) or ".", f"drift_report_{repo_name}.html")
        report.save_html(html_path)

        repo_result = {
            "status": "ok",
            "reference_rows": len(reference),
            "current_rows": len(current),
            "dataset_drift": dataset_drift,
            "drift_share": round(drift_share, 4),
            "drifted_features": drifted_features,
            "drifted_count": len(drifted_features),
            "total_features": len(feature_cols),
            "report_html": html_path,
        }
        results["repos"][repo_name] = repo_result
        print(f"  Drift detected: {dataset_drift} (share: {drift_share:.2%})")
        print(f"  Drifted features: {len(drifted_features)}/{len(feature_cols)}")
        if drifted_features:
            print(f"    {', '.join(drifted_features[:5])}")

    # Overall status
    all_drifted = [r for r in results["repos"].values()
                   if r.get("status") == "ok" and r.get("dataset_drift")]
    total_repos = len([r for r in results["repos"].values() if r.get("status") == "ok"])
    results["summary"] = {
        "repos_analyzed": total_repos,
        "repos_with_drift": len(all_drifted),
        "needs_retraining": len(all_drifted) >= total_repos * retrain_threshold,
        "retrain_threshold": retrain_threshold,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/commit_features.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/drift_results.json"
    run_drift_per_repo(csv_path, output_path)
