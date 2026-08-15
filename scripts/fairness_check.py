#!/usr/bin/env python3
"""
Fairlearn fairness check for Gatekeeper model.

Checks whether the model is systematically harsher on commits from
new/inexperienced contributors vs experienced ones.
Bins author_prior_commits into "new" (<5) vs "experienced" (>=5).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
import yaml
from sklearn.model_selection import train_test_split


def main():
    # Load data
    df = pd.read_csv("data/commit_features.csv")
    with open("ml/config.yaml") as f:
        config = yaml.safe_load(f)
    feature_cols = config["feature_columns"]

    X = df[feature_cols].values
    y = df["risky"].values

    # Same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reconstruct test_df properly
    _, test_indices = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
    )
    test_df = df.iloc[test_indices].reset_index(drop=True)

    # Load model
    model_path = Path("models/gatekeeper_risk_model.skops")
    trusted = sio.get_untrusted_types(file=str(model_path))
    model = sio.load(str(model_path), trusted=trusted)

    # Predictions
    y_pred = model.predict(X_test)

    # Create sensitive attribute: new (<5 prior commits) vs experienced (>=5)
    sensitive = test_df["author_prior_commits"].values
    groups = np.where(sensitive < 5, "new", "experienced")

    # Compute metrics per group
    new_mask = groups == "new"
    exp_mask = groups == "experienced"

    results = {}
    for name, mask in [("new", new_mask), ("experienced", exp_mask)]:
        if mask.sum() == 0:
            continue
        total = mask.sum()
        pred_pos_rate = y_pred[mask].mean()

        results[name] = {
            "count": int(total),
            "actual_positive_rate": float(y_test[mask].mean()),
            "predicted_positive_rate": float(pred_pos_rate),
            "accuracy": float((y_pred[mask] == y_test[mask]).mean()),
        }

    # Demographic parity difference
    if "new" in results and "experienced" in results:
        dp_diff = abs(
            results["new"]["predicted_positive_rate"]
            - results["experienced"]["predicted_positive_rate"]
        )
    else:
        dp_diff = None

    # Print results
    print("=== Fairlearn Fairness Check ===")
    print(f"Test set: {len(y_test)} commits")
    print(f"New contributors (<5 prior commits): {new_mask.sum()}")
    print(f"Experienced contributors (>=5): {exp_mask.sum()}")
    print()

    for name, r in results.items():
        print(f"  [{name}]")
        print(f"    Count: {r['count']}")
        print(f"    Actual positive rate: {r['actual_positive_rate']:.4f}")
        print(f"    Predicted positive rate: {r['predicted_positive_rate']:.4f}")
        print(f"    Accuracy: {r['accuracy']:.4f}")
        print()

    if dp_diff is not None:
        print(f"  Demographic parity difference: {dp_diff:.4f}")
        threshold = 0.1
        if dp_diff < threshold:
            print(f"  ✅ PASS — difference < {threshold} threshold")
        else:
            print(f"  ⚠️  WARN — difference >= {threshold} threshold")
    else:
        print("  Cannot compute — one group is empty")

    # Save results
    output = {
        "groups": results,
        "demographic_parity_difference": dp_diff,
        "threshold": 0.1,
    }
    with open("governance/fairness_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to governance/fairness_results.json")

    return output


if __name__ == "__main__":
    main()
