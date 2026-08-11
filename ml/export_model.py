#!/usr/bin/env python3
"""
Export the latest GatekeeperRiskPredictor model from MLflow to a standalone .skops file.

This creates a self-contained model file that can be used without MLflow,
suitable for GitHub Actions and other environments.
"""

import glob
import os

import skops.io as sio

# Trusted types for LightGBM model deserialization
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "numpy.dtype",
    "numpy.ndarray",
    "pandas.core.frame.DataFrame",
    "pandas.core.series.Series",
]


def export_model():
    """Find the latest model artifact and export it to models/gatekeeper_risk_model.skops"""
    # Find model artifacts
    pattern = os.path.join("mlruns", "*", "models", "*", "artifacts", "model.skops")
    skops_files = glob.glob(pattern)

    if not skops_files:
        print("ERROR: No model.skops files found in mlruns/")
        print("Make sure you've trained a model first with: python ml/train.py")
        return False

    # Pick the most recently modified one (latest training run)
    latest_file = max(skops_files, key=os.path.getmtime)
    print(f"Loading model from: {latest_file}")

    # Load the model
    model_data = open(latest_file, "rb").read()
    model = sio.loads(model_data, trusted=TRUSTED_TYPES)
    print(f"Model loaded successfully: {type(model).__name__}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Export to standalone file
    output_path = os.path.join("models", "gatekeeper_risk_model.skops")
    sio.dump(model, output_path)

    # Get file size
    file_size = os.path.getsize(output_path)
    print(f"Model exported to: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    return True


if __name__ == "__main__":
    success = export_model()
    if success:
        print("\nExport complete!")
    else:
        print("\nExport failed!")
        exit(1)
