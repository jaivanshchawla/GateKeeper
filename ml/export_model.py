#!/usr/bin/env python3
"""
Export the CURRENTLY tagged Production model from MLflow Model Registry
to a standalone models/gatekeeper_risk_model.skops file.

This is the single source of truth for what Gate 2 and the API serve.
Always run this after promoting a new model to Production.
"""

import os
import sys

import mlflow
import mlflow.sklearn
import skops.io as sio

MODEL_NAME = "GatekeeperRiskPredictor"
OUTPUT_PATH = os.path.join("models", "gatekeeper_risk_model.skops")

# Trusted types for model deserialization
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "numpy.dtype",
    "numpy.ndarray",
    "pandas.core.frame.DataFrame",
    "pandas.core.series.Series",
    # LightGBM (legacy)
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    # RandomForest / sklearn ensemble
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.utils._tags._TagsDict",
]


def export_production_model():
    """Load the Production model from MLflow registry and export to skops."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    client = mlflow.MlflowClient()

    # Find the version tagged Production
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_versions = [v for v in all_versions if v.current_stage == "Production"]

    if not prod_versions:
        print(f"ERROR: No version tagged Production for {MODEL_NAME}")
        print("Available versions:")
        for v in sorted(all_versions, key=lambda x: int(x.version)):
            print(f"  v{v.version}: stage={v.current_stage}")
        return False

    prod = prod_versions[0]
    print(f"Production model: v{prod.version}, run_id={prod.run_id}")

    # Load the model using sklearn loader (preserves model type)
    model_uri = f"models:/{MODEL_NAME}/Production"
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Model type: {type(model).__name__}")

    # Show key metrics from the associated run
    run = client.get_run(prod.run_id)
    metrics = run.data.metrics
    print(f"Metrics: {', '.join(f'{k}={v:.4f}' for k, v in metrics.items())}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Export to skops
    sio.dump(model, OUTPUT_PATH)
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"Exported to: {OUTPUT_PATH} ({file_size:,} bytes / {file_size / 1024:.1f} KB)")

    # Verify the export by re-loading
    verify = sio.load(OUTPUT_PATH, trusted=TRUSTED_TYPES)
    print(f"Verification: loaded model type is {type(verify).__name__}")

    if type(verify).__name__ != type(model).__name__:
        print(f"WARNING: exported model type mismatch! Expected {type(model).__name__}, got {type(verify).__name__}")
        return False

    return True


if __name__ == "__main__":
    success = export_production_model()
    if success:
        print("\nExport complete — models/gatekeeper_risk_model.skops is now in sync with Production.")
    else:
        print("\nExport FAILED.")
        sys.exit(1)
