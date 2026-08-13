"""
Register model component for the Gatekeeper retraining pipeline.
Registers the winning model to MLflow Model Registry, compares with
the current production version, and only promotes if better.
"""

import os
from pathlib import Path

from kfp import dsl


@dsl.component(
    packages_to_install=[
        "mlflow",
        "scikit-learn",
        "lightgbm",
        "skops",
        "joblib",
    ],
)
def register_model(
    model_path: str,
) -> str:
    """
    Register the best model from AutoML to MLflow Model Registry.

    Compares the new model's F1 score against the currently registered
    production version. Only promotes to "Production" if it's actually better.

    Args:
        model_path: Path to the best model file (.joblib)

    Returns:
        Status message with registration details
    """
    import json as json_mod
    import mlflow
    import mlflow.sklearn
    import joblib

    # Set MLflow tracking URI
    project_root = str(Path(__file__).parent.parent.parent)
    tracking_uri = f"sqlite:///{os.path.join(project_root, 'mlflow.db')}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("gatekeeper")

    print(f"MLflow tracking URI: {tracking_uri}")
    print("Experiment: gatekeeper")

    # Load the new model
    print(f"Loading model from {model_path}")
    new_model = joblib.load(model_path)
    print(f"Model type: {type(new_model).__name__}")

    # Load AutoML results for metrics
    results_path = Path(project_root) / "models" / "automl_results.json"
    new_metrics = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            automl_results = json_mod.load(f)
        best_model_name = automl_results["best_model"]
        new_metrics = automl_results["results"].get(best_model_name, {})
        print(f"New model ({best_model_name}) F1: {new_metrics.get('f1', 'N/A')}")

    # Check if a model is already registered
    model_name = "GatekeeperRiskPredictor"
    current_version = None
    current_metrics = {}

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            # Get the latest version
            latest_version = max(versions, key=lambda v: int(v.version))
            current_version = latest_version.version
            print(f"Current registered model: version {current_version}")

            # Try to get metrics from the latest run
            run_id = latest_version.run_id
            run = client.get_run(run_id)
            current_metrics = run.data.metrics
            print(f"Current model F1: {current_metrics.get('f1', 'N/A')}")
    except Exception as e:
        print(f"No existing model registered: {e}")

    # Compare metrics
    new_f1 = new_metrics.get("f1", 0)
    current_f1 = current_metrics.get("f1", 0)

    should_promote = new_f1 > current_f1 if current_version else True

    print(f"\n{'='*60}")
    print("Model Promotion Decision:")
    print(f"  New model F1: {new_f1:.4f}")
    print(f"  Current model F1: {current_f1:.4f}")
    if should_promote:
        print("  Decision: PROMOTE (new model is better)")
    else:
        print("  Decision: KEEP current model (new model is not better)")
    print(f"{'='*60}")

    # Register the new model version
    with mlflow.start_run() as run:
        # Log parameters
        if new_metrics:
            mlflow.log_params({
                "model_type": type(new_model).__name__,
                "source": "automl_pipeline",
                "promoted": str(should_promote),
            })
            mlflow.log_metrics(new_metrics)

        # Log the model
        mlflow.sklearn.log_model(
            new_model,
            "model",
            registered_model_name=model_name,
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "numpy.dtype",
                "numpy.ndarray",
            ],
        )
        print(f"Model logged. Run ID: {run.info.run_id}")

    # If better, transition to Production
    if should_promote:
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            latest = max(versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(
                name=model_name,
                version=latest.version,
                stage="Production",
            )
            print(f"Model version {latest.version} promoted to Production")
        except Exception as e:
            print(f"Note: Could not transition to Production stage: {e}")

    status = (
        f"Model registered. "
        f"{'Promoted to Production' if should_promote else 'Kept current model'}. "
        f"New F1={new_f1:.4f}, Previous F1={current_f1:.4f}"
    )
    print(status)
    return status
