"""
Register model component for the Gatekeeper retraining pipeline.
Registers the winning model to MLflow, promotes it only if it beats the
current Production version, and updates the standalone serving artifact.
"""

from kfp import dsl


@dsl.component(
    base_image="gatekeeper-kfp-base",
    packages_to_install=[
        "mlflow",
        "scikit-learn",
        "lightgbm",
        "skops",
    ],
)
def register_model(
    model_path: dsl.InputPath("Model"),
    automl_results_path: dsl.InputPath("JSON"),
) -> str:
    """
    Register the best model from AutoML to MLflow Model Registry.

    Args:
        model_path: Path to the best model file (.skops).
        automl_results_path: Path to the automl results JSON from
            the automl_search component.

    Returns:
        Status message with registration details.
    """
    import json as json_mod
    import os
    from pathlib import Path

    import mlflow
    import mlflow.sklearn
    import skops.io as sio
    from mlflow.exceptions import MlflowException

    trusted_types = [
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
        "numpy.dtype",
        "numpy.ndarray",
        "pandas.core.frame.DataFrame",
        "pandas.core.series.Series",
    ]

    # MLflow tracking URI: use env var (set by run_retrain.py or docker-compose)
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("gatekeeper")

    print(f"MLflow tracking URI: {tracking_uri}")
    print("Experiment: gatekeeper")

    print(f"Loading model from {model_path}")
    new_model = sio.load(model_path, trusted=trusted_types)
    print(f"Model type: {type(new_model).__name__}")

    # Read automl results from KFP artifact (not Path.cwd())
    new_metrics = {}
    best_model_name = "unknown"
    if Path(automl_results_path).exists():
        with open(automl_results_path, "r") as f:
            automl_results = json_mod.load(f)
        best_model_name = automl_results.get("best_model", "unknown")
        new_metrics = automl_results.get("results", {}).get(best_model_name, {})
        print(f"New model ({best_model_name}) F1: {new_metrics.get('f1', 'N/A')}")

    model_name = "GatekeeperRiskPredictor"
    client = mlflow.tracking.MlflowClient()
    current_version = None
    current_metrics = {}

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        production_versions = [
            version for version in versions
            if version.current_stage == "Production"
        ]
        if production_versions:
            current_model_version = max(
                production_versions,
                key=lambda version: int(version.version),
            )
            current_version = current_model_version.version
            print(f"Current Production model: version {current_version}")

            run = client.get_run(current_model_version.run_id)
            current_metrics = run.data.metrics
            print(f"Current Production F1: {current_metrics.get('f1', 'N/A')}")
        else:
            print("No existing Production model version found")
    except MlflowException as exc:
        print(f"No existing model registered: {exc}")

    new_f1 = new_metrics.get("f1", 0)
    current_f1 = current_metrics.get("f1", 0)
    should_promote = new_f1 > current_f1 if current_version else True

    print(f"\n{'=' * 60}")
    print("Model Promotion Decision:")
    print(f"  New model F1: {new_f1:.4f}")
    print(f"  Current Production F1: {current_f1:.4f}")
    print(
        "  Decision: "
        + (
            "PROMOTE (new model is better)"
            if should_promote
            else "KEEP current Production model (new model is not better)"
        )
    )
    print(f"{'=' * 60}")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_type": type(new_model).__name__,
                "source": "automl_pipeline",
                "promoted": str(should_promote),
            }
        )
        if new_metrics:
            mlflow.log_metrics(new_metrics)

        mlflow.sklearn.log_model(
            new_model,
            "model",
            serialization_format="skops",
            registered_model_name=model_name,
            skops_trusted_types=trusted_types,
        )
        print(f"Model logged. Run ID: {run.info.run_id}")

    promoted_version = None
    if should_promote:
        versions = client.search_model_versions(f"name='{model_name}'")
        run_versions = [
            version for version in versions
            if version.run_id == run.info.run_id
        ]
        if not run_versions:
            raise RuntimeError(
                f"Could not find registered model version for run {run.info.run_id}"
            )

        promoted_version = max(run_versions, key=lambda version: int(version.version))
        client.transition_model_version_stage(
            name=model_name,
            version=promoted_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Model version {promoted_version.version} promoted to Production")

        # Export production skops model to pipeline_root-mounted path
        # In Docker, we write to the same directory as the model_path artifact
        production_model_path = Path(model_path).parent / "gatekeeper_risk_model.skops"
        sio.dump(new_model, production_model_path)
        print(f"Production skops model exported to {production_model_path}")

    status = (
        f"Model registered. "
        f"{'Promoted to Production' if should_promote else 'Kept current model'}. "
        f"New F1={new_f1:.4f}, Previous F1={current_f1:.4f}"
    )
    if promoted_version is not None:
        status += f", Production version={promoted_version.version}"
    print(status)
    return status
