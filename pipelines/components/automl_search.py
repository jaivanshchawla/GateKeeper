"""
AutoML search component for the Gatekeeper retraining pipeline.
Uses PyCaret to compare candidate classifiers and exports the winner as skops.
"""

from kfp import dsl


@dsl.component(
    packages_to_install=[
        "pandas",
        "lightgbm",
        "pycaret",
        "pyyaml",
        "skops",
    ],
)
def automl_search(
    features_path: dsl.InputPath("Dataset"),
    model_path: dsl.OutputPath("Model"),
    automl_results_path: dsl.OutputPath("JSON"),
) -> None:
    """
    Run PyCaret AutoML over the supported model families.

    Args:
        features_path: Path to the validated features CSV.
        model_path: KFP output path for the best skops model.
        automl_results_path: KFP output path for the automl results JSON
            (best model name, all model metrics, feature columns).
    """
    import json
    from pathlib import Path

    import pandas as pd
    import skops.io as sio
    import yaml

    try:
        from pycaret.classification import compare_models, finalize_model, pull, setup
    except ImportError as exc:
        raise RuntimeError(
            "PyCaret is required for the retraining AutoML search. "
            "Install project requirements before running the pipeline."
        ) from exc

    # Derive project root from model_path parent if available, else cwd
    # model_path is at {pipeline_root}/{run_name}/automl_search/model
    # In Docker, Path.cwd() may be / so we try to find config relative to
    # the output artifact location, or fall back to cwd.
    model_dir = Path(model_path).parent
    project_root = model_dir.parent.parent.parent if model_dir.exists() else Path.cwd()

    # Try to load feature columns from config
    config_path = project_root / "ml" / "config.yaml"
    feature_columns = []
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        feature_columns = config.get("feature_columns", [])

    print(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows")

    if not feature_columns:
        exclude = {"risky", "hash", "author", "date", "commit_msg"}
        feature_columns = [
            col
            for col in df.columns
            if col not in exclude
            and df[col].dtype in ["int64", "float64", "int32", "float32"]
        ]

    training_df = df[feature_columns + ["risky"]].copy()
    print(f"Features: {feature_columns}")

    setup(
        data=training_df,
        target="risky",
        train_size=0.8,
        session_id=42,
        fold=3,
        html=False,
        verbose=True,
        system_log=False,
        n_jobs=-1,
    )

    candidate_model_ids = ["lightgbm", "rf", "lr"]
    print(f"Running PyCaret compare_models(include={candidate_model_ids}, sort='F1')")
    best_model = compare_models(
        include=candidate_model_ids,
        sort="F1",
        n_select=1,
        fold=3,
    )
    leaderboard = pull()
    print("\nPyCaret leaderboard:")
    print(leaderboard.to_string())

    final_model = finalize_model(best_model)
    serving_model = final_model
    if hasattr(final_model, "steps") and final_model.steps:
        serving_model = final_model.steps[-1][1]
        print(
            "Extracted fitted estimator from PyCaret pipeline for serving: "
            f"{type(serving_model).__name__}"
        )

    best_model_name = str(leaderboard.iloc[0].get("Model", type(serving_model).__name__))
    print(f"Best model: {best_model_name} ({type(serving_model).__name__})")

    metric_columns = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "recall": "Recall",
        "precision": "Prec.",
        "f1": "F1",
        "kappa": "Kappa",
        "mcc": "MCC",
    }
    results = {}
    for _, row in leaderboard.iterrows():
        model_name = str(row.get("Model", "unknown"))
        results[model_name] = {
            output_name: float(row[column])
            for output_name, column in metric_columns.items()
            if column in leaderboard.columns and pd.notna(row[column])
        }

    output_model_path = Path(model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(serving_model, output_model_path)
    print(f"Best model saved to {output_model_path}")

    # Write automl results as KFP artifact (not to Path.cwd())
    automl_output = {
        "best_model": best_model_name,
        "results": results,
        "feature_columns": feature_columns,
    }
    output_results_path = Path(automl_results_path)
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path, "w") as f:
        json.dump(automl_output, f, indent=2)
    print(f"Results saved to {output_results_path}")
