"""
AutoML search component for the Gatekeeper retraining pipeline.
Compares LightGBM, RandomForest, and LogisticRegression models,
selects the best by F1 score.
"""

import json
from pathlib import Path

from kfp import dsl


@dsl.component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "lightgbm",
        "pyyaml",
    ],
)
def automl_search(
    features_path: str,
) -> str:
    """
    Run AutoML search comparing multiple model families.

    Compares:
    - LightGBM (gradient boosting)
    - RandomForest (ensemble)
    - LogisticRegression (linear baseline)

    Selects the best model by F1 score and saves it to disk.

    Args:
        features_path: Path to the features CSV

    Returns:
        Path to the best model file (.joblib)
    """
    import yaml
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    import joblib

    # Try importing LightGBM
    try:
        import lightgbm as lgb

        HAS_LGBM = True
    except ImportError:
        HAS_LGBM = False
        print("WARNING: LightGBM not available, skipping it")

    # Load config
    project_root = str(Path(__file__).parent.parent.parent)
    config_path = Path(project_root) / "ml" / "config.yaml"

    feature_columns = []
    lgbm_params = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        feature_columns = config.get("feature_columns", [])
        lgbm_params = config.get("lightgbm_params", {})

    # Load features
    print(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows")

    if not feature_columns:
        # Auto-detect: all numeric columns except 'risky', 'hash', 'author', 'date', 'commit_msg'
        exclude = {"risky", "hash", "author", "date", "commit_msg"}
        feature_columns = [
            col for col in df.columns
            if col not in exclude and df[col].dtype in ["int64", "float64", "int32", "float32"]
        ]

    X = df[feature_columns]
    y = df["risky"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features: {feature_columns}")

    # Define models to compare
    models = {}

    # LightGBM
    if HAS_LGBM and lgbm_params:
        models["lightgbm"] = lgb.LGBMClassifier(**lgbm_params)
    elif HAS_LGBM:
        models["lightgbm"] = lgb.LGBMClassifier(
            num_leaves=31, learning_rate=0.05, n_estimators=100
        )

    # Random Forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    # Logistic Regression
    models["logistic_regression"] = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )

    # Train and evaluate each model
    results = {}
    best_f1 = -1
    best_model_name = None
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        results[name] = metrics
        print(f"  {name}: F1={metrics['f1']:.4f}, "
              f"Acc={metrics['accuracy']:.4f}, "
              f"Prec={metrics['precision']:.4f}, "
              f"Rec={metrics['recall']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model

    # Print summary
    print(f"\n{'='*60}")
    print("AutoML Search Results:")
    print(f"{'='*60}")
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        marker = " ← BEST" if name == best_model_name else ""
        print(f"  {name}: F1={metrics['f1']:.4f}{marker}")
    print(f"{'='*60}")
    print(f"Best model: {best_model_name} (F1={best_f1:.4f})")

    # Save the best model and results
    output_dir = Path(project_root) / "models"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to {model_path}")

    # Save results as JSON
    results_path = output_dir / "automl_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "results": results,
                "feature_columns": feature_columns,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {results_path}")

    return str(model_path)
