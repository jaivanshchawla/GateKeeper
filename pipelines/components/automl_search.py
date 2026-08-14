"""
AutoML search component for the Gatekeeper retraining pipeline.
Compares LightGBM, RandomForest, and LogisticRegression, selects best by F1.
"""

from kfp import dsl


@dsl.component(
    base_image="gatekeeper-kfp-base",
    packages_to_install=[
        "pandas",
        "lightgbm",
        "scikit-learn",
        "skops",
    ],
)
def automl_search(
    features_path: dsl.InputPath("Dataset"),
    model_path: dsl.OutputPath("Model"),
    automl_results_path: dsl.OutputPath("JSON"),
) -> None:
    """
    Compare LightGBM, RandomForest, LogisticRegression and pick the best by F1.

    Args:
        features_path: Path to the validated features CSV.
        model_path: KFP output path for the best skops model.
        automl_results_path: KFP output path for the automl results JSON.
    """
    import json
    from pathlib import Path

    import pandas as pd
    import skops.io as sio
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = None

    try:
        import yaml
        config_path = Path.cwd().parent.parent.parent / "ml" / "config.yaml"
        feature_columns = []
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            feature_columns = config.get("feature_columns", [])
    except (OSError, ValueError, KeyError):
        feature_columns = []

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

    X = df[feature_columns].values
    y = df["risky"].values
    print(f"Features: {feature_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    candidates = {}
    if LGBMClassifier is not None:
        candidates["lightgbm"] = LGBMClassifier(
            num_leaves=31, learning_rate=0.05, n_estimators=100,
            random_state=42, verbose=-1
        )
    candidates["random_forest"] = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    candidates["logistic_regression"] = LogisticRegression(
        max_iter=1000, random_state=42
    )

    results = {}
    best_model = None
    best_f1 = -1.0
    best_name = ""

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        print(f"  F1={f1:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
        results[name] = {
            "f1": float(f1),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
        }
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

    # Save best model as skops
    output_model_path = Path(model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(best_model, output_model_path)
    print(f"Best model saved to {output_model_path}")

    # Save automl results as KFP artifact
    automl_output = {
        "best_model": best_name,
        "results": results,
        "feature_columns": feature_columns,
    }
    output_results_path = Path(automl_results_path)
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path, "w") as f:
        json.dump(automl_output, f, indent=2)
    print(f"Results saved to {output_results_path}")
