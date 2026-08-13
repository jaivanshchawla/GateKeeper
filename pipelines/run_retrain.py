#!/usr/bin/env python3
"""
Gatekeeper Retraining Pipeline Runner.

This script runs the retraining pipeline locally by executing each component
sequentially. It serves two purposes:

1. **Local testing**: Validates the pipeline end-to-end without requiring
   a KFP server or Kubernetes cluster.
2. **CI/CD**: Can be run in GitHub Actions (Phase 9 will deploy this on a
   machine with Django cloned).

When a KFP server is available, use `pipelines/retrain_pipeline.py` instead,
which defines the genuine @dsl.pipeline for deployment.

The components in pipelines/components/ are genuine @dsl.component-decorated
functions designed for KFP deployment. This runner calls them as regular
Python functions for local execution.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def log_step(step_num, name):
    """Print a formatted step header."""
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {name}")
    print(f"{'='*60}")
    start = time.time()
    return start


def log_complete(start_time, name):
    """Print step completion with timing."""
    elapsed = time.time() - start_time
    print(f"\n[PASS] {name} completed in {elapsed:.1f}s")


def step_ingest():
    """Step 1: Clone repo and extract features."""
    start = log_step(1, "INGEST - Clone & Extract Features")

    # Load params
    params_path = PROJECT_ROOT / "params.yaml"
    import yaml
    with open(params_path) as f:
        params = yaml.safe_load(f)

    repo_url = "https://github.com/django/django.git"
    since_date = params.get("since", "2023-08-09")
    label_window_days = 7

    # Clone/pull the repo
    repo_path = Path("../../django").resolve()
    if not repo_path.exists():
        print(f"Cloning {repo_url}...")
        subprocess.run(
            ["git", "clone", repo_url, str(repo_path)],
            check=True,
            timeout=600,
        )
    else:
        print(f"Pulling existing repo at {repo_path}...")
        subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)

    # Extract features
    from ml.extract_features import CommitFeatureExtractor

    extractor = CommitFeatureExtractor(
        repo_path=str(repo_path),
        since=since_date,
        label_window_days=label_window_days,
    )

    output_path = str(PROJECT_ROOT / "data" / "commit_features.csv")
    extractor.extract_and_save(output_path)

    log_complete(start, "INGEST")
    return output_path


def step_feature_eng(features_path):
    """Step 2: Light feature engineering pass-through."""
    start = log_step(2, "FEATURE ENG - Pass-through")

    import pandas as pd

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Existing feature set is sufficient — no transformation needed
    print("Feature set is well-designed from Phase 1. No transformation needed.")

    log_complete(start, "FEATURE ENG")
    return features_path


def step_validate(features_path):
    """Step 3: Validate features (row count, class balance, schema)."""
    start = log_step(3, "VALIDATE - Sanity Checks")

    import pandas as pd
    import yaml

    df = pd.read_csv(features_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    # 1. Row count
    min_rows = 100
    if total_rows < min_rows:
        raise ValueError(f"FAIL: Only {total_rows} rows, need >= {min_rows}")
    print(f"[PASS] Row count OK: {total_rows} >= {min_rows}")

    # 2. Class balance
    positive_count = int(df["risky"].sum())
    positive_pct = positive_count / total_rows
    print(f"Class balance: {positive_count} positive ({positive_pct:.2%}), "
          f"{total_rows - positive_count} negative ({1 - positive_pct:.2%})")

    if positive_pct < 0.05:
        print(f"[WARN] Positive class below 5% ({positive_pct:.2%})")
    else:
        print(f"[PASS] Class balance OK: {positive_pct:.2%} >= 5%")

    # 3. Schema check
    config_path = PROJECT_ROOT / "ml" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    expected = config.get("feature_columns", []) + ["risky", "hash"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"FAIL: Missing columns: {missing}")
    print(f"[PASS] Schema OK: All {len(expected)} expected columns present")

    print("\nAll validations passed!")
    log_complete(start, "VALIDATE")
    return features_path


def step_automl(features_path):
    """Step 4: AutoML search — compare LightGBM, RandomForest, LogisticRegression."""
    start = log_step(4, "AUTOML SEARCH - Model Comparison")

    import yaml
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )

    try:
        import lightgbm as lgb
        HAS_LGBM = True
    except ImportError:
        HAS_LGBM = False
        print("WARNING: LightGBM not available, skipping it")

    # Load config
    config_path = PROJECT_ROOT / "ml" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    feature_columns = config.get("feature_columns", [])
    lgbm_params = config.get("lightgbm_params", {})

    # Load features
    df = pd.read_csv(features_path)
    X = df[feature_columns]
    y = df["risky"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features: {feature_columns}")

    # Define models
    models = {}
    if HAS_LGBM:
        models["lightgbm"] = lgb.LGBMClassifier(**lgbm_params)
    models["random_forest"] = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    models["logistic_regression"] = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )

    # Train and evaluate
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
        print(f"  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model

    print(f"\n{'='*60}")
    print("AutoML Search Results:")
    print(f"{'='*60}")
    for name, m in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        marker = " << BEST" if name == best_model_name else ""
        print(f"  {name}: F1={m['f1']:.4f}{marker}")
    print(f"{'='*60}")
    print(f"Best model: {best_model_name} (F1={best_f1:.4f})")

    # Save model and results
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    results_path = output_dir / "automl_results.json"
    with open(results_path, "w") as f:
        json.dump({"best_model": best_model_name, "results": results, "feature_columns": feature_columns}, f, indent=2)

    print(f"\nBest model saved to {model_path}")
    log_complete(start, "AUTOML SEARCH")
    return str(model_path)


def step_register(model_path):
    """Step 5: Register the best model to MLflow, compare with current production."""
    start = log_step(5, "REGISTER MODEL - MLflow Registry")

    import mlflow
    import mlflow.sklearn
    import joblib

    tracking_uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("gatekeeper")

    # Load the new model
    new_model = joblib.load(model_path)
    print(f"Model type: {type(new_model).__name__}")

    # Load AutoML results
    results_path = PROJECT_ROOT / "models" / "automl_results.json"
    new_metrics = {}
    if results_path.exists():
        with open(results_path) as f:
            automl_results = json.load(f)
        best_model_name = automl_results["best_model"]
        new_metrics = automl_results["results"].get(best_model_name, {})
        print(f"New model ({best_model_name}) F1: {new_metrics.get('f1', 'N/A')}")

    # Check existing registered model
    model_name = "GatekeeperRiskPredictor"
    current_f1 = 0
    current_version = None

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            current_version = latest.version
            run = client.get_run(latest.run_id)
            current_f1 = run.data.metrics.get("f1", 0)
            print(f"Current registered model: v{current_version}, F1={current_f1:.4f}")
    except Exception as e:
        print(f"No existing model registered: {e}")

    new_f1 = new_metrics.get("f1", 0)
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

    # Log to MLflow
    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": type(new_model).__name__,
            "source": "retrain_pipeline",
            "promoted": str(should_promote),
        })
        if new_metrics:
            mlflow.log_metrics(new_metrics)
        mlflow.sklearn.log_model(
            new_model, "model",
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

    # Promote to Production if better
    if should_promote:
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            latest = max(versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(
                name=model_name, version=latest.version, stage="Production",
            )
            print(f"Model v{latest.version} promoted to Production")
        except Exception as e:
            print(f"Note: Could not transition to Production: {e}")

    status = (f"{'Promoted to Production' if should_promote else 'Kept current model'}. "
              f"New F1={new_f1:.4f}, Previous F1={current_f1:.4f}")
    log_complete(start, "REGISTER MODEL")
    return status


def main():
    """Run the full retraining pipeline."""
    print("=" * 60)
    print("GATEKEEPER RETRAINING PIPELINE")
    print(f"Started at: {datetime.now()}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)

    total_start = time.time()

    try:
        # Step 1: Ingest
        features_path = step_ingest()

        # Step 2: Feature Engineering
        features_path = step_feature_eng(features_path)

        # Step 3: Validate
        features_path = step_validate(features_path)

        # Step 4: AutoML Search
        model_path = step_automl(features_path)

        # Step 5: Register Model
        status = step_register(model_path)

        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Status: {status}")
        print(f"Completed at: {datetime.now()}")
        print(f"{'='*60}")

    except Exception as e:
        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"PIPELINE FAILED after {total_elapsed:.1f}s")
        print(f"Error: {e}")
        print(f"{'='*60}")
        raise


if __name__ == "__main__":
    main()
