#!/usr/bin/env python3
"""
Training script for Gatekeeper MLOps quality gate.
Trains a LightGBM binary classifier to predict risky commits.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_features(features_path: str) -> pd.DataFrame:
    """Load commit features from CSV."""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} commits from {features_path}")
    return df


def prepare_data(
    df: pd.DataFrame,
    feature_columns: list
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector."""
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in features: {missing_cols}")
    
    X = df[feature_columns]
    y = df["risky"]
    
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier."""
    # Create LightGBM classifier
    model = lgb.LGBMClassifier(**params)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """Evaluate the trained model."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    return metrics


def log_to_mlflow(
    model: lgb.LGBMClassifier,
    params: Dict,
    metrics: Dict,
    X_train: pd.DataFrame
) -> None:
    """Log experiment to MLflow."""
    # Set MLflow tracking URI to local SQLite database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Set experiment name
    mlflow.set_experiment("gatekeeper")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model with trusted types for LightGBM
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="GatekeeperRiskPredictor",
            skops_trusted_types=[
                'collections.OrderedDict',
                'lightgbm.basic.Booster',
                'lightgbm.sklearn.LGBMClassifier',
                'numpy.dtype',
                'numpy.ndarray',
            ]
        )
        
        # Log feature importance
        importance = model.feature_importances_
        feature_names = X_train.columns.tolist()
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        # Use a temporary file for feature importance
        temp_file = "feature_importance_temp.csv"
        importance_df.to_csv(temp_file, index=False)
        mlflow.log_artifact(temp_file)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print(f"MLflow run logged. Run ID: {mlflow.active_run().info.run_id}")


def main():
    # Paths
    config_path = "ml/config.yaml"
    features_path = "data/commit_features.csv"
    
    # Load config
    print("Loading configuration...")
    config = load_config(config_path)
    
    feature_columns = config.get("feature_columns", [])
    lgbm_params = config.get("lightgbm_params", {})
    
    # Load features
    print("Loading features...")
    df = load_features(features_path)
    
    # Prepare data
    print("Preparing data...")
    X, y = prepare_data(df, feature_columns)
    
    # Print class balance
    total = len(y)
    positive = y.sum()
    negative = total - positive
    pos_pct = (positive / total) * 100
    
    print(f"\nClass Balance:")
    print(f"Total commits: {total}")
    print(f"Risky (1): {positive} ({pos_pct:.2f}%)")
    print(f"Safe (0): {negative} ({100 - pos_pct:.2f}%)")
    
    # Warn if positives are under 5%
    if pos_pct < 5:
        print("\n⚠️  WARNING: Positive class is under 5% of the data!")
        print("Consider resampling or adjusting the labeling criteria.")
    
    # Split data
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Train model
    print("\nTraining LightGBM model...")
    model = train_model(X_train, y_train, lgbm_params)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Log to MLflow
    print("\nLogging to MLflow...")
    log_to_mlflow(model, lgbm_params, metrics, X_train)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()