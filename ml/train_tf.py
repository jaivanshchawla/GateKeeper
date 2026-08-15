#!/usr/bin/env python3
"""
TF Serving benchmark: Train a small Keras NN on commit features.
Logged to MLflow as a benchmark run — NOT promoted to Production.
Purpose: Does a neural net beat gradient boosting on this tabular data?
"""

import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_features(path: str) -> tuple:
    df = pd.read_csv(path)
    with open("ml/config.yaml") as f:
        config = yaml.safe_load(f)
    feature_cols = config["feature_columns"]
    X = df[feature_cols].values
    y = df["risky"].values
    return X, y, feature_cols


def build_model(n_features: int) -> tf.keras.Model:
    """Small feedforward NN — intentionally modest to test if NNs help at all."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading data...")
    X, y, _ = load_features("data/commit_features.csv")

    # Same split logic as ml/train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Standardize features (NNs need this, tree models don't)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Building model...")
    model = build_model(X_train_scaled.shape[1])
    model.summary()

    print("\nTraining...")
    model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    # Evaluate
    y_pred_prob = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print("\n=== TF Serving Benchmark Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Log to MLflow as a benchmark (NOT promoted to Production)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("gatekeeper")

    with mlflow.start_run(run_name="tf_serving_benchmark"):
        mlflow.log_params({
            "model_type": "Keras Sequential NN",
            "source": "benchmark",
            "promoted": "False",
            "hidden_layers": "64->32->1",
            "dropout": "0.3->0.2",
            "epochs_trained": "50 (early stopping)",
        })
        mlflow.log_metrics(metrics)

        # Export SavedModel for TF Serving (versioned directory: 1/)
        export_path = Path("models/tf_model")
        versioned_path = export_path / "1"
        versioned_path.mkdir(parents=True, exist_ok=True)
        model.export(str(versioned_path))
        mlflow.log_artifacts(str(export_path), artifact_path="tf_model")

        print(f"\nMLflow run: {mlflow.active_run().info.run_id}")
        print(f"SavedModel exported to {versioned_path}")

    # Also save scaler for inference
    joblib.dump(scaler, "models/tf_scaler.joblib")

    print("\nDone. This is a BENCHMARK — not promoted to Production.")


if __name__ == "__main__":
    main()
