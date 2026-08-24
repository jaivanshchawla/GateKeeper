"""Determinism tests: training reproducibility and serving consistency."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import skops.io as sio
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def test_training_reproducibility():
    """Same seed produces identical F1 to 8 decimal places."""
    with open("ml/config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv("data/commit_features.csv")
    X = df[config["feature_columns"]]
    y = df["risky"]

    f1s = []
    for _ in range(2):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = lgb.LGBMClassifier(**config["lightgbm_params"])
        model.fit(X_tr, y_tr)
        f1s.append(f1_score(y_te, model.predict(X_te), zero_division=0))

    assert f1s[0] == f1s[1], f"Non-deterministic: {f1s[0]} != {f1s[1]}"
    assert f"{f1s[0]:.8f}" == f"{f1s[1]:.8f}"


def test_serving_determinism():
    """10 consecutive predict_proba calls return identical scores."""
    model = sio.load(
        "models/gatekeeper_risk_model.skops",
        trusted=sio.get_untrusted_types(file="models/gatekeeper_risk_model.skops"),
    )
    test_input = np.array([[10, 5, 3, 2, 10, 14, 1, 50, 1]])
    scores = [float(model.predict_proba(test_input)[0, 1]) for _ in range(10)]
    assert len(set(scores)) == 1, f"Non-deterministic serving: {scores}"
