#!/usr/bin/env python3
"""
W7.5c: Test that the production model artifact (models/gatekeeper_risk_model.skops)
is loadable and consistent with the MLflow registry's Production version.

In a full pipeline test, we'd compare md5 hashes. Here we verify the serving
artifact exists, is loadable, and has the expected structure.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import skops.io as sio


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKOPS_PATH = PROJECT_ROOT / "models" / "gatekeeper_risk_model.skops"

TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "numpy.dtype",
    "numpy.ndarray",
    "pandas.core.frame.DataFrame",
    "pandas.core.series.Series",
]


def test_production_model_exists():
    """The serving artifact must exist."""
    assert SKOPS_PATH.exists(), (
        f"Production model not found at {SKOPS_PATH}. "
        "Run ml/export_model.py or the retraining pipeline."
    )


def test_production_model_loadable():
    """The serving artifact must load without errors."""
    if not SKOPS_PATH.exists():
        pytest.skip("Production model not found")
    model = sio.load(SKOPS_PATH, trusted=TRUSTED_TYPES)
    assert model is not None


def test_production_model_has_predict():
    """The serving model must have predict_proba."""
    if not SKOPS_PATH.exists():
        pytest.skip("Production model not found")
    model = sio.load(SKOPS_PATH, trusted=TRUSTED_TYPES)
    assert hasattr(model, "predict_proba"), "Model must have predict_proba"


def test_production_model_predicts_correctly():
    """The serving model must produce valid predictions."""
    if not SKOPS_PATH.exists():
        pytest.skip("Production model not found")
    model = sio.load(SKOPS_PATH, trusted=TRUSTED_TYPES)
    n_features = getattr(model, "n_features_in_", 35)
    dummy = np.zeros((1, n_features))
    proba = model.predict_proba(dummy)
    assert proba.shape == (1, 2), f"Expected (1, 2) output, got {proba.shape}"
    assert 0 <= proba[0][1] <= 1, f"Score out of range: {proba[0][1]}"


def test_export_model_importable():
    """The export_model module must be importable and callable."""
    from ml.export_model import export_production_model
    assert callable(export_production_model)
