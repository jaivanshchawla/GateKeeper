#!/usr/bin/env python3
"""
Tests for the SHAP explanation engine (ml/explainer.py).

Verifies:
- SHAP values sum to the model prediction (log-odds + sigmoid)
- Explanation latency is under 200ms
- Output structure is correct
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops"
)


@pytest.fixture(scope="module")
def explainer_and_cols():
    """Load the explainer once for all tests in this module."""
    from ml.explainer import _load_model_and_explainer
    return _load_model_and_explainer()


@pytest.fixture(scope="module")
def feature_columns(explainer_and_cols):
    _, cols = explainer_and_cols
    return cols


@pytest.fixture(scope="module")
def model():
    """Load the model directly."""
    import skops.io as sio
    TRUSTED = [
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
        "sklearn.ensemble._forest.RandomForestClassifier",
        "sklearn.tree._classes.DecisionTreeClassifier",
        "sklearn.utils._tags._TagsDict",
        "numpy.dtype",
        "numpy.ndarray",
        "pandas.core.frame.DataFrame",
        "pandas.core.series.Series",
    ]
    return sio.loads(open(MODEL_PATH, "rb").read(), trusted=TRUSTED)


def _make_random_features(n_features: int) -> np.ndarray:
    """Create a random but valid feature vector."""
    rng = np.random.RandomState(42)
    features = rng.rand(1, n_features) * 50
    # Ensure non-negative for count features
    features = np.abs(features)
    return features.astype(np.float64)


class TestSHAPSumMatchesPrediction:
    """SHAP values in log-odds space should sum with expected_value
    to produce a value whose sigmoid matches predict_proba."""

    @pytest.mark.parametrize("seed", range(5))
    def test_shap_sum_matches_prediction(
        self, explainer_and_cols, feature_columns, model, seed
    ):
        explainer, cols = explainer_and_cols
        rng = np.random.RandomState(seed)
        features = rng.rand(1, len(cols)).astype(np.float64) * 50
        features = np.abs(features)

        # Compute SHAP values (log-odds space for LightGBM)
        sv = explainer.shap_values(features)
        if isinstance(sv, list):
            sv_class1 = np.array(sv[1])
            ev = explainer.expected_value[1] if hasattr(explainer.expected_value, '__getitem__') else explainer.expected_value
        else:
            sv_class1 = np.array(sv)
            ev = explainer.expected_value

        # Convert log-odds to probability via sigmoid
        log_odds_sum = float(np.sum(sv_class1))
        expected_val = float(ev)
        shap_prob = 1.0 / (1.0 + np.exp(-(log_odds_sum + expected_val)))

        # Model prediction
        model_prob = float(model.predict_proba(features)[0][1])

        # They should match closely
        assert abs(shap_prob - model_prob) < 0.001, (
            f"SHAP-derived probability {shap_prob:.6f} != "
            f"model prediction {model_prob:.6f} (diff={abs(shap_prob - model_prob):.6f})"
        )


class TestSHAPExplanationOutput:
    """Test the explain() function output structure."""

    def test_explain_returns_top_k_factors(
        self, explainer_and_cols, feature_columns
    ):
        from ml.explainer import explain
        features = _make_random_features(len(feature_columns))
        factors = explain(features, top_k=3)

        assert len(factors) == 3
        for f in factors:
            assert "feature" in f
            assert "shap_value" in f
            assert "direction" in f
            assert f["direction"] in ("increases", "decreases")
            assert "feature_value" in f
            assert "description" in f

    def test_format_explanation_returns_strings(
        self, explainer_and_cols, feature_columns
    ):
        from ml.explainer import explain, format_explanation
        features = _make_random_features(len(feature_columns))
        factors = explain(features, top_k=3)
        formatted = format_explanation(factors)

        assert len(formatted) == 3
        for s in formatted:
            assert isinstance(s, str)
            assert len(s) > 10  # Non-trivial description


class TestSHAPLatency:
    """SHAP explanations must complete in under 200ms."""

    def test_single_explanation_latency(
        self, explainer_and_cols, feature_columns
    ):
        from ml.explainer import explain
        features = _make_random_features(len(feature_columns))

        # Warmup
        explain(features, top_k=3)

        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            explain(features, top_k=3)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        p99 = np.percentile(times, 99)
        assert p99 < 200, f"SHAP p99 latency {p99:.1f}ms exceeds 200ms target"
