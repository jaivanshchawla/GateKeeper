#!/usr/bin/env python3
"""
Unit tests for Gatekeeper FastAPI API.

Tests /predict and /health endpoints using FastAPI's TestClient
with a mocked model to avoid needing a real MLflow model.
"""

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def create_mock_model():
    """Create a mock model that returns predictable predictions."""
    mock_model = MagicMock()
    # Return high risk probability (0.85 for class 1, 0.15 for class 0)
    mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
    return mock_model


def create_test_app_with_model(mock_model):
    """Create a test FastAPI app with a mocked model."""
    import api.main as app_module

    # Store original values
    original_model = app_module.model
    original_lifespan = app_module.app.router.lifespan_context

    # Set the model
    app_module.model = mock_model

    # Replace lifespan with a no-op
    @asynccontextmanager
    async def mock_lifespan(app):
        yield

    app_module.app.router.lifespan_context = mock_lifespan

    return app_module.app, original_model, original_lifespan


@pytest.fixture(scope="module")
def client():
    """Create a test client with a mocked model."""
    import api.main as app_module

    # Store original values
    original_model = app_module.model
    original_lifespan = app_module.app.router.lifespan_context

    # Set the model directly
    mock_model = create_mock_model()
    app_module.model = mock_model

    # Replace lifespan with a no-op
    @asynccontextmanager
    async def mock_lifespan(app):
        yield

    app_module.app.router.lifespan_context = mock_lifespan

    # Create test client
    with TestClient(app_module.app) as c:
        yield c

    # Restore original values
    app_module.model = original_model
    app_module.app.router.lifespan_context = original_lifespan


def make_features(**overrides):
    """Build a complete features payload with sensible defaults for all fields."""
    defaults = {
        "lines_added": 10, "lines_deleted": 0, "files_touched": 1,
        "dirs_touched": 1, "author_prior_commits": 0, "hour_of_day": 10,
        "day_of_week": 2, "commit_msg_length": 30, "is_fix_bug_revert": 0,
        "file_prior_changes_max": 5, "file_prior_changes_mean": 3.0,
        "file_prior_risky_max": 1, "file_prior_risky_mean": 0.5,
        "file_revert_count_max": 0, "file_revert_count_mean": 0.0,
        "file_age_days_max": 30, "file_age_days_mean": 20.0,
        "churn_ratio": 0.0, "change_entropy": 0.0, "max_file_churn": 10.0,
        "is_test_only": 0, "test_to_code_ratio": 0.0, "config_touch": 0,
    }
    defaults.update(overrides)
    return {"features": defaults}


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_without_model(self):
        """Health endpoint should work even without a loaded model."""
        import api.main as app_module

        original_model = app_module.model
        app_module.model = None

        try:
            with TestClient(app_module.app) as c:
                response = c.get("/health")
                assert response.status_code == 200
        finally:
            app_module.model = original_model


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_malformed_input_returns_422(self, client):
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_string_input_returns_422(self, client):
        response = client.post("/predict", json="not a dict")
        assert response.status_code == 422

    def test_predict_valid_input_returns_200(self, client):
        valid_payload = make_features()
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_predict_risk_score_in_valid_range(self, client):
        valid_payload = make_features()
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_predict_risk_label_matches_score_thresholds(self, client):
        """Risk label should match the percentile-based score thresholds.

        Since W1.2, /predict calls evaluate_commit_full() which loads
        the real model. We mock evaluate_commit_full to control the output.
        """
        from ml.scoring import ScoringResult

        mock_result = ScoringResult(
            risk_score=0.85,
            band="medium",
            shap_top3=[{
                "feature": "test",
                "shap_value": 0.1,
                "direction": "increases risk",
                "feature_value": 1.0,
                "description": "test",
                "human_readable": "test factor",
            }],
            rule_results=[],
            blocked=False,
            warning_count=0,
        )

        with patch("ml.scoring.evaluate_commit_full", return_value=mock_result):
            valid_payload = make_features(is_fix_bug_revert=1)
            response = client.post("/predict", json=valid_payload)
            data = response.json()

            # Mock returns 0.85 → medium under percentile thresholds
            assert data["risk_label"] == "medium"
            assert data["risk_score"] == 0.85

    def test_predict_missing_features_returns_422(self, client):
        """Missing required features should return 422 (Pydantic validation)."""
        incomplete_payload = {
            "features": {
                "lines_added": 10,
                "lines_deleted": 0,
                # Missing most required fields
            }
        }
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 422

    def test_predict_returns_commit_hash_if_provided(self, client):
        valid_payload = make_features(hash="abc123def456")
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert data["commit_hash"] == "abc123def456"


class TestModelNotLoaded:
    """Tests for when the model is not loaded."""

    def test_predict_without_model_returns_503(self):
        import api.main as app_module

        original_model = app_module.model
        app_module.model = None

        try:
            with TestClient(app_module.app) as c:
                valid_payload = make_features()
                response = c.post("/predict", json=valid_payload)
                assert response.status_code == 503
        finally:
            app_module.model = original_model
