#!/usr/bin/env python3
"""
Unit tests for Gatekeeper FastAPI API.

Tests /predict and /health endpoints using FastAPI's TestClient
with a mocked model to avoid needing a real MLflow model.
"""

import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def create_mock_model():
    """Create a mock model that returns predictable predictions."""
    mock_model = MagicMock()
    # Return high risk score (0.85)
    mock_model.predict.return_value = np.array([0.85])
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


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        """Health check should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have status and model_loaded fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_without_model(self):
        """Health should report model_loaded=False when no model is loaded."""
        import api.main as app_module

        original_model = app_module.model
        original_lifespan = app_module.app.router.lifespan_context

        try:
            app_module.model = None

            # Replace lifespan with a no-op
            @asynccontextmanager
            async def mock_lifespan(app):
                yield

            app_module.app.router.lifespan_context = mock_lifespan

            with TestClient(app_module.app) as c:
                response = c.get("/health")
                data = response.json()
                assert data["model_loaded"] is False
        finally:
            app_module.model = original_model
            app_module.app.router.lifespan_context = original_lifespan


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_malformed_input_returns_422(self, client):
        """Missing 'features' key should return 422."""
        response = client.post(
            "/predict",
            json={"wrong_key": "value"},
        )
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, client):
        """Empty body should return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_string_input_returns_422(self, client):
        """String instead of object should return 422."""
        response = client.post(
            "/predict",
            json="not an object",
        )
        assert response.status_code == 422

    def test_predict_valid_input_returns_200(self, client):
        """Valid features should return 200 with risk_score and risk_label."""
        valid_payload = {
            "features": {
                "lines_added": 42,
                "lines_deleted": 10,
                "files_touched": 3,
                "dirs_touched": 2,
                "author_prior_commits": 5,
                "hour_of_day": 14,
                "day_of_week": 1,
                "commit_msg_length": 45,
                "is_fix_bug_revert": 0,
            }
        }

        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

        data = response.json()
        assert "risk_score" in data
        assert "risk_label" in data
        assert isinstance(data["risk_score"], (int, float))
        assert data["risk_label"] in ["low", "medium", "high"]

    def test_predict_risk_score_in_valid_range(self, client):
        """Risk score should be between 0 and 1."""
        valid_payload = {
            "features": {
                "lines_added": 100,
                "lines_deleted": 50,
                "files_touched": 10,
                "dirs_touched": 5,
                "author_prior_commits": 20,
                "hour_of_day": 9,
                "day_of_week": 0,
                "commit_msg_length": 80,
                "is_fix_bug_revert": 1,
            }
        }

        response = client.post("/predict", json=valid_payload)
        data = response.json()

        assert 0.0 <= data["risk_score"] <= 1.0

    def test_predict_risk_label_matches_score_thresholds(self, client):
        """Risk label should match the score thresholds."""
        # Test with high risk score (mock returns 0.85)
        valid_payload = {
            "features": {
                "lines_added": 10,
                "lines_deleted": 0,
                "files_touched": 1,
                "dirs_touched": 1,
                "author_prior_commits": 0,
                "hour_of_day": 10,
                "day_of_week": 2,
                "commit_msg_length": 30,
                "is_fix_bug_revert": 1,
            }
        }

        response = client.post("/predict", json=valid_payload)
        data = response.json()

        # Mock returns 0.85, which is > 0.6, so label should be "high"
        assert data["risk_label"] == "high"

    def test_predict_missing_features_returns_400(self, client):
        """Missing required features should return 400."""
        incomplete_payload = {
            "features": {
                "lines_added": 10,
                # Missing other required features
            }
        }

        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 400
        assert "Missing required features" in response.json()["detail"]

    def test_predict_returns_commit_hash_if_provided(self, client):
        """Response should include commit_hash if provided in features."""
        valid_payload = {
            "features": {
                "hash": "abc123def456",
                "lines_added": 10,
                "lines_deleted": 0,
                "files_touched": 1,
                "dirs_touched": 1,
                "author_prior_commits": 0,
                "hour_of_day": 10,
                "day_of_week": 2,
                "commit_msg_length": 30,
                "is_fix_bug_revert": 0,
            }
        }

        response = client.post("/predict", json=valid_payload)
        data = response.json()

        assert data["commit_hash"] == "abc123def456"


class TestModelNotLoaded:
    """Tests for behavior when model is not loaded."""

    def test_predict_without_model_returns_503(self):
        """Prediction should return 503 when model is not loaded."""
        import api.main as app_module

        original_model = app_module.model
        original_lifespan = app_module.app.router.lifespan_context

        try:
            app_module.model = None

            # Replace lifespan with a no-op
            @asynccontextmanager
            async def mock_lifespan(app):
                yield

            app_module.app.router.lifespan_context = mock_lifespan

            with TestClient(app_module.app) as c:
                valid_payload = {
                    "features": {
                        "lines_added": 10,
                        "lines_deleted": 0,
                        "files_touched": 1,
                        "dirs_touched": 1,
                        "author_prior_commits": 0,
                        "hour_of_day": 10,
                        "day_of_week": 2,
                        "commit_msg_length": 30,
                        "is_fix_bug_revert": 0,
                    }
                }

                response = c.post("/predict", json=valid_payload)
                assert response.status_code == 503
                assert "Model not loaded" in response.json()["detail"]
        finally:
            app_module.model = original_model
            app_module.app.router.lifespan_context = original_lifespan
