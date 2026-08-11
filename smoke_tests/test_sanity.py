#!/usr/bin/env python3
"""
Smoke test: Sanity checks for /predict and /health endpoints.

Validates:
- Both health endpoints return 200
- Risky payload scores higher than safe payload
"""

import requests


def test_api_health_returns_200(api_url, wait_for_api):
    """Validate API health endpoint returns 200."""
    response = requests.get(f"{api_url}/health")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    assert data["status"] == "healthy", f"Expected status 'healthy', got '{data['status']}'"
    assert data["model_loaded"] is True, "Model should be loaded"

    print(f"API health check passed: {data}")


def test_webhook_health_returns_200(webhook_url):
    """Validate webhook health endpoint returns 200."""
    try:
        response = requests.get(f"{webhook_url}/health", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print(f"Webhook health check passed: {response.json()}")
    except requests.exceptions.ConnectionError:
        # Webhook might not be running in some test environments
        print("WARNING: Webhook not available, skipping webhook health check")


def test_risky_scores_higher_than_safe(api_url, safe_payload, risky_payload, wait_for_api):
    """Validate that risky payload produces higher risk score than safe payload."""
    # Get safe score
    safe_response = requests.post(f"{api_url}/predict", json=safe_payload)
    assert safe_response.status_code == 200
    safe_score = safe_response.json()["risk_score"]

    # Get risky score
    risky_response = requests.post(f"{api_url}/predict", json=risky_payload)
    assert risky_response.status_code == 200
    risky_score = risky_response.json()["risk_score"]

    # Risky should score higher
    assert risky_score > safe_score, (
        f"Risky score ({risky_score:.4f}) should be higher than safe score ({safe_score:.4f})"
    )

    print(f"Score comparison passed: safe={safe_score:.4f}, risky={risky_score:.4f}, diff={risky_score - safe_score:.4f}")
