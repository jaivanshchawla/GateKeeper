#!/usr/bin/env python3
"""
Smoke test: Schema validation for /predict endpoint.

Validates that the response has exactly the expected keys with correct types.
"""

import requests


def test_predict_response_schema(api_url, safe_payload, wait_for_api):
    """Validate /predict response has correct schema and types."""
    response = requests.post(f"{api_url}/predict", json=safe_payload)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()

    # Check all expected keys are present
    expected_keys = {"risk_score", "risk_label", "commit_hash", "explanations"}
    actual_keys = set(data.keys())
    assert actual_keys == expected_keys, f"Expected keys {expected_keys}, got {actual_keys}"

    # Validate risk_score is a float between 0 and 1
    assert isinstance(data["risk_score"], float), f"risk_score should be float, got {type(data['risk_score'])}"
    assert 0.0 <= data["risk_score"] <= 1.0, f"risk_score should be 0-1, got {data['risk_score']}"

    # Validate risk_label is one of the expected values
    valid_labels = {"low", "medium", "high"}
    assert data["risk_label"] in valid_labels, f"risk_label should be one of {valid_labels}, got {data['risk_label']}"

    # Validate commit_hash is a string
    assert isinstance(data["commit_hash"], str), f"commit_hash should be string, got {type(data['commit_hash'])}"

    # Validate explanations is a list of dicts
    assert isinstance(data["explanations"], list), f"explanations should be list, got {type(data['explanations'])}"
    for exp in data["explanations"]:
        assert "feature" in exp, f"explanation missing 'feature' key: {exp}"
        assert "shap_value" in exp, f"explanation missing 'shap_value' key: {exp}"
        assert "human_readable" in exp, f"explanation missing 'human_readable' key: {exp}"

    print(f"Schema validation passed: {data}")


def test_predict_response_matches_score_label(api_url, safe_payload, wait_for_api):
    """Validate that risk_label matches percentile-based score thresholds."""
    response = requests.post(f"{api_url}/predict", json=safe_payload)

    assert response.status_code == 200
    data = response.json()

    score = data["risk_score"]
    label = data["risk_label"]

    # Global percentile thresholds from config.yaml
    # high: >= 0.8619 (top 10%), medium: >= 0.7536 (next 15%), low: < 0.7536
    if score >= 0.8619:
        assert label == "high", f"Score {score} should map to 'high', got '{label}'"
    elif score >= 0.7536:
        assert label == "medium", f"Score {score} should map to 'medium', got '{label}'"
    else:
        assert label == "low", f"Score {score} should map to 'low', got '{label}'"

    print(f"Score-label consistency passed: score={score:.4f}, label={label}")
