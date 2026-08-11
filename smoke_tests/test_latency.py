#!/usr/bin/env python3
"""
Smoke test: Latency check for /predict endpoint.

Validates that a single /predict call completes under 2 seconds.
Reports actual time if it fails so thresholds can be adjusted.
"""

import time

import requests

# Maximum acceptable latency in seconds
MAX_LATENCY_SECONDS = 2.0


def test_predict_latency_under_threshold(api_url, safe_payload, wait_for_api):
    """Validate /predict responds within latency threshold."""
    start = time.time()
    response = requests.post(f"{api_url}/predict", json=safe_payload)
    elapsed = time.time() - start

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    if elapsed > MAX_LATENCY_SECONDS:
        pytest.fail(
            f"Latency too high: {elapsed:.3f}s (threshold: {MAX_LATENCY_SECONDS}s). "
            f"Consider adjusting threshold or optimizing the model."
        )

    print(f"Latency check passed: {elapsed:.3f}s (threshold: {MAX_LATENCY_SECONDS}s)")


def test_predict_p95_latency(api_url, safe_payload, wait_for_api):
    """Measure latency over 10 calls and report p95."""
    latencies = []

    for i in range(10):
        start = time.time()
        response = requests.post(f"{api_url}/predict", json=safe_payload)
        elapsed = time.time() - start
        latencies.append(elapsed)
        assert response.status_code == 200

    # Calculate p95
    latencies_sorted = sorted(latencies)
    p95_index = int(len(latencies_sorted) * 0.95)
    p95_latency = latencies_sorted[p95_index]
    avg_latency = sum(latencies) / len(latencies)

    print(f"Latency stats over 10 calls: avg={avg_latency:.3f}s, p95={p95_latency:.3f}s, min={min(latencies):.3f}s, max={max(latencies):.3f}s")

    # Warn but don't fail on p95
    if p95_latency > MAX_LATENCY_SECONDS:
        print(f"WARNING: p95 latency ({p95_latency:.3f}s) exceeds threshold ({MAX_LATENCY_SECONDS}s)")


# Import pytest for pytest.fail
import pytest
