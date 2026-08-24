#!/usr/bin/env python3
"""
Pytest configuration for smoke tests.
Provides fixtures for API/webhook URLs, test payloads, and dashboard logging.
"""

import os
import time

import pytest
import requests

# --- Fixtures ---

@pytest.fixture
def api_url():
    """API base URL for testing."""
    return os.environ.get("API_URL", "http://localhost:8000")


@pytest.fixture
def webhook_url():
    """Webhook base URL for testing."""
    return os.environ.get("WEBHOOK_URL", "http://localhost:5000")


@pytest.fixture
def safe_payload():
    """A clearly low-risk commit payload."""
    return {
        "features": {
            "lines_added": 5,
            "lines_deleted": 2,
            "files_touched": 1,
            "dirs_touched": 1,
            "author_prior_commits": 500,
            "hour_of_day": 14,
            "day_of_week": 1,
            "commit_msg_length": 25,
            "is_fix_bug_revert": 0,
        }
    }


@pytest.fixture
def risky_payload():
    """A clearly high-risk commit payload."""
    return {
        "features": {
            "lines_added": 500,
            "lines_deleted": 300,
            "files_touched": 25,
            "dirs_touched": 10,
            "author_prior_commits": 0,
            "hour_of_day": 2,
            "day_of_week": 6,
            "commit_msg_length": 150,
            "is_fix_bug_revert": 1,
        }
    }


@pytest.fixture
def wait_for_api(api_url):
    """Wait for the API to be ready before running tests."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"API at {api_url} not ready after {max_retries} seconds")


# --- Dashboard logging ---

def log_to_dashboard(issue_type: str, repo: str, details: str):
    """Log an issue to the dashboard if DASHBOARD_URL is set.
    
    This function gracefully degrades if the dashboard is unavailable.
    It should never cause the smoke tests to fail.
    """
    dashboard_url = os.environ.get("DASHBOARD_URL")
    if not dashboard_url:
        print(f"WARNING: DASHBOARD_URL not set, skipping dashboard logging for {issue_type}")
        return
    
    try:
        response = requests.post(
            f"{dashboard_url}/issues",
            json={
                "gate": 3,
                "type": issue_type,
                "repo": repo,
                "details": details,
            },
            timeout=5  # Short timeout to avoid blocking
        )
        if response.status_code == 201:
            print(f"Logged issue to dashboard: {issue_type}")
        else:
            print(f"WARNING: Dashboard returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Failed to log to dashboard: {e}")


# --- Failure hook ---

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to log test failures to dashboard."""
    outcome = yield
    report = outcome.get_result()
    
    # Only log failures (not errors during setup/teardown)
    if report.when == "call" and report.failed:
        # Determine issue type based on test name
        test_name = item.name
        if "schema" in test_name:
            issue_type = "schema_validation_failed"
        elif "sanity" in test_name:
            issue_type = "sanity_check_failed"
        elif "latency" in test_name:
            issue_type = "latency_threshold_exceeded"
        elif "drift" in test_name:
            issue_type = "data_drift_detected"
        else:
            issue_type = "smoke_test_failed"
        
        # Get repo name from environment or default
        repo = os.environ.get("GITHUB_REPOSITORY", "local")
        
        # Log to dashboard
        log_to_dashboard(
            issue_type=issue_type,
            repo=repo,
            details=f"Test {test_name} failed: {str(report.longrepr)[:500]}"
        )
