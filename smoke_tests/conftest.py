#!/usr/bin/env python3
"""
Shared fixtures for Gatekeeper smoke tests.
"""

import os

import pytest
import requests

# API base URL - defaults to localhost:8000 for local testing
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "http://localhost:5000")


@pytest.fixture(scope="session")
def api_url():
    """Get the API base URL."""
    return API_BASE_URL


@pytest.fixture(scope="session")
def webhook_url():
    """Get the webhook base URL."""
    return WEBHOOK_BASE_URL


@pytest.fixture(scope="session")
def safe_payload():
    """A clearly-safe commit payload (small change, daytime, experienced author)."""
    return {
        "features": {
            "lines_added": 5,
            "lines_deleted": 2,
            "files_touched": 1,
            "dirs_touched": 1,
            "author_prior_commits": 50,
            "hour_of_day": 10,
            "day_of_week": 1,
            "commit_msg_length": 25,
            "is_fix_bug_revert": 0,
        }
    }


@pytest.fixture(scope="session")
def risky_payload():
    """A clearly-risky commit payload (large change, late night, new author, fix keywords)."""
    return {
        "features": {
            "lines_added": 500,
            "lines_deleted": 200,
            "files_touched": 25,
            "dirs_touched": 10,
            "author_prior_commits": 0,
            "hour_of_day": 2,
            "day_of_week": 6,
            "commit_msg_length": 150,
            "is_fix_bug_revert": 1,
        }
    }


@pytest.fixture(scope="session")
def wait_for_api(api_url, timeout=30):
    """Wait for the API to be ready."""
    import time

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{api_url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    pytest.fail(f"API not ready after {timeout} seconds")
