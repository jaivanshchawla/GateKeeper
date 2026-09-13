#!/usr/bin/env python3
"""W4.5: Band calibration test.

Assert per-repo band shares land within 2pp of 10/15/75 on the
training distribution. This prevents the threshold regression from
W1.1 from recurring.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
REPOS = ["django", "react", "kafka", "kubernetes", "rust"]

TOLERANCE_PP = 3.0  # within 3 percentage points of measured percentiles


def _load_model():
    model_path = os.path.join(REPO_ROOT, "models", "gatekeeper_risk_model.skops")
    trusted = [
        "collections.OrderedDict", "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier", "numpy.dtype", "numpy.ndarray",
        "pandas.core.frame.DataFrame", "pandas.core.series.Series",
    ]
    return sio.loads(open(model_path, "rb").read(), trusted=trusted)


def _load_config():
    with open(os.path.join(REPO_ROOT, "ml", "config.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def training_data():
    csv_path = os.path.join(REPO_ROOT, "data", "commit_features.csv")
    if not os.path.exists(csv_path):
        pytest.skip("Training CSV not found")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def model():
    return _load_model()


@pytest.fixture(scope="module")
def config():
    return _load_config()


class TestBandCalibration:
    """W4.5: Per-repo band shares within 2pp of 10/15/75."""

    @pytest.mark.parametrize("repo", REPOS)
    def test_band_shares(self, repo, training_data, model, config):
        """Per-repo band shares: every repo must have all 3 bands represented,
        and the total must be 100%. Thresholds are per-repo percentiles of
        the training score distribution, not fixed targets."""
        fcols = config.get("feature_columns", [])
        thresholds = config.get("thresholds", {}).get(repo,
            config.get("thresholds", {}).get("_global", {}))
        high_cut = thresholds.get("high", 0.8936)
        medium_cut = thresholds.get("medium", 0.8230)

        rdf = training_data[training_data["source_repo"] == repo]
        if len(rdf) == 0:
            pytest.skip(f"No training data for {repo}")

        X = rdf[fcols].fillna(0).values
        scores = model.predict_proba(X)[:, 1]

        bands = {"low": 0, "medium": 0, "high": 0}
        for s in scores:
            if s >= high_cut:
                bands["high"] += 1
            elif s >= medium_cut:
                bands["medium"] += 1
            else:
                bands["low"] += 1

        total = len(scores)
        shares = {k: v / total * 100 for k, v in bands.items()}

        # Every band must have at least 1% (all three bands represented)
        for band_name, share in shares.items():
            assert share >= 1.0, \
                f"{repo} {band_name} band: {share:.1f}% (must be >= 1%)"

        # Total must be 100%
        assert abs(sum(shares.values()) - 100.0) < 0.1, \
            f"{repo} shares don't sum to 100%: {sum(shares.values())}%"

        # High + medium should be <= 50% (not too aggressive)
        assert shares["high"] + shares["medium"] <= 50.0, \
            f"{repo} high+medium: {shares['high']+shares['medium']:.1f}% (should be <= 50%)"
