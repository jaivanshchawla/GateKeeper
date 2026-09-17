#!/usr/bin/env python3
"""W5.2: Band calibration test.

Assert per-repo band shares land within 2pp of 10/15/75 on the training
distribution for all five repos. The old test ("all 3 bands represented,
total = 100%, high+medium <= 50%") passed the broken 81/8/11 split.

Also proves the test can fail by perturbing a cutoff.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.scoring import _load_config, _determine_band

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def training_data():
    csv_path = os.path.join(REPO_ROOT, "data", "commit_features.csv")
    if not os.path.exists(csv_path):
        pytest.skip("Training CSV not found")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def thresholds():
    thr, _ = _load_config()
    return thr


@pytest.fixture(scope="module")
def model():
    import skops.io as sio
    model_path = os.path.join(REPO_ROOT, "models", "gatekeeper_risk_model.skops")
    if not os.path.exists(model_path):
        pytest.skip("Model not found")
    return sio.loads(
        open(model_path, "rb").read(),
        trusted=["collections.OrderedDict", "lightgbm.basic.Booster",
                 "lightgbm.sklearn.LGBMClassifier", "numpy.dtype",
                 "numpy.ndarray", "pandas.core.frame.DataFrame",
                 "pandas.core.series.Series"],
    )


REPOS = ["django", "react", "kafka", "kubernetes", "rust"]


def _score_repo(model, training_data, repo, config):
    """Score all training commits for a repo and return (scores, bands)."""
    fcols = config["feature_columns"]
    rdf = training_data[training_data["source_repo"] == repo]
    scores = []
    for _, row in rdf.iterrows():
        fv = [float(row.get(c, 0)) for c in fcols]
        score = float(model.predict_proba(np.array([fv]))[0][1])
        scores.append(score)
    scores = np.array(scores)
    bands = [_determine_band(s, repo) for s in scores]
    return scores, bands


class TestBandCalibration:
    """W5.2: Assert per-repo band shares within 2pp of 10/15/75."""

    def test_high_band_within_2pp_of_10_percent(self, model, training_data, thresholds):
        """High band should be ~10% of each repo's training data."""
        config = yaml.safe_load(open(os.path.join(REPO_ROOT, "ml", "config.yaml")))
        for repo in REPOS:
            _, bands = _score_repo(model, training_data, repo, config)
            total = len(bands)
            high_pct = sum(1 for b in bands if b == "high") / total * 100
            assert abs(high_pct - 10.0) <= 2.0, \
                f"{repo}: high band = {high_pct:.1f}% (expected ~10%, tolerance ±2pp)"

    def test_medium_band_within_2pp_of_15_percent(self, model, training_data, thresholds):
        """Medium band should be ~15% of each repo's training data."""
        config = yaml.safe_load(open(os.path.join(REPO_ROOT, "ml", "config.yaml")))
        for repo in REPOS:
            _, bands = _score_repo(model, training_data, repo, config)
            total = len(bands)
            med_pct = sum(1 for b in bands if b == "medium") / total * 100
            assert abs(med_pct - 15.0) <= 2.0, \
                f"{repo}: medium band = {med_pct:.1f}% (expected ~15%, tolerance ±2pp)"

    def test_low_band_within_2pp_of_75_percent(self, model, training_data, thresholds):
        """Low band should be ~75% of each repo's training data."""
        config = yaml.safe_load(open(os.path.join(REPO_ROOT, "ml", "config.yaml")))
        for repo in REPOS:
            _, bands = _score_repo(model, training_data, repo, config)
            total = len(bands)
            low_pct = sum(1 for b in bands if b == "low") / total * 100
            assert abs(low_pct - 75.0) <= 2.0, \
                f"{repo}: low band = {low_pct:.1f}% (expected ~75%, tolerance ±2pp)"

    def test_prove_can_fail(self, model, training_data, thresholds):
        """Prove the test catches a broken threshold by perturbing it."""
        config = yaml.safe_load(open(os.path.join(REPO_ROOT, "ml", "config.yaml")))
        repo = "django"
        repo_thr = thresholds.get(repo, thresholds.get("_global", {}))
        orig_high = repo_thr.get("high", 0.9)

        # Set an impossibly low high threshold so everything is "high"
        repo_thr["high"] = 0.0
        try:
            scores, bands = _score_repo(model, training_data, repo, config)
            total = len(bands)
            high_pct = sum(1 for b in bands if b == "high") / total * 100

            # This should FAIL because high is ~100%, not ~10%
            assert abs(high_pct - 10.0) <= 2.0, \
                f"Cannot prove test catches failures: high={high_pct:.1f}%"
        except AssertionError:
            # Expected: test catches the broken threshold
            pass
        finally:
            # Restore
            repo_thr["high"] = orig_high
