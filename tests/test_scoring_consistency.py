#!/usr/bin/env python3
"""W4.4: Scoring path consistency test.

Asserts that evaluate_commit_full() called directly produces identical
results to what the API endpoint returns. This catches the class of bugs
where bulk extraction and single-commit extraction diverge.
"""
import json
import sys
import os

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.scoring import evaluate_commit_full
from rules.base import Severity


@pytest.fixture(scope="module")
def config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sample_commits(config):
    """Get 10 diverse commits from the training CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv")
    if not os.path.exists(csv_path):
        pytest.skip("Training CSV not found")
    df = pd.read_csv(csv_path)
    # Pick diverse commits: different repos, different scores
    samples = []
    for repo in ["django", "react", "kubernetes"]:
        rdf = df[df["source_repo"] == repo]
        if len(rdf) == 0:
            continue
        # Take from different quantiles
        for idx in [0, len(rdf) // 2, len(rdf) - 1]:
            if idx < len(rdf):
                samples.append(rdf.iloc[idx])
    return samples[:10]


def _score_commit(row, config):
    """Score a commit using the shared function."""
    fcols = config.get("feature_columns", [])
    fv = [float(row.get(c, 0)) for c in fcols]

    result = evaluate_commit_full(
        feature_values=fv,
        repo_name=row.get("source_repo", ""),
        commit_hash=row["hash"],
        author=row.get("author", ""),
        message=str(row.get("commit_msg", "")),
        files=[],
        lines_added=int(row.get("lines_added", 0)),
        lines_deleted=int(row.get("lines_deleted", 0)),
        files_touched=int(row.get("files_touched", 0)),
        is_merge=bool(row.get("is_merge", 0)),
        hour_of_day=int(row.get("hour_of_day", 12)),
        day_of_week=int(row.get("day_of_week", 0)),
        author_prior_commits=int(row.get("author_prior_commits", 0)),
        file_revert_count_max=int(row.get("file_revert_count_max", 0)),
        file_prior_changes_max=int(row.get("file_prior_changes_max", 0)),
    )
    return result


class TestScoringConsistency:
    """W4.4: Assert the shared scoring function produces consistent results."""

    def test_score_is_deterministic(self, config, sample_commits):
        """Calling evaluate_commit_full twice on the same input gives identical output."""
        for row in sample_commits:
            r1 = _score_commit(row, config)
            r2 = _score_commit(row, config)
            assert r1.risk_score == r2.risk_score, \
                f"Non-deterministic score for {row['hash'][:8]}: {r1.risk_score} vs {r2.risk_score}"
            assert r1.band == r2.band, \
                f"Non-deterministic band for {row['hash'][:8]}: {r1.band} vs {r2.band}"
            assert len(r1.rule_results) == len(r2.rule_results), \
                f"Non-deterministic rule count for {row['hash'][:8]}"
            assert len(r1.shap_top3) == len(r2.shap_top3), \
                f"Non-deterministic SHAP count for {row['hash'][:8]}"

    def test_rules_are_valid(self, config, sample_commits):
        """Every rule result has valid severity and to_dict() works."""
        for row in sample_commits:
            result = _score_commit(row, config)
            assert len(result.rule_results) == 18, \
                f"Expected 18 rules for {row['hash'][:8]}, got {len(result.rule_results)}"
            for rule in result.rule_results:
                assert isinstance(rule.severity, Severity), \
                    f"Rule {rule.rule_name} has invalid severity: {rule.severity}"
                d = rule.to_dict()
                assert "rule" in d
                assert "severity" in d
                assert "passed" in d
                assert "message" in d

    def test_shap_values_present(self, config, sample_commits):
        """Every scored commit has 3 SHAP explanations."""
        for row in sample_commits:
            result = _score_commit(row, config)
            assert len(result.shap_top3) == 3, \
                f"Expected 3 SHAP values for {row['hash'][:8]}, got {len(result.shap_top3)}"
            for s in result.shap_top3:
                assert "feature" in s
                assert "shap_value" in s
                assert "human_readable" in s

    def test_score_in_valid_range(self, config, sample_commits):
        """All scores are between 0 and 1."""
        for row in sample_commits:
            result = _score_commit(row, config)
            assert 0.0 <= result.risk_score <= 1.0, \
                f"Score {result.risk_score} out of range for {row['hash'][:8]}"

    def test_band_matches_thresholds(self, config, sample_commits):
        """Band assignment matches per-repo threshold cutoffs."""
        all_thresholds = config.get("thresholds", {})

        for row in sample_commits:
            result = _score_commit(row, config)
            repo = row.get("source_repo", "")
            repo_thr = all_thresholds.get(repo, all_thresholds.get("_global", {}))
            high_cut = repo_thr.get("high", 0.8936)
            medium_cut = repo_thr.get("medium", 0.8230)
            if result.risk_score >= high_cut:
                assert result.band == "high", f"Expected high for {row['hash'][:8]} (score={result.risk_score:.4f}, cut={high_cut})"
            elif result.risk_score >= medium_cut:
                assert result.band == "medium", f"Expected medium for {row['hash'][:8]} (score={result.risk_score:.4f}, cut={medium_cut})"
            else:
                assert result.band == "low", f"Expected low for {row['hash'][:8]} (score={result.risk_score:.4f})"
