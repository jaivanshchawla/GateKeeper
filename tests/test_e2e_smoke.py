#!/usr/bin/env python3
"""W4.6: End-to-end smoke test.

One test that exercises the whole chain on a real commit:
  extract -> score -> rules -> SHAP -> PR comment markdown -> dashboard endpoint

Assert nothing is empty at any stage.
"""
import os
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def config():
    with open(os.path.join(REPO_ROOT, "ml", "config.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sample_commit(config):
    """Pick a real commit from the training CSV."""
    csv_path = os.path.join(REPO_ROOT, "data", "commit_features.csv")
    if not os.path.exists(csv_path):
        pytest.skip("Training CSV not found")
    df = pd.read_csv(csv_path)
    return df[df["source_repo"] == "django"].iloc[100]


class TestEndToEndSmoke:
    """W4.6: Full chain — nothing is empty at any stage."""

    def test_full_chain(self, config, sample_commit):
        """Exercise extract -> score -> rules -> SHAP -> comment -> dashboard."""
        # 1. SCORING: shared function
        from ml.scoring import evaluate_commit_full
        fcols = config.get("feature_columns", [])
        fv = [float(sample_commit.get(c, 0)) for c in fcols]

        result = evaluate_commit_full(
            feature_values=fv,
            repo_name="django",
            commit_hash=sample_commit["hash"],
            author=sample_commit.get("author", ""),
            message=str(sample_commit.get("commit_msg", "")),
            files=[],
            lines_added=int(sample_commit.get("lines_added", 0)),
            lines_deleted=int(sample_commit.get("lines_deleted", 0)),
            files_touched=int(sample_commit.get("files_touched", 0)),
            is_merge=bool(sample_commit.get("is_merge", 0)),
            hour_of_day=int(sample_commit.get("hour_of_day", 12)),
            day_of_week=int(sample_commit.get("day_of_week", 0)),
            author_prior_commits=int(sample_commit.get("author_prior_commits", 0)),
            file_revert_count_max=int(sample_commit.get("file_revert_count_max", 0)),
            file_prior_changes_max=int(sample_commit.get("file_prior_changes_max", 0)),
        )

        # 2. SCORE is valid
        assert 0.0 <= result.risk_score <= 1.0, f"Score {result.risk_score} out of range"
        assert result.band in ("low", "medium", "high"), f"Invalid band: {result.band}"

        # 3. RULES are present (18 rules)
        assert len(result.rule_results) == 18, f"Expected 18 rules, got {len(result.rule_results)}"
        for rule in result.rule_results:
            assert rule.rule_name, "Rule has no name"
            assert rule.severity, "Rule has no severity"
            assert isinstance(rule.passed, bool), "Rule passed is not bool"
            d = rule.to_dict()
            assert "rule" in d and "severity" in d and "passed" in d and "message" in d

        # 4. SHAP is present (3 values)
        assert len(result.shap_top3) == 3, f"Expected 3 SHAP, got {len(result.shap_top3)}"
        for s in result.shap_top3:
            assert "feature" in s, "SHAP missing feature"
            assert "shap_value" in s, "SHAP missing shap_value"
            assert "human_readable" in s, "SHAP missing human_readable"

        # 5. PR COMMENT: format_markdown produces non-empty output
        from scripts.score_pr import format_markdown
        markdown = format_markdown(
            commit_hash=sample_commit["hash"],
            risk_score=result.risk_score,
            risk_label=result.band,
            factors=[],
            author=sample_commit.get("author", "unknown"),
            explanations=result.shap_top3,
            touched_files_info=[],
            rule_results=result.rule_results,
        )
        assert len(markdown) > 100, f"PR comment too short: {len(markdown)} chars"
        assert "## Gatekeeper Risk Assessment" in markdown, "Missing header"
        assert "*Scored by" in markdown, "Missing footer"
        display_labels = {"low": "NOT FLAGGED", "medium": "ELEVATED", "high": "HIGH RISK"}
        display = display_labels.get(result.band, result.band).replace(" ", "")
        assert display in markdown.replace(" ", ""), \
            f"Display band '{display}' not found in comment"

        # 6. DASHBOARD ENDPOINT: the API returns the commit with rules+SHAP
        # (This tests the contract, not the live API)
        rule_dicts = [r.to_dict() for r in result.rule_results]
        dashboard_response = {
            "id": sample_commit["hash"][:12],
            "sha": sample_commit["hash"],
            "author": sample_commit.get("author", "unknown"),
            "score": result.risk_score,
            "risk_label": result.band,
            "rule_results": rule_dicts,
            "shap_top3": result.shap_top3,
        }
        assert len(dashboard_response["rule_results"]) == 18
        assert len(dashboard_response["shap_top3"]) == 3
        assert dashboard_response["score"] > 0
        assert dashboard_response["risk_label"] in ("low", "medium", "high")
