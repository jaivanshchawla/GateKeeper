#!/usr/bin/env python3
"""
W1.2: Shared scoring function — ONE implementation for all call sites.

evaluate_commit_full() combines:
- ML model prediction → risk_score + band
- SHAP explanations → top 3 features
- Rule engine → rule results (metadata + content rules)
- Reviewer suggestions (optional)

Every scoring path (API /predict, POST /score_pr, scripts/score_pr.py,
gatekeeper check) calls this function. Two implementations of this
contract is what cost us B2.4, O.1, and Q-T.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rules.base import CommitContext, Severity
from rules.engine import RuleEngine, load_config as load_rules_config


@dataclass
class ScoringResult:
    """Complete scoring output for a single commit."""
    # ML output
    risk_score: float = 0.0
    band: str = "low"  # low / medium / high

    # SHAP
    shap_top3: list[dict] = field(default_factory=list)

    # Rules
    rule_results: list = field(default_factory=list)
    blocked: bool = False
    warning_count: int = 0

    # Reviewers (optional)
    suggested_reviewers: list[dict] = field(default_factory=list)


# Cached model and explainer (loaded once per process)
_model = None
_explainer = None
_rule_engine = None
_thresholds = None
_feature_columns = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    import skops.io as sio
    model_path = Path(__file__).parent.parent / "models" / "gatekeeper_risk_model.skops"
    trusted = [
        "collections.OrderedDict", "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier", "numpy.dtype", "numpy.ndarray",
        "pandas.core.frame.DataFrame", "pandas.core.series.Series",
    ]
    _model = sio.loads(model_path.read_bytes(), trusted=trusted)
    return _model


def _load_config():
    global _thresholds, _feature_columns
    if _thresholds is not None:
        return _thresholds, _feature_columns
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    _thresholds = config.get("thresholds", {})
    _feature_columns = config.get("feature_columns", [])
    return _thresholds, _feature_columns


def _load_rule_engine():
    global _rule_engine
    if _rule_engine is not None:
        return _rule_engine
    try:
        config = load_rules_config()
        _rule_engine = RuleEngine(config)
    except Exception:
        _rule_engine = RuleEngine()
    return _rule_engine


def _load_explainer():
    global _explainer
    if _explainer is not None:
        return _explainer
    try:
        from ml.explainer import _load_model_and_explainer
        _explainer, _ = _load_model_and_explainer()
    except Exception:
        pass
    return _explainer


def _determine_band(risk_score: float, repo_name: str = "") -> str:
    """Determine band using per-repo percentile thresholds."""
    thresholds, _ = _load_config()
    repo_thr = thresholds.get(repo_name, thresholds.get("_global", {}))
    high_cut = repo_thr.get("high", 0.8936)
    medium_cut = repo_thr.get("medium", 0.8230)
    if risk_score >= high_cut:
        return "high"
    elif risk_score >= medium_cut:
        return "medium"
    return "low"


def evaluate_commit_full(
    feature_values: list[float] | np.ndarray,
    repo_name: str = "",
    commit_hash: str = "",
    author: str = "",
    message: str = "",
    files: list[str] | None = None,
    lines_added: int = 0,
    lines_deleted: int = 0,
    files_touched: int = 0,
    dirs_touched: int = 0,
    is_merge: bool = False,
    hour_of_day: int = 12,
    day_of_week: int = 0,
    author_prior_commits: int = 0,
    file_revert_count_max: int = 0,
    file_prior_changes_max: int = 0,
    diff_text: str = "",
    repo_path: str = "",
    default_branch: str = "main",
    is_direct_push: bool = False,
) -> ScoringResult:
    """Score a single commit: ML + SHAP + rules + band.

    This is THE single implementation. Every scoring path calls this.
    """
    result = ScoringResult()

    # 1. ML prediction
    model = _load_model()
    _, feature_columns = _load_config()
    arr = np.array([feature_values])
    result.risk_score = float(model.predict_proba(arr)[0][1])
    result.band = _determine_band(result.risk_score, repo_name)

    # 2. SHAP explanations
    explainer = _load_explainer()
    if explainer is not None:
        try:
            from ml.explainer import explain, format_explanation
            features_dict = {col: float(feature_values[i]) for i, col in enumerate(feature_columns)}
            factors = explain(arr, top_k=3)
            human_readable = format_explanation(factors, features_dict)
            result.shap_top3 = [
                {**f, "human_readable": hr}
                for f, hr in zip(factors, human_readable)
            ]
        except Exception as e:
            import sys
            print(f"[scoring] SHAP explain failed (non-fatal): {e}", file=sys.stderr)

    # 3. Rules (metadata + content)
    engine = _load_rule_engine()
    files = files or []
    added_lines = []
    removed_lines = []
    deleted_files = []
    added_files = []

    # Parse diff_text if provided (for content rules)
    if diff_text:
        current_file = ""
        for line in diff_text.split("\n"):
            if line.startswith("diff --git"):
                parts = line.split(" b/")
                if len(parts) > 1:
                    current_file = parts[1]
            elif line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])
                if current_file and current_file not in added_files:
                    added_files.append(current_file)
            elif line.startswith("-") and not line.startswith("---"):
                removed_lines.append(line[1:])
                if current_file and current_file not in deleted_files:
                    deleted_files.append(current_file)

    ctx = CommitContext(
        hash=commit_hash,
        author=author,
        message=message,
        files=files,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        files_touched=files_touched or len(files),
        dirs_touched=dirs_touched,
        is_merge=is_merge,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        author_prior_commits=author_prior_commits,
        file_revert_count_max=file_revert_count_max,
        file_prior_changes_max=file_prior_changes_max,
        diff_text=diff_text,
        added_lines=added_lines,
        removed_lines=removed_lines,
        deleted_files=deleted_files,
        added_files=added_files,
        repo_path=repo_path,
        repo_name=repo_name,
        default_branch=default_branch,
        is_direct_push=is_direct_push,
        risk_score=result.risk_score,
        risk_label=result.band,
    )

    try:
        result.rule_results = engine.evaluate(ctx)
        result.blocked = engine.should_block(result.rule_results)
        result.warning_count = sum(
            1 for r in result.rule_results if not r.passed and r.severity == Severity.WARN
        )
    except Exception as e:
        import sys
        print(f"[scoring] Rule engine failed (non-fatal): {e}", file=sys.stderr)

    return result


def reset_cache():
    """Reset cached model/engine (for testing or reload)."""
    global _model, _explainer, _rule_engine, _thresholds, _feature_columns
    _model = None
    _explainer = None
    _rule_engine = None
    _thresholds = None
    _feature_columns = None
