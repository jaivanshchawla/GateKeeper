#!/usr/bin/env python3
"""
FastAPI application for Gatekeeper MLOps quality gate.
Provides prediction endpoint for commit risk assessment.
"""

import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
import skops.io as sio
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:
    Instrumentator = None

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")

def load_config():
    """Load feature configuration from YAML."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
FEATURE_COLUMNS = config.get("feature_columns", [])
THRESHOLDS = config.get("thresholds", {})
DEFAULT_THRESHOLDS = THRESHOLDS.get("_global", {"high": 0.8619, "medium": 0.7536})

# Pydantic models for request/response
class Features(BaseModel):
    """Feature values with validation constraints.
    
    All count-based features must be >= 0.
    Temporal features have specific valid ranges.
    is_fix_bug_revert is a binary flag (0 or 1).
    """
    lines_added: int = Field(..., ge=0, description="Lines of code added")
    lines_deleted: int = Field(..., ge=0, description="Lines of code deleted")
    files_touched: int = Field(..., ge=0, description="Number of files modified")
    dirs_touched: int = Field(..., ge=0, description="Number of directories touched")
    author_prior_commits: int = Field(..., ge=0, description="Author's total prior commits in repo")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    commit_msg_length: int = Field(..., ge=0, description="Commit message length in characters")
    is_fix_bug_revert: int = Field(..., ge=0, le=1, description="1 if commit contains fix/bug/revert keywords, 0 otherwise")
    # M.1a: File-level history
    file_prior_changes_max: float = Field(0.0, ge=0)
    file_prior_changes_mean: float = Field(0.0, ge=0)
    file_prior_risky_max: float = Field(0.0, ge=0)
    file_prior_risky_mean: float = Field(0.0, ge=0)
    file_revert_count_max: float = Field(0.0, ge=0)
    file_revert_count_mean: float = Field(0.0, ge=0)
    file_age_days_max: float = Field(0.0, ge=0)
    file_age_days_mean: float = Field(0.0, ge=0)
    file_authors_count_max: float = Field(0.0, ge=0)
    file_authors_count_mean: float = Field(0.0, ge=0)
    days_since_last_change_max: float = Field(0.0, ge=0)
    days_since_last_change_mean: float = Field(0.0, ge=0)
    # M.1b: Author-file familiarity
    author_file_prior_commits_max: float = Field(0.0, ge=0)
    author_file_prior_commits_mean: float = Field(0.0, ge=0)
    author_dir_prior_commits_max: float = Field(0.0, ge=0)
    author_dir_prior_commits_mean: float = Field(0.0, ge=0)
    is_author_first_touch_dir: int = Field(0, ge=0, le=1)
    author_days_since_last_commit: float = Field(0.0, ge=0)
    # M.1c: Change-shape
    churn_ratio: float = Field(0.0, ge=0)
    change_entropy: float = Field(0.0, ge=0)
    max_file_churn: float = Field(0.0, ge=0)
    is_test_only: int = Field(0, ge=0, le=1)
    test_to_code_ratio: float = Field(0.0, ge=0, le=1)
    config_touch: int = Field(0, ge=0, le=1)
    is_merge: int = Field(0, ge=0, le=1)
    files_per_dir_ratio: float = Field(0.0, ge=0)
    
    class Config:
        extra = "allow"  # Allow extra fields (ignored) for backward compatibility

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: Features

class ExplanationItem(BaseModel):
    """A single SHAP explanation factor."""
    feature: str
    description: str
    shap_value: float
    direction: str
    feature_value: float
    human_readable: str

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    risk_score: float
    risk_label: str
    commit_hash: str = ""
    explanations: list[ExplanationItem] = []

class CommitScoreRequest(BaseModel):
    """A single commit to score within a PR."""
    hash: str
    features: Features


class ScorePRRequest(BaseModel):
    """Request model for PR-level scoring endpoint."""
    commits: list[CommitScoreRequest]
    repo_name: str = ""


class PRVerdictResponse(BaseModel):
    """Response model for PR-level scoring endpoint."""
    verdict: str  # low/medium/high
    mean_score: float
    max_score: float
    band_counts: dict[str, int]
    total_commits: int
    total_files: int
    total_lines_added: int
    total_lines_deleted: int
    riskiest_commit_hash: str = ""
    should_block: bool
    blocked_rules: list[dict]
    warned_rules: list[dict]
    info_rules: list[dict]
    patterns: list[dict]
    comment_markdown: str = ""


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model_loaded: bool

# Global variables
model = None
explainer = None  # SHAP TreeExplainer, loaded at startup


def _load_model():
    """Load the model from the standalone .skops file.

    In production (Docker), always load from models/gatekeeper_risk_model.skops.
    For local dev with MLflow, try the registry first as a convenience.
    """
    global model

    # Standalone model path (primary — works in Docker and local)
    standalone_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")

    # In Docker (no mlflow.db), load directly from standalone
    mlflow_db = os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
    if not os.path.exists(mlflow_db) and os.path.exists(standalone_path):
        try:
            trusted_types = [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "numpy.dtype",
                "numpy.ndarray",
                "pandas.core.frame.DataFrame",
                "pandas.core.series.Series",
            ]
            model = sio.loads(open(standalone_path, "rb").read(), trusted=trusted_types)
            print(f"Loaded standalone model: {type(model).__name__} ({model.n_features_in_} features)")
            return
        except Exception as e:
            print(f"Standalone load failed: {e}")

    # Local dev: try MLflow Model Registry first
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        model_uri = "models:/GatekeeperRiskPredictor/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        print("Loaded model from MLflow Model Registry")
        return
    except Exception as e:
        print(f"Model Registry load failed: {e}")

    # Fallback: standalone file (even if mlflow.db exists)
    if model is None and os.path.exists(standalone_path):
        try:
            trusted_types = [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "numpy.dtype",
                "numpy.ndarray",
                "pandas.core.frame.DataFrame",
                "pandas.core.series.Series",
            ]
            model = sio.loads(open(standalone_path, "rb").read(), trusted=trusted_types)
            print(f"Loaded standalone model (fallback): {type(model).__name__}")
        except Exception as e:
            print(f"FATAL: Could not load model: {e}")
            model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model and warmup on startup."""
    _load_model()
    # Warmup: run one prediction to trigger lazy init (fixes p99 latency)
    if model is not None:
        try:
            dummy = np.zeros((1, len(FEATURE_COLUMNS)))
            model.predict_proba(dummy)
            print("Model warmup complete")
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}")
    # Initialize SHAP explainer
    global explainer
    try:
        from ml.explainer import _load_model_and_explainer
        _load_model_and_explainer()
        explainer = True  # Signal that explainer is available
        print("SHAP explainer initialized")
    except Exception as e:
        print(f"SHAP explainer init failed (non-fatal): {e}")
        explainer = None

    yield
    # No cleanup needed


# Create FastAPI app
app = FastAPI(
    title="Gatekeeper API",
    description="ML-based commit risk prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus metrics — exposed at /metrics
if Instrumentator is not None:
    Instrumentator().instrument(app).expose(app)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict commit risk score.
    
    Pydantic automatically validates:
    - All required features are present (Field(...))
    - Values are integers (int type hints)
    - Values are within valid ranges (ge, le constraints)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Prepare feature array in correct order (Pydantic already validated)
        feature_values = [getattr(request.features, col) for col in FEATURE_COLUMNS]
        features_array = np.array([feature_values])
        
        # Get prediction probability of risky class (class 1)
        risk_score = float(model.predict_proba(features_array)[0][1])
        
        # Determine risk label using percentile-based thresholds.
        # Per-repo cutoffs from config.yaml; fallback to _global for unknown repos.
        repo_name = getattr(request.features, "source_repo", "")
        repo_thresholds = THRESHOLDS.get(repo_name, DEFAULT_THRESHOLDS)
        high_cutoff = repo_thresholds["high"]
        medium_cutoff = repo_thresholds["medium"]

        if risk_score >= high_cutoff:
            risk_label = "high"
        elif risk_score >= medium_cutoff:
            risk_label = "medium"
        else:
            risk_label = "low"
        
        # Get commit hash if provided (optional field)
        commit_hash = getattr(request.features, "hash", "")

        features_dict = {col: getattr(request.features, col) for col in FEATURE_COLUMNS}

        # SHAP explanations
        explanations = []
        if explainer is not None:
            try:
                from ml.explainer import explain, format_explanation
                factors = explain(features_array, top_k=3)
                human_readable = format_explanation(factors, features_dict)
                explanations = [
                    ExplanationItem(
                        feature=f["feature"],
                        description=f["description"],
                        shap_value=f["shap_value"],
                        direction=f["direction"],
                        feature_value=f["feature_value"],
                        human_readable=hr,
                    )
                    for f, hr in zip(factors, human_readable)
                ]
            except Exception:
                pass  # Non-fatal: prediction still works without explanations

        return PredictionResponse(
            risk_score=risk_score,
            risk_label=risk_label,
            commit_hash=str(commit_hash) if commit_hash else "",
            explanations=explanations,
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {e!s}"
        )

@app.post("/score_pr", response_model=PRVerdictResponse)
async def score_pr(request: ScorePRRequest):
    """Score every commit in a PR and return an aggregated verdict.

    Takes a list of commits with their features, scores each one,
    aggregates to a PR-level verdict, detects PR-level patterns,
    and returns a formatted GitHub comment.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from ml.pr_scoring import (
            CommitScore,
            aggregate_commits_to_pr,
            detect_pr_patterns,
            format_pr_comment,
        )

        commit_scores = []
        rule_engine = None
        try:
            from rules.engine import RuleEngine, load_config as load_rules_config
            rule_engine = RuleEngine(load_rules_config())
        except Exception:
            pass

        for req in request.commits:
            # Score with model
            feature_values = [getattr(req.features, col, 0) for col in FEATURE_COLUMNS]
            features_array = np.array([feature_values])
            risk_score = float(model.predict_proba(features_array)[0][1])

            # Determine band
            repo_name = getattr(req.features, "source_repo", request.repo_name, "")
            repo_thresholds = THRESHOLDS.get(repo_name, DEFAULT_THRESHOLDS)
            if risk_score >= repo_thresholds["high"]:
                risk_label = "high"
            elif risk_score >= repo_thresholds["medium"]:
                risk_label = "medium"
            else:
                risk_label = "low"

            # SHAP explanations
            explanations = []
            if explainer is not None:
                try:
                    from ml.explainer import explain, format_explanation
                    features_dict = {col: getattr(req.features, col, 0) for col in FEATURE_COLUMNS}
                    factors = explain(features_array, top_k=3)
                    human_readable = format_explanation(factors, features_dict)
                    explanations = [
                        {**f, "human_readable": hr}
                        for f, hr in zip(factors, human_readable)
                    ]
                except Exception:
                    pass

            # Run rules
            rule_results = []
            if rule_engine is not None:
                try:
                    touched = getattr(req.features, "touched_files", "")
                    file_list = [f.strip() for f in str(touched).split("|") if f.strip()] if touched else []
                    ctx = CommitContext(
                        hash=req.hash,
                        author=getattr(req.features, "author", ""),
                        message=getattr(req.features, "commit_message", ""),
                        files=file_list,
                        lines_added=req.features.lines_added,
                        lines_deleted=req.features.lines_deleted,
                        files_touched=req.features.files_touched,
                        dirs_touched=req.features.dirs_touched,
                        is_merge=bool(getattr(req.features, "is_merge", 0)),
                        hour_of_day=req.features.hour_of_day,
                        day_of_week=req.features.day_of_week,
                        author_prior_commits=req.features.author_prior_commits,
                        file_revert_count_max=int(getattr(req.features, "file_revert_count_max", 0)),
                        file_prior_changes_max=int(getattr(req.features, "file_prior_changes_max", 0)),
                        repo_name=repo_name,
                        risk_score=risk_score,
                        risk_label=risk_label,
                    )
                    rule_results = rule_engine.evaluate(ctx)
                except Exception:
                    pass

            cs = CommitScore(
                hash=req.hash,
                author=getattr(req.features, "author", ""),
                message=getattr(req.features, "commit_message", ""),
                risk_score=risk_score,
                risk_label=risk_label,
                files=file_list if 'file_list' in dir() else [],
                lines_added=req.features.lines_added,
                lines_deleted=req.features.lines_deleted,
                files_touched=req.features.files_touched,
                rule_results=rule_results,
                explanations=explanations,
                blocked=rule_engine.should_block(rule_results) if rule_engine else False,
                warning_count=sum(1 for r in rule_results if not r.passed and r.severity == Severity.WARN),
            )
            commit_scores.append(cs)

        # Aggregate to PR verdict
        verdict = aggregate_commits_to_pr(commit_scores)
        comment = format_pr_comment(verdict, request.repo_name)

        return PRVerdictResponse(
            verdict=verdict.verdict,
            mean_score=verdict.mean_score,
            max_score=verdict.max_score,
            band_counts=verdict.band_counts,
            total_commits=verdict.total_commits,
            total_files=verdict.total_files,
            total_lines_added=verdict.total_lines_added,
            total_lines_deleted=verdict.total_lines_deleted,
            riskiest_commit_hash=verdict.riskiest_commit.hash if verdict.riskiest_commit else "",
            should_block=verdict.should_block,
            blocked_rules=verdict.blocked_rules,
            warned_rules=verdict.warned_rules,
            info_rules=verdict.info_rules,
            patterns=verdict.patterns,
            comment_markdown=comment,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PR scoring error: {e!s}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)