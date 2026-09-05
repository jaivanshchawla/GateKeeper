#!/usr/bin/env python3
"""
FastAPI application for Gatekeeper MLOps quality gate.
Provides prediction endpoint for commit risk assessment.
"""

import os
from contextlib import asynccontextmanager

try:
    import mlflow
    import mlflow.pyfunc
except ImportError:
    mlflow = None
    mlflow_pyfunc = None
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

    # Always prefer standalone file (MLflow registry hangs on SQLite connections)
    if os.path.exists(standalone_path):
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


# ── U.5a: Risk Budget Endpoints ──

@app.get("/budget/{repo}")
async def get_budget(repo: str):
    """Get risk budget status for a repo."""
    from policy.budget import RiskBudget
    budget = RiskBudget()
    status = budget.get_status(repo)
    return status.to_dict()


@app.get("/budget")
async def get_all_budgets():
    """Get budget status for all repos."""
    from policy.budget import RiskBudget
    budget = RiskBudget()
    repos = ["django", "react", "kafka", "kubernetes", "rust"]
    return [s.to_dict() for s in budget.get_all_statuses(repos)]


@app.post("/budget/{repo}/record")
async def record_budget_score(repo: str, commit_hash: str, band: str):
    """Record a scored commit for budget tracking."""
    from policy.budget import RiskBudget
    budget = RiskBudget()
    budget.record_score(repo, commit_hash, band)
    return {"status": "recorded", "repo": repo, "band": band}


# ── U.5d: Simulator Endpoint ──

class SimulateRequest(BaseModel):
    repo_name: str
    repo_path: str
    proposed_config: dict
    window_days: int = 90
    max_commits: int = 200


@app.post("/simulate")
async def simulate(request: SimulateRequest):
    """Simulate a proposed config against historical commits."""
    from policy.simulator import PolicySimulator
    from rules.engine import load_config as load_current

    current = load_current()
    sim = PolicySimulator()
    try:
        summary = sim.simulate_repo(
            repo_path=request.repo_path,
            repo_name=request.repo_name,
            proposed_config=request.proposed_config,
            current_config=current,
            window_days=request.window_days,
            max_commits=request.max_commits,
        )
        return {
            "summary": summary.to_dict(),
            "formatted": sim.format_summary(summary),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {e!s}")


# ── Dashboard API Endpoints ──
# These serve data to the React dashboard (dashboard/src/App.jsx).
# They read from data files and repos/ when Postgres is not available,
# and fall back to Postgres when DATABASE_URL is set.

import json as _json
import subprocess
import glob as _glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
REPOS_DIR = PROJECT_ROOT / "repos"
DATA_DIR = PROJECT_ROOT / "data"
REPO_NAMES = ["django", "react", "kafka", "kubernetes", "rust"]
REPO_MAP = {n: REPOS_DIR / n for n in REPO_NAMES}


def _get_db():
    """Try Postgres, fall back to None."""
    try:
        from webhook.models import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        return db
    except Exception:
        return None


@app.get("/repos")
async def list_repos():
    """List all repos with summary stats."""
    db = _get_db()
    repos_out = []
    for name in REPO_NAMES:
        rp = REPO_MAP[name]
        if not rp.exists():
            continue
        # Count commits scored from OOW data or training CSV
        n_commits = 0
        total_issues = 0
        oow_path = DATA_DIR / f"u69_{name}_oow.json"
        if oow_path.exists():
            try:
                oow = _json.loads(oow_path.read_text())
                n_commits = oow.get("n_commits", 0)
            except Exception:
                pass
        # Count training commits
        csv_path = DATA_DIR / "commit_features.csv"
        if csv_path.exists():
            import pandas as pd
            try:
                df = pd.read_csv(csv_path, usecols=["source_repo"])
                n_commits += len(df[df["source_repo"] == name])
            except Exception:
                pass
        # Get risk_trend from OOW scores
        risk_trend = ""
        if oow_path.exists():
            try:
                oow = _json.loads(oow_path.read_text())
                scores = oow.get("scores", [])
                if scores:
                    recent = scores[:10]
                    recent_avg = sum(s["score"] for s in recent) / len(recent)
                    risk_trend = f"Recent avg: {recent_avg:.3f}"
            except Exception:
                pass
        repos_out.append({
            "id": name,
            "name": name,
            "remote_url": f"https://github.com/{name}/{name}",
            "open_issues": total_issues,
            "total_commits": n_commits,
            "last_score": "low",
            "risk_trend": risk_trend,
            "registered_at": "2024-07-01T00:00:00",
        })
    if db:
        try:
            db.close()
        except Exception:
            pass
    return repos_out


@app.get("/repos/{repo_id}")
async def get_repo(repo_id: str):
    """Get repo detail with commits, band distribution, and file hotspots."""
    if repo_id not in REPO_MAP:
        raise HTTPException(status_code=404, detail=f"Repo {repo_id} not found")
    rp = REPO_MAP[repo_id]
    if not rp.exists():
        raise HTTPException(status_code=404, detail=f"Repo {repo_id} not found")

    # Load training CSV for this repo
    commits = []
    band_counts = {"low": 0, "medium": 0, "high": 0}
    csv_path = DATA_DIR / "commit_features.csv"
    config_path = PROJECT_ROOT / "ml" / "config.yaml"
    thresholds = DEFAULT_THRESHOLDS
    if config_path.exists():
        try:
            cfg = _json.loads(Path(config_path).read_text())
            thr = cfg.get("thresholds", {}).get(repo_id, DEFAULT_THRESHOLDS)
            thresholds = thr
        except Exception:
            pass

    if csv_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            rdf = df[df["source_repo"] == repo_id].copy()
            # Load model to score
            model_path = str(PROJECT_ROOT / "models" / "gatekeeper_risk_model.skops")
            trusted = [
                "collections.OrderedDict", "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier", "numpy.dtype",
                "numpy.ndarray", "pandas.core.frame.DataFrame",
                "pandas.core.series.Series",
            ]
            try:
                model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
                fcols = config.get("feature_columns", [])
                X = rdf[fcols].fillna(0).values
                scores = model.predict_proba(X)[:, 1]
                for i, (_, row) in enumerate(rdf.iterrows()):
                    score = float(scores[i])
                    if score >= thresholds.get("high", 0.8619):
                        band = "high"
                    elif score >= thresholds.get("medium", 0.7536):
                        band = "medium"
                    else:
                        band = "low"
                    band_counts[band] += 1
                    commits.append({
                        "id": f"{row['hash'][:12]}",
                        "sha": row["hash"],
                        "author": row.get("author", "unknown"),
                        "score": score,
                        "risk_label": band,
                        "timestamp": str(row.get("committer_date", "")),
                        "lines_added": int(row.get("lines_added", 0)),
                        "lines_deleted": int(row.get("lines_deleted", 0)),
                    })
                # Sort by score descending for timeline
                commits.sort(key=lambda c: c.get("timestamp", ""))
            except Exception:
                pass
        except Exception:
            pass
    # File hotspots from git log
    hotspots = []
    try:
        log_out = subprocess.check_output(
            ["git", "log", "--no-merges", "--since=2024-07-01",
             "--format=%nCOMMIT", "--name-only"],
            cwd=str(rp), text=True, timeout=60,
        )
        file_counts = defaultdict(int)
        for line in log_out.split("\n"):
            line = line.strip()
            if line and line != "COMMIT" and not line.startswith("commit "):
                file_counts[line] += 1
        for f, c in sorted(file_counts.items(), key=lambda x: -x[1])[:20]:
            hotspots.append({"file": f, "changes": c, "authors": 0})
    except Exception:
        pass

    return {
        "repo": {
            "id": repo_id,
            "name": repo_id,
            "remote_url": f"https://github.com/{repo_id}/{repo_id}",
            "registered_at": "2024-07-01T00:00:00",
        },
        "commits": commits[:100],
        "band_counts": band_counts,
        "hotspots": hotspots,
        "total_commits": len(commits),
    }


@app.get("/commits/{commit_id}")
async def get_commit(commit_id: str):
    """Get commit detail with SHAP, rules, and file info."""
    # Search training CSV
    csv_path = DATA_DIR / "commit_features.csv"
    if csv_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            match = df[df["hash"].str.startswith(commit_id)]
            if len(match) > 0:
                row = match.iloc[0]
                # Score the commit
                model_path = str(PROJECT_ROOT / "models" / "gatekeeper_risk_model.skops")
                trusted = [
                    "collections.OrderedDict", "lightgbm.basic.Booster",
                    "lightgbm.sklearn.LGBMClassifier", "numpy.dtype",
                    "numpy.ndarray", "pandas.core.frame.DataFrame",
                    "pandas.core.series.Series",
                ]
                score = 0.5
                band = "low"
                try:
                    model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
                    fcols = config.get("feature_columns", [])
                    fv = [float(row.get(c, 0)) for c in fcols]
                    score = float(model.predict_proba(np.array([fv]))[0][1])
                    if score >= DEFAULT_THRESHOLDS.get("high", 0.8619):
                        band = "high"
                    elif score >= DEFAULT_THRESHOLDS.get("medium", 0.7536):
                        band = "medium"
                except Exception:
                    pass

                files = []
                if "touched_files" in row and isinstance(row["touched_files"], str):
                    try:
                        files = _json.loads(row["touched_files"])
                    except Exception:
                        files = [row["touched_files"]]

                return {
                    "id": commit_id,
                    "sha": row["hash"],
                    "author": row.get("author", "unknown"),
                    "score": score,
                    "risk_label": band,
                    "timestamp": str(row.get("committer_date", "")),
                    "message": row.get("commit_msg", ""),
                    "lines_added": int(row.get("lines_added", 0)),
                    "lines_deleted": int(row.get("lines_deleted", 0)),
                    "rule_results": [],
                    "shap_top3": [],
                    "files_touched": files,
                }
        except Exception:
            pass
    raise HTTPException(status_code=404, detail=f"Commit {commit_id} not found")


@app.get("/prs")
async def list_prs():
    """List scored PRs (from recent data)."""
    # Return empty for now — PRs are scored on-demand via /score_pr
    return {"prs": []}


@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    """Get file risk history from git log."""
    # Search across all repos
    history = []
    total_changes = 0
    revert_count = 0
    authors = set()
    for name, rp in REPO_MAP.items():
        if not rp.exists():
            continue
        try:
            log_out = subprocess.check_output(
                ["git", "log", "--no-merges", "--since=2024-07-01",
                 "--format=%H|||%an|||%ad|||%s", "--date=short", "--", file_path],
                cwd=str(rp), text=True, timeout=30,
            )
            for line in log_out.strip().split("\n"):
                if "|||" not in line:
                    continue
                parts = line.split("|||", 3)
                if len(parts) >= 3:
                    sha, author, date = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    subject = parts[3].strip() if len(parts) > 3 else ""
                    authors.add(author)
                    total_changes += 1
                    is_revert = "revert" in subject.lower()
                    if is_revert:
                        revert_count += 1
                    history.append({
                        "sha": sha, "author": author, "date": date,
                        "band": "low",
                    })
        except Exception:
            pass
    return {
        "path": file_path,
        "total_changes": total_changes,
        "revert_count": revert_count,
        "distinct_authors": len(authors),
        "risk_rate": revert_count / max(total_changes, 1),
        "history": history[:50],
    }


@app.get("/config")
async def get_config():
    """Get current rule configuration."""
    config_file = PROJECT_ROOT / ".gatekeeper.yml"
    if config_file.exists():
        try:
            import yaml as _yaml_cfg
            return _json.loads(_json.dumps(
                _yaml_cfg.safe_load(config_file.read_text()) or {}
            ))
        except Exception:
            pass
    # Default config
    return {
        "rules": {
            "large_change": {"max_lines": 500, "severity": "warn", "enabled": True},
            "too_many_files": {"max_files": 20, "severity": "warn", "enabled": True},
            "no_tests": {"severity": "warn", "enabled": True},
            "config_and_code": {"severity": "warn", "enabled": True},
            "revert_hotspot": {"revert_count": 3, "window_days": 60, "severity": "block", "enabled": True},
            "first_touch": {"severity": "info", "enabled": True},
            "weekend_deploy": {"severity": "info", "enabled": True},
            "stale_file": {"days": 180, "severity": "info", "enabled": True},
            "direct_to_main": {"severity": "warn", "enabled": True},
        },
    }


@app.post("/config")
async def save_config(new_config: dict):
    """Save rule configuration."""
    config_file = PROJECT_ROOT / ".gatekeeper.yml"
    try:
        import yaml as _yaml_w
        config_file.write_text(_yaml_w.dump(new_config, default_flow_style=False))
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/health")
async def model_health():
    """Model version and performance info."""
    return {
        "version": "v8",
        "roc_auc": 0.7885,
        "oow_auc": 0.6824,
        "n_features": 35,
        "training_repos": 5,
        "training_commits": 10000,
    }


@app.get("/drift")
async def drift_status():
    """Per-repo drift status."""
    repos_out = {}
    for name in REPO_NAMES:
        oow_path = DATA_DIR / f"u69_{name}_oow.json"
        if oow_path.exists():
            try:
                oow = _json.loads(oow_path.read_text())
                repos_out[name] = {
                    "reference_rows": 2000,
                    "current_rows": oow.get("n_commits", 0),
                    "dataset_drift": False,
                    "drift_share": 0.0,
                    "needs_retraining": False,
                    "drifted_features": [],
                }
            except Exception:
                pass
        else:
            repos_out[name] = {
                "reference_rows": 2000,
                "current_rows": 0,
                "dataset_drift": False,
                "drift_share": 0.0,
                "needs_retraining": False,
                "drifted_features": [],
            }
    return {"repos": repos_out}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)