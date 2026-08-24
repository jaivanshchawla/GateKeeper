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
    # L.1: File-level history
    file_prior_changes_max: float = Field(0.0, ge=0)
    file_prior_changes_mean: float = Field(0.0, ge=0)
    file_prior_risky_max: float = Field(0.0, ge=0)
    file_prior_risky_mean: float = Field(0.0, ge=0)
    file_revert_count_max: float = Field(0.0, ge=0)
    file_revert_count_mean: float = Field(0.0, ge=0)
    file_age_days_max: float = Field(0.0, ge=0)
    file_age_days_mean: float = Field(0.0, ge=0)
    # L.3: Change-shape
    churn_ratio: float = Field(0.0, ge=0)
    change_entropy: float = Field(0.0, ge=0)
    max_file_churn: float = Field(0.0, ge=0)
    is_test_only: int = Field(0, ge=0, le=1)
    test_to_code_ratio: float = Field(0.0, ge=0, le=1)
    config_touch: int = Field(0, ge=0, le=1)
    
    class Config:
        extra = "allow"  # Allow extra fields (ignored) for backward compatibility

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: Features

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    risk_score: float
    risk_label: str
    commit_hash: str = ""

class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model_loaded: bool

# Global variable for loaded model
model = None


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
        
        return PredictionResponse(
            risk_score=risk_score,
            risk_label=risk_label,
            commit_hash=str(commit_hash) if commit_hash else ""
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {e!s}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)