#!/usr/bin/env python3
"""
FastAPI application for Gatekeeper MLOps quality gate.
Provides prediction endpoint for commit risk assessment.
"""

import glob
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
    """Load the model from MLflow Model Registry or filesystem."""
    global model

    # Set MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    model_name = "GatekeeperRiskPredictor"

    # Strategy 1: Try loading from Model Registry via models:/ URI
    try:
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model '{model_name}' from MLflow Model Registry")
        return
    except Exception as e:
        print(f"Model Registry load failed: {e}")

    # Strategy 2: Find model artifacts on filesystem and load directly with skops
    # This handles cases where artifact URIs contain platform-specific absolute paths
    try:
        pattern = os.path.join(os.path.dirname(__file__), "..", "mlruns", "*", "models", "*", "artifacts", "model.skops")
        skops_files = glob.glob(pattern)

        if not skops_files:
            raise FileNotFoundError("No model.skops files found in mlruns")

        # Pick the most recently modified one (latest training run)
        latest_file = max(skops_files, key=os.path.getmtime)
        print(f"Loading model directly from: {latest_file}")

        trusted_types = [
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier",
            "sklearn.ensemble._forest.RandomForestClassifier",
            "sklearn.tree._classes.DecisionTreeClassifier",
            "sklearn.utils._tags._TagsDict",
            "numpy.dtype",
            "numpy.ndarray",
            "pandas.core.frame.DataFrame",
            "pandas.core.series.Series",
        ]
        model = sio.loads(open(latest_file, "rb").read(), trusted=trusted_types)
        print(f"Loaded model '{model_name}' via direct filesystem fallback")
    except Exception as e2:
        print(f"Strategy 2 failed: {e2}")

    # Strategy 3: Load standalone model file (for Docker/GitHub Actions environments)
    if model is None:
        try:
            standalone_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
            if os.path.exists(standalone_model_path):
                print(f"Loading standalone model from: {standalone_model_path}")
                trusted_types = [
                    "collections.OrderedDict",
                    "lightgbm.basic.Booster",
                    "lightgbm.sklearn.LGBMClassifier",
                    "numpy.dtype",
                    "numpy.ndarray",
                    "pandas.core.frame.DataFrame",
                    "pandas.core.series.Series",
                ]
                model = sio.loads(open(standalone_model_path, "rb").read(), trusted=trusted_types)
                print(f"Loaded model '{model_name}' from standalone file")
            else:
                raise FileNotFoundError(f"Standalone model not found: {standalone_model_path}")
        except Exception as e3:
            print(f"FATAL: Could not load model from any source: {e3}")
            model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup."""
    _load_model()
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
        
        # Determine risk label based on thresholds
        # Note: <= 0.6 for medium means exactly 0.60 is medium (upper edge),
        # not high. This matches the stated definition: <0.3 low, 0.3-0.6
        # medium, >0.6 high. Thresholds were originally set for LightGBM
        # and not recalibrated after Phase 7 promoted RandomForest.
        if risk_score < 0.3:
            risk_label = "low"
        elif risk_score <= 0.6:
            risk_label = "medium"
        else:
            risk_label = "high"
        
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