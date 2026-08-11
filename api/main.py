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
from pydantic import BaseModel

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")

def load_config():
    """Load feature configuration from YAML."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
FEATURE_COLUMNS = config.get("feature_columns", [])

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: dict  # Dictionary of feature_name: value pairs

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
            "numpy.dtype",
            "numpy.ndarray",
            "pandas.core.frame.DataFrame",
            "pandas.core.series.Series",
        ]
        model = sio.loads(open(latest_file, "rb").read(), trusted=trusted_types)
        print(f"Loaded model '{model_name}' via direct filesystem fallback")
    except Exception as e2:
        print(f"FATAL: Could not load model from any source: {e2}")
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict commit risk score."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Validate that all required features are present
    missing_features = [f for f in FEATURE_COLUMNS if f not in request.features]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing_features}"
        )
    
    try:
        # Prepare feature array in correct order
        feature_values = [request.features[col] for col in FEATURE_COLUMNS]
        features_array = np.array([feature_values])
        
        # Get prediction probability of risky class (class 1)
        risk_score = float(model.predict_proba(features_array)[0][1])
        
        # Determine risk label based on thresholds
        if risk_score < 0.3:
            risk_label = "low"
        elif risk_score < 0.6:
            risk_label = "medium"
        else:
            risk_label = "high"
        
        return PredictionResponse(
            risk_score=risk_score,
            risk_label=risk_label,
            commit_hash=str(request.features.get("hash", ""))
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {e!s}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)