# Gatekeeper - MLOps Quality Gate

Phase 1: Core ML component for predicting risky commits in ML projects.

## Overview

Gatekeeper analyzes git commit history to predict which commits are likely to be risky (requiring reverts or causing issues). This enables proactive quality gates in MLOps pipelines.

## Features

- Extracts per-commit metrics using PyDriller
- Labels commits as risky/safe based on reverts and file touch patterns
- Trains a LightGBM binary classifier
- Logs experiments to MLflow

## Setup

### 1. Create virtual environment

```bash
cd gatekeeper
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Clone Django repository (target for analysis)

```bash
git clone https://github.com/django/django.git ../django
```

### 4. Edit params.yaml

Set the `repo_path` in `params.yaml` to point at your cloned Django folder:

```yaml
repo_path: ../django
since: "2023-08-09"
```

### 5. Initialize DVC

```bash
dvc init
```

### 6. Run feature extraction

```bash
dvc repro
```

Or run directly:

```bash
python ml/extract_features.py --repo-path ../django --since 2023-08-09
```

**Note:** The first run will take a while as it mines commit history and analyzes file patterns for labeling.

### 7. Train the model

```bash
python ml/train.py
```

This will:
- Load the extracted features
- Train a LightGBM classifier
- Evaluate the model
- Log everything to MLflow

### 8. View MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser to view:
- Training parameters
- Evaluation metrics
- Model artifacts
- Feature importance

## Configuration

Edit `ml/config.yaml` to adjust:
- Feature columns
- LightGBM hyperparameters
- Label window (days after commit to check for re-risks)

Edit `params.yaml` to adjust:
- Repository path
- Since date for mining

## DVC Pipeline

```bash
dvc repro          # Run the pipeline
dvc status         # Check pipeline status
dvc dag            # View pipeline DAG
```

## Project Structure

```
gatekeeper/
├── requirements.txt        # Python dependencies
├── params.yaml             # DVC parameters (repo path, since date)
├── ml/
│   ├── extract_features.py # Commit feature extraction
│   ├── config.yaml         # Feature & model configuration
│   └── train.py            # Model training
├── dvc.yaml                # DVC pipeline definition
├── data/                   # Extracted features (gitignored)
└── README.md               # This file
```

## Labeling Criteria

A commit is labeled as "risky" (1) if:
1. It's a revert commit (message contains "revert")
2. Any of its touched files are modified again within the label window (default: 7 days)

## Next Phases

- Phase 2: API for real-time predictions
- Phase 3: Docker containerization
- Phase 4: CI/CD integrationThis is a test.
