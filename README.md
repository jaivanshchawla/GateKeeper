# Gatekeeper — MLOps Quality & Safety Gate

A push-to-deploy quality/safety gate for ML projects. Gatekeeper analyzes git commit history to predict which commits are likely risky, then enforces quality gates at three stages:

- **Gate 1 (Pre-push):** Local pre-commit hook that blocks risky pushes
- **Gate 2 (Pre-merge):** GitHub Action that scores PR commits and posts risk assessments as PR comments
- **Gate 3 (Post-deploy):** Smoke test suite that validates deployed services and detects data drift

A **dashboard** tracks all issues from Gates 1-3, showing open vs. resolved status over time.

## Quick Start

### 1. Create virtual environment and install dependencies

```bash
cd gatekeeper
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Clone the target repository (for training data)

```bash
git clone https://github.com/django/django.git ../django
```

### 3. Edit `params.yaml`

Set `repo_path` to point at your cloned Django folder:

```yaml
repo_path: ../django
since: "2023-08-09"
```

### 4. Initialize DVC and extract features

```bash
dvc init
dvc repro
```

Or run directly:

```bash
python ml/extract_features.py --repo-path ../django --since 2023-08-09
```

### 5. Train the model

```bash
python ml/train.py
```

### 6. Export the standalone model

```bash
python ml/export_model.py
```

### 7. Run the full stack with Docker

```bash
docker compose up -d
```

This starts 4 services:

| Service    | Port | Description                     |
|------------|------|---------------------------------|
| api        | 8000 | FastAPI prediction service      |
| webhook    | 5000 | Flask webhook + dashboard API   |
| postgres   | 5432 | PostgreSQL for issue tracking   |
| dashboard  | 3000 | React dashboard (Vite)          |

### 8. Access the dashboard

Open http://localhost:3000 to view the Gatekeeper Dashboard.

## Phase 1: Core ML

- `ml/extract_features.py` — Feature extraction using PyDriller
- `ml/train.py` — LightGBM training with MLflow tracking
- `ml/config.yaml` — Feature columns and model hyperparameters
- `ml/export_model.py` — Export standalone model as `.skops`

## Phase 2: Serving

- `api/main.py` — FastAPI app with `/predict` and `/health` endpoints
- `webhook/app.py` — Flask app with webhook receiver and dashboard API
- `api/Dockerfile` and `webhook/Dockerfile` — Container images

## Phase 3: Gate 1 (Pre-push Hook)

- `tests/` — Unit tests for feature extraction and API
- `scripts/check_data_leakage.py` — Validates train/test split integrity
- `.pre-commit-config.yaml` — Pre-push hooks (detect-secrets, ruff, leakage, pytest)

### Install pre-push hook

```bash
pre-commit install --hook-type pre-push
```

## Phase 4: Gate 2 (GitHub Action)

- `scripts/score_pr.py` — Standalone commit risk scorer
- `.github/workflows/gate2_pr_risk.yml` — Scores PR commits and posts risk comments

## Phase 5: Gate 3 (Post-deploy Smoke Tests)

- `smoke_tests/test_schema.py` — Validates `/predict` response schema
- `smoke_tests/test_sanity.py` — Health checks and score comparison
- `smoke_tests/test_latency.py` — Response time validation
- `smoke_tests/test_drift.py` — Evidently data drift detection
- `.github/workflows/gate3_post_deploy.yml` — Automated smoke tests

## Phase 6: Dashboard

- `webhook/models.py` — SQLAlchemy model for issues table
- `webhook/routes/dashboard.py` — CRUD API for issues
- `dashboard/` — React app (Vite + Recharts)

### Dashboard API

| Endpoint              | Method | Description                        |
|-----------------------|--------|------------------------------------|
| `/issues`             | POST   | Log a new issue                    |
| `/issues`             | GET    | List issues (filter by `?status=` and `?repo=`) |
| `/issues/<id>`        | PATCH  | Toggle status (open/resolved)      |
| `/issues/stats`       | GET    | Daily counts for last 30 days      |

### Graceful degradation for Gates 2 & 3

Both `score_pr.py` and the smoke test conftest check for `DASHBOARD_URL`. If unset or unreachable, they log a warning and continue — the gate never fails because of the dashboard.

```bash
# Set in GitHub Actions secrets for Gate 2/3:
DASHBOARD_URL=https://your-deployed-webhook-url
```

## View MLflow UI

```bash
mlflow ui
```

Open http://localhost:5000 to view experiment runs, metrics, and the registered `GatekeeperRiskPredictor` model.

## Configuration

| File                | Purpose                                      |
|---------------------|----------------------------------------------|
| `params.yaml`       | DVC parameters (repo path, since date)       |
| `ml/config.yaml`    | Feature columns, LightGBM hyperparameters    |
| `requirements.txt`  | Python dependencies                          |
| `docker-compose.yml`| Service definitions (API, webhook, postgres, dashboard) |

## Labeling Criteria

A commit is labeled as **risky** (1) if:
1. It's a revert commit (message contains "revert"), OR
2. Any of its touched files are modified again within the label window (default: 7 days)

## Project Structure

```
gatekeeper/
├── requirements.txt            # Python dependencies
├── params.yaml                 # DVC parameters
├── docker-compose.yml          # All 4 services
├── ml/
│   ├── extract_features.py     # Commit feature extraction
│   ├── config.yaml             # Feature & model configuration
│   ├── train.py                # Model training + MLflow logging
│   └── export_model.py         # Export standalone .skops model
├── api/
│   ├── main.py                 # FastAPI prediction service
│   └── Dockerfile
├── webhook/
│   ├── app.py                  # Flask webhook + dashboard API
│   ├── models.py               # SQLAlchemy issues model
│   ├── routes/dashboard.py     # Dashboard CRUD endpoints
│   └── Dockerfile
├── dashboard/                  # React dashboard (Vite)
│   ├── src/App.jsx
│   └── Dockerfile
├── scripts/
│   ├── score_pr.py             # Gate 2: PR commit risk scorer
│   ├── check_data_leakage.py   # Train/test overlap check
│   └── export_sample.py        # Export reference data sample
├── smoke_tests/                # Gate 3: Post-deploy smoke tests
│   ├── test_schema.py
│   ├── test_sanity.py
│   ├── test_latency.py
│   └── test_drift.py
├── tests/                      # Unit tests
├── models/
│   └── gatekeeper_risk_model.skops  # Standalone model (committed)
├── .github/workflows/
│   ├── gate2_pr_risk.yml       # Gate 2: PR risk scoring
│   └── gate3_post_deploy.yml   # Gate 3: Post-deploy smoke tests
├── .pre-commit-config.yaml     # Pre-push hooks
└── data/                       # Training data (gitignored, DVC-tracked)
```
