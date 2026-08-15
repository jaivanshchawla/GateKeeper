# Deploying Gatekeeper API to Render

This guide deploys **only the FastAPI API service** to Render's free tier.
No credit card required for standard web services.

The dashboard/webhook/Postgres stack remains local (docker-compose).

## Prerequisites

- A GitHub account
- The GateKeeper repo cloned locally

## Step-by-Step Deployment

### 1. Sign up / Log in to Render

1. Go to [render.com](https://render.com)
2. Click **Get Started for Free**
3. Choose **Sign up with GitHub** — this is the easiest path (no email/password needed)
4. Authorize Render to access your GitHub repositories

### 2. Create a New Blueprint

1. From the Render dashboard, click **New +** (top right)
2. Select **Blueprint**
3. Connect your GitHub account if prompted
4. Select the **GateKeeper** repository
5. Render will detect `render.yaml` automatically — confirm it shows the `gatekeeper-api` service
6. Click **Apply** to start the deployment

### 3. Wait for the Build

- Render will clone the repo, build the Docker image (installs Python deps + scikit-learn + skops), and start the API
- First build takes 3-5 minutes on free tier
- Once deployed, you'll see a green "Live" badge

### 4. Get Your Live URL

1. Click on the `gatekeeper-api` service in the dashboard
2. Your URL is shown near the top (format: `https://gatekeeper-api-xxxx.onrender.com`)
3. Test it:
   ```bash
   curl https://gatekeeper-api-xxxx.onrender.com/health
   # Should return: {"status":"healthy","model_loaded":true}
   ```

### 5. Activate the Keep-Warm Workflow

Render free tier puts services to sleep after 15 minutes of inactivity.
A GitHub Actions workflow pings the API every 10 minutes to keep it alive.

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `DEPLOYED_API_URL`
4. Value: your Render URL **without** the trailing slash
   (e.g. `https://gatekeeper-api-xxxx.onrender.com`)
5. Click **Add secret**

The `keep_warm.yml` workflow will start pinging on the next scheduled run.

### 6. Verify

```bash
# Health check
curl https://gatekeeper-api-xxxx.onrender.com/health

# Prediction test
curl -X POST https://gatekeeper-api-xxxx.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"lines_added":50,"lines_deleted":10,"files_touched":3,"dirs_touched":2,"author_prior_commits":50,"hour_of_day":14,"day_of_week":1,"commit_msg_length":45,"is_fix_bug_revert":0}}'
```

## How It Works (No MLflow in Production)

The deployed API does **not** need a local MLflow database. On startup:

1. **Strategy 1:** Tries to load from MLflow Model Registry → fails (no DB)
2. **Strategy 2:** Tries to find `mlruns/` artifacts → fails (no artifacts)
3. **Strategy 3:** Loads `models/gatekeeper_risk_model.skops` directly → **succeeds**

This standalone model file is committed to git (~5MB, same as a small image).
The Dockerfile copies it into the container image during build.

## Updating the Model

When a new model is promoted via the retraining pipeline:

1. `pipelines/components/register_model.py` updates the MLflow registry locally
2. `ml/export_model.py` re-exports `models/gatekeeper_risk_model.skops`
3. Commit and push the updated skops file
4. Render auto-redeploys from the new commit

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `MLFLOW_TRACKING_URI` | `sqlite:///nonexistent/mlflow.db` | Forces Strategy 3 (standalone file) |

## Limitations

- **Free tier sleep:** Service sleeps after 15 min inactivity. The keep-warm workflow mitigates this.
- **Cold start:** First request after sleep takes 30-60s (Python + sklearn import).
- **No GPU:** RandomForest is CPU-only; no GPU benefit anyway.
- **No persistent storage:** Each deploy gets a fresh filesystem. The model is baked into the image.
