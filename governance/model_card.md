# Gatekeeper Risk Prediction Model — Model Card

## Model Purpose

Gatekeeper predicts whether a git commit is "risky" — likely to be reverted or have its files touched again within 7 days. This is used as a quality/safety gate in CI/CD pipelines:

- **Gate 1 (Pre-push):** Blocks pushes containing high-risk commits
- **Gate 2 (Pre-merge):** Scores PR commits and posts risk comments
- **Gate 3 (Post-deploy):** Smoke tests validate the model in production

## Training Data

| Property | Value |
|----------|-------|
| Source | 5 repositories across 5 languages: django/django (Python), facebook/react (JavaScript), rust-lang/rust (Rust), kubernetes/kubernetes (Go), apache/kafka (Java) |
| Mining tool | PyDriller |
| Date range | 2024-08-18 to 2026-08-18 (~2 years) |
| Total commits | 7,896 |
| Features | 9 (lines_added, lines_deleted, files_touched, dirs_touched, author_prior_commits, hour_of_day, day_of_week, commit_msg_length, is_fix_bug_revert) |
| Label definition | 1 (risky) if: commit message contains "revert", OR any of its files were touched again by another commit within 7 days |
| Class balance | 48.3% risky (3,815), 51.7% safe (4,081) |

### Per-Repo Breakdown

| Repo | Language | Commits | Risky % |
|------|----------|---------|--------|
| django/django | Python | 2,038 | 45.2% |
| facebook/react | JavaScript | 2,358 | 66.0% |
| rust-lang/rust | Rust | 1,500 | 39.3% |
| kubernetes/kubernetes | Go | 1,500 | 30.1% |
| apache/kafka | Java | 500 | 59.2% |

Note: Rust and Kubernetes were expanded to 1,500 commits each (from an initial 500) after a labeling-window bug fix — earlier runs mined only the oldest commits, missing the 7-day forward-look needed for correct labeling. A 500-commit buffer is now mined beyond the training cap to ensure every row has complete label context.

## Model Architecture

| Property | Value |
|----------|-------|
| Type | LGBMClassifier (LightGBM) |
| Hyperparameters | num_leaves=31, learning_rate=0.05, n_estimators=100 |
| Selection | Best among LightGBM, RandomForest, LogisticRegression via AutoML comparison on F1 |
| Serialized as | skops (models/gatekeeper_risk_model.skops) |

## Performance Metrics

Evaluated on 20% held-out test set (stratified, random_state=42):

| Metric | Value |
|--------|-------|
| Accuracy | 0.7127 |
| Precision | 0.6988 |
| Recall | 0.7117 |
| F1 | 0.7052 |

### Leave-One-Repo-Out Generalization

To test whether the model generalizes beyond its training repos, we trained on 4 repos and tested on the held-out 5th:

| Held-out Repo | Accuracy | Precision | Recall | F1 |
|---------------|----------|-----------|--------|----|
| django | 0.6021 | 0.5433 | 0.7492 | 0.6298 |
| react | 0.5992 | 0.7235 | 0.6356 | 0.6767 |
| rust | 0.7053 | 0.6529 | 0.5356 | 0.5885 |
| kubernetes | 0.7593 | 0.6051 | 0.5796 | 0.5921 |
| kafka | 0.6860 | 0.7473 | 0.7095 | 0.7279 |
| **Average** | | | | **0.6430** |

**Interpretation:** With the corrected labeling and 3x more Rust/K8s data, all repos now achieve >0.58 F1. The model generalizes reasonably across Python, JavaScript, and Java (all >0.62 F1). Rust and Kubernetes remain the weakest points (~0.59 F1) — their commit patterns (Rust's bors merge automation, Kubernetes' bot-heavy workflow with large batch merges) differ structurally from the application-layer repos. When held out, the model defaults to predicting "low risk" for 40-49% of Rust/K8s commits (vs 4% for Django), suggesting it hasn't fully learned systems-language risk signatures. This is a genuine domain gap documented as a known limitation.

### Benchmark: Neural Network Comparison

A small feedforward Keras NN (64→32→1 with dropout) was trained on the same data as a benchmark. Gradient boosting / tree ensembles consistently outperform neural networks on this type of tabular data with categorical-like features, confirming the choice of tree-based models.

## Fairness Analysis

**Sensitive attribute:** Author experience (new: <5 prior commits, experienced: ≥5)

A Fairlearn-based check was run on the test set to evaluate whether the model is systematically harsher on new contributors. Results are saved in `governance/fairness_results.json`.

**Known finding:** The model may assign higher risk scores to commits from new contributors because the `author_prior_commits` feature directly encodes experience. A new contributor with 0 prior commits has no history of reliable contributions, so the model treats this as a risk factor. This is a **design decision**, not necessarily bias — but it means new contributors' commits will tend to score higher, which could feel punitive if not communicated clearly.

## Known Limitations

1. **Noisy labels:** The 7-day re-touch label is somewhat noisy. A commit that scores right at the medium/low boundary (risk_score ≈ 0.50) may have been labeled "risky" simply because a follow-up bugfix touched the same files — not because the original commit was inherently dangerous. This is an acknowledged source of label noise.

2. **Partial generalization:** The model generalizes well across application-layer repos (Python, JavaScript, Java) but shows weaker performance on systems/infrastructure languages (Rust F1=0.589, Kubernetes F1=0.592). Systems-language commits follow different conventions (merge automation, bot workflows, large batch merges) that the current feature set doesn't fully capture.

3. **Temporal drift:** Code review practices, CI/CD tooling, and contributor behavior evolve over time. The model was trained on 3 years of data (2023-2026) and may not capture recent shifts.

4. **Feature limitations:** The model uses only commit-level metadata (diff size, timing, author history, message keywords). It does not analyze code content, test coverage, or review quality.

5. **Class imbalance handling:** The model does not explicitly address class imbalance (45/55 split). More sophisticated approaches (SMOTE, class weights) could improve recall for the risky class.

6. **Binary risk framing:** The model outputs a continuous probability but the gate logic uses hard thresholds (<0.3 low, 0.3-0.6 medium, >0.6 high). Commits near these boundaries are inherently uncertain.

7. **Threshold-recalibration gap:** The risk thresholds (0.3/0.6) are fixed global values. When the model is swapped (e.g., LightGBM → RandomForest → LightGBM), the score distribution shifts and the thresholds may no longer produce the intended 27/40/32 split. **Documented MLOps gap:** register_model.py could compute new threshold boundaries from the new model's score percentiles on each promotion, rather than using fixed global values.

8. **Shallow clone constraint:** CI environments (GitHub Actions for Gates 2, 3) use shallow-cloned repos, so `author_prior_commits` is relative to the shallow clone depth rather than the repo's full history. This limits the feature's accuracy in production scoring.

## Version History

| Version | Date | Model Type | F1 | Training Data | Notes |
|---------|------|------------|-----|---------------|-------|
| v1-v3 | 2026-08-09 | LightGBM | 0.6588 | Django only | Initial training |
| v5 (current) | 2026-08-24 | LGBMClassifier | 0.7052 | 5 repos (7,896 commits) | Buffer labeling fix, expanded Rust/K8s to 1,500 each |
| v4 | 2026-08-18 | LGBMClassifier | 0.6954 | 5 repos (5,896 commits) | Multi-repo retraining (had labeling bug) |

## Cloud Deployment

The API is deployed as a standalone Docker container, making it portable across cloud providers with **zero code changes** — only the deployment target configuration differs.

| Provider | Service | Free Tier | Notes |
|----------|---------|-----------|-------|
| **Render** | Web Service | ✅ Yes (no card) | Current target. Blueprint via `render.yaml`. |
| AWS | ECS Fargate / App Runner | 12-month free tier | Requires AWS account + billing setup |
| Azure | Container Apps | Free tier available | Requires Azure account |
| GCP | Cloud Run | 2M requests/month free | Requires GCP account |

**Why Render:** Chosen specifically for zero-card free hosting suitable for a student project. The Docker-based deployment is identical across all targets — `render.yaml` is the only provider-specific file.

**Model loading in production:** The deployed API does not have a local MLflow database. On startup, it falls back to loading `models/gatekeeper_risk_model.skops` directly (Strategy 3 in `api/main.py`). This standalone file is committed to git and baked into the Docker image (~5MB).

See [DEPLOY.md](../DEPLOY.md) for step-by-step deployment instructions.

## Maintenance

- **Retraining:** Run `python pipelines/run_retrain.py` weekly via GitHub Actions
- **Monitoring:** Prometheus/Grafana dashboards track API latency, error rates, and request volume
- **Drift detection:** Evidently-based drift reports compare recent vs. historical feature distributions
- **Model registry:** MLflow Model Registry tracks versions; only models that improve F1 over Production are promoted
