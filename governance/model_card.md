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
| Total commits | 5,896 |
| Features | 9 (lines_added, lines_deleted, files_touched, dirs_touched, author_prior_commits, hour_of_day, day_of_week, commit_msg_length, is_fix_bug_revert) |
| Label definition | 1 (risky) if: commit message contains "revert", OR any of its files were touched again by another commit within 7 days |
| Class balance | 51.4% risky (3,033), 48.6% safe (2,863) |

### Per-Repo Breakdown

| Repo | Language | Commits | Risky % |
|------|----------|---------|--------|
| django/django | Python | 2,038 | 45.2% |
| facebook/react | JavaScript | 2,358 | 66.0% |
| rust-lang/rust | Rust | 500 | 25.2% |
| kubernetes/kubernetes | Go | 500 | 26.8% |
| apache/kafka | Java | 500 | 59.2% |

Note: Rust, Kubernetes, and Kafka were capped at 500 commits each due to the O(n²) labeling cost on large monorepos. Django and React used larger samples (2,000+).

## Model Architecture

| Property | Value |
|----------|-------|
| Type | RandomForestClassifier (scikit-learn) |
| Hyperparameters | n_estimators=100, random_state=42 |
| Selection | Best among LightGBM, RandomForest, LogisticRegression via AutoML comparison on F1 |
| Serialized as | skops (models/gatekeeper_risk_model.skops) |

## Performance Metrics

Evaluated on 20% held-out test set (stratified, random_state=42):

| Metric | Value |
|--------|-------|
| Accuracy | 0.6822 |
| Precision | 0.6859 |
| Recall | 0.7051 |
| F1 | 0.6954 |

### Leave-One-Repo-Out Generalization

To test whether the model generalizes beyond its training repos, we trained on 4 repos and tested on the held-out 5th:

| Held-out Repo | Accuracy | Precision | Recall | F1 |
|---------------|----------|-----------|--------|----|
| django | 0.621 | 0.568 | 0.678 | 0.618 |
| react | 0.607 | 0.734 | 0.634 | 0.681 |
| rust | 0.722 | 0.446 | 0.429 | 0.437 |
| kubernetes | 0.748 | 0.547 | 0.351 | 0.427 |
| kafka | 0.706 | 0.788 | 0.689 | 0.735 |

**Interpretation:** The model generalizes well to repositories with similar development patterns (django, react, kafka all >0.6 F1). Performance drops for Rust and Kubernetes — repositories with fundamentally different commit conventions (Rust's bors merge commits, Kubernetes' bot-heavy workflow). The per-author-history feature likely contributes to this gap: the model learned Django/React contributor patterns that don't transfer to Rust's contributor model.

### Benchmark: Neural Network Comparison

A small feedforward Keras NN (64→32→1 with dropout) was trained on the same data as a benchmark. Gradient boosting / tree ensembles consistently outperform neural networks on this type of tabular data with categorical-like features, confirming the choice of tree-based models.

## Fairness Analysis

**Sensitive attribute:** Author experience (new: <5 prior commits, experienced: ≥5)

A Fairlearn-based check was run on the test set to evaluate whether the model is systematically harsher on new contributors. Results are saved in `governance/fairness_results.json`.

**Known finding:** The model may assign higher risk scores to commits from new contributors because the `author_prior_commits` feature directly encodes experience. A new contributor with 0 prior commits has no history of reliable contributions, so the model treats this as a risk factor. This is a **design decision**, not necessarily bias — but it means new contributors' commits will tend to score higher, which could feel punitive if not communicated clearly.

## Known Limitations

1. **Noisy labels:** The 7-day re-touch label is somewhat noisy. A commit that scores right at the medium/low boundary (risk_score ≈ 0.50) may have been labeled "risky" simply because a follow-up bugfix touched the same files — not because the original commit was inherently dangerous. This is an acknowledged source of label noise.

2. **Single-repo training:** The model is trained exclusively on django/django commits. It may not generalize well to repositories with different commit conventions, team sizes, or development workflows.

3. **Temporal drift:** Code review practices, CI/CD tooling, and contributor behavior evolve over time. The model was trained on 3 years of data (2023-2026) and may not capture recent shifts.

4. **Feature limitations:** The model uses only commit-level metadata (diff size, timing, author history, message keywords). It does not analyze code content, test coverage, or review quality.

5. **Class imbalance handling:** The model does not explicitly address class imbalance (45/55 split). More sophisticated approaches (SMOTE, class weights) could improve recall for the risky class.

6. **Binary risk framing:** The model outputs a continuous probability but the gate logic uses hard thresholds (<0.3 low, 0.3-0.6 medium, >0.6 high). Commits near these boundaries are inherently uncertain.

7. **Train/test overfitting:** RandomForest (max_depth=None) shows more extreme scores on training data (51.7% labeled "low") than test data (27.4% labeled "low") — classic unpruned-tree overfitting. The 100 unpruned decision trees memorize training patterns, producing confident extreme scores on seen data, but revert to more moderate predictions on unseen data. Real-world confidence is likely closer to test-set behavior. **Documented future improvement:** set max_depth=10 or increase min_samples_leaf to reduce overfitting without retraining.

8. **Threshold-recalibration gap:** The risk thresholds (0.3/0.6) were set for LightGBM and never recalibrated when Phase 7 promoted a RandomForest instead. Currently still produces a reasonable 27/40/32 score distribution split, but nothing guarantees this holds for future model swaps — the pipeline selects by F1 but doesn't validate or recalibrate downstream thresholds. **Documented MLOps gap:** register_model.py could compute new threshold boundaries from the new model's score percentiles on each promotion, rather than using fixed global values. This is a natural next-iteration improvement.

## Version History

| Version | Date | Model Type | F1 | Training Data | Notes |
|---------|------|------------|-----|---------------|-------|
| v1-v3 | 2026-08-09 | LightGBM | 0.6588 | Django only | Initial training |
| v4 (current) | 2026-08-18 | LGBMClassifier | 0.6954 | 5 repos (5,896 commits) | Multi-repo retraining, promoted to Production |

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
