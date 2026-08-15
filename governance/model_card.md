# Gatekeeper Risk Prediction Model — Model Card

## Model Purpose

Gatekeeper predicts whether a git commit is "risky" — likely to be reverted or have its files touched again within 7 days. This is used as a quality/safety gate in CI/CD pipelines:

- **Gate 1 (Pre-push):** Blocks pushes containing high-risk commits
- **Gate 2 (Pre-merge):** Scores PR commits and posts risk comments
- **Gate 3 (Post-deploy):** Smoke tests validate the model in production

## Training Data

| Property | Value |
|----------|-------|
| Source | django/django GitHub repository |
| Mining tool | PyDriller |
| Date range | 2023-08-09 to 2026-08-13 (~3 years) |
| Total commits | 2,953 |
| Features | 9 (lines_added, lines_deleted, files_touched, dirs_touched, author_prior_commits, hour_of_day, day_of_week, commit_msg_length, is_fix_bug_revert) |
| Label definition | 1 (risky) if: commit message contains "revert", OR any of its files were touched again by another commit within 7 days |
| Class balance | 45.48% risky (1,343), 54.52% safe (1,610) |

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
| Accuracy | 0.7107 |
| Precision | 0.7042 |
| Recall | 0.6283 |
| F1 | 0.6640 |

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

| Version | Date | Model Type | F1 | Notes |
|---------|------|------------|-----|-------|
| v1-v3 | 2026-08-09 | LightGBM | 0.6588 | Initial training |
| v4 | 2026-08-13 | RandomForest | 0.6814 | AutoML comparison, promoted to Production |
| v2 (current) | 2026-08-14 | RandomForest | 0.6640 | Re-registered after mlflow.db incident, same architecture |

## Maintenance

- **Retraining:** Run `python pipelines/run_retrain.py` weekly via GitHub Actions
- **Monitoring:** Prometheus/Grafana dashboards track API latency, error rates, and request volume
- **Drift detection:** Evidently-based drift reports compare recent vs. historical feature distributions
- **Model registry:** MLflow Model Registry tracks versions; only models that improve F1 over Production are promoted
