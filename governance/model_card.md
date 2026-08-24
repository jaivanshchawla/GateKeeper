# Gatekeeper Risk Prediction Model — Model Card

## Model Purpose

Gatekeeper predicts whether a git commit is "risky" — likely to be reverted or have its files touched again within 7 days. It is a **ranking model**, not a binary classifier: it assigns a continuous risk score used to prioritize review attention, not to make hard accept/reject decisions.

- **Gate 1 (Pre-push):** Flags high-risk commits for human review
- **Gate 2 (Pre-merge):** Scores PR commits and posts risk comments
- **Gate 3 (Post-deploy):** Smoke tests validate the model in production

## Headline Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC-AUC | **0.674** | Cross-repo LORO mean — honest generalization |
| PR-AUC lift | +0.157 | Model ranks risky commits above base rate |
| Top-decile lift | ~1.5-2x | Top 10% of scores have 1.5-2x the precision of random |
| Brier score | 0.224 | Moderately well-calibrated |

**Why not F1:** The constant classifier (predict everything as risky) beats the model on F1 for 4/5 repos at these base rates:

| Repo | Positive rate p | Constant F1 = 2p/(1+p) | Model F1 | Winner |
|------|----------------|------------------------|----------|--------|
| django | 0.4765 | 0.6454 | 0.6519 | Model (barely) |
| react | 0.6725 | 0.8042 | 0.7368 | Constant |
| rust | 0.6940 | 0.8194 | 0.6828 | Constant |
| kubernetes | 0.5750 | 0.7302 | 0.7039 | Constant |
| kafka | 0.5585 | 0.7167 | 0.7106 | Constant |

F1 penalizes the model for not having perfect recall, which the constant classifier trivially achieves. ROC-AUC and PR-AUC lift are the honest metrics for a ranking model.

## Training Data

| Property | Value |
|----------|-------|
| Source | 5 repositories, 5 languages |
| Repos | django/django (Python), facebook/react (JS), rust-lang/rust (Rust), kubernetes/kubernetes (Go), apache/kafka (Java) |
| Window | 2024-07-01 to 2026-06-30 (24 months, identical for all repos) |
| Total commits mined | 10,000 (2,000 per repo, every-Nth sampling) |
| Features | 9 (see below) |
| Label | V1: any-file retouch within 7 days, or "revert" in commit message |
| Class balance | 59.5% risky, 40.5% safe |

### Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | lines_added | Lines of code added |
| 2 | lines_deleted | Lines of code deleted |
| 3 | files_touched | Number of files modified |
| 4 | dirs_touched | Number of directories touched |
| 5 | author_prior_commits | Author's commit count before the training window (seeded from full repo history) |
| 6 | hour_of_day | Hour of day (UTC, 0-23) |
| 7 | day_of_week | Day of week (0=Monday, 6=Sunday) |
| 8 | commit_msg_length | Commit message length in characters |
| 9 | is_fix_bug_revert | 1 if message contains fix/bug/revert keywords |

### Per-Repo Breakdown

| Repo | Language | Commits | Risky rate | Commits/week |
|------|----------|---------|------------|--------------|
| django | Python | 2,000 | 47.7% | 19.6 |
| react | JavaScript | 2,000 | 67.3% | 24.8 |
| rust | Rust | 2,000 | 69.4% | 399.5 |
| kubernetes | Go | 2,000 | 57.5% | 94.4 |
| kafka | Java | 2,000 | 55.9% | 44.2 |

## Model Architecture

| Property | Value |
|----------|-------|
| Type | LGBMClassifier (LightGBM) |
| Hyperparameters | num_leaves=31, learning_rate=0.05, n_estimators=100 |
| Serialized as | skops (models/gatekeeper_risk_model.skops) |
| MLflow registry | GatekeeperRiskPredictor v5 (Production) |

## Evaluation Protocol

The headline metric uses **cross-repo leave-one-repo-out (LORO)** evaluation: train on 4 repos, test on the held-out 5th. This is the only protocol that measures real generalization to unseen repos and languages.

| Held-out Repo | ROC-AUC | F1 |
|---------------|---------|-----|
| django | 0.684 | 0.652 |
| react | 0.659 | 0.737 |
| rust | 0.651 | 0.683 |
| kubernetes | 0.684 | 0.716 |
| kafka | 0.715 | 0.711 |
| **Mean** | **0.674** | **0.700** |

### Protocol Comparison

| Protocol | ROC-AUC | F1 | Interpretation |
|----------|---------|-----|----------------|
| Pooled random 80/20 | ~0.70 | ~0.72 | Overestimates (leakage from same-author/same-file overlap) |
| Purged time-ordered | ~0.68 | ~0.69 | More honest, temporal embargo reduces leakage |
| Cross-repo LORO | **0.674** | **0.700** | Honest: tests generalization to unseen repos |

The pooled-random number (0.72 F1) is higher because temporally adjacent commits from the same author land on both sides of the split, and `author_prior_commits` is a running counter. The cross-repo number (0.674 ROC-AUC) is the honest headline.

## Percentile-Based Thresholds

Absolute thresholds (0.3/0.6) failed because the score distribution shifts when the model changes and differs per repo. Instead, per-repo percentile bands are used:

| Repo | High (top 10%) | Medium (next 15%) | Low (bottom 75%) |
|------|----------------|-------------------|------------------|
| django | >= 0.8029 | >= 0.6841 | < 0.6841 |
| react | >= 0.8839 | >= 0.8042 | < 0.8042 |
| rust | >= 0.8632 | >= 0.7659 | < 0.7659 |
| kubernetes | >= 0.8543 | >= 0.7301 | < 0.7301 |
| kafka | >= 0.8752 | >= 0.7573 | < 0.7573 |
| _global (fallback) | >= 0.8619 | >= 0.7536 | < 0.7536 |

Cutoffs are persisted in `ml/config.yaml` and used by both `api/main.py` and `scripts/score_pr.py`. Unknown repos fall back to `_global`.

## Label Density and Repo Velocity

The label partly encodes **repo velocity** rather than commit quality alone. Pearson correlation between graph commits/week and risky rate: **+0.62**.

| Repo | Commits/week | Risky rate |
|------|-------------|------------|
| django | 19.6 | 47.7% |
| react | 24.8 | 67.3% |
| kafka | 44.2 | 55.9% |
| kubernetes | 94.4 | 57.5% |
| rust | 399.5 | 69.4% |

Denser repos have more file re-touches, so more commits get labeled risky. This is a structural property of the label, not a model bug.

## Leakage Analysis: Why V4 Was Rejected

A candidate label variant V4 ("fix-keyword retouch") showed ROC-AUC 0.861 — suspiciously high. Investigation revealed label leakage:

| Variant | ROC-AUC (with feature) | ROC-AUC (without is_fix_bug_revert) | Delta |
|---------|----------------------|-------------------------------------|-------|
| V1 (any retouch) | 0.676 | 0.674 | -0.002 |
| V4 (fix retouch) | 0.861 | 0.568 | **-0.293** |

V4 labels a commit risky if a later fix/bug/revert commit re-touches its files. The feature `is_fix_bug_revert` uses the same regex on this commit's own message. Fix commits cluster, so the feature partly reads the label. V1's label doesn't share the regex, so removing the feature barely moves the needle.

**Additional V4 flaws:**
- **Subset violation:** V4=1 AND V1=0 exists for all repos (530 django, 177 react, 121 rust, 157 k8s, 175 kafka). Code bug: `risky = set(fix_hashes)` adds ALL fix commits unconditionally, not just when re-touched.
- **Perfect separation artifact:** Django top-10% showed 100% precision, but this was due to distribution shift (training 25% positive, testing 65%) causing the model to predict all-negative.

**Lesson:** A high ROC-AUC with a feature that shares its regex with the label definition is leakage until proven otherwise.

## Merge-Commit Trap

The labeling graph was built from `git log --name-only --no-merges`, which excludes merge commits entirely. A later fix attempted `--numstat` to include merges, but **100% of sampled Rust bors auto-merges have 0 files in `git log --numstat`** (verified on 200 samples). The same commits show files in `git show --numstat -m` (the merge diff), but `git log --numstat` reports the trivial diff (0 files).

**Impact:** Merge commits' file touches are not captured in the labeling graph. For Rust (15.9% merges) and Kubernetes (36.7% merges), this means some file re-touches by merge commits are missed, slightly deflating positive rates. Known limitation, not fixed.

## Provenance History

The dataset has been rebuilt multiple times as bugs were found:

1. **Original (Phases 1-8):** Django-only, `since=2023-08-09`, capped at 2,948 commits. Carried a tail-truncation labeling bug (commits near the end of the mined window couldn't be observed being re-touched).
2. **First multi-repo rebuild:** 5 repos, `since=2024-08-15`, mixed shallow/full clones. Rust/K8s had shallow clones that contradicted the script's "no --depth" claim. The `since` date was silently changed from the original.
3. **B2 rebuild:** Full clones, fixed window [2024-07-01, 2026-06-30), unified extractor. But PyDriller was too slow for large repos; fell back to git-log feature extraction with a different code path.
4. **Current (Part I):** Full clones, git-log labeling graph, PyDriller bulk feature extraction, every-Nth sampling. 10,000 rows (2,000 per repo).

All pre-rebuild numbers (Phases 1-8) are superseded by the current dataset.

## Known Limitations

1. **Modest discriminative power:** ROC-AUC 0.674 means the model ranks risky commits only moderately better than random. It is not a reliable standalone decision-maker — use it as one signal among several.

2. **Label encodes repo velocity:** +0.62 correlation between commits/week and risky rate. The label partly measures "how active is this repo" rather than "how risky is this commit."

3. **Merge-commit blind spot:** The labeling graph misses file touches by merge commits (100% of Rust bors merges have 0 files in `git log --numstat`). Positive rates for Rust/K8s are slightly deflated.

4. **author_prior_commits train/serve skew:** Training seeds the counter from full repo history; single-commit extraction (Gate 2) has no window context and returns 0 for all authors. Dropping the feature actually improves F1 by +0.013.

5. **No code understanding:** The model uses only commit metadata (diff size, timing, author history, message keywords). It does not analyze code content, test coverage, or review quality.

6. **Calibration gaps:** 10-bin reliability analysis shows overconfidence in mid-range bins (predicted probabilities systematically higher than observed frequencies for rust, lower for react).

7. **Temporal drift:** Code review practices, CI/CD tooling, and contributor behavior evolve. The model was trained on a 2-year window and may not capture recent shifts.

8. **Line-level revert label (V8) rejected:** Evaluated tracking whether specific lines introduced by a commit are later modified by fix/bug/revert commits within 7 days. Near-zero positive rate (0-0.5% across all repos) — the intersection of exact line content matching and fix-commit overlap is vanishingly rare in practice. Commit metadata features remain the ceiling for what can be extracted without LLM-based code understanding.

## Version History

| Version | Model Type | ROC-AUC | Training Data | Notes |
|---------|------------|---------|---------------|-------|
| v1 | LGBMClassifier | ? | Django only | Initial (orphaned params) |
| v2 | RandomForestClassifier | ? | Django only | Archived |
| v4 | LGBMClassifier | ? | 5 repos (pre-rebuild) | Archived, had labeling bugs |
| **v5 (current)** | **LGBMClassifier** | **0.674** | **5 repos, 10K commits** | **Current Production** |

## Cloud Deployment

The API is deployed as a Docker container (1.31GB slim image) portable across cloud providers with zero code changes.

| Provider | Service | Free Tier |
|----------|---------|-----------|
| **Render** | Web Service | Yes (no card) |
| AWS | ECS Fargate / App Runner | 12-month free tier |
| Azure | Container Apps | Free tier |
| GCP | Cloud Run | 2M req/month free |

See [DEPLOY.md](../DEPLOY.md) for deployment instructions.

## Maintenance

- **Retraining:** `python pipelines/run_retrain.py` (weekly via GitHub Actions)
- **Promotion gate:** Compares cross-repo LORO ROC-AUC; only promotes if new model beats Production
- **Monitoring:** Prometheus/Grafana dashboards (request rate, latency, error rate)
- **Drift detection:** Evidently reports comparing recent vs. training feature distributions
- **Backup:** mlflow.db is backed up before every registry write
