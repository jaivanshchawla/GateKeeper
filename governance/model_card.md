# Gatekeeper Risk Prediction Model — Model Card

## Model Purpose

Gatekeeper predicts whether a git commit is "risky" — likely to be reverted or have its files touched again within 7 days. It is a **ranking model**, not a binary classifier: it assigns a continuous risk score used to prioritize review attention, not to make hard accept/reject decisions.

- **Gate 1 (Pre-push):** Flags high-risk commits for human review
- **Gate 2 (Pre-merge):** Scores PR commits and posts risk comments
- **Gate 3 (Post-deploy):** Smoke tests validate the model in production

## Headline Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC-AUC (cross-repo LORO) | **0.7885** | Generalization to unseen repos within the training time window |
| ROC-AUC (out-of-window, excl React) | **0.7563** | Production figure for 4/5 repos — React divergence excluded |
| PR-AUC lift | +0.246 | Model ranks risky commits above base rate |
| Top-decile lift | ~1.5-2x | Top 10% of scores have 1.5-2x the precision of random |
| Brier score | 0.224 | Moderately well-calibrated |

**Two numbers, not one:** The cross-repo LORO (0.7885) tests generalization to unseen repos within the same time window. The out-of-window ROC-AUC tests generalization to future commits. Excluding React, the OOW mean is 0.7563 (gap -0.034, within noise). React diverged due to structural contributor-base changes, not temporal drift.

**Pooled AUC caveat:** The pooled ROC-AUC (~0.80) is higher than the per-repo mean (0.7885) because between-repo separation inflates the number. The per-repo table below is the honest presentation. See Protocol Comparison for details.

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
| Features | 35 (see below) |
| Label | V1: any-file retouch within 7 days, or "revert" in commit message |
| Class balance | 59.5% risky, 40.5% safe |

### Features (35 total)

**Base features (9):** lines_added, lines_deleted, files_touched, dirs_touched, author_prior_commits, hour_of_day, day_of_week, commit_msg_length, is_fix_bug_revert

**File history (12):** file_prior_changes_max/mean, file_prior_risky_max/mean, file_revert_count_max/mean, file_age_days_max/mean, file_authors_count_max/mean, days_since_last_change_max/mean

**Author-file familiarity (6):** author_file_prior_commits_max/mean, author_dir_prior_commits_max/mean, is_author_first_touch_dir, author_days_since_last_commit

**Change-shape (8):** churn_ratio, change_entropy, max_file_churn, is_test_only, test_to_code_ratio, config_touch, is_merge, files_per_dir_ratio

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
| Features | 35 (9 base + 12 file history + 6 author-file + 8 change-shape) |
| Serialized as | skops (models/gatekeeper_risk_model.skops) |
| MLflow registry | GatekeeperRiskPredictor v8 (Production) |
| Rule engine | 9 deterministic rules in `.gatekeeper.yml` (separate from ML score) |

## Evaluation Protocol

The headline metric uses **cross-repo leave-one-repo-out (LORO)** evaluation: train on 4 repos, test on the held-out 5th. This is the only protocol that measures real generalization to unseen repos and languages.

| Held-out Repo | ROC-AUC | 95% CI | Notes |
|---------------|---------|--------|-------|
| django | 0.7607 | [0.7372, 0.7818] | |
| kafka | 0.8247 | [0.8037, 0.8404] | |
| kubernetes | 0.7952 | [0.7742, 0.8139] | |
| react | 0.7579 | [0.7361, 0.7789] | |
| rust | 0.8038 | [0.7833, 0.8223] | |
| **Mean** | **0.7885** | | **v8 Production — parity-verified extraction** |

### Out-of-Window Evaluation (Y.1)

The most honest metric: test on commits whose committer_date is AFTER the training window end (2026-06-30). These commits were never seen during training or evaluation.

| Repo | N (OOW) | Base Rate | OOW ROC-AUC | 95% CI | Offline LORO | Gap |
|------|---------|-----------|------------|--------|-------------|-----|
| django | 145 | 39.3% | 0.7547 | [0.6715, 0.8313] | 0.7607 | -0.006 |
| kafka | 100 | 36.0% | 0.7363 | [0.6823, 0.7874] | 0.8247 | -0.088 |
| kubernetes | 100 | 49.0% | 0.8812 | [0.8319, 0.9240] | 0.7952 | +0.086 |
| rust | 390 | 51.3% | 0.7237 | [0.6725, 0.7701] | 0.8038 | -0.080 |
| react | 102 | 52.0% | **0.5542** | [0.4300, 0.6740] | 0.7579 | **-0.204** |
| **Mean** | | | **0.7300** | | 0.7885 | **-0.059** |

**Key findings:**
- **Mean OOW ROC-AUC is 0.730 vs 0.7885 LORO** — a 0.059 gap from temporal drift alone.
- **React is broken out-of-window** (0.5542, CI includes 0.5 = no signal). React's codebase and contributor patterns shifted enough post-training that the model cannot generalize.
- **Kubernetes OOW exceeds LORO** (0.8812 vs 0.7952) — likely because k8s's high commit density makes the ranking task easier on recent commits.
- **Django has too few OOW commits** (145, CI width ±0.08) for precise measurement.

Excluding React (where the contributor base shifted structurally), the OOW gap is -0.034 — within noise. The model generalizes to future commits for 4/5 repos. Right-censoring at HEAD was ruled out as a cause (Z.1).

### Protocol Comparison

| Protocol | ROC-AUC | Notes |
|----------|---------|-------|
| Pooled random 80/20 | ~0.80 | **Inflated** — see below |
| Cross-repo LORO | **0.7885** | Generalization to unseen repos, same time window |
| Out-of-window | **0.7300** | Generalization to future commits — what users experience |

**Why pooled AUC inflates:** Pooling predictions across repos with different score distributions counts between-repo separation as within-repo discrimination. The per-repo AUCs range from 0.738 to 0.810, but the pooled number (0.80) sits above three of five repos. The honest headline is the per-repo table and its mean, not the pooled figure. Additionally, temporally adjacent commits from the same author land on both sides of a random split, and `author_prior_commits` is a running counter — creating leakage that cross-repo LORO avoids entirely.

## Percentile-Based Thresholds

Absolute thresholds (0.3/0.6) failed because the score distribution shifts when the model changes and differs per repo. Instead, per-repo percentile bands are used:

| Repo | High Risk (top 10%) | Elevated (next 15%) | Not Flagged (bottom 75%) |
|------|---------------------|---------------------|-------------------------|
| django | >= 0.8029 | >= 0.6841 | < 0.6841 |
| react | >= 0.8839 | >= 0.8042 | < 0.8042 |
| rust | >= 0.8632 | >= 0.7659 | < 0.7659 |
| kubernetes | >= 0.8543 | >= 0.7301 | < 0.7301 |
| kafka | >= 0.8752 | >= 0.7573 | < 0.7573 |
| _global (fallback) | >= 0.8619 | >= 0.7536 | < 0.7536 |

Cutoffs are persisted in `ml/config.yaml` and used by both `api/main.py` and `scripts/score_pr.py`. Unknown repos fall back to `_global`.

### Band Semantics (X.1 Out-of-Window Backfill)

The band names reflect what the model actually measures on **unseen commits** (after training window end 2026-06-30). The original W.2 backfill had 75.5% Django overlap with training data, inflating precision to 100%. The numbers below are the honest production figures.

| Repo | N | Base Rate | High Prec | High Lift | Med Prec | Med Lift |
|------|---|-----------|-----------|-----------|----------|----------|
| kafka | 100 | 36.0% | 80.0% [37.6%, 96.4%] | 2.22x | 63.6% [35.4%, 84.8%] | 1.77x |
| kubernetes | 100 | 49.0% | 89.5% [68.6%, 97.1%] | 1.83x | 57.1% [25.0%, 84.2%] | 1.17x |
| rust | 100 | 52.0% | 76.0% [56.6%, 88.5%] | 1.46x | 37.5% [13.7%, 69.4%] | 0.72x |
| django | 145 | 39.3% | 66.7% [20.8%, 93.9%] | 1.70x | 57.1% [25.0%, 84.2%] | 1.45x |
| react | 102 | 52.0% | 33.3% [6.1%, 79.2%] | 0.64x | 37.5% [13.7%, 69.4%] | 0.72x |

**Key findings:**
- **Kafka and Kubernetes are strong**: high-band precision 80-90%, genuine lift.
- **Rust works**: high-band 76%, CI includes 56.6%-88.5%.
- **Django has too few out-of-window commits** (145 total, only 3 in high band) for reliable statistics.
- **React is broken out-of-window**: high-band 33.3% is below the 52% base rate, meaning the model is actively wrong on React commits it hasn't seen.

"Not Flagged" means the commit is in the bottom 75% of the score distribution — the model is a **ranking signal**, not a binary classifier. The PR comment footer includes this caveat.

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

## Fairness Check

Using Fairlearn on the 35-feature model, comparing authors with <5 prior commits ("new") vs >=5 ("experienced"):

| Group | Count | Predicted-positive rate |
|-------|-------|------------------------|
| New (<5 commits) | 1,526 | 41.2% |
| Experienced (>=5) | 8,474 | 64.9% |

**Demographic parity difference: 0.2369** (experienced authors flagged risky at 1.6x the rate of new authors).

This is expected and defensible: the model uses `author_prior_commits` as a feature, and experienced authors have higher file-author overlap, making their commits more likely to be re-touched. The feature is not measuring author identity but author-file familiarity — a genuine signal. However, it does mean the model systematically assigns lower risk to new contributors, which could mask risky first-time commits.

Per-repo parity differences range from 0.119 (django) to 0.357 (react).

### Equalized Odds (per-repo)

| Repo | New TPR | Exp TPR | New FPR | Exp FPR | Amplification |
|------|---------|---------|---------|---------|---------------|
| django | 70.4% | 77.0% | 11.0% | 19.3% | +4.4% |
| kafka | 72.6% | 85.8% | 13.6% | 27.6% | +8.6% |
| kubernetes | 68.8% | 83.3% | 16.8% | 27.8% | +8.7% |
| react | 74.5% | 90.8% | 26.3% | 46.2% | -2.1% |
| rust | 75.0% | 89.1% | 22.6% | 34.9% | +10.5% |

The model AMPLIFIES actual disparity in 4/5 repos (experienced contributors get higher false-positive rates). `is_author_first_touch_file` was dropped in O.2 to eliminate the double-penalization of new contributors via model feature + rule.

## Known Limitations

1. **Modest discriminative power:** ROC-AUC 0.7885 means the model ranks risky commits moderately better than random. It is not a reliable standalone decision-maker — use it as one signal among several.

2. **Label encodes repo velocity:** +0.62 correlation between commits/week and risky rate. The label partly measures "how active is this repo" rather than "how risky is this commit."

3. **Merge-commit blind spot:** The labeling graph misses file touches by merge commits (100% of Rust bors merges have 0 files in `git log --numstat`). Positive rates for Rust/K8s are slightly deflated.

4. **author_prior_commits residual skew (T.1):** Author identity is now keyed on normalized email (%aE + NFKD + casefold), fixing the display-name splitting bug ("Esteban Küber" vs "Esteban Kuber", "Lauren Tan" vs "lauren"). PyDriller and git log still occasionally report different emails for the same commit (e.g. `tg@trevorgross.com` vs `tmgross@umich.edu`); the SC path overrides with the graph email. Feature parity: 34/35 features at 0/50, author_prior_commits off-by-1-2 at timestamp edges (0.8% of cells).

5. **No code understanding:** The model uses only commit metadata (diff size, timing, author history, message keywords). It does not analyze code content, test coverage, or review quality.

6. **Calibration gaps:** 10-bin reliability analysis shows overconfidence in mid-range bins (predicted probabilities systematically higher than observed frequencies for rust, lower for react).

7. **React-specific divergence, not general temporal drift (Z.1-Z.5):** The mean out-of-window ROC-AUC is 0.730 vs 0.7885 LORO. However, Z.1 proved this is NOT caused by right-censoring at HEAD, and **excluding React, the gap is -0.034 (within noise)**. React is the sole driver: its contributor base shifted structurally post-training (activity dropped from 129/month to 35/month, top-3 authors do 60% of commits, same author has dual emails causing feature inconsistency). K8s actually improved OOW (0.8812 vs 0.7952 LORO). Z.5 confirmed more training data helps (-0.10 AUC when using less), so the path to fixing React is retraining with OOW data — not urgent for the other 4 repos.

8. **Pooled AUC inflates:** Pooling predictions across repos with different score distributions counts between-repo separation as within-repo discrimination. The pooled ROC-AUC (~0.80) sits above three of five per-repo AUCs. The per-repo table is the honest presentation.

9. **Line-level revert label (V8) rejected:** Evaluated tracking whether specific lines introduced by a commit are later modified by fix/bug/revert commits within 7 days. Near-zero positive rate (0-0.5% across all repos) — the intersection of exact line content matching and fix-commit overlap is vanishingly rare in practice. Commit metadata features remain the ceiling for what can be extracted without LLM-based code understanding.

## Feature Importance (from LORO evaluation)

| Feature Group | ROC-AUC when removed | Contribution |
|---------------|---------------------|--------------|
| All 35 (baseline) | 0.7885 | — |
| Minus file history (12) | 0.7347 | **+0.0538** (biggest winner) |
| Minus author-file familiarity (6) | 0.7847 | +0.0038 |
| Minus change-shape (8) | 0.7863 | +0.0022 |
| Minus individual suspects | 0.7857 | <0.001 each |

File history features contribute 73% of the total ROC-AUC gain over the 9-feature baseline.

## Version History

| Version | Model Type | ROC-AUC | Training Data | Notes |
|---------|------------|---------|---------------|-------|
| v1 | LGBMClassifier | ? | Django only | Initial (orphaned params) |
| v2 | RandomForestClassifier | ? | Django only | Archived |
| v4 | LGBMClassifier | ? | 5 repos (pre-rebuild) | Archived, had labeling bugs |
| v5 | LGBMClassifier | 0.6744 (cross-repo) | 5 repos, 10K commits, 19 features | Archived — leaky pooled-F1 baseline |
| v7 | LGBMClassifier | 0.7784 | 5 repos, 10K commits, 35 features | Archived — promoted on cross-repo LORO ROC-AUC |
| **v8 (current)** | **LGBMClassifier** | **0.7885** | **5 repos, 10K commits, 35 features** | **Current Production — parity-verified extraction, author identity via normalized email** |

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
- **Promotion gate:** Compares cross-repo LORO ROC-AUC with `eval_protocol` recorded in run params; only promotes if new model beats Production on the same metric (fixed in K.3, verified in P.3)
- **Monitoring:** Prometheus/Grafana dashboards (request rate, latency, error rate)
- **Drift detection:** Evidently reports comparing recent vs. training feature distributions
- **Backup:** mlflow.db is backed up before every registry write
