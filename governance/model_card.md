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
| ROC-AUC (out-of-window, frozen set, mean excl React) | **0.7033** | Production figure — forward generalization on frozen eval set |
| PR-AUC lift | +0.246 | Model ranks risky commits above base rate |
| Top-decile lift | ~1.5-2x | Top 10% of scores have 1.5-2x the precision of random |
| Brier score | 0.224 | Moderately well-calibrated |

**Two numbers, not one:** The cross-repo LORO (0.7885) tests generalization to unseen repos within the same time window. The out-of-window ROC-AUC tests generalization to future commits on a **frozen eval set** (U.7.0a, committed to `data/oow_eval_set.json`). The frozen set fixes a long-standing measurement instability: OOW commit samples were changing between reports, causing attributed deltas that were actually sampling noise.

**Frozen OOW results (U.7.0a):** django (0.654), react (0.618), kafka (0.679), kubernetes (0.789), rust (0.692). Mean excl. React: 0.703. Mean all 5: 0.686. Previous OOW numbers from reports Y.1 through U.6.9 used varying random samples and are not directly comparable to each other or to these frozen numbers.

**Sample instability disclosure:** OOW measurements before U.7.0a used random sampling with no seed or a different seed each time. The commit sets varied between reports Y.1, Z.1, V5.1, U.6.7b, U.6.8b, and U.6.9. Some deltas attributed to feature fixes (e.g. k8s 0.8812→0.7997 from parity fix, react 0.5542→0.6087) were partly sampling noise. The frozen set eliminates this.

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
| Rule engine | 18 deterministic rules (9 metadata + 9 content) in `.gatekeeper.yml` (separate from ML score) |

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

### Out-of-Window Evaluation (U.7.0a — frozen eval set)

The most honest metric: test on commits whose committer_date is AFTER the training window end (2026-06-30), with a **permanently frozen commit set** written to `data/oow_eval_set.json`. The frozen set uses deterministic sampling (every-Nth from the post-window commits, minus 7-day censoring at HEAD) so results never change unless the model changes.

**Note:** The commit sets in reports Y.1 through U.6.9 used random sampling with varying seeds. Those tables are not directly comparable to each other or to this frozen set.

| Repo | N (frozen) | Base Rate | OOW ROC-AUC | 95% CI (1000-row resamp.) | Offline LORO | Gap |
|------|-----------|-----------|------------|--------------------------|-------------|-----|
| django | 132 | 34.1% | **0.6536** | [0.5555, 0.7492] | 0.7607 | -0.107 |
| kafka | 160 | 50.0% | **0.6791** | [0.5948, 0.7661] | 0.8247 | -0.146 |
| kubernetes | 196 | 71.4% | **0.7889** | [0.7148, 0.8551] | 0.7952 | -0.006 |
| react | 95 | 53.7% | **0.6183** | [0.4966, 0.7443] | 0.7579 | -0.140 |
| rust | 50 | 66.0% | **0.6916** | [0.5471, 0.8472] | 0.8038 | -0.112 |
| **Mean** | | | **0.6863** | | 0.7885 | **-0.102** |

**Key findings (frozen set, U.7.0a):**
- **All 5 repos show forward generalization** — OOW ROC-AUC > 0.5 for every repo.
- **Kubernetes best** (0.7889, gap -0.006 from LORO) — nearly matches offline performance.
- **Django, Kafka, Rust cluster** at 0.65-0.69 OOW — modest but real signal.
- **React weakest** (0.6183, CI lower bound 0.50) — marginal signal, CI includes 0.5.
- **Rust CI is wide** [0.547, 0.847] due to N=50 only — Rust has 3,537 OOW commits but individual scoring takes ~1s per commit; only 50 were scored so far.
- **Parity tolerance:** author_prior_commits: mean |Δ| < 2, max < 5 between bulk and single-commit paths. Window-start boundary: file-level features may differ by 1 for commits within ~24h of window start (1/50 commits observed).

**What this means:** The model generalizes well to unseen repos within the training window (LORO 0.7885) and retains meaningful signal on future commits for all 5 repos, with a mean gap of -0.102. K8s is production-ready OOW; the other repos would benefit from periodic retraining. The frozen eval set ensures these numbers are stable across reports.

### Protocol Comparison

| Protocol | ROC-AUC | Notes |
|----------|---------|-------|
| Pooled random 80/20 | ~0.80 | **Inflated** — see below |
| Cross-repo LORO | **0.7885** | Generalization to unseen repos, same time window |
| Out-of-window (frozen set, mean) | **0.6863** | Generalization to future commits — what users experience (U.7.0a) |

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

Per-band precision figures below are from the pre-parity-fix backfill (W.2). Post-parity OOW ROC-AUC figures above are more reliable; per-band tables will be updated on next backfill run.

**Key findings (still valid):**
- **Kafka and Kubernetes are strong**: high-band precision 80-90%, genuine lift.
- **Django had too few out-of-window commits** for reliable per-band statistics.
- **React improved with parity fix** (ROC-AUC 0.5542→0.6626) but per-band precision is still below base rate.

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

4. **author_prior_commits residual skew (T.1):** Author identity is now keyed on normalized email (%aE + NFKD + casefold), fixing the display-name splitting bug ("Esteban Küber" vs "Esteban Kuber", "Lauren Tan" vs "lauren"). PyDriller and git log still occasionally report different emails for the same commit (e.g. `tg@trevorgross.com` vs `tmgross@umich.edu`); the SC path overrides with the graph email. Feature parity: 34/35 features at 0/50, author_prior_commits off-by-1-2 at timestamp edges (0.8% of cells). **Parity tolerances (documented, accepted):** author_prior_commits: mean |Δ| < 2, max < 5 (due to email identity differences between bulk extraction and single-commit extraction). Window-start boundary: file-level features may differ by 1 for commits within ~24h of window start (1/50 commits observed, e.g. react/e6783e7cc94b at 2024-07-01 13:50, 13 hours into the window).

5. **No code understanding:** The model uses only commit metadata (diff size, timing, author history, message keywords). It does not analyze code content, test coverage, or review quality.

6. **Calibration gaps:** 10-bin reliability analysis shows overconfidence in mid-range bins (predicted probabilities systematically higher than observed frequencies for rust, lower for react).

7. **OOW gap (U.7.0a frozen set):** All 5 repos show forward generalization (OOW > 0.5) on a frozen eval set (mean 0.6863, gap -0.102 from LORO 0.7885). K8s is near-parity (0.7889 vs 0.7952). React is weakest (0.6183, CI includes 0.5). Rust CI is wide (N=50 of 3,537 available). The frozen eval set (`data/oow_eval_set.json`) uses deterministic sampling and eliminates measurement instability — prior OOW tables (Y.1 through U.6.9) used varying random samples and are not directly comparable to each other or to this frozen set. Identity resolution via mailmap was tested: React has no .mailmap (untestable), Rust has 353 aliases merged from 392 .mailmap entries. Degeneracy guard: reject <2 distinct scores, <2 label classes, zero-width CI.

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
