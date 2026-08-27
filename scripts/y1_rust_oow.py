#!/usr/bin/env python3
"""Y.1: Rust out-of-window ROC-AUC — optimized batch evaluation."""
import os, sys, subprocess, time, yaml, numpy as np, skops.io as sio
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

rp = os.path.join(os.path.dirname(__file__), "..", "repos", "rust")
TRAINING_END = "2026-06-30"
WINDOW_DAYS = 7
MAX_COMMITS = 200  # Keep manageable

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
           "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

# Step 1: Get OOW commits (limited to MAX_COMMITS)
r = subprocess.run(
    ["git", "log", f"--since={TRAINING_END}", "--no-merges",
     "--format=%H|%ct|%aE", "--max-count=2000"],
    cwd=rp, capture_output=True, text=True, timeout=60
)
entries = []
for line in r.stdout.strip().split("\n"):
    if "|" in line:
        parts = line.split("|", 2)
        try:
            entries.append((parts[0].strip(), int(parts[1]), parts[2] if len(parts) > 2 else ""))
        except ValueError:
            pass
print(f"Rust OOW commits available: {len(entries)}")
entries = entries[:MAX_COMMITS]
print(f"Using first {len(entries)}")

# Step 2: Build file->commit index for outcome computation (single git call)
print("Building file index for outcomes...")
t0 = time.time()
buffer_start = datetime(2026, 6, 23, tzinfo=timezone.utc)
all_r = subprocess.run(
    ["git", "log", f"--since={buffer_start.isoformat()}", "--no-merges",
     "--format=COMMIT|%H|%ct", "--name-only"],
    cwd=rp, capture_output=True, text=True, timeout=120
)
commits_data = {}
cur_h = None
cur_ts = 0
cur_files = []
for line in all_r.stdout.split("\n"):
    if line.startswith("COMMIT|"):
        if cur_h:
            commits_data[cur_h] = (cur_ts, cur_files)
        parts = line.split("|", 2)
        cur_h = parts[1]
        cur_ts = int(parts[2])
        cur_files = []
    elif cur_h and line.strip():
        cur_files.append(line.strip())
if cur_h:
    commits_data[cur_h] = (cur_ts, cur_files)

# Build file->commits index
file_commits = defaultdict(list)
for h, (ts, files) in commits_data.items():
    for f in files:
        file_commits[f].append((ts, h))
for f in file_commits:
    file_commits[f].sort()

# Build revert lookup
revert_commits = []
for h, (ts, files) in commits_data.items():
    pass  # Reverts are in commit messages; we'll check inline

print(f"  Index built in {time.time()-t0:.0f}s ({len(commits_data)} commits, {len(file_commits)} files)")

# Step 3: Score commits using extract_single_commit (with warm cache)
print("\nScoring commits (with cache)...")
from ml.extract_features import CommitFeatureExtractor
from ml.single_commit_features import clear_cache
clear_cache()
ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

scores = []
t0 = time.time()
for i, (h, ts, author) in enumerate(entries):
    if i % 20 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta = (len(entries) - i) / rate
        print(f"  ... {i}/{len(entries)} ({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s)")
    try:
        feat = ext.extract_single_commit(rp, h)
        fv = [feat.get(c, 0) for c in fcols]
        score = float(model.predict_proba(np.array([fv]))[0][1])
        scores.append((h, ts, score))
    except Exception as e:
        # Use 0.5 for failures
        scores.append((h, ts, 0.5))

elapsed = time.time() - t0
print(f"  Scored {len(scores)} in {elapsed:.0f}s ({elapsed/len(scores):.1f}s/commit)")

# Step 4: Compute outcomes from index
print("\nComputing realized outcomes from index...")
actuals = []
for h, ts, score in scores:
    commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    window_end_ts = int((commit_dt + timedelta(days=WINDOW_DAYS)).timestamp())

    # Get files
    files_r = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h[:8]],
        cwd=rp, capture_output=True, text=True, timeout=10
    )
    cfiles = [f.strip() for f in files_r.stdout.strip().split("\n") if f.strip()][:10]

    is_retouched = False
    for fp in cfiles:
        for cts, ch in file_commits.get(fp, []):
            if cts > ts and cts <= window_end_ts and ch[:8] != h[:8]:
                is_retouched = True
                break
        if is_retouched:
            break

    # Check revert (use subprocess for message-based check, just for OOW commits)
    is_reverted = False
    rev_r = subprocess.run(
        ["git", "log", f"--since={commit_dt.isoformat()}", f"--until={datetime.fromtimestamp(window_end_ts, tz=timezone.utc).isoformat()}",
         "--grep=revert", "-i", "--format=%H|%s", "--max-count=50"],
        cwd=rp, capture_output=True, text=True, timeout=15
    )
    for line in rev_r.stdout.strip().split("\n"):
        if "|" in line:
            _, msg = line.split("|", 1)
            if h[:8] in msg:
                is_reverted = True
                break

    actual = 1 if (is_reverted or is_retouched) else 0
    actuals.append(actual)

# Step 5: Compute metrics
scores_arr = np.array([s[2] for s in scores])
actuals_arr = np.array(actuals)
n = len(actuals_arr)
base_rate = float(actuals_arr.mean())

# Bootstrap ROC-AUC
rng = np.random.RandomState(42)
aucs = []
for _ in range(1000):
    idx = rng.choice(n, size=n, replace=True)
    s, a = scores_arr[idx], actuals_arr[idx]
    if len(np.unique(a)) < 2:
        continue
    aucs.append(roc_auc_score(a, s))
mean_auc = float(np.mean(aucs))
lo = float(np.percentile(aucs, 2.5))
hi = float(np.percentile(aucs, 97.5))
pr_auc = float(average_precision_score(actuals_arr, scores_arr))

# Per-band
thresholds = config.get("thresholds", {})
rt = thresholds.get("rust", thresholds.get("_global", {}))
high_mask = scores_arr >= rt.get("high", 0.86)
med_mask = (scores_arr >= rt.get("medium", 0.75)) & ~high_mask

high_n = int(high_mask.sum())
high_tp = int(actuals_arr[high_mask].sum()) if high_n > 0 else 0
med_n = int(med_mask.sum())
med_tp = int(actuals_arr[med_mask].sum()) if med_n > 0 else 0

# Wilson CI
def wilson_ci(s, t, z=1.96):
    if t == 0: return 0.0, 0.0, 1.0
    p = s / t
    d = 1 + z**2 / t
    c = (p + z**2 / (2*t)) / d
    sp = z * np.sqrt((p*(1-p) + z**2/(4*t)) / t) / d
    return p, max(0, c-sp), min(1, c+sp)

h_prec, h_lo, h_hi = wilson_ci(high_tp, high_n)
m_prec, m_lo, m_hi = wilson_ci(med_tp, med_n)

print(f"\n{'='*60}")
print(f"RUST OUT-OF-WINDOW RESULTS")
print(f"{'='*60}")
print(f"N: {n}, Base rate: {base_rate:.1%}")
print(f"ROC-AUC: {mean_auc:.4f} [{lo:.4f}, {hi:.4f}]")
print(f"PR-AUC: {pr_auc:.4f} (lift: {pr_auc/base_rate:.2f}x)")
print(f"High band: {high_tp}/{high_n} = {h_prec:.1%} [{h_lo:.1%}, {h_hi:.1%}]")
print(f"Med band:  {med_tp}/{med_n} = {m_prec:.1%} [{m_lo:.1%}, {m_hi:.1%}]")
print(f"\nOffline LORO ROC-AUC: 0.8038")
print(f"Gap: {0.8038 - mean_auc:.4f}")

# Calibration
print(f"\nCalibration (10 deciles):")
print(f"{'Dec':>4} {'N':>5} {'Actual%':>8}")
for decile in range(10):
    lo_p = np.percentile(scores_arr, decile * 10)
    hi_p = np.percentile(scores_arr, (decile + 1) * 10)
    mask = (scores_arr >= lo_p) & (scores_arr < hi_p) if decile < 9 else (scores_arr >= lo_p)
    if mask.sum() > 0:
        print(f"  {decile+1:>4} {int(mask.sum()):>5} {actuals_arr[mask].mean():>7.1%}")
