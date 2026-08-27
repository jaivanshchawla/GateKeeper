#!/usr/bin/env python3
"""Y.1: Rust out-of-window evaluation — batched with checkpointing."""
import os, sys, subprocess, time, yaml, json, numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = "rust"
rp = os.path.join(os.path.dirname(__file__), "..", "repos", REPO)
TRAINING_END = "2026-06-30"
WINDOW_DAYS = 7
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "data", "y1_rust_checkpoint.json")
N_COMMITS = 200

# Resume from checkpoint if exists
results = {}
if os.path.exists(CHECKPOINT):
    results = json.load(open(CHECKPOINT))
    print(f"Resuming from checkpoint: {len(results)} already scored")
else:
    print("Starting fresh")

# Get OOW commits
r = subprocess.run(
    ["git", "log", f"--since={TRAINING_END}", "--no-merges",
     "--format=%H|%ct", f"--max-count={N_COMMITS * 10}"],
    cwd=rp, capture_output=True, text=True, timeout=60
)
entries = []
for line in r.stdout.strip().split("\n"):
    if "|" in line:
        h, ts = line.split("|", 1)
        try:
            hh = h.strip()
            if hh not in results:
                entries.append((hh, int(ts)))
        except ValueError:
            pass

print(f"OOW commits to score: {len(entries)}")

if len(entries) == 0:
    print("All done! Computing metrics...")
else:
    # Build file index for outcomes
    print("Building file index...")
    t0 = time.time()
    buffer_start = datetime(2026, 6, 23, tzinfo=timezone.utc)
    all_r = subprocess.run(
        ["git", "log", f"--since={buffer_start.isoformat()}", "--no-merges",
         "--format=COMMIT|%H|%ct", "--name-only"],
        cwd=rp, capture_output=True, text=True, timeout=120
    )
    commits_data = {}
    cur_h, cur_ts, cur_files = None, 0, []
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

    file_commits = defaultdict(list)
    for h, (ts, files) in commits_data.items():
        for f in files:
            file_commits[f].append((ts, h))
    for f in file_commits:
        file_commits[f].sort()
    print(f"  Built in {time.time()-t0:.0f}s ({len(file_commits)} files)")

    # Load model
    import skops.io as sio
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
    trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
               "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
    model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
    config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
    fcols = config["feature_columns"]

    # Initialize extractor (warm cache)
    from ml.extract_features import CommitFeatureExtractor
    from ml.single_commit_features import clear_cache
    clear_cache()
    ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

    # Score + compute outcomes in batches
    t0 = time.time()
    for i, (h, ts) in enumerate(entries):
        if i > 0 and i % 10 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(entries) - i) / rate
            print(f"  ... {i}/{len(entries)} ({elapsed:.0f}s, ETA {eta:.0f}s)")
            # Save checkpoint every 10
            json.dump(results, open(CHECKPOINT, "w"))

        try:
            feat = ext.extract_single_commit(rp, h)
            fv = [feat.get(c, 0) for c in fcols]
            score = float(model.predict_proba(np.array([fv]))[0][1])
        except:
            score = 0.5

        # Outcome
        commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        window_end_ts = int((commit_dt + timedelta(days=WINDOW_DAYS)).timestamp())

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

        is_reverted = False
        rev_r = subprocess.run(
            ["git", "log", f"--since={commit_dt.isoformat()}",
             f"--until={datetime.fromtimestamp(window_end_ts, tz=timezone.utc).isoformat()}",
             "--grep=revert", "-i", "--format=%H|%s", "--max-count=50"],
            cwd=rp, capture_output=True, text=True, timeout=15
        )
        for line in rev_r.stdout.strip().split("\n"):
            if "|" in line:
                _, msg = line.split("|", 1)
                if h[:8] in msg:
                    is_reverted = True
                    break

        results[h] = {"ts": ts, "score": score, "actual": 1 if (is_reverted or is_retouched) else 0}

    json.dump(results, open(CHECKPOINT, "w"))
    print(f"  Scoring+outcomes done in {time.time()-t0:.0f}s")

# Compute metrics
print(f"\n{'='*60}")
print("RUST OUT-OF-WINDOW RESULTS")
print(f"{'='*60}")

from sklearn.metrics import roc_auc_score, average_precision_score

scores_arr = np.array([v["score"] for v in results.values()])
actuals_arr = np.array([v["actual"] for v in results.values()])
n = len(actuals_arr)
base_rate = float(actuals_arr.mean())

rng = np.random.RandomState(42)
aucs = []
for _ in range(1000):
    idx = rng.choice(n, size=n, replace=True)
    s, a = scores_arr[idx], actuals_arr[idx]
    if len(np.unique(a)) < 2:
        continue
    aucs.append(roc_auc_score(a, s))
mean_auc = float(np.mean(aucs))
lo_ci = float(np.percentile(aucs, 2.5))
hi_ci = float(np.percentile(aucs, 97.5))
pr_auc = float(average_precision_score(actuals_arr, scores_arr))

thresholds = config.get("thresholds", {})
rt = thresholds.get("rust", thresholds.get("_global", {}))
high_mask = scores_arr >= rt.get("high", 0.86)
med_mask = (scores_arr >= rt.get("medium", 0.75)) & ~high_mask
high_n = int(high_mask.sum())
high_tp = int(actuals_arr[high_mask].sum()) if high_n > 0 else 0
med_n = int(med_mask.sum())
med_tp = int(actuals_arr[med_mask].sum()) if med_n > 0 else 0

def wilson_ci(s, t, z=1.96):
    if t == 0: return 0.0, 0.0, 1.0
    p = s / t
    d = 1 + z**2 / t
    c = (p + z**2 / (2*t)) / d
    sp = z * np.sqrt((p*(1-p) + z**2/(4*t)) / t) / d
    return p, max(0, c-sp), min(1, c+sp)

h_prec, h_lo, h_hi = wilson_ci(high_tp, high_n)
m_prec, m_lo, m_hi = wilson_ci(med_tp, med_n)

print(f"N: {n}, Base rate: {base_rate:.1%}")
print(f"ROC-AUC: {mean_auc:.4f} [{lo_ci:.4f}, {hi_ci:.4f}]")
print(f"PR-AUC: {pr_auc:.4f} (lift: {pr_auc/base_rate:.2f}x)")
print(f"High: {high_tp}/{high_n} = {h_prec:.1%} [{h_lo:.1%}, {h_hi:.1%}]")
print(f"Med:  {med_tp}/{med_n} = {m_prec:.1%} [{m_lo:.1%}, {m_hi:.1%}]")
print(f"Offline LORO ROC-AUC: 0.8038")
print(f"Gap: {0.8038 - mean_auc:.4f}")

print(f"\nCalibration (10 deciles):")
print(f"{'Dec':>4} {'N':>5} {'Actual%':>8}")
for decile in range(10):
    lo_p = np.percentile(scores_arr, decile * 10)
    hi_p = np.percentile(scores_arr, (decile + 1) * 10)
    mask = (scores_arr >= lo_p) & (scores_arr < hi_p) if decile < 9 else (scores_arr >= lo_p)
    if mask.sum() > 0:
        print(f"  {decile+1:>4} {int(mask.sum()):>5} {actuals_arr[mask].mean():>7.1%}")
