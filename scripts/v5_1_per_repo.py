#!/usr/bin/env python3
"""V5.1: Re-extract OOW for ONE repo with identity resolution."""
import os, sys, subprocess, time, yaml, json, numpy as np, skops.io as sio
from datetime import datetime, timezone, timedelta
from collections import defaultdict

repo_name = sys.argv[1] if len(sys.argv) > 1 else "react"
MAX_COMMITS = 200

REPOS = {
    "django": "repos/django", "react": "repos/react",
    "kafka": "repos/kafka", "kubernetes": "repos/kubernetes", "rust": "repos/rust",
}
rp_rel = REPOS.get(repo_name)
if not rp_rel:
    print(f"Unknown repo: {repo_name}")
    sys.exit(1)

rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
TRAINING_END = "2026-06-30"
WINDOW_DAYS = 7

# Ensure we can import from gatekeeper root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check if already done
out_path = os.path.join(os.path.dirname(__file__), "..", "data", f"v5_1_{repo_name}_oow.json")
if os.path.exists(out_path):
    d = json.load(open(out_path))
    print(f"{repo_name}: already done ({len(d)} commits)")
    sys.exit(0)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
           "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

# Force-clear identity cache
import ml.m1_shared as m1s
m1s._identity_map = {}
m1s._identity_loaded_for = ""

# Check identity resolution
from policy.identity import build_identity_map
imap = build_identity_map(rp)
print(f"{repo_name}: {len(imap)} identity aliases resolved")
for v, c in list(imap.items())[:10]:
    print(f"  {v} -> {c}")

# Get OOW commits
r = subprocess.run(
    ["git", "log", f"--since={TRAINING_END}", "--no-merges",
     "--format=%H|%ct", f"--max-count={MAX_COMMITS}"],
    cwd=rp, capture_output=True, text=True, timeout=60
)
entries = []
for line in r.stdout.strip().split("\n"):
    if "|" in line:
        h, ts = line.split("|", 1)
        try: entries.append((h.strip(), int(ts)))
        except: pass

head_r = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD"],
                        cwd=rp, capture_output=True, text=True, timeout=10)
head_ts = int(head_r.stdout.strip())
cutoff_ts = head_ts - WINDOW_DAYS * 86400

print(f"OOW commits: {len(entries)}")

# Build file index for outcomes
from ml.extract_features import CommitFeatureExtractor
from ml.single_commit_features import clear_cache
clear_cache()
ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

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
        if cur_h: commits_data[cur_h] = (cur_ts, cur_files)
        parts = line.split("|", 2)
        cur_h = parts[1]; cur_ts = int(parts[2]); cur_files = []
    elif cur_h and line.strip():
        cur_files.append(line.strip())
if cur_h: commits_data[cur_h] = (cur_ts, cur_files)

file_commits = defaultdict(list)
for h, (ts, files) in commits_data.items():
    for f in files:
        file_commits[f].append((ts, h))
for f in file_commits:
    file_commits[f].sort()

# Extract and score
results = {}
t0 = time.time()
for i, (h, ts) in enumerate(entries):
    if i > 0 and i % 10 == 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta = (len(entries) - i) / rate
        print(f"  ... {i}/{len(entries)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    try:
        feat = ext.extract_single_commit(rp, h)
        fv = [feat.get(c, 0) for c in fcols]
        score = float(model.predict_proba(np.array([fv]))[0][1])
        apc = feat.get("author_prior_commits", 0)
    except:
        score = 0.5; apc = 0

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
                is_retouched = True; break
        if is_retouched: break

    actual = 1 if is_retouched else 0
    results[h] = {"ts": ts, "score": score, "actual": actual,
                   "within_7d": ts >= cutoff_ts, "apc": apc}

# Save
json.dump(results, open(out_path, "w"))
print(f"Done: {len(results)} commits in {time.time()-t0:.0f}s")

# Quick AUC
from sklearn.metrics import roc_auc_score
scores_arr = np.array([v["score"] for v in results.values()])
actuals_arr = np.array([v["actual"] for v in results.values()])
if len(set(actuals_arr)) >= 2:
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(500):
        idx = rng.choice(len(actuals_arr), size=len(actuals_arr), replace=True)
        if len(np.unique(actuals_arr[idx])) < 2: continue
        aucs.append(roc_auc_score(actuals_arr[idx], scores_arr[idx]))
    if aucs:
        print(f"ROC-AUC: {np.mean(aucs):.4f} [{np.percentile(aucs,2.5):.4f},{np.percentile(aucs,97.5):.4f}]")
