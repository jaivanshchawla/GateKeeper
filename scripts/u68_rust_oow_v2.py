#!/usr/bin/env python3
"""U.6.8b: Rust OOW scoring with actual files passed to feature extractor."""
import os, sys, json, time, subprocess, numpy as np, skops.io as sio, yaml
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

rp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "repos", "rust"))

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
           "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
model = sio.loads(open(model_path,"rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

from ml.single_commit_features import (
    clear_cache, compute_single_commit_m1_features,
    _get_full_graph, _precompute_author_prior, _ensure_walk_snapshots, _hot_state,
)
clear_cache()

t0 = time.time()
graph, risky, sorted_graph = _get_full_graph(rp)
_precompute_author_prior(rp)
_ensure_walk_snapshots(rp)
print(f"Cold start: {time.time()-t0:.1f}s ({len(graph)} commits)")

# Get OOW commits
result = subprocess.check_output(
    ["git", "log", "--no-merges", "--format=%H|%ct", "--since=2026-07-01", "--max-count=400"],
    cwd=rp, text=True, timeout=60,
).strip().split("\n")
entries = []
for line in result:
    if "|" not in line: continue
    parts = line.split("|", 1)
    try: entries.append((parts[0].strip(), int(parts[1])))
    except: pass
step = max(1, len(entries) // 200)
entries = entries[::step][:200]
print(f"OOW commits: {len(entries)}")

# Build file index for outcome computation
buffer_start = datetime(2026, 6, 23)
all_r = subprocess.run(
    ["git", "log", f"--since={buffer_start.isoformat()}", "--no-merges", "--format=COMMIT|%H|%ct", "--name-only"],
    cwd=rp, capture_output=True, text=True, timeout=120,
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
    for f in files: file_commits[f].append((ts, h))
for f in file_commits: file_commits[f].sort()

# Score
scores_list = []
_hot_state.clear()
t0 = time.time()

for i, (h, ts) in enumerate(entries):
    if i > 0 and i % 20 == 0:
        elapsed = time.time() - t0
        print(f"  [{i}/{len(entries)}] ({elapsed:.0f}s)")

    # Get actual files touched by this commit
    try:
        files_r = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h],
            cwd=rp, capture_output=True, text=True, timeout=10,
        )
        touched_files = {f.strip() for f in files_r.stdout.strip().split("\n") if f.strip()}
    except Exception:
        touched_files = set()

    # Get author
    try:
        author_r = subprocess.run(
            ["git", "log", "-1", "--format=%aE", h],
            cwd=rp, capture_output=True, text=True, timeout=10,
        )
        author = author_r.stdout.strip()
    except Exception:
        author = ""

    try:
        feats = compute_single_commit_m1_features(
            rp, h, datetime.fromtimestamp(ts), author, touched_files, 0, 0
        )
        fv = [feats.get(c, 0) for c in fcols]
        score = float(model.predict_proba(np.array([fv]))[0][1])
    except Exception as e:
        print(f"  WARN {h[:12]}: {e}")
        score = 0.5

    # Outcome
    commit_dt = datetime.fromtimestamp(ts)
    window_end_ts = int((commit_dt + timedelta(days=7)).timestamp())
    cfiles = list(touched_files)[:10]

    is_retouched = False
    for fp in cfiles:
        for cts, ch in file_commits.get(fp, []):
            if cts > ts and cts <= window_end_ts and ch[:8] != h[:8]:
                is_retouched = True
                break
        if is_retouched: break

    is_reverted = False
    try:
        rev_r = subprocess.run(
            ["git", "log", f"--since={commit_dt.isoformat()}",
             f"--until={datetime.fromtimestamp(window_end_ts).isoformat()}",
             "--grep=revert", "-i", "--format=%H|%s", "--max-count=50"],
            cwd=rp, capture_output=True, text=True, timeout=15,
        )
        for line in rev_r.stdout.strip().split("\n"):
            if "|" in line:
                _, msg = line.split("|", 1)
                if h[:8] in msg:
                    is_reverted = True
                    break
    except Exception:
        pass

    actual = 1 if (is_reverted or is_retouched) else 0
    scores_list.append({"hash": h, "ts": ts, "score": score, "actual": actual})

elapsed = time.time() - t0
y_true = [s["actual"] for s in scores_list]
y_score = [s["score"] for s in scores_list]
pos_rate = float(np.mean(y_true))
print(f"\nScored {len(scores_list)} in {elapsed:.0f}s, rate={pos_rate:.3f}")

from sklearn.metrics import roc_auc_score
if len(np.unique(y_true)) >= 2:
    auc = float(roc_auc_score(y_true, y_score))
    n = len(y_true); aucs = []
    rng = np.random.RandomState(42)
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        yt, ys = np.array(y_true)[idx], np.array(y_score)[idx]
        if len(np.unique(yt)) < 2: continue
        aucs.append(roc_auc_score(yt, ys))
    mean, lo, hi = float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
    print(f"ROC-AUC: {auc:.4f}  mean={mean:.4f}  CI=[{lo:.4f},{hi:.4f}]")
    output = {"repo": "rust", "n_commits": len(scores_list), "positive_rate": pos_rate,
              "roc_auc": auc, "bootstrap_mean": mean, "bootstrap_ci": [lo, hi], "scores": scores_list}
else:
    print(f"Only one class (rate={pos_rate:.3f})")
    output = {"repo": "rust", "n_commits": len(scores_list), "positive_rate": pos_rate, "scores": scores_list}

Path(os.path.join(os.path.dirname(__file__), "..", "data", "u68_rust_oow.json")).write_text(json.dumps(output, indent=2))
print("Saved")
