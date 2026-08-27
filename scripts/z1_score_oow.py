#!/usr/bin/env python3
"""Score OOW commits for Django/React/Kafka/K8s and save to JSON checkpoint."""
import os, sys, subprocess, time, yaml, json, numpy as np, skops.io as sio
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "kubernetes": "repos/kubernetes",
}
TRAINING_END = "2026-06-30"
WINDOW_DAYS = 7
MAX_PER_REPO = 150

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
           "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

for repo_name, rp_rel in REPOS.items():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "data", f"z1_{repo_name}_oow.json")
    if os.path.exists(ckpt_path):
        existing = json.load(open(ckpt_path))
        print(f"{repo_name}: {len(existing)} already scored, skipping")
        continue

    rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
    if not os.path.exists(rp):
        continue

    print(f"\n{'='*60}")
    print(f"  {repo_name}")
    print(f"{'='*60}")

    # Get HEAD
    head_r = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD"], cwd=rp, capture_output=True, text=True, timeout=10)
    head_ts = int(head_r.stdout.strip())
    head_dt = datetime.fromtimestamp(head_ts, tz=timezone.utc)
    cutoff_ts = int((head_dt - timedelta(days=WINDOW_DAYS)).timestamp())

    # Get OOW commits
    r = subprocess.run(
        ["git", "log", f"--since={TRAINING_END}", "--no-merges", "--format=%H|%ct", f"--max-count={MAX_PER_REPO*2}"],
        cwd=rp, capture_output=True, text=True, timeout=60
    )
    entries = []
    for line in r.stdout.strip().split("\n"):
        if "|" in line:
            h, ts = line.split("|", 1)
            try: entries.append((h.strip(), int(ts)))
            except: pass
    print(f"  OOW commits: {len(entries)}")

    # Load extractor
    from ml.extract_features import CommitFeatureExtractor
    from ml.single_commit_features import clear_cache
    clear_cache()
    ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

    # Build file index for outcome computation
    buffer_start = datetime(2026, 6, 23, tzinfo=timezone.utc)
    all_r = subprocess.run(
        ["git", "log", f"--since={buffer_start.isoformat()}", "--no-merges", "--format=COMMIT|%H|%ct", "--name-only"],
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

    # Score and compute outcomes
    results = {}
    t0 = time.time()
    for i, (h, ts) in enumerate(entries):
        if i > 0 and i % 20 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(entries) - i) / rate
            print(f"  ... {i}/{len(entries)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

        # Feature extraction
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
            if is_retouched: break

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

        actual = 1 if (is_reverted or is_retouched) else 0
        results[h] = {"ts": ts, "score": score, "actual": actual,
                       "within_7d": ts >= cutoff_ts}

    # Save
    json.dump(results, open(ckpt_path, "w"))
    print(f"  Done: {len(results)} commits in {time.time()-t0:.0f}s")
    print(f"  Within 7d: {sum(1 for v in results.values() if v['within_7d'])}")
    print(f"  Beyond 7d: {sum(1 for v in results.values() if not v['within_7d'])}")
