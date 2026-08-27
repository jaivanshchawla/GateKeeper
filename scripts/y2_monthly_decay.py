#!/usr/bin/env python3
"""Y.2: Monthly decay analysis — does performance decay with distance from training window?"""
import os, sys, subprocess, time, yaml, numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "kubernetes": "repos/kubernetes",
    "rust": "repos/rust",
}
TRAINING_END = datetime(2026, 6, 30, tzinfo=timezone.utc)
TRAINING_END_TS = int(TRAINING_END.timestamp())

import skops.io as sio
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
           "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

from sklearn.metrics import roc_auc_score

def bootstrap_auc(scores, actuals, n_resamples=200, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.choice(len(actuals), size=len(actuals), replace=True)
        s, a = scores[idx], actuals[idx]
        if len(np.unique(a)) < 2: continue
        aucs.append(roc_auc_score(a, s))
    if len(aucs) < 5: return 0.0, 0.0, 1.0
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


print("=" * 80)
print("Y.2: MONTHLY DECAY ANALYSIS")
print("=" * 80)

for repo_name, rp_rel in REPOS.items():
    rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
    if not os.path.exists(rp):
        continue

    print(f"\n{'─' * 60}")
    print(f"  {repo_name}")
    print(f"{'─' * 60}")

    # Get ALL non-merge commits with timestamps
    r = subprocess.run(
        ["git", "log", "--since=2024-07-01", "--no-merges", "--format=%H|%ct", "--max-count=50000"],
        cwd=rp, capture_output=True, text=True, timeout=60
    )
    all_entries = []
    for line in r.stdout.strip().split("\n"):
        if "|" in line:
            h, ts = line.split("|", 1)
            try: all_entries.append((h.strip(), int(ts)))
            except: pass

    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv"))
    repo_df = df[df["source_repo"] == repo_name].copy()
    csv_hashes = set(repo_df["hash"].values)

    # Build monthly buckets
    buckets = defaultdict(list)
    for h, ts in all_entries:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        bucket = dt.strftime("%Y-%m")
        buckets[bucket].append((h, ts))

    MAX_PER_BUCKET = 50

    print(f"  {'Month':>8} {'N':>5} {'Base':>6} {'ROC-AUC':>12} {'95% CI':>20} {'Type':>8}")
    print(f"  {'─' * 60}")

    monthly_results = {}
    in_aucs = []
    out_aucs = []

    for bucket_name in sorted(buckets.keys()):
        entries = buckets[bucket_name]
        bucket_dt = datetime(int(bucket_name[:4]), int(bucket_name[5:]), 1, tzinfo=timezone.utc)
        is_oow = bucket_dt >= TRAINING_END

        if len(entries) > MAX_PER_BUCKET:
            step = max(1, len(entries) // MAX_PER_BUCKET)
            sampled = entries[::step][:MAX_PER_BUCKET]
        else:
            sampled = entries

        scores = []
        actuals = []
        for h, ts in sampled:
            if h not in csv_hashes:
                continue  # Skip OOW commits needing extraction for speed

            csv_row = repo_df[repo_df["hash"] == h]
            if len(csv_row) == 0:
                continue
            fv = [csv_row.iloc[0].get(c, 0) for c in fcols]
            score = float(model.predict_proba(np.array([fv]))[0][1])

            commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            window_end_ts = int((commit_dt + timedelta(days=7)).timestamp())

            files_r = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h[:8]],
                cwd=rp, capture_output=True, text=True, timeout=10
            )
            cfiles = [f.strip() for f in files_r.stdout.strip().split("\n") if f.strip()][:5]

            is_retouched = False
            for fp in cfiles:
                retouch_r = subprocess.run(
                    ["git", "log", f"--since={commit_dt.isoformat()}",
                     f"--until={datetime.fromtimestamp(window_end_ts, tz=timezone.utc).isoformat()}",
                     "--format=%H", "--", fp],
                    cwd=rp, capture_output=True, text=True, timeout=15
                )
                for line in retouch_r.stdout.strip().split("\n"):
                    if line.strip() and line.strip()[:8] != h[:8]:
                        is_retouched = True
                        break
                if is_retouched:
                    break

            actual = 1 if is_retouched else 0
            scores.append(score)
            actuals.append(actual)

        n = len(actuals)
        label = "OUT" if is_oow else "IN"
        if n < 10 or len(set(actuals)) < 2:
            print(f"  {bucket_name:>8} {n:>5} {'':>6} {'N/A':>12} {'':>20} {label:>8}")
            continue

        scores_arr = np.array(scores)
        actuals_arr = np.array(actuals)
        base_rate = actuals_arr.mean()
        mean_auc, lo, hi = bootstrap_auc(scores_arr, actuals_arr, n_resamples=200)

        print(f"  {bucket_name:>8} {n:>5} {base_rate:>5.1%} {mean_auc:>8.4f} [{lo:.4f},{hi:.4f}] {label:>8}")

        monthly_results[bucket_name] = {"n": n, "roc_auc": mean_auc, "is_oow": is_oow}
        if is_oow:
            out_aucs.append(mean_auc)
        else:
            in_aucs.append(mean_auc)

    if in_aucs:
        print(f"  IN-window mean AUC: {np.mean(in_aucs):.4f} ({len(in_aucs)} months)")
    if out_aucs:
        print(f"  OUT-window mean AUC: {np.mean(out_aucs):.4f} ({len(out_aucs)} months)")
    if in_aucs and out_aucs:
        gap = np.mean(in_aucs) - np.mean(out_aucs)
        if gap > 0.05:
            print(f"  → TEMPORAL DECAY: gap={gap:.4f}, retraining should help")
        elif gap < -0.05:
            print(f"  → REPO-SPECIFIC: OOW BETTER (gap={gap:.4f}), variance not decay")
        else:
            print(f"  → NO SIGNIFICANT DECAY: gap={gap:.4f}")
    elif in_aucs and not out_aucs:
        print(f"  → No OOW commits with CSV features (extraction too slow)")
