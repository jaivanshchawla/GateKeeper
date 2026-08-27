#!/usr/bin/env python3
"""Z.1: Right-censoring at HEAD — the decisive test for the OOW gap."""
import os, sys, subprocess, time, yaml, json, numpy as np, skops.io as sio
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "kubernetes": "repos/kubernetes",
    "rust": "repos/rust",
}
TRAINING_END = "2026-06-30"
WINDOW_DAYS = 7

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = ["collections.OrderedDict","lightgbm.basic.Booster","lightgbm.sklearn.LGBMClassifier",
           "numpy.dtype","numpy.ndarray","pandas.core.frame.DataFrame","pandas.core.series.Series"]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

def bootstrap_auc(scores, actuals, n_resamples=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.choice(len(actuals), size=len(actuals), replace=True)
        s, a = scores[idx], actuals[idx]
        if len(np.unique(a)) < 2: continue
        aucs.append(roc_auc_score(a, s))
    if len(aucs) < 10: return 0.0, 0.0, 1.0
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def wilson_ci(s, t, z=1.96):
    if t == 0: return 0.0, 0.0, 1.0
    p = s / t; d = 1 + z**2 / t
    c = (p + z**2 / (2*t)) / d
    sp = z * np.sqrt((p*(1-p) + z**2/(4*t)) / t) / d
    return p, max(0, c-sp), min(1, c+sp)


print("=" * 80)
print("Z.1: RIGHT-CENSORING AT HEAD")
print("=" * 80)

# Load checkpoint if available (Rust)
rust_ckpt = {}
ckpt_path = os.path.join(os.path.dirname(__file__), "..", "data", "y1_rust_checkpoint.json")
if os.path.exists(ckpt_path):
    rust_ckpt = json.load(open(ckpt_path))
    print(f"Loaded Rust checkpoint: {len(rust_ckpt)} commits")

for repo_name, rp_rel in REPOS.items():
    rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
    if not os.path.exists(rp):
        continue

    print(f"\n{'─' * 60}")
    print(f"  {repo_name}")
    print(f"{'─' * 60}")

    # Get HEAD date
    head_r = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "HEAD"],
        cwd=rp, capture_output=True, text=True, timeout=10
    )
    head_ts = int(head_r.stdout.strip())
    head_dt = datetime.fromtimestamp(head_ts, tz=timezone.utc)

    # Get OOW commits
    r = subprocess.run(
        ["git", "log", f"--since={TRAINING_END}", "--no-merges",
         "--format=%H|%ct", "--max-count=2000"],
        cwd=rp, capture_output=True, text=True, timeout=60
    )
    entries = []
    for line in r.stdout.strip().split("\n"):
        if "|" in line:
            h, ts = line.split("|", 1)
            try: entries.append((h.strip(), int(ts)))
            except: pass

    n_total = len(entries)
    cutoff_ts = int((head_dt - timedelta(days=WINDOW_DAYS)).timestamp())
    within_7d = [(h, ts) for h, ts in entries if ts >= cutoff_ts]
    beyond_7d = [(h, ts) for h, ts in entries if ts < cutoff_ts]

    print(f"  HEAD: {head_dt.strftime('%Y-%m-%d')} (ts={head_ts})")
    print(f"  OOW commits total: {n_total}")
    print(f"  Within 7d of HEAD: {len(within_7d)} (censored — cannot observe retouch)")
    print(f"  Beyond 7d from HEAD: {len(beyond_7d)} (valid for labeling)")

    # Score commits using checkpoint (Rust) or extract_single_commit (others)
    if repo_name == "rust" and rust_ckpt:
        # Use checkpoint
        scores_all = []
        actuals_all = []
        scores_valid = []
        actuals_valid = []

        for h, ts in entries:
            if h in rust_ckpt:
                v = rust_ckpt[h]
                scores_all.append(v["score"])
                actuals_all.append(v["actual"])
                if ts < cutoff_ts:
                    scores_valid.append(v["score"])
                    actuals_valid.append(v["actual"])

        n_all = len(scores_all)
        n_valid = len(scores_valid)
        print(f"  Scored from checkpoint: {n_all} total, {n_valid} beyond 7d")

        if n_all > 0 and len(set(actuals_all)) >= 2:
            s_all = np.array(scores_all); a_all = np.array(actuals_all)
            mean_all, lo_all, hi_all = bootstrap_auc(s_all, a_all)
            pr_all = float(average_precision_score(a_all, s_all))
            br_all = float(a_all.mean())
            print(f"  ALL OOW:  N={n_all} Base={br_all:.1%} ROC-AUC={mean_all:.4f} [{lo_all:.4f},{hi_all:.4f}] PR-AUC={pr_all:.4f}")
        else:
            print(f"  ALL OOW:  N={n_all} — insufficient for AUC")

        if n_valid > 0 and len(set(actuals_valid)) >= 2:
            s_v = np.array(scores_valid); a_v = np.array(actuals_valid)
            mean_v, lo_v, hi_v = bootstrap_auc(s_v, a_v)
            pr_v = float(average_precision_score(a_v, s_v))
            br_v = float(a_v.mean())
            print(f"  VALID (>7d): N={n_valid} Base={br_v:.1%} ROC-AUC={mean_v:.4f} [{lo_v:.4f},{hi_v:.4f}] PR-AUC={pr_v:.4f}")
        else:
            print(f"  VALID (>7d): N={n_valid} — insufficient for AUC")
    else:
        # For other repos, use CSV features (fast) + outcome from index
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv"))
        repo_df = df[df["source_repo"] == repo_name].copy()
        csv_hashes = set(repo_df["hash"].values)

        # Score from CSV
        scores_all = []
        actuals_all = []
        scores_valid = []
        actuals_valid = []
        scored_all = 0
        scored_valid = 0

        for h, ts in entries:
            if h not in csv_hashes:
                continue

            csv_row = repo_df[repo_df["hash"] == h]
            if len(csv_row) == 0:
                continue
            fv = [csv_row.iloc[0].get(c, 0) for c in fcols]
            score = float(model.predict_proba(np.array([fv]))[0][1])

            # Outcome
            commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            window_end_ts = int((commit_dt + timedelta(days=WINDOW_DAYS)).timestamp())

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
                if is_retouched: break
            actual = 1 if is_retouched else 0

            scores_all.append(score)
            actuals_all.append(actual)
            scored_all += 1
            if ts < cutoff_ts:
                scores_valid.append(score)
                actuals_valid.append(actual)
                scored_valid += 1

        print(f"  Scored: {scored_all} total, {scored_valid} beyond 7d")

        if scored_all > 0 and len(set(actuals_all)) >= 2:
            s_all = np.array(scores_all); a_all = np.array(actuals_all)
            mean_all, lo_all, hi_all = bootstrap_auc(s_all, a_all)
            pr_all = float(average_precision_score(a_all, s_all))
            br_all = float(a_all.mean())
            print(f"  ALL OOW:  N={scored_all} Base={br_all:.1%} ROC-AUC={mean_all:.4f} [{lo_all:.4f},{hi_all:.4f}] PR-AUC={pr_all:.4f}")
        else:
            print(f"  ALL OOW:  N={scored_all} — insufficient")

        if scored_valid > 0 and len(set(actuals_valid)) >= 2:
            s_v = np.array(scores_valid); a_v = np.array(actuals_valid)
            mean_v, lo_v, hi_v = bootstrap_auc(s_v, a_v)
            pr_v = float(average_precision_score(a_v, s_v))
            br_v = float(a_v.mean())
            print(f"  VALID (>7d): N={scored_valid} Base={br_v:.1%} ROC-AUC={mean_v:.4f} [{lo_v:.4f},{hi_v:.4f}] PR-AUC={pr_v:.4f}")
        else:
            print(f"  VALID (>7d): N={scored_valid} — insufficient")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("Compare ALL OOW vs VALID (>7d) per repo.")
print("If VALID AUC closes the gap to LORO, the gap is a censoring artifact.")
print("If VALID AUC remains far from LORO, there is genuine temporal drift.")
