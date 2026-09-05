#!/usr/bin/env python3
"""U.6.9: Re-score all 5 repos OOW with CORRECT feature extraction.

The u68_oow_all.py script passed empty files/author to compute_single_commit_m1_features,
causing constant features and degenerate AUC for django, react, and kafka.
This script passes actual files from git diff-tree, matching what u68_rust_oow_v2.py did.
"""
import os
import sys
import json
import time
import subprocess
import numpy as np
import skops.io as sio
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "kubernetes": "repos/kubernetes",
    "rust": "repos/rust",
}
TRAINING_END = "2026-06-30"
WINDOW_DAYS = 7
MAX_PER_REPO = 200

# Load model
model_path = str(Path(__file__).parent.parent / "models" / "gatekeeper_risk_model.skops")
trusted = [
    "collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
    "numpy.dtype", "numpy.ndarray", "pandas.core.frame.DataFrame", "pandas.core.series.Series",
]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)

config = yaml.safe_load(open(str(Path(__file__).parent.parent / "ml" / "config.yaml")))
fcols = config["feature_columns"]


def bootstrap_auc_ci(y_true, y_score, n_resamples=1000):
    n = len(y_true)
    aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_t, y_s = np.array(y_true)[idx], np.array(y_score)[idx]
        if len(np.unique(y_t)) < 2:
            continue
        aucs.append(roc_auc_score(y_t, y_s))
    if not aucs:
        return 0.5, 0.5, 0.5
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    from ml.single_commit_features import (
        clear_cache, compute_single_commit_m1_features,
        _get_full_graph, _precompute_author_prior, _ensure_walk_snapshots,
        _hot_state,
    )

    all_results = {}

    for repo_name, rp_rel in REPOS.items():
        rp = str(Path(__file__).parent.parent / rp_rel)
        if not os.path.exists(rp):
            print(f"{repo_name}: repo not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  {repo_name}")
        print(f"{'='*60}")

        clear_cache()

        # Build graph + caches
        t0 = time.time()
        graph, risky, sorted_graph = _get_full_graph(rp)
        _precompute_author_prior(rp)
        _ensure_walk_snapshots(rp)
        print(f"  Cold start: {time.time()-t0:.1f}s ({len(graph)} commits)")

        # Get OOW commits
        result = subprocess.check_output(
            ["git", "log", "--no-merges", "--format=%H|%ct",
             f"--since={TRAINING_END}", f"--max-count={MAX_PER_REPO*2}"],
            cwd=rp, text=True, timeout=60,
        ).strip().split("\n")

        entries = []
        for line in result:
            if "|" not in line:
                continue
            parts = line.split("|", 1)
            try:
                entries.append((parts[0].strip(), int(parts[1])))
            except (ValueError, IndexError):
                pass

        # Sample evenly
        if len(entries) > MAX_PER_REPO:
            step = max(1, len(entries) // MAX_PER_REPO)
            entries = entries[::step][:MAX_PER_REPO]

        print(f"  OOW commits: {len(entries)}")

        # Build file index for outcome computation
        buffer_start = datetime(2026, 6, 23, tzinfo=timezone.utc)
        all_r = subprocess.run(
            ["git", "log", f"--since={buffer_start.isoformat()}",
             "--no-merges", "--format=COMMIT|%H|%ct", "--name-only"],
            cwd=rp, capture_output=True, text=True, timeout=120,
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

        # Score and compute outcomes
        scores_list = []
        _hot_state.clear()
        t0 = time.time()

        for i, (h, ts) in enumerate(entries):
            if i > 0 and i % 20 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(entries) - i) / rate if rate > 0 else 0
                print(f"  ... {i}/{len(entries)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

            try:
                commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)

                # KEY FIX: Get actual files touched by this commit
                cfiles = []
                try:
                    files_r = subprocess.run(
                        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h],
                        cwd=rp, capture_output=True, text=True, timeout=10,
                    )
                    cfiles = [f.strip() for f in files_r.stdout.strip().split("\n") if f.strip()]
                except Exception:
                    pass

                # Get author name
                try:
                    author_r = subprocess.run(
                        ["git", "log", "-1", "--format=%an", h],
                        cwd=rp, capture_output=True, text=True, timeout=10,
                    )
                    author = author_r.stdout.strip()
                except Exception:
                    author = ""

                # Pass ACTUAL files and author — the fix for u68_oow_all.py
                feats = compute_single_commit_m1_features(
                    rp, h, commit_dt, author, set(cfiles), 0, 0
                )
                fv = [feats.get(c, 0) for c in fcols]
                score = float(model.predict_proba(np.array([fv]))[0][1])
            except Exception as e:
                print(f"    WARN {h[:12]}: {e}")
                import traceback; traceback.print_exc()
                score = 0.5
                cfiles = []

            # Outcome
            commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
            window_end_ts = int((commit_dt + timedelta(days=WINDOW_DAYS)).timestamp())

            is_retouched = False
            for fp in cfiles:
                for cts, ch in file_commits.get(fp, []):
                    if cts > ts and cts <= window_end_ts and ch[:8] != h[:8]:
                        is_retouched = True
                        break
                if is_retouched:
                    break

            actual = 1 if is_retouched else 0
            scores_list.append({"hash": h, "ts": ts, "score": score, "actual": actual})

        elapsed = time.time() - t0
        print(f"  Scored {len(scores_list)} commits in {elapsed:.0f}s")

        # Compute metrics
        y_true = [s["actual"] for s in scores_list]
        y_score = [s["score"] for s in scores_list]
        pos_rate = float(np.mean(y_true)) if y_true else 0

        unique_scores = len(set(y_score))
        unique_labels = len(set(y_true))

        output = {
            "repo": repo_name,
            "n_commits": len(scores_list),
            "positive_rate": pos_rate,
            "scores": scores_list,
            "unique_scores": unique_scores,
        }

        # DEGENERACY GUARD
        if unique_scores < 2:
            print(f"  DEGENERATE: only {unique_scores} distinct score(s) — AUC undefined")
            output["roc_auc"] = None
            output["bootstrap_mean"] = None
            output["bootstrap_ci"] = [None, None]
            output["degenerate_reason"] = f"only {unique_scores} distinct predicted score(s)"
        elif unique_labels < 2:
            print(f"  DEGENERATE: only {unique_labels} label class(es) — AUC undefined")
            output["roc_auc"] = None
            output["bootstrap_mean"] = None
            output["bootstrap_ci"] = [None, None]
            output["degenerate_reason"] = f"only {unique_labels} label class(es)"
        else:
            auc = float(roc_auc_score(y_true, y_score))
            mean, lo, hi = bootstrap_auc_ci(y_true, y_score, n_resamples=1000)
            # Check for zero-width CI
            if abs(hi - lo) < 1e-10:
                print(f"  DEGENERATE: zero-width bootstrap CI — AUC unreliable")
                output["degenerate_reason"] = "zero-width bootstrap CI"
            output["roc_auc"] = auc
            output["bootstrap_mean"] = mean
            output["bootstrap_ci"] = [lo, hi]
            print(f"  ROC-AUC: {auc:.4f}  bootstrap mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

        print(f"  Positive rate: {pos_rate:.3f} ({sum(y_true)}/{len(y_true)})")
        print(f"  Unique scores: {unique_scores}, Unique labels: {unique_labels}")
        print(f"  Score range: {min(y_score):.6f} — {max(y_score):.6f}")

        ckpt_path = str(Path(__file__).parent.parent / "data" / f"u69_{repo_name}_oow.json")
        Path(ckpt_path).write_text(json.dumps(output, indent=2))
        all_results[repo_name] = output

    # Summary table
    print(f"\n{'='*80}")
    print(f"  U.6.9 OOW SUMMARY TABLE (CORRECTED)")
    print(f"{'='*80}")
    print(f"{'Repo':12s} {'N':>6s} {'Rate':>8s} {'ROC-AUC':>10s} {'CI Lower':>10s} {'CI Upper':>10s} {'Unique':>7s}")
    print("-" * 70)
    for name in REPOS:
        if name not in all_results:
            continue
        r = all_results[name]
        n = r["n_commits"]
        rate = r["positive_rate"]
        auc = r.get("roc_auc")
        ci = r.get("bootstrap_ci", [None, None])
        uniq = r.get("unique_scores", "?")
        if auc is not None:
            print(f"{name:12s} {n:6d} {rate:8.3f} {auc:10.4f} {ci[0]:10.4f} {ci[1]:10.4f} {uniq:7d}")
        else:
            reason = r.get("degenerate_reason", "unknown")
            print(f"{name:12s} {n:6d} {rate:8.3f} {'UNDEFINED':>10s} {'—':>10s} {'—':>10s} {uniq:7d}  ({reason})")


if __name__ == "__main__":
    main()
