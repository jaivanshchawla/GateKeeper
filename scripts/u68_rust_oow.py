#!/usr/bin/env python3
"""U.6.8b: Rust OOW scoring with parity-correct features."""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.single_commit_features import (
    clear_cache, compute_single_commit_m1_features,
    _get_full_graph, _precompute_author_prior, _ensure_walk_snapshots,
    _hot_state, WINDOW_START, FORWARD_LOOK_END, LABEL_WINDOW_DAYS,
)

REPO = "repos/rust"
OOW_START = "2026-07-01"
SAMPLE_SIZE = 200


def get_oow_commits():
    """Get non-merge OOW commits sampled evenly."""
    result = subprocess.check_output(
        ["git", "log", "--no-merges", "--format=%H|%ct|%aE",
         f"--since={OOW_START}", "HEAD"],
        cwd=REPO, text=True, timeout=60,
    ).strip().split("\n")
    
    commits = []
    for line in result:
        if "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        h, ts, email = parts[0], int(parts[1]), parts[2]
        commits.append((h, ts, email))
    
    # Sample every Nth
    if len(commits) > SAMPLE_SIZE:
        step = len(commits) // SAMPLE_SIZE
        commits = commits[::step][:SAMPLE_SIZE]
    
    return commits


def compute_label(commit_hash, commit_date, graph, label_window_days=7):
    """Check if commit is risky: files re-touched within label_window_days."""
    # Revert in subject
    if commit_hash in graph and "revert" in graph[commit_hash].get("subject", "").lower():
        return 1
    
    # Get files touched by this commit
    files = set()
    try:
        result = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash],
            cwd=REPO, text=True, timeout=10,
        )
        files = {f.strip() for f in result.strip().split("\n") if f.strip()}
    except Exception:
        pass
    
    if not files:
        return 0
    
    # Check if any file was re-touched within window
    cd = datetime.fromtimestamp(commit_date)
    for fp in files:
        # Find all touches of this file
        try:
            log_result = subprocess.check_output(
                ["git", "log", "--no-merges", "--format=%ct", "--follow",
                 f"--since={WINDOW_START}", "--until={FORWARD_LOOK_END}", "--", fp],
                cwd=REPO, text=True, timeout=10,
            )
            touch_dates = []
            for line in log_result.strip().split("\n"):
                if line.strip():
                    touch_dates.append(datetime.fromtimestamp(int(line.strip())))
            touch_dates.sort()
            
            # Check if any later touch is within window
            for td in touch_dates:
                if td > cd and (td - cd).days <= label_window_days:
                    return 1
        except Exception:
            pass
    
    return 0


def bootstrap_auc_ci(y_true, y_score, n_resamples=1000):
    """Bootstrap 95% CI for ROC-AUC, resampling ROWS."""
    from sklearn.metrics import roc_auc_score
    
    n = len(y_true)
    aucs = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_t = np.array(y_true)[idx]
        y_s = np.array(y_score)[idx]
        
        if len(np.unique(y_t)) < 2:
            continue
        aucs.append(roc_auc_score(y_t, y_s))
    
    if not aucs:
        return 0.5, 0.5, 0.5
    
    mean = np.mean(aucs)
    lo = np.percentile(aucs, 2.5)
    hi = np.percentile(aucs, 97.5)
    return mean, lo, hi


def main():
    from sklearn.metrics import roc_auc_score
    
    clear_cache()
    
    print("Building graph + caches...")
    t0 = time.time()
    graph, risky_hashes, sorted_graph = _get_full_graph(REPO)
    _precompute_author_prior(REPO)
    _ensure_walk_snapshots(REPO)
    print(f"Cold start: {time.time()-t0:.1f}s ({len(graph)} commits)")
    
    print(f"\nGetting OOW commits (since {OOW_START})...")
    commits = get_oow_commits()
    print(f"Sampled {len(commits)} commits")
    
    # Score each commit
    print("Scoring commits...")
    scores = []
    _hot_state.clear()
    
    for i, (h, ts, email) in enumerate(commits):
        cd = datetime.fromtimestamp(ts)
        
        t_start = time.time()
        try:
            feats = compute_single_commit_m1_features(
                REPO, h, cd, email, set(), 0, 0
            )
            # Use a simple score based on features (need model for real score)
            # For now, use a proxy: sum of file_prior_risky + file_revert_count
            score = (
                feats.get("file_prior_risky_max", 0) * 0.3 +
                feats.get("file_revert_count_max", 0) * 0.2 +
                feats.get("author_file_prior_commits_max", 0) * 0.1 +
                feats.get("file_prior_changes_max", 0) * 0.1 +
                feats.get("days_since_last_change_max", 0) * 0.01
            )
        except Exception as e:
            print(f"  ERROR on {h[:12]}: {e}")
            score = 0
        
        elapsed = time.time() - t_start
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(commits)}] {h[:12]} score={score:.3f} ({elapsed:.1f}s)")
        
        scores.append({
            "hash": h,
            "timestamp": ts,
            "score": score,
        })
    
    # Compute labels (expensive — use graph-based labeling)
    print("\nComputing labels (file re-touch within 7d)...")
    # For OOW commits, check if their files were re-touched later
    # This requires looking at commits AFTER each OOW commit
    labeled = []
    for i, s in enumerate(scores):
        h = s["hash"]
        ts = s["timestamp"]
        cd = datetime.fromtimestamp(ts)
        
        # Get files touched by this commit
        try:
            files_result = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h],
                cwd=REPO, text=True, timeout=10,
            )
            files = {f.strip() for f in files_result.strip().split("\n") if f.strip()}
        except Exception:
            files = set()
        
        # Check if any file was re-touched within 7 days
        is_risky = 0
        if "revert" in subprocess.check_output(
            ["git", "log", "-1", "--format=%s", h],
            cwd=REPO, text=True, timeout=10,
        ).strip().lower():
            is_risky = 1
        else:
            for fp in files:
                try:
                    later = subprocess.check_output(
                        ["git", "log", "--no-merges", "--format=%ct", 
                         f"--since={cd.strftime('%Y-%m-%d')}",
                         f"--until={(cd + timedelta(days=7)).strftime('%Y-%m-%d')}",
                         "--", fp],
                        cwd=REPO, text=True, timeout=10,
                    ).strip()
                    if later and any(l.strip() and int(l.strip()) > ts for l in later.split("\n") if l.strip()):
                        is_risky = 1
                        break
                except Exception:
                    pass
        
        labeled.append({**s, "label": is_risky})
        
        if (i + 1) % 50 == 0:
            risky_so_far = sum(1 for l in labeled if l["label"] == 1)
            print(f"  [{i+1}/{len(scores)}] risky so far: {risky_so_far}/{len(labeled)}")
    
    # Compute metrics
    y_true = [l["label"] for l in labeled]
    y_score = [l["score"] for l in labeled]
    
    pos_rate = np.mean(y_true)
    print(f"\n=== RUST OOW RESULTS ({len(labeled)} commits) ===")
    print(f"Positive rate: {pos_rate:.3f} ({sum(y_true)}/{len(y_true)})")
    
    if len(np.unique(y_true)) < 2:
        print("WARNING: Only one class present — cannot compute AUC")
    else:
        auc = roc_auc_score(y_true, y_score)
        mean, lo, hi = bootstrap_auc_ci(y_true, y_score, n_resamples=1000)
        print(f"ROC-AUC: {auc:.4f}  bootstrap mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    
    # Save results
    output = {
        "repo": "rust",
        "n_commits": len(labeled),
        "positive_rate": float(pos_rate),
        "scores": labeled,
    }
    if len(np.unique(y_true)) >= 2:
        output["roc_auc"] = float(auc)
        output["bootstrap_mean"] = float(mean)
        output["bootstrap_ci"] = [float(lo), float(hi)]
    
    Path("data/u68_rust_oow.json").write_text(json.dumps(output, indent=2))
    print(f"\nSaved to data/u68_rust_oow.json")


if __name__ == "__main__":
    main()
