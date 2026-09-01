#!/usr/bin/env python3
"""U.6.7b: Re-score OOW commits with identity resolution ON.

Previous V5.1 ran while resolve() was hanging on Rust's self-cycles.
Now it's fixed. Re-score all repos and compute OOW ROC-AUC with
1000-resample row bootstrap CIs.
"""
import os, sys, subprocess, time, json, yaml
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import skops.io as sio

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "kafka": "repos/kafka",
    "kubernetes": "repos/kubernetes",
    "rust": "repos/rust",
}
TRAINING_END = "2026-06-30"
LABEL_WINDOW = 7
MAX_OOW = 200

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
trusted = [
    "collections.OrderedDict", "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier", "numpy.dtype", "numpy.ndarray",
    "pandas.core.frame.DataFrame", "pandas.core.series.Series",
]
model = sio.loads(open(model_path, "rb").read(), trusted=trusted)
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
fcols = config["feature_columns"]

def bootstrap_ci(scores, labels, n_resamples=1000, ci=0.95):
    """Bootstrap CI for ROC-AUC, resampling ROWS."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(42)
    scores_np = np.array(scores)
    labels_np = np.array(labels)
    n = len(labels_np)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(labels_np[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels_np[idx], scores_np[idx]))
    if not aucs:
        return 0.5, 0.5, 0.5
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return float(np.mean(aucs)), float(lo), float(hi)


def get_oow_commits(repo_path, max_count=MAX_OOW):
    """Get OOW commits (after training window) with their labels."""
    # Get commits after training end
    r = subprocess.run(
        ["git", "log", f"--since={TRAINING_END}", "--no-merges",
         "--format=%H|%ct|%aE", f"--max-count={max_count * 2}"],
        cwd=repo_path, capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace",
    )
    commits = []
    for line in r.stdout.strip().split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        h, ts, email = parts[0], int(parts[1]), parts[2]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        commits.append((h, dt, email))

    # Compute labels: risky if any file re-touched within LABEL_WINDOW days
    # For OOW, we need to check forward from the commit
    file_touches = defaultdict(list)
    for h, dt, email in commits:
        # Get files for this commit
        fr = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h],
            cwd=repo_path, capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        files = [f for f in fr.stdout.strip().split("\n") if f.strip()]
        for f in files:
            file_touches[f].append((h, dt))

    # Label risky: file re-touched within LABEL_WINDOW days
    risky_hashes = set()
    for touches in file_touches.values():
        if len(touches) < 2:
            continue
        touches.sort(key=lambda x: x[1])
        for i, (h_i, d_i) in enumerate(touches):
            if h_i in risky_hashes:
                continue
            for j in range(i + 1, len(touches)):
                h_j, d_j = touches[j]
                if (d_j - d_i).days <= LABEL_WINDOW:
                    risky_hashes.add(h_i)
                    break
                elif (d_j - d_i).days > LABEL_WINDOW:
                    break

    # Also check revert in subject
    for h, dt, email in commits:
        sr = subprocess.run(
            ["git", "log", "-1", "--format=%s", h],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        if "revert" in sr.stdout.strip().lower():
            risky_hashes.add(h)

    # Trim to max_count
    commits = commits[:max_count]
    return commits, risky_hashes


def main():
    from ml.extract_features import CommitFeatureExtractor

    results = {}
    for repo_name, rp_rel in REPOS.items():
        rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
        if not os.path.exists(rp):
            print(f"\n{repo_name}: SKIP (not cloned)")
            continue

        print(f"\n{'='*60}")
        print(f"  {repo_name}")
        print(f"{'='*60}")

        t0 = time.time()
        commits, risky_hashes = get_oow_commits(rp)
        dt = time.time() - t0
        print(f"  OOW commits: {len(commits)}, risky: {len([h for h, _, _ in commits if h in risky_hashes])}, time: {dt:.1f}s")

        # Score each commit
        scores = []
        labels = []
        extractor = CommitFeatureExtractor(rp, since="2024-07-01")

        for h, dt_commit, email in commits:
            try:
                features = extractor.extract_single_commit(rp, h)
                # Build feature vector
                feat_dict = {}
                for col in fcols:
                    feat_dict[col] = features.get(col, 0)
                feat_vector = [feat_dict[col] for col in fcols]
                score = model.predict_proba([feat_vector])[0][1]
                scores.append(score)
                labels.append(1 if h in risky_hashes else 0)
            except Exception as e:
                print(f"  ERROR {h[:12]}: {e}")

        if not scores:
            print(f"  No valid scores")
            continue

        scores = np.array(scores)
        labels = np.array(labels)
        pos_rate = labels.mean()

        # Bootstrap CI
        mean_auc, lo, hi = bootstrap_ci(scores, labels)
        from sklearn.metrics import roc_auc_score, average_precision_score
        point_auc = roc_auc_score(labels, scores)
        pr_auc = average_precision_score(labels, scores)

        print(f"  Base rate: {pos_rate:.3f}")
        print(f"  ROC-AUC: {point_auc:.4f} (bootstrap mean: {mean_auc:.4f} [{lo:.4f}, {hi:.4f}])")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  PR-AUC lift: {pr_auc - pos_rate:.4f}")

        results[repo_name] = {
            "n": len(scores),
            "base_rate": float(pos_rate),
            "roc_auc": float(point_auc),
            "roc_auc_bootstrap_mean": float(mean_auc),
            "roc_auc_ci_lo": float(lo),
            "roc_auc_ci_hi": float(hi),
            "pr_auc": float(pr_auc),
            "pr_auc_lift": float(pr_auc - pos_rate),
        }

    # Summary table
    print(f"\n{'='*80}")
    print("OOW ROC-AUC WITH IDENTITY RESOLUTION (1000-resample bootstrap)")
    print(f"{'='*80}")
    prior = {
        "django": (0.7547, 0.67, 0.83),
        "react": (0.5542, 0.43, 0.67),
        "kafka": (0.7363, 0.68, 0.79),
        "kubernetes": (0.8812, 0.83, 0.92),
        "rust": (0.7237, 0.67, 0.77),
    }
    print(f"{'Repo':<12s} {'Prior AUC':>10s} {'New AUC':>10s} {'CI':>20s} {'Δ':>8s} {'N':>6s}")
    print("-" * 70)
    for repo, r in results.items():
        p = prior.get(repo, (0, 0, 0))
        delta = r["roc_auc"] - p[0]
        ci_str = f"[{r['roc_auc_ci_lo']:.3f}, {r['roc_auc_ci_hi']:.3f}]"
        print(f"{repo:<12s} {p[0]:>10.4f} {r['roc_auc']:>10.4f} {ci_str:>20s} {delta:>+8.4f} {r['n']:>6d}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "u67_oow_identity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
