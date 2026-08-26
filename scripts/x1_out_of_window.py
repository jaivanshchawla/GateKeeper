#!/usr/bin/env python3
"""
X.1: Re-run W.2 backfill using ONLY commits outside the training window.
"""
import os
import sys
import subprocess
import time
import yaml
import numpy as np
import skops.io as sio
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "rust": "repos/rust",
    "kubernetes": "repos/kubernetes",
    "kafka": "repos/kafka",
}

WINDOW_DAYS = 7
TRAINING_END = "2026-06-30"  # exclusive


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return p, max(0, center - spread), min(1, center + spread)


def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
    trusted = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
               "numpy.dtype", "numpy.ndarray", "pandas.core.frame.DataFrame", "pandas.core.series.Series"]
    model = sio.loads(open(model_path, "rb").read(), trusted=trusted)

    config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
    fcols = config["feature_columns"]
    thresholds = config.get("thresholds", {})

    print("=" * 80)
    print("X.1: OUT-OF-WINDOW BACKFILL")
    print(f"Training window ends: {TRAINING_END}")
    print("=" * 80)

    all_data = []

    for repo_name, rp_rel in REPOS.items():
        rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
        if not os.path.exists(rp):
            print(f"\n{repo_name}: SKIPPED (not cloned)")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {repo_name}")
        print(f"{'─' * 60}")

        # Get ALL non-merge commits AFTER training window
        r = subprocess.run(
            ["git", "log", f"--since={TRAINING_END}", "--no-merges", "--format=%H|%ct",
             "--max-count=500"],
            cwd=rp, capture_output=True, text=True, timeout=60,
        )
        all_entries = []
        for line in r.stdout.strip().split("\n"):
            if "|" in line:
                h, ts = line.split("|", 1)
                try:
                    all_entries.append((h.strip(), int(ts)))
                except ValueError:
                    pass

        print(f"  Available out-of-window: {len(all_entries)} commits")

        if len(all_entries) < 20:
            print(f"  TOO FEW commits for reliable statistics. Using all {len(all_entries)}.")
            sampled = all_entries
        else:
            # Sample up to 200
            step = max(1, len(all_entries) // min(200, len(all_entries)))
            sampled = all_entries[::step][:200]

        print(f"  Sampling {len(sampled)} commits")

        from ml.extract_features import CommitFeatureExtractor
        from ml.single_commit_features import clear_cache

        clear_cache()
        ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

        bands = {"high": [], "medium": [], "low": []}
        t0 = time.time()

        for i, (h, ts) in enumerate(sampled):
            if i % 50 == 0 and i > 0:
                print(f"  ... {i}/{len(sampled)} ({time.time()-t0:.0f}s)")

            try:
                feat = ext.extract_single_commit(rp, h)
                fv = [feat.get(c, 0) for c in fcols]
                score = float(model.predict_proba(np.array([fv]))[0][1])

                repo_thresh = thresholds.get(repo_name, thresholds.get("_global", {}))
                if score >= repo_thresh.get("high", 0.86):
                    band = "high"
                elif score >= repo_thresh.get("medium", 0.75):
                    band = "medium"
                else:
                    band = "low"

                # Realized outcome from git history
                commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                window_end = commit_dt + timedelta(days=WINDOW_DAYS)

                revert_r = subprocess.run(
                    ["git", "log", f"--since={commit_dt.isoformat()}",
                     f"--until={window_end.isoformat()}", "--format=%H|%s", "--all"],
                    cwd=rp, capture_output=True, text=True, timeout=15,
                )
                is_reverted = False
                for line in revert_r.stdout.strip().split("\n"):
                    if "|" in line:
                        rh, msg = line.split("|", 1)
                        if "revert" in msg.lower() and h[:8] in msg:
                            is_reverted = True
                            break

                files_r = subprocess.run(
                    ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h],
                    cwd=rp, capture_output=True, text=True, timeout=10,
                )
                commit_files = [f.strip() for f in files_r.stdout.strip().split("\n") if f.strip()][:5]

                is_retouched = False
                for fp in commit_files:
                    retouch_r = subprocess.run(
                        ["git", "log", f"--since={commit_dt.isoformat()}",
                         f"--until={window_end.isoformat()}", "--format=%H", "--", fp],
                        cwd=rp, capture_output=True, text=True, timeout=15,
                    )
                    for line in retouch_r.stdout.strip().split("\n"):
                        if line.strip() and line.strip()[:8] != h[:8]:
                            is_retouched = True
                            break
                    if is_retouched:
                        break

                actual = 1 if (is_reverted or is_retouched) else 0
                bands[band].append({"hash": h, "ts": ts, "score": score, "actual": actual})
                all_data.append((repo_name, score, band, actual))

            except Exception:
                pass

        elapsed = time.time() - t0
        print(f"  Scored {sum(len(b) for b in bands.values())} commits in {elapsed:.0f}s")

        base_rate = np.mean([d["actual"] for b in bands.values() for d in b]) if any(bands.values()) else 0
        print(f"\n  Base rate: {base_rate:.1%}")
        print(f"  {'Band':<8} {'N':>5} {'Risky':>6} {'Precision':>10} {'95% CI':>20} {'Lift':>8}")
        print(f"  {'─'*55}")
        for band in ["high", "medium", "low"]:
            b = bands[band]
            if b:
                tp = sum(d["actual"] for d in b)
                prec, lo, hi = wilson_ci(tp, len(b))
                lift = prec / base_rate if base_rate > 0 else 0
                print(f"  {band:<8} {len(b):>5} {tp:>6} {prec:>9.1%} [{lo:.1%}, {hi:.1%}] {lift:>7.2f}x")

    # Pooled
    print(f"\n{'=' * 60}")
    print("POOLED (all repos, out-of-window only)")
    print(f"{'=' * 60}")

    pooled = {"high": [], "medium": [], "low": []}
    for _, score, band, actual in all_data:
        pooled[band].append(actual)

    total_n = sum(len(b) for b in pooled.values())
    total_tp = sum(sum(b) for b in pooled.values())
    base_rate = total_tp / total_n if total_n > 0 else 0

    print(f"  Total: {total_n} commits, base rate: {base_rate:.1%}")
    print(f"  {'Band':<8} {'N':>5} {'Risky':>6} {'Precision':>10} {'95% CI':>20} {'Lift':>8}")
    print(f"  {'─'*55}")
    for band in ["high", "medium", "low"]:
        b = pooled[band]
        if b:
            tp = sum(b)
            prec, lo, hi = wilson_ci(tp, len(b))
            lift = prec / base_rate if base_rate > 0 else 0
            print(f"  {band:<8} {len(b):>5} {tp:>6} {prec:>9.1%} [{lo:.1%}, {hi:.1%}] {lift:>7.2f}x")

    # ROC-AUC
    scores = np.array([d[1] for d in all_data])
    actuals = np.array([d[3] for d in all_data])
    if len(set(actuals)) > 1:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(actuals, scores)
        print(f"\n  ROC-AUC on realized outcomes: {auc:.4f}")

    # Per-repo overlap report
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv"))
    train_hashes = set(df["hash"].str[:8])

    print(f"\n{'=' * 60}")
    print("OVERLAP REPORT")
    print(f"{'=' * 60}")
    for repo_name, rp_rel in REPOS.items():
        rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
        if not os.path.exists(rp):
            continue
        r = subprocess.run(
            ["git", "log", f"--since={TRAINING_END}", "--no-merges", "--format=%H", "--max-count=500"],
            cwd=rp, capture_output=True, text=True, timeout=30,
        )
        oow_hashes = set(l.strip()[:8] for l in r.stdout.strip().split("\n") if l.strip())
        repo_train = df[df["source_repo"] == repo_name]
        train_hashes_repo = set(repo_train["hash"].str[:8])
        overlap = len(oow_hashes & train_hashes_repo)
        print(f"  {repo_name}: {len(oow_hashes)} out-of-window, {overlap} in training ({overlap/max(len(oow_hashes),1)*100:.1f}% overlap)")


if __name__ == "__main__":
    main()
