#!/usr/bin/env python3
"""
Y.1: Out-of-window ROC-AUC per repo with bootstrap CIs.
Uses ALL commits after training window end (2026-06-30), not a sample.
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

TRAINING_END = "2026-06-30"
WINDOW_DAYS = 7


def bootstrap_auc(scores, actuals, n_resamples=1000, seed=42):
    """Bootstrap 95% CI for ROC-AUC, resampling ROWS."""
    rng = np.random.RandomState(seed)
    from sklearn.metrics import roc_auc_score
    n = len(actuals)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        s, a = scores[idx], actuals[idx]
        if len(np.unique(a)) < 2:
            continue
        aucs.append(roc_auc_score(a, s))
    if not aucs:
        return 0.0, 0.0, 1.0
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return p, max(0, center - spread), min(1, center + spread)


def main():
    repo_filter = sys.argv[1] if len(sys.argv) > 1 else None

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
    trusted = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
               "numpy.dtype", "numpy.ndarray", "pandas.core.frame.DataFrame", "pandas.core.series.Series"]
    model = sio.loads(open(model_path, "rb").read(), trusted=trusted)

    config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))
    fcols = config["feature_columns"]
    thresholds = config.get("thresholds", {})

    print("=" * 80)
    print("Y.1: OUT-OF-WINDOW ROC-AUC (FULL SET)")
    print(f"Training window ends: {TRAINING_END}")
    print("=" * 80)

    all_repo_results = {}

    for repo_name, rp_rel in REPOS.items():
        if repo_filter and repo_name != repo_filter:
            continue

        rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
        if not os.path.exists(rp):
            print(f"\n{repo_name}: SKIPPED (not cloned)")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {repo_name}")
        print(f"{'─' * 60}")

        # Get ALL non-merge commits after training window
        r = subprocess.run(
            ["git", "log", f"--since={TRAINING_END}", "--no-merges", "--format=%H|%ct",
             "--max-count=2000"],
            cwd=rp, capture_output=True, text=True, timeout=60,
        )
        entries = []
        for line in r.stdout.strip().split("\n"):
            if "|" in line:
                h, ts = line.split("|", 1)
                try:
                    entries.append((h.strip(), int(ts)))
                except ValueError:
                    pass

        print(f"  Out-of-window commits: {len(entries)}")

        from ml.extract_features import CommitFeatureExtractor
        from ml.single_commit_features import clear_cache

        clear_cache()
        ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

        scores = []
        actuals = []
        t0 = time.time()

        for i, (h, ts) in enumerate(entries):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"  ... {i}/{len(entries)} scored ({elapsed:.0f}s)")

            try:
                feat = ext.extract_single_commit(rp, h)
                fv = [feat.get(c, 0) for c in fcols]
                score = float(model.predict_proba(np.array([fv]))[0][1])

                # Realized outcome
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
                scores.append(score)
                actuals.append(actual)

            except Exception:
                pass

        elapsed = time.time() - t0
        n = len(scores)
        print(f"  Scored {n} commits in {elapsed:.0f}s ({elapsed/max(n,1):.1f}s/commit)")

        if n == 0 or len(set(actuals)) < 2:
            print("  Cannot compute ROC-AUC (need both classes)")
            continue

        scores_arr = np.array(scores)
        actuals_arr = np.array(actuals)

        # ROC-AUC with bootstrap CIs
        from sklearn.metrics import roc_auc_score, average_precision_score
        mean_auc, lo, hi = bootstrap_auc(scores_arr, actuals_arr, n_resamples=1000)
        pr_auc = average_precision_score(actuals_arr, scores_arr)
        base_rate = actuals_arr.mean()

        # Per-band precision
        rt = thresholds.get(repo_name, thresholds.get("_global", {}))
        high_mask = scores_arr >= rt.get("high", 0.86)
        med_mask = (scores_arr >= rt.get("medium", 0.75)) & ~high_mask

        high_n = high_mask.sum()
        high_tp = actuals_arr[high_mask].sum() if high_n > 0 else 0
        high_prec, high_lo, high_hi = wilson_ci(int(high_tp), int(high_n))

        med_n = med_mask.sum()
        med_tp = actuals_arr[med_mask].sum() if med_n > 0 else 0
        med_prec, med_lo, med_hi = wilson_ci(int(med_tp), int(med_n))

        # 10-decile calibration
        print(f"\n  Results:")
        print(f"  N: {n}, Base rate: {base_rate:.1%}")
        print(f"  ROC-AUC: {mean_auc:.4f} [{lo:.4f}, {hi:.4f}] (1000-resample bootstrap)")
        print(f"  PR-AUC: {pr_auc:.4f} (lift: {pr_auc/base_rate:.2f}x)")
        print(f"  High band: {int(high_tp)}/{int(high_n)} = {high_prec:.1%} [{high_lo:.1%}, {high_hi:.1%}]")
        print(f"  Med band:  {int(med_tp)}/{int(med_n)} = {med_prec:.1%} [{med_lo:.1%}, {med_hi:.1%}]")

        print(f"\n  Calibration (10 deciles):")
        print(f"  {'Dec':>4} {'Range':>20} {'N':>5} {'Actual%':>8}")
        for decile in range(10):
            lo_p = np.percentile(scores_arr, decile * 10)
            hi_p = np.percentile(scores_arr, (decile + 1) * 10)
            mask = (scores_arr >= lo_p) & (scores_arr < hi_p) if decile < 9 else (scores_arr >= lo_p)
            if mask.sum() > 0:
                actual_rate = actuals_arr[mask].mean()
                print(f"  {decile+1:>4} [{lo_p:.4f}, {hi_p:.4f}]  {mask.sum():>5} {actual_rate:>7.1%}")

        all_repo_results[repo_name] = {
            "n": n, "base_rate": float(base_rate),
            "roc_auc": mean_auc, "ci_lo": lo, "ci_hi": hi,
            "pr_auc": float(pr_auc),
        }

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: Out-of-window vs Offline LORO")
    print(f"{'=' * 80}")

    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "commit_features.csv"))

    print(f"{'Repo':<12} {'N':>5} {'Base':>6} {'OOW ROC-AUC':>14} {'Offline LORO':>14} {'Gap':>8}")
    print("─" * 65)
    for repo_name in all_repo_results:
        r = all_repo_results[repo_name]
        repo_df = df[df["source_repo"] == repo_name]
        offline_n = len(repo_df)
        print(f"{repo_name:<12} {r['n']:>5} {r['base_rate']:>5.1%} {r['roc_auc']:>8.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] {'':>14}")


if __name__ == "__main__":
    main()
