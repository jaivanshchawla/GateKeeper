#!/usr/bin/env python3
"""
W.2: Backfill outcomes for 200+ commits per repo across 12 months.
Reports per-band precision with Wilson CIs, lift over base rate, and calibration.

Usage:
  python w2_backfill.py                     # all repos, saves to data/w2_results.json
  python w2_backfill.py --repo django       # single repo
  python w2_backfill.py --combine           # combine saved per-repo results
"""
import os
import sys
import json
import subprocess
import argparse
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

TARGET_COMMITS_PER_REPO = 200
WINDOW_DAYS = 7
BACKFILL_MONTHS = 12
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "w2_results.json")


def wilson_ci(successes, total, z=1.96):
    """Wilson score 95% CI for a proportion."""
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return p, max(0, center - spread), min(1, center + spread)


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
    trusted = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
               "numpy.dtype", "numpy.ndarray", "pandas.core.frame.DataFrame", "pandas.core.series.Series"]
    return sio.loads(open(model_path, "rb").read(), trusted=trusted)


def load_config():
    return yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")))


def sample_commits(rp, start_date, target):
    """Sample commits evenly across time."""
    r = subprocess.run(
        ["git", "log", f"--since={start_date}", "--no-merges", "--format=%H|%ct",
         f"--max-count={target * 3}"],
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

    if len(all_entries) > target:
        step = max(1, len(all_entries) // target)
        sampled = all_entries[::step][:target]
    else:
        sampled = all_entries[:target]
    return all_entries, sampled


def score_and_label(repo_name, rp, sampled, model, fcols, thresholds):
    """Score commits and compute realized outcomes."""
    from ml.extract_features import CommitFeatureExtractor
    from ml.single_commit_features import clear_cache

    # Build extractor ONCE (graph cached after first call)
    clear_cache()
    ext = CommitFeatureExtractor(repo_path=rp, since="2024-07-01", label_window_days=7)

    results = []
    scored = 0
    t0 = time.time()

    for i, (h, ts) in enumerate(sampled):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  ... {i}/{len(sampled)} scored ({elapsed:.0f}s)")

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

            # Realized outcome
            commit_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            window_end = commit_dt + timedelta(days=WINDOW_DAYS)

            # Revert check
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

            # File retouch check
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
            results.append({"hash": h, "ts": ts, "score": score, "band": band, "actual": actual})
            scored += 1

        except Exception as e:
            pass

    elapsed = time.time() - t0
    print(f"  Scored {scored} commits in {elapsed:.0f}s ({elapsed/max(scored,1):.1f}s/commit)")
    return results


def analyze_per_repo(repo_name, results):
    """Compute per-repo statistics."""
    bands = {"high": [], "medium": [], "low": []}
    for r in results:
        bands[r["band"]].append(r)

    base_rate = np.mean([r["actual"] for r in results]) if results else 0

    stats = {"repo": repo_name, "total": len(results), "base_rate": float(base_rate), "bands": {}}

    for band_name, band_results in bands.items():
        n = len(band_results)
        tp = sum(r["actual"] for r in band_results)
        prec, lo, hi = wilson_ci(tp, n)
        lift = prec / base_rate if base_rate > 0 else 0
        stats["bands"][band_name] = {
            "n": n, "tp": tp, "precision": float(prec),
            "ci_lo": float(lo), "ci_hi": float(hi), "lift": float(lift),
        }

    return stats


def print_repo_stats(stats):
    """Print formatted repo statistics."""
    print(f"\n  {stats['repo']}: {stats['total']} commits, base rate: {stats['base_rate']:.1%}")
    print(f"  {'Band':<8} {'N':>5} {'Risky':>6} {'Precision':>10} {'95% CI':>20} {'Lift':>8}")
    print(f"  {'─'*55}")
    for band in ["high", "medium", "low"]:
        b = stats["bands"].get(band, {})
        if b.get("n", 0) > 0:
            print(f"  {band:<8} {b['n']:>5} {b['tp']:>6} {b['precision']:>9.1%} [{b['ci_lo']:.1%}, {b['ci_hi']:.1%}] {b['lift']:>7.2f}x")


def analyze_pooled(all_results):
    """Compute pooled statistics across all repos."""
    base_rate = np.mean([r["actual"] for r in all_results]) if all_results else 0
    bands = {"high": [], "medium": [], "low": []}
    for r in all_results:
        bands[r["band"]].append(r)

    stats = {"repo": "POOLED", "total": len(all_results), "base_rate": float(base_rate), "bands": {}}

    for band_name, band_results in bands.items():
        n = len(band_results)
        tp = sum(r["actual"] for r in band_results)
        prec, lo, hi = wilson_ci(tp, n)
        lift = prec / base_rate if base_rate > 0 else 0
        stats["bands"][band_name] = {
            "n": n, "tp": tp, "precision": float(prec),
            "ci_lo": float(lo), "ci_hi": float(hi), "lift": float(lift),
        }

    return stats


def print_calibration(all_results):
    """Print 10-decile calibration table."""
    scores = np.array([r["score"] for r in all_results])
    actuals = np.array([r["actual"] for r in all_results])

    print(f"\n  Calibration (10 deciles):")
    print(f"  {'Decile':>8} {'Score Range':>20} {'Count':>6} {'Actual%':>8}")
    print(f"  {'─'*45}")
    for decile in range(10):
        lo = np.percentile(scores, decile * 10)
        hi = np.percentile(scores, (decile + 1) * 10)
        mask = (scores >= lo) & (scores < hi) if decile < 9 else (scores >= lo)
        if mask.sum() > 0:
            actual_rate = actuals[mask].mean()
            print(f"  {decile+1:>8} [{lo:.4f}, {hi:.4f}]  {mask.sum():>6} {actual_rate:>7.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="Run only this repo")
    parser.add_argument("--combine", action="store_true", help="Combine saved per-repo results")
    args = parser.parse_args()

    if args.combine:
        if not os.path.exists(RESULTS_FILE):
            print(f"No results file at {RESULTS_FILE}")
            return
        with open(RESULTS_FILE) as f:
            saved = json.load(f)

        all_results = []
        print("=" * 80)
        print("W.2: COMBINED RESULTS")
        print("=" * 80)

        for repo_name, results in saved.items():
            stats = analyze_per_repo(repo_name, results)
            print_repo_stats(stats)
            all_results.extend(results)

        pooled = analyze_pooled(all_results)
        print(f"\n{'=' * 60}")
        print_repo_stats(pooled)

        if len(set(r["actual"] for r in all_results)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score([r["actual"] for r in all_results], [r["score"] for r in all_results])
            print(f"\n  ROC-AUC on realized outcomes: {auc:.4f}")

        print_calibration(all_results)
        return

    model = load_model()
    config = load_config()
    fcols = config["feature_columns"]
    thresholds = config.get("thresholds", {})

    print("=" * 80)
    print(f"W.2: BACKFILL — {TARGET_COMMITS_PER_REPO} commits/repo, {BACKFILL_MONTHS} months, window={WINDOW_DAYS}d")
    print("=" * 80)

    # Load existing results if any
    saved = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            saved = json.load(f)

    repos_to_run = {args.repo: REPOS[args.repo]} if args.repo else REPOS

    for repo_name, rp_rel in repos_to_run.items():
        rp = os.path.join(os.path.dirname(__file__), "..", rp_rel)
        if not os.path.exists(rp):
            print(f"\n{repo_name}: SKIPPED (not cloned)")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {repo_name}")
        print(f"{'─' * 60}")

        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=BACKFILL_MONTHS * 30)).strftime("%Y-%m-%d")

        all_entries, sampled = sample_commits(rp, start_date, TARGET_COMMITS_PER_REPO)
        print(f"  Sampled {len(sampled)} commits from {len(all_entries)} available")

        results = score_and_label(repo_name, rp, sampled, model, fcols, thresholds)

        # Save per-repo
        saved[repo_name] = results
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(saved, f, indent=2)
        print(f"  Saved to {RESULTS_FILE}")

        stats = analyze_per_repo(repo_name, results)
        print_repo_stats(stats)

    print(f"\nDone. Run with --combine to see pooled results.")


if __name__ == "__main__":
    main()
