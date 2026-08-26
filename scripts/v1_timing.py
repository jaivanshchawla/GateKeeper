#!/usr/bin/env python3
"""
V.1: Measure end-to-end scoring timing on real commits.
Measures extraction + model inference + rules, cold and warm cache.
"""
import os
import sys
import time
import subprocess
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPOS = {
    "django": "repos/django",
    "react": "repos/react",
    "rust": "repos/rust",
    "kubernetes": "repos/kubernetes",
    "kafka": "repos/kafka",
}

def get_recent_commits(repo_path: str, count: int) -> list[str]:
    """Get N recent commit hashes."""
    r = subprocess.run(
        ["git", "log", f"-{count}", "--format=%H"],
        cwd=repo_path, capture_output=True, text=True, timeout=30,
    )
    return [h.strip() for h in r.stdout.strip().split("\n") if h.strip()][:count]


def measure_scoring_time(repo_name: str, repo_path: str, commit_hashes: list[str]) -> dict:
    """Measure full scoring pipeline: extraction + model + rules."""
    from ml.extract_features import CommitFeatureExtractor
    import numpy as np
    import yaml
    import skops.io as sio

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
    trusted = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
               "numpy.dtype", "numpy.ndarray", "pandas.core.frame.DataFrame", "pandas.core.series.Series"]
    model = sio.loads(open(model_path, "rb").read(), trusted=trusted)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    feature_cols = config["feature_columns"]

    # Load rules
    from rules.engine import RuleEngine, load_config as load_rules_config
    engine = RuleEngine(load_rules_config())

    from rules.base import CommitContext
    from ml.pr_scoring import CommitScore, aggregate_commits_to_pr

    times = []
    for h in commit_hashes:
        t0 = time.time()
        try:
            extractor = CommitFeatureExtractor(repo_path=repo_path, since="2024-07-01", label_window_days=7)
            features = extractor.extract_single_commit(repo_path, h)

            feature_vals = [features.get(col, 0) for col in feature_cols]
            features_array = np.array([feature_vals])
            risk_score = float(model.predict_proba(features_array)[0][1])

            # Determine band
            thresholds = config.get("thresholds", {}).get(repo_name, config.get("thresholds", {}).get("_global", {}))
            if risk_score >= thresholds.get("high", 0.86):
                risk_label = "high"
            elif risk_score >= thresholds.get("medium", 0.75):
                risk_label = "medium"
            else:
                risk_label = "low"

            # Run rules
            rule_results = []
            try:
                ctx = CommitContext(
                    hash=h, author=features.get("author", ""),
                    message=features.get("commit_message", ""),
                    files=[f.strip() for f in str(features.get("touched_files", "")).split("|") if f.strip()],
                    lines_added=features.get("lines_added", 0),
                    lines_deleted=features.get("lines_deleted", 0),
                    files_touched=features.get("files_touched", 0),
                    dirs_touched=features.get("dirs_touched", 0),
                    risk_score=risk_score, risk_label=risk_label,
                )
                rule_results = engine.evaluate(ctx)
            except Exception:
                pass

            t1 = time.time()
            times.append(t1 - t0)
        except Exception as e:
            t1 = time.time()
            times.append(t1 - t0)
            print(f"  ERROR {h[:8]}: {e}")

    if not times:
        return {"p50": 0, "p95": 0, "max": 0, "min": 0, "mean": 0, "n": 0}

    times_sorted = sorted(times)
    p50_idx = int(len(times_sorted) * 0.5)
    p95_idx = int(len(times_sorted) * 0.95)

    return {
        "p50": times_sorted[p50_idx],
        "p95": times_sorted[min(p95_idx, len(times_sorted) - 1)],
        "max": max(times),
        "min": min(times),
        "mean": statistics.mean(times),
        "n": len(times),
    }


def main():
    print("=" * 80)
    print("V.1: END-TO-END SCORING TIMING (extraction + model + rules)")
    print("=" * 80)

    for repo_name, repo_path in REPOS.items():
        full_path = os.path.join(os.path.dirname(__file__), "..", repo_path)
        if not os.path.exists(full_path):
            print(f"\n{repo_name}: SKIPPED (repo not cloned)")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {repo_name}")
        print(f"{'─' * 60}")

        for count in [1, 5, 20]:
            commits = get_recent_commits(full_path, count)
            if len(commits) < count:
                print(f"  {count}-commit PR: only {len(commits)} available")
                continue

            # Cold cache
            from ml.single_commit_features import clear_cache
            clear_cache()
            stats = measure_scoring_time(repo_name, full_path, commits)
            print(f"  {count:>2}-commit PR (COLD): p50={stats['p50']:.3f}s  p95={stats['p95']:.3f}s  max={stats['max']:.3f}s  mean={stats['mean']:.3f}s")

            # Warm cache (run again — cache is now populated)
            stats = measure_scoring_time(repo_name, full_path, commits)
            print(f"  {count:>2}-commit PR (WARM): p50={stats['p50']:.3f}s  p95={stats['p95']:.3f}s  max={stats['max']:.3f}s  mean={stats['mean']:.3f}s")


if __name__ == "__main__":
    main()
