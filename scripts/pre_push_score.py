#!/usr/bin/env python3
"""
Gate 1: Pre-push hook that scores outgoing commits locally.

Reads incoming refs from stdin (git pre-push format), scores each new
commit being pushed, and prints the band + top SHAP reasons.

Warns on high risk but NEVER blocks by default. Set GATE1_BLOCK=1 to
make it blocking.

Must complete in under 2s for a typical push.
"""

import os
import subprocess
import sys
import time

import numpy as np
import skops.io as sio
import yaml

# Trusted types for skops deserialization
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.utils._tags._TagsDict",
    "numpy.dtype",
    "numpy.ndarray",
    "pandas.core.frame.DataFrame",
    "pandas.core.series.Series",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "config.yaml")


def get_new_commits():
    """Parse stdin for new commits being pushed (git pre-push format).

    Each line: <local ref> <local sha> <remote ref> <remote sha>
    We want commits between local sha and remote sha.
    """
    commits = []
    for line in sys.stdin:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        _local_ref, local_sha = parts[0], parts[1]
        _remote_ref, remote_sha = parts[2], parts[3] if len(parts) > 3 else "0" * 40

        # Skip delete pushes
        if local_sha == "0" * 40:
            continue

        # Get commits between remote and local (new commits being pushed)
        if remote_sha == "0" * 40:
            # New branch — get recent commits (limit to 10 to stay fast)
            result = subprocess.run(
                ["git", "log", "--oneline", "-10", "--format=%H", local_sha],
                capture_output=True, text=True, timeout=5
            )
        else:
            # Existing branch — get new commits
            result = subprocess.run(
                ["git", "log", "--oneline", "--format=%H", f"{remote_sha}..{local_sha}"],
                capture_output=True, text=True, timeout=5
            )

        if result.returncode == 0:
            new_hashes = [h.strip() for h in result.stdout.strip().split("\n") if h.strip()]
            commits.extend(new_hashes[:10])  # Cap at 10 commits

    return commits


def score_commit(model, feature_columns, thresholds, repo_path, commit_hash):
    """Score a single commit. Returns (score, label, reasons) or None on error."""
    try:
        from ml.extract_features import CommitFeatureExtractor
        extractor = CommitFeatureExtractor(repo_path=repo_path, since="2020-01-01")
        features = extractor.extract_single_commit(repo_path, commit_hash)
    except Exception:
        return None

    # Prepare feature array
    feature_values = [features.get(col, 0) for col in feature_columns]
    features_array = np.array([feature_values])

    # Predict
    risk_score = float(model.predict_proba(features_array)[0][1])

    # Label
    repo_name = os.path.basename(repo_path) if repo_path else ""
    repo_thresh = thresholds.get(repo_name, thresholds.get("_global", {}))
    high_cutoff = repo_thresh.get("high", 0.8619)
    medium_cutoff = repo_thresh.get("medium", 0.7536)
    if risk_score >= high_cutoff:
        risk_label = "high"
    elif risk_score >= medium_cutoff:
        risk_label = "medium"
    else:
        risk_label = "low"

    # SHAP reasons
    reasons = []
    try:
        from ml.explainer import explain, format_explanation
        raw_factors = explain(features_array, feature_columns=feature_columns, top_k=3)
        human_readable = format_explanation(raw_factors, features)
        reasons = human_readable
    except Exception:
        # Fallback: basic heuristics
        if features.get("files_touched", 0) > 10:
            reasons.append(f"{features['files_touched']} files touched")
        if features.get("author_prior_commits", 0) == 0:
            reasons.append("first-time contributor")
        if features.get("is_fix_bug_revert", 0) == 1:
            reasons.append("commit message contains fix/bug/revert keywords")
        reasons = reasons[:3]

    return risk_score, risk_label, reasons


def main():
    start = time.perf_counter()

    # Load model
    if not os.path.exists(MODEL_PATH):
        print("[Gate 1] Model not found, skipping scoring")
        return 0

    try:
        model = sio.loads(open(MODEL_PATH, "rb").read(), trusted=TRUSTED_TYPES)
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        feature_columns = config.get("feature_columns", [])
        thresholds = config.get("thresholds", {})
    except Exception as e:
        print(f"[Gate 1] Failed to load model: {e}")
        return 0

    # Determine repo root
    repo_path = os.getcwd()

    # Get new commits
    commits = get_new_commits()
    if not commits:
        return 0  # Nothing to score

    # Score each commit (limit to 5 most recent for speed)
    scored = 0
    high_count = 0
    for commit_hash in commits[:5]:
        result = score_commit(model, feature_columns, thresholds, repo_path, commit_hash)
        if result is None:
            continue

        risk_score, risk_label, reasons = result
        scored += 1
        short_hash = commit_hash[:8]

        # Band display
        band_emoji = {"low": "LOW", "medium": "MED", "high": "HIGH"}.get(risk_label, "UNK")

        # Print result
        reasons_text = "; ".join(reasons) if reasons else "no notable factors"
        print(f"  [{band_emoji}] {short_hash} — {reasons_text}")

        if risk_label == "high":
            high_count += 1

    elapsed_ms = (time.perf_counter() - start) * 1000

    if scored > 0:
        print(f"\n[Gate 1] Scored {scored} commit(s) in {elapsed_ms:.0f}ms")
        if high_count > 0:
            print(f"[Gate 1] WARNING: {high_count} commit(s) scored HIGH risk")
            print("[Gate 1] Review these changes carefully before pushing")
            # Check if blocking is enabled
            if os.environ.get("GATE1_BLOCK") == "1":
                print("[Gate 1] BLOCKED: Set GATE1_BLOCK=0 to allow this push")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
