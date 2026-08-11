#!/usr/bin/env python3
"""
Standalone script to score a single commit's risk.

Used by GitHub Actions (Gate 2) to evaluate PR commits.
No MLflow dependency - loads the standalone .skops model file directly.

Usage:
    python scripts/score_pr.py --repo-path <path> --commit-hash <hash>
    # Or in GitHub Actions (reads from environment):
    python scripts/score_pr.py
"""

import argparse
import os
import sys

import numpy as np
import skops.io as sio
import yaml

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.extract_features import CommitFeatureExtractor

# Trusted types for model deserialization
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "numpy.dtype",
    "numpy.ndarray",
    "pandas.core.frame.DataFrame",
    "pandas.core.series.Series",
]


def load_model(model_path: str = "models/gatekeeper_risk_model.skops"):
    """Load the standalone model file."""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    model = sio.loads(open(model_path, "rb").read(), trusted=TRUSTED_TYPES)
    return model


def load_config(config_path: str = "ml/config.yaml"):
    """Load feature configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_risk_label(score: float) -> str:
    """Determine risk label based on score thresholds."""
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    else:
        return "high"


def get_notable_factors(features: dict) -> list[str]:
    """Extract 2-3 most notable contributing factors from features."""
    factors = []

    # Files touched
    files = features.get("files_touched", 0)
    if files > 10:
        factors.append(f"{files} files touched")
    elif files > 5:
        factors.append(f"{files} files touched")

    # Lines changed
    lines_added = features.get("lines_added", 0)
    lines_deleted = features.get("lines_deleted", 0)
    total_lines = lines_added + lines_deleted
    if total_lines > 200:
        factors.append(f"{total_lines} lines changed")
    elif total_lines > 100:
        factors.append(f"{total_lines} lines changed")

    # Time of day
    hour = features.get("hour_of_day", 12)
    if hour < 6 or hour > 22:
        factors.append(f"changed at {hour}:00 (late night/early morning)")
    elif hour < 8:
        factors.append(f"changed at {hour}:00 (early morning)")

    # Author experience
    prior_commits = features.get("author_prior_commits", 0)
    if prior_commits == 0:
        factors.append("first-time contributor")
    elif prior_commits < 5:
        factors.append(f"new contributor ({prior_commits} prior commits)")

    # Fix/bug/revert keywords
    if features.get("is_fix_bug_revert", 0) == 1:
        factors.append("commit message contains fix/bug/revert keywords")

    # Directories touched
    dirs = features.get("dirs_touched", 0)
    if dirs > 5:
        factors.append(f"spans {dirs} directories")

    # Return top 3 factors
    return factors[:3] if factors else ["no notable risk factors"]


def format_markdown(
    commit_hash: str,
    risk_score: float,
    risk_label: str,
    factors: list[str],
    author: str = "",
) -> str:
    """Format the risk assessment as markdown."""
    # Emoji for risk level (use ASCII-safe characters for Windows compatibility)
    emoji = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]"}.get(risk_label, "[UNK]")

    # Format factors as bullet points
    factors_text = "\n".join(f"  - {f}" for f in factors)

    author_text = f" by **{author}**" if author else ""

    markdown = f"""## Gatekeeper Risk Assessment{author_text}

| Metric | Value |
|--------|-------|
| **Risk Score** | {risk_score:.4f} |
| **Risk Level** | {emoji} **{risk_label.upper()}** |
| **Commit** | `{commit_hash[:12]}` |

### Notable Factors
{factors_text}

---
*Scored by [Gatekeeper](https://github.com/jaivanshchawla/GateKeeper) - automated commit risk analysis*"""

    return markdown


def main():
    parser = argparse.ArgumentParser(description="Score a commit's risk level")
    parser.add_argument("--repo-path", help="Path to the git repository")
    parser.add_argument("--commit-hash", help="Commit hash to score")
    parser.add_argument("--model-path", default="models/gatekeeper_risk_model.skops")
    parser.add_argument("--config", default="ml/config.yaml")
    args = parser.parse_args()

    # Get repo path and commit hash from args or environment
    repo_path = args.repo_path or os.environ.get("GITHUB_WORKSPACE")
    commit_hash = args.commit_hash or os.environ.get("GITHUB_SHA")

    if not repo_path:
        print("ERROR: --repo-path or GITHUB_WORKSPACE required")
        sys.exit(1)

    if not commit_hash:
        print("ERROR: --commit-hash or GITHUB_SHA required")
        sys.exit(1)

    print(f"Scoring commit: {commit_hash[:12]}")
    print(f"Repository: {repo_path}")

    # Load model
    print("Loading model...")
    model = load_model(args.model_path)

    # Load config
    config = load_config(args.config)
    feature_columns = config.get("feature_columns", [])

    # Extract features
    print("Extracting features...")
    extractor = CommitFeatureExtractor(
        repo_path=repo_path,
        since="2020-01-01",  # We only need the specific commit
    )
    features = extractor.extract_single_commit(repo_path, commit_hash)

    # Prepare feature array
    feature_values = [features.get(col, 0) for col in feature_columns]
    features_array = np.array([feature_values])

    # Get prediction
    risk_score = float(model.predict_proba(features_array)[0][1])
    risk_label = get_risk_label(risk_score)

    # Get notable factors
    factors = get_notable_factors(features)

    # Get author name
    author = features.get("author", "")

    # Generate markdown
    markdown = format_markdown(commit_hash, risk_score, risk_label, factors, author)

    # Print markdown to stdout
    print("\n" + "=" * 60)
    print(markdown)
    print("=" * 60)

    # Also write to a file for GitHub Actions to pick up
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"risk_score={risk_score:.4f}\n")
            f.write(f"risk_label={risk_label}\n")

    return 0


if __name__ == "__main__":
    exit(main())
# TODO: fix this later
x = 42  # unused variable
def unused_function(): pass
