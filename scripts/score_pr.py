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
import requests
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
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.utils._tags._TagsDict",
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


def get_risk_label(score: float, repo_name: str = "") -> str:
    """Determine risk label using percentile-based thresholds.

    Per-repo cutoffs from config.yaml; fallback to _global for unknown repos.
    high: top 10% of that repo's score distribution,
    medium: next 15%, low: bottom 75%.
    """
    config = load_config()
    thresholds = config.get("thresholds", {})
    repo_thresh = thresholds.get(repo_name, thresholds.get("_global", {}))
    high_cutoff = repo_thresh.get("high", 0.8619)
    medium_cutoff = repo_thresh.get("medium", 0.7536)

    if score >= high_cutoff:
        return "high"
    elif score >= medium_cutoff:
        return "medium"
    else:
        return "low"


def log_to_dashboard(issue_type: str, repo: str, details: str, 
                     commit_hash: str = "", risk_score: int = None):
    """Log an issue to the dashboard if DASHBOARD_URL is set.
    
    This function gracefully degrades if the dashboard is unavailable.
    It should never cause the gate to fail.
    """
    dashboard_url = os.environ.get("DASHBOARD_URL")
    if not dashboard_url:
        print("WARNING: DASHBOARD_URL not set, skipping dashboard logging")
        return
    
    try:
        response = requests.post(
            f"{dashboard_url}/issues",
            json={
                "gate": 2,
                "type": issue_type,
                "repo": repo,
                "details": details,
                "commit_hash": commit_hash,
                "risk_score": risk_score,
            },
            timeout=5  # Short timeout to avoid blocking
        )
        if response.status_code == 201:
            print(f"Logged issue to dashboard: {issue_type}")
        else:
            print(f"WARNING: Dashboard returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Failed to log to dashboard: {e}")


def get_notable_factors(features: dict, features_array: np.ndarray = None,
                         feature_columns: list[str] = None) -> list[str]:
    """Extract 3 most notable contributing factors using SHAP.

    Falls back to heuristic if SHAP is unavailable.
    """
    # Try SHAP first
    if features_array is not None and feature_columns is not None:
        try:
            from ml.explainer import explain, format_explanation
            factors = explain(features_array, feature_columns=feature_columns, top_k=3)
            return format_explanation(factors, features)
        except Exception:
            pass  # Fall through to heuristic

    # Heuristic fallback (same as before)
    factors = []
    files = features.get("files_touched", 0)
    if files > 10 or files > 5:
        factors.append(f"{files} files touched")

    lines_added = features.get("lines_added", 0)
    lines_deleted = features.get("lines_deleted", 0)
    total_lines = lines_added + lines_deleted
    if total_lines > 200 or total_lines > 100:
        factors.append(f"{total_lines} lines changed")

    hour = features.get("hour_of_day", 12)
    if hour < 6 or hour > 22:
        factors.append(f"changed at {hour}:00 (late night/early morning)")
    elif hour < 8:
        factors.append(f"changed at {hour}:00 (early morning)")

    prior_commits = features.get("author_prior_commits", 0)
    if prior_commits == 0:
        factors.append("first-time contributor")
    elif prior_commits < 5:
        factors.append(f"new contributor ({prior_commits} prior commits)")

    if features.get("is_fix_bug_revert", 0) == 1:
        factors.append("commit message contains fix/bug/revert keywords")

    dirs = features.get("dirs_touched", 0)
    if dirs > 5:
        factors.append(f"spans {dirs} directories")

    return factors[:3] if factors else ["no notable risk factors"]


def format_markdown(
    commit_hash: str,
    risk_score: float,
    risk_label: str,
    factors: list[str],
    author: str = "",
    explanations: list[dict] = None,
    touched_files_info: list[dict] = None,
) -> str:
    """Format the risk assessment as markdown.

    Args:
        commit_hash: commit SHA
        risk_score: model score (0-1)
        risk_label: low/medium/high
        factors: list of plain-language factor strings (legacy fallback)
        author: commit author name
        explanations: list of SHAP explanation dicts (preferred over factors)
        touched_files_info: list of dicts with file metadata for per-file table
    """
    # Emoji for risk level
    emoji = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]"}.get(risk_label, "[UNK]")

    author_text = f" by **{author}**" if author else ""

    # SHAP explanations section (preferred) or fallback to factors
    if explanations:
        reasons_lines = []
        for i, exp in enumerate(explanations, 1):
            reasons_lines.append(f"{i}. {exp.get('human_readable', exp.get('description', ''))}")
        reasons_text = "\n".join(reasons_lines)
    else:
        reasons_text = "\n".join(f"  - {f}" for f in factors)

    # Per-file table (if file metadata available)
    file_table = ""
    if touched_files_info:
        file_rows = []
        for fi in touched_files_info[:10]:  # Cap at 10 files
            name = fi.get("name", "unknown")
            # Truncate long paths
            if len(name) > 60:
                name = "..." + name[-57:]
            prior = fi.get("prior_changes", 0)
            reverts = fi.get("revert_count", 0)
            risk_marker = " !" if reverts > 0 else ""
            file_rows.append(f"| `{name}` | {prior} | {reverts}{risk_marker} |")
        file_table = (
            "\n### File History\n\n"
            "| File | Prior Changes | Reverts |\n"
            "|------|---------------|----------|\n"
            + "\n".join(file_rows)
            + ("\n\n_...and more files" if len(touched_files_info) > 10 else "")
        )

    markdown = f"""## Gatekeeper Risk Assessment{author_text}

| Metric | Value |
|--------|-------|
| **Band** | {emoji} **{risk_label.upper()}** |
| **Commit** | `{commit_hash[:12]}` |

### Why this score?
{reasons_text}
{file_table}

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
    repo_name = os.path.basename(repo_path) if repo_path else ""
    risk_label = get_risk_label(risk_score, repo_name)

    # Get notable factors (SHAP-based with heuristic fallback)
    factors = get_notable_factors(features, features_array=features_array, feature_columns=feature_columns)

    # Get SHAP explanations (structured)
    explanations = []
    try:
        from ml.explainer import explain, format_explanation as fmt_exp
        raw_factors = explain(features_array, feature_columns=feature_columns, top_k=3)
        human_readable = fmt_exp(raw_factors, features)
        explanations = [
            {**f, "human_readable": hr}
            for f, hr in zip(raw_factors, human_readable)
        ]
    except Exception:
        pass  # Non-fatal: use heuristic factors instead

    # Get author name
    author = features.get("author", "")

    # Get file metadata for per-file table
    touched_files_info = []
    try:
        touched_files = features.get("touched_files", "")
        if touched_files:
            file_list = [f.strip() for f in str(touched_files).split(",") if f.strip()]
            # Try to get file history from extractor
            if hasattr(extractor, "file_touches") and extractor.file_touches:
                for fp in file_list[:10]:
                    touches = extractor.file_touches.get(fp, [])
                    prior = len(touches)
                    reverts = sum(1 for _, _, msg in touches if "revert" in msg.lower()) if any(len(t) == 3 for t in touches) else 0
                    touched_files_info.append({
                        "name": fp,
                        "prior_changes": prior,
                        "revert_count": reverts,
                    })
            else:
                for fp in file_list[:10]:
                    touched_files_info.append({
                        "name": fp,
                        "prior_changes": 0,
                        "revert_count": 0,
                    })
    except Exception:
        pass

    # Generate markdown
    markdown = format_markdown(
        commit_hash, risk_score, risk_label, factors, author,
        explanations=explanations, touched_files_info=touched_files_info,
    )

    # Print markdown to stdout
    print("\n" + "=" * 60)
    print(markdown)
    print("=" * 60)

    # Log to dashboard if risk is medium or high
    if risk_label in ["medium", "high"]:
        repo_name = os.path.basename(repo_path) if repo_path else "unknown"
        log_to_dashboard(
            issue_type="high_risk_pr",
            repo=repo_name,
            details=f"Risk score: {risk_score:.4f}. Notable factors: {', '.join(factors)}",
            commit_hash=commit_hash,
            risk_score=int(risk_score * 100)
        )

    # Also write to a file for GitHub Actions to pick up
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"risk_score={risk_score:.4f}\n")
            f.write(f"risk_label={risk_label}\n")

    return 0


if __name__ == "__main__":
    exit(main())