"""
SHAP explanation engine for Gatekeeper.

Caches the TreeExplainer at startup. Provides top-3 contributing features
with direction and magnitude, phrased in plain language.

Latency: ~1-5ms per explanation after initial setup (TreeExplainer is fast
on LightGBM).
"""

import os

import numpy as np
import shap
import skops.io as sio

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

# Human-readable feature descriptions
FEATURE_DESCRIPTIONS = {
    "lines_added": "lines added",
    "lines_deleted": "lines deleted",
    "files_touched": "files touched",
    "dirs_touched": "directories touched",
    "author_prior_commits": "author's prior commits",
    "hour_of_day": "time of day",
    "day_of_week": "day of week",
    "commit_msg_length": "commit message length",
    "is_fix_bug_revert": "fix/bug/revert keywords",
    "file_prior_changes_max": "max file change count",
    "file_prior_changes_mean": "avg file change count",
    "file_prior_risky_max": "max prior risky changes to touched files",
    "file_prior_risky_mean": "avg prior risky changes to touched files",
    "file_revert_count_max": "max revert count for touched files",
    "file_revert_count_mean": "avg revert count for touched files",
    "file_age_days_max": "oldest touched file age (days)",
    "file_age_days_mean": "avg touched file age (days)",
    "churn_ratio": "churn ratio (deleted/added)",
    "change_entropy": "change spread across files",
    "max_file_churn": "largest single-file change",
    "is_test_only": "test-only change",
    "test_to_code_ratio": "test-to-code ratio",
    "config_touch": "touches config/CI files",
}

# Singleton: explainer and feature names cached at module load
_explainer = None
_feature_columns = None


def _logits_to_prob(logits: float) -> float:
    """Convert log-odds to probability via sigmoid."""
    return 1.0 / (1.0 + np.exp(-logits))


def _load_model_and_explainer(model_path: str = None):
    """Load the model and initialize SHAP TreeExplainer (cached)."""
    global _explainer, _feature_columns

    if _explainer is not None:
        return _explainer, _feature_columns

    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "gatekeeper_risk_model.skops"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = sio.loads(open(model_path, "rb").read(), trusted=TRUSTED_TYPES)

    # Load feature columns from config
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    _feature_columns = config.get("feature_columns", [])

    # Create background dataset (small, for TreeExplainer initialization)
    # Use zeros as background — SHAP values are relative, so this works
    # for LightGBM which handles missing values natively
    background = np.zeros((1, len(_feature_columns)))

    _explainer = shap.TreeExplainer(model, background)
    return _explainer, _feature_columns


def explain(features_array: np.ndarray, feature_columns: list[str] = None,
            top_k: int = 3, model_path: str = None) -> list[dict]:
    """Get SHAP-based explanations for a prediction.

    Args:
        features_array: (1, n_features) numpy array
        feature_columns: list of feature names (auto-loaded if None)
        top_k: number of top factors to return
        model_path: override model path (used by score_pr.py)

    Returns:
        List of dicts: [{"feature": str, "description": str, "shap_value": float,
                         "direction": "increases"|"decreases", "feature_value": float}]
    """
    explainer, columns = _load_model_and_explainer(model_path)

    if feature_columns is not None:
        columns = feature_columns

    # Compute SHAP values
    shap_values = explainer.shap_values(features_array)

    # For binary classification, shap_values may be a list of 2 arrays
    # or a single array. Handle both.
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 (risky) SHAP values
    else:
        sv = shap_values

    # sv shape: (n_samples, n_features) or (n_features,)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    row = sv[0]
    feat_vals = features_array[0]

    # Rank by absolute SHAP value
    indices = np.argsort(np.abs(row))[::-1][:top_k]

    results = []
    for idx in indices:
        col_name = columns[idx] if idx < len(columns) else f"feature_{idx}"
        desc = FEATURE_DESCRIPTIONS.get(col_name, col_name)
        shap_val = float(row[idx])
        feat_val = float(feat_vals[idx])

        results.append({
            "feature": col_name,
            "description": desc,
            "shap_value": shap_val,
            "direction": "increases" if shap_val > 0 else "decreases",
            "feature_value": feat_val,
        })

    return results


def format_explanation(factors: list[dict], features: dict = None) -> list[str]:
    """Format SHAP factors as plain-language strings.

    Args:
        factors: list of dicts from explain()
        features: original feature dict (for context in descriptions)

    Returns:
        List of human-readable strings.
    """
    if features is None:
        features = {}

    descriptions = []
    for f in factors:
        shap_val = f["shap_value"]
        feat_val = f["feature_value"]

        # Build contextual description
        if f["feature"] == "file_prior_changes_max":
            descriptions.append(
                f"Most-changed file has {int(feat_val)} prior modifications "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )
        elif f["feature"] == "file_revert_count_max":
            descriptions.append(
                f"File has been reverted {int(feat_val)} time(s) "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )
        elif f["feature"] == "file_prior_risky_max":
            descriptions.append(
                f"File has {int(feat_val)} prior risky change(s) "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )
        elif f["feature"] == "author_prior_commits":
            if feat_val == 0:
                descriptions.append(
                    f"First-time contributor ({'elevates' if shap_val > 0 else 'lowers'} risk)"
                )
            else:
                descriptions.append(
                    f"Author has {int(feat_val)} prior commits "
                    f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
                )
        elif f["feature"] == "files_touched":
            descriptions.append(
                f"Touches {int(feat_val)} files "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )
        elif f["feature"] == "hour_of_day":
            descriptions.append(
                f"Committed at hour {int(feat_val)} "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )
        elif f["feature"] == "is_fix_bug_revert":
            if feat_val == 1:
                descriptions.append(
                    f"Commit message contains fix/bug/revert keywords "
                    f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
                )
            else:
                descriptions.append(
                    f"No fix/bug/revert keywords in message "
                    f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
                )
        elif f["feature"] == "config_touch":
            descriptions.append(
                f"{'Touches' if feat_val == 1 else 'Does not touch'} config/CI files "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )
        elif f["feature"] == "churn_ratio":
            descriptions.append(
                f"Churn ratio {feat_val:.2f} (deleted vs added) "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )
        elif f["feature"] == "change_entropy":
            descriptions.append(
                f"Change entropy {feat_val:.2f} (spread across files) "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )
        else:
            descriptions.append(
                f"{f['description']}: {feat_val:.1f} "
                f"({'elevates' if shap_val > 0 else 'lowers'} risk)"
            )

    return descriptions
