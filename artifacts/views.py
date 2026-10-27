"""The projections the API serves for model and drift health.

Route handlers return these dicts verbatim. That keeps the interesting part --
which artifact field feeds which response field -- in one file that a test can
call without standing up FastAPI, and leaves no room in the route for a literal
to appear.
"""

from __future__ import annotations

from typing import Any

from .store import list_artifacts, load

MODEL_METRICS = "model_metrics"
MODEL_DRIFT = "model_drift"


def model_health() -> dict[str, Any]:
    """Model identity and measured performance, or an error -- never a default."""
    artifact = load(MODEL_METRICS)
    metrics = artifact.payload["metrics"]

    stale = artifact.input_drift()
    return {
        # A content digest, so deploying a new model changes the version without
        # anyone remembering to bump a string.
        "version": artifact.payload["model"]["version"],
        "roc_auc": metrics["loro_roc_auc"],
        "roc_auc_claim": metrics.get("loro_roc_auc_claimed"),
        "oow_auc": metrics["oow_auc_mean"],
        "n_features": metrics["n_features"],
        "training_repos": metrics["training_repos"],
        "training_commits": metrics["training_commits"],
        "per_repo_oow": {
            repo: values["roc_auc"]
            for repo, values in artifact.payload["per_repo_oow"].items()
        },
        "model_sha256": artifact.payload["model"]["sha256"],
        "artifact": "artifacts/model_metrics.json",
        "artifact_stale": bool(stale),
        "warnings": list(artifact.payload.get("warnings") or []),
    }


_DRIFT_FIELDS = (
    "reference_rows",
    "current_rows",
    "dataset_drift",
    "drift_share",
    "drifted_features",
    "drifted_count",
    "total_features",
    "needs_retraining",
    "report",
)


def drift_status() -> dict[str, Any]:
    """Per-repository drift exactly as the committed Evidently run reported it."""
    artifact = load(MODEL_DRIFT)
    repos = artifact.payload["repos"]
    return {
        "generated_at": artifact.payload.get("evidently_generated_at"),
        "repos": {
            name: {field: values[field] for field in _DRIFT_FIELDS}
            for name, values in repos.items()
        },
        "summary": artifact.payload["summary"],
        "artifact": "artifacts/model_drift.json",
        "warnings": list(artifact.payload.get("warnings") or []),
    }


def artifact_manifest() -> dict[str, Any]:
    """Every committed artifact with its digest and whether its inputs moved."""
    artifacts = list_artifacts()
    return {
        "artifacts": artifacts,
        "stale": [a["name"] for a in artifacts if a["stale_inputs"]],
    }
