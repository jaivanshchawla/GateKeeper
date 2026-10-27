"""Declarations: every leaf a metric response may contain, and its authority.

Read this file to know what the dashboard is allowed to say. ``tests/
test_artifact_truth.py`` runs :func:`artifacts.prove.prove` with these tuples,
so:

* a field added to a response without an entry here **fails the suite**;
* an entry citing the wrong artifact pointer **fails the suite**;
* a literal that disagrees with its artifact **fails the suite**.

The wildcard ``*`` matches one path segment and binds it; a ``*`` in the
source pointer receives the same binding, which is how
``/per_repo_oow/django`` is proven from the artifact's own
``/per_repo_oow/django/roc_auc``.
"""

from __future__ import annotations

from .provenance import Claimed, Literal, Source

_LITERAL_WARNING = Literal(
    "generator warning, served verbatim; the generator records its own sources"
)

MODEL_HEALTH: tuple[Claimed, ...] = (
    Claimed("/version", Source("model_metrics", "/model/version")),
    Claimed("/roc_auc", Source("model_metrics", "/metrics/loro_roc_auc")),
    # The model card's figure travels beside the artifact's, clearly labelled,
    # so a reader can see the gap rather than whichever number the route chose.
    Claimed("/roc_auc_claim", Source("model_metrics", "/metrics/loro_roc_auc_claimed")),
    Claimed("/oow_auc", Source("model_metrics", "/metrics/oow_auc_mean")),
    Claimed("/n_features", Source("model_metrics", "/metrics/n_features")),
    Claimed("/training_repos", Source("model_metrics", "/metrics/training_repos")),
    Claimed("/training_commits", Source("model_metrics", "/metrics/training_commits")),
    Claimed(
        "/per_repo_oow/*",
        Source("model_metrics", "/per_repo_oow/*/roc_auc"),
    ),
    Claimed("/model_sha256", Source("model_metrics", "/model/sha256")),
    Claimed(
        "/artifact",
        Literal("names the file this response was read from; not a measurement"),
    ),
    Claimed(
        "/artifact_stale",
        Literal("computed at request time by hashing the artifact's recorded inputs"),
    ),
    Claimed("/warnings", _LITERAL_WARNING),
    Claimed("/warnings/*", _LITERAL_WARNING),
)

_DRIFT_REPO = tuple(
    Claimed(f"/repos/*/{field}", Source("model_drift", f"/repos/*/{field}"))
    for field in (
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
) + (
    # A non-empty list's element paths are one segment deeper and need their own
    # claim; an empty list is covered by the container entry above.
    Claimed(
        "/repos/*/drifted_features/*",
        Source("model_drift", "/repos/*/drifted_features/*"),
    ),
)

DRIFT: tuple[Claimed, ...] = (
    Claimed(
        "/generated_at",
        Source("model_drift", "/evidently_generated_at"),
    ),
    *_DRIFT_REPO,
    Claimed("/summary/repos_analyzed", Source("model_drift", "/summary/repos_analyzed")),
    Claimed(
        "/summary/repos_with_drift",
        Source("model_drift", "/summary/repos_with_drift"),
    ),
    Claimed(
        "/summary/needs_retraining",
        Source("model_drift", "/summary/needs_retraining"),
    ),
    Claimed(
        "/artifact",
        Literal("names the file this response was read from; not a measurement"),
    ),
    Claimed("/warnings", _LITERAL_WARNING),
    Claimed("/warnings/*", _LITERAL_WARNING),
)

ENDPOINTS = {
    "/model/health": MODEL_HEALTH,
    "/drift": DRIFT,
}
