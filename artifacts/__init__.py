"""Committed artifacts and the machinery that proves a served number came from one.

The public surface is deliberately small:

    from artifacts import load, input_drift
    metrics = load("model_metrics")
    metrics.get("/metrics/oow_auc_mean")

    from artifacts.claims import MODEL_HEALTH
    from artifacts import prove
    prove("/model/health", payload, MODEL_HEALTH)

Everything else here exists to support those two calls. See `store.py` for why
this package exists at all.
"""

from .provenance import (
    Claimed,
    Literal,
    MismatchedProvenance,
    MissingSource,
    Proof,
    ProveError,
    ProvenanceError,
    Source,
    UnprovenancedMetric,
    flatten,
    prove,
    prove_all,
)
from .store import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    SCHEMA_VERSION,
    Artifact,
    ArtifactError,
    InputChanged,
    InvalidArtifact,
    MissingArtifact,
    input_drift,
    list_artifacts,
    load,
    sha256_file,
    write,
)

__all__ = [
    "ARTIFACT_DIR",
    "PROJECT_ROOT",
    "SCHEMA_VERSION",
    "Artifact",
    "ArtifactError",
    "Claimed",
    "InputChanged",
    "InvalidArtifact",
    "Literal",
    "MismatchedProvenance",
    "MissingArtifact",
    "MissingSource",
    "Proof",
    "ProveError",
    "ProvenanceError",
    "Source",
    "UnprovenancedMetric",
    "flatten",
    "input_drift",
    "list_artifacts",
    "load",
    "prove",
    "prove_all",
    "sha256_file",
    "write",
]
