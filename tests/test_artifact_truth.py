#!/usr/bin/env python3
"""Prove the dashboard's numbers exist, and that they are the right numbers.

The regression this file exists for: ``/model/health`` used to return
``"roc_auc": 0.7885`` and ``/drift`` used to return ``dataset_drift: false`` for
every repository -- literals in the route handlers, while the committed
artifacts said something else. A test that compares the endpoint with itself
cannot catch that, so these tests compare each *leaf of the response* with the
artifact it claims to come from, and fail when a leaf has no such claim.

What is asserted
----------------
* every leaf of every modelled endpoint is declared in ``artifacts/claims.py``
  and equals its artifact pointer (``test_*_is_proven``);
* an undeclared field or a drifted literal fails (the proof layer works);
* the artifacts are reproducible byte-for-byte from the committed inputs
  (``ml/build_metrics_artifact.py --check``);
* the route handlers and the projection contain no numeric literals, so a
  number cannot be reintroduced without a claim;
* a missing artifact is an error rather than a default.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifacts import (
    ArtifactError,
    Claimed,
    Literal,
    MismatchedProvenance,
    MissingArtifact,
    ProvenanceError,
    Source,
    load,
    prove,
)
from artifacts import claims as artifact_claims
from artifacts import store as artifact_store
from artifacts.views import drift_status, model_health

# ── The served payloads are proven against their artifacts ──────────


def test_model_health_is_proven():
    """Every leaf of /model/health equals the artifact pointer it cites."""
    proof = prove("/model/health", model_health(), artifact_claims.MODEL_HEALTH)
    assert proof.leaves >= 10
    assert proof.artifacts == ("model_metrics",)


def test_drift_status_is_proven():
    """Every leaf of /drift equals the artifact pointer it cites."""
    proof = prove("/drift", drift_status(), artifact_claims.DRIFT)
    # 5 repos x 9 fields, plus summary and warnings: enough that a silently
    # dropped branch would show up here.
    assert proof.leaves >= 45


def test_every_declared_endpoint_has_claims():
    assert set(artifact_claims.ENDPOINTS) == {"/model/health", "/drift"}
    for endpoint, claims in artifact_claims.ENDPOINTS.items():
        assert claims, f"{endpoint} has no claims"


# ── The proof layer actually rejects things ─────────────────────────


def test_undeclared_field_is_rejected():
    """A number added without a source is exactly the bug this layer prevents."""
    payload = model_health()
    payload["sneaky_metric"] = 0.99
    with pytest.raises(ProvenanceError) as excinfo:
        prove("/model/health", payload, artifact_claims.MODEL_HEALTH)
    problems = excinfo.value.problems
    assert any(
        getattr(problem, "path", "") == "/sneaky_metric" for problem in problems
    ), problems


def test_literal_that_disagrees_with_artifact_is_rejected():
    """The old hardcoded 0.7885 must fail against the artifact's 0.7784."""
    payload = model_health()
    payload["roc_auc"] = 0.7885
    with pytest.raises(ProvenanceError) as excinfo:
        prove("/model/health", payload, artifact_claims.MODEL_HEALTH)
    mismatches = [
        p for p in excinfo.value.problems if isinstance(p, MismatchedProvenance)
    ]
    assert len(mismatches) == 1
    assert mismatches[0].path == "/roc_auc"
    assert mismatches[0].expected == pytest.approx(0.7783761339784208)


def test_all_problems_are_reported_not_just_the_first():
    payload = model_health()
    payload["a"] = 1
    payload["b"] = 2
    with pytest.raises(ProvenanceError) as excinfo:
        prove("/model/health", payload, artifact_claims.MODEL_HEALTH)
    paths = {getattr(problem, "path", "") for problem in excinfo.value.problems}
    assert {"/a", "/b"} <= paths


def test_claim_must_be_source_or_literal():
    """A bare string would erase the difference between measured and constant."""
    with pytest.raises(TypeError):
        Claimed("/x", "because")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Claimed("x", Literal("no leading slash"))


def test_wildcard_binds_to_the_same_key_in_the_source():
    payload = {"per_repo_oow": {"django": 0.66, "rust": 0.5743}}
    # Written against the real artifact so the binding is exercised end to end.
    claims = (Claimed("/per_repo_oow/*", Source("model_metrics", "/per_repo_oow/*/roc_auc")),)
    assert prove("/x", payload, claims).leaves == 2


def test_pointer_typo_is_a_missing_source_not_a_zero():
    claims = (Claimed("/roc_auc", Source("model_metrics", "/metrics/does_not_exist")),)
    with pytest.raises(ProvenanceError) as excinfo:
        prove("/x", {"roc_auc": 0.5}, claims)
    assert "does_not_exist" in str(excinfo.value)


# ── The artifacts describe the files on disk ────────────────────────


def test_committed_artifacts_have_no_stale_inputs():
    for name in ("model_metrics", "model_drift"):
        artifact = load(name)
        drift = artifact.input_drift()
        assert drift == [], (
            f"{name} cites inputs that changed since it was built: {drift}. "
            "Run `python ml/build_metrics_artifact.py` and commit the result."
        )


def test_artifacts_regenerate_identically():
    """Staleness is checked by rebuilding, not by trusting a timestamp.

    If a model is retrained, a drift report regenerated, or a card edited
    without rebuilding the artifact, this fails with the fields that differ.
    """
    result = subprocess.run(
        [sys.executable, "ml/build_metrics_artifact.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, (
        "artifacts are stale:\n" + (result.stdout or "") + (result.stderr or "")
    )


def test_text_inputs_hash_the_same_with_crlf_and_lf(monkeypatch, tmp_path):
    """CI checks these files out with different line endings than Windows does.

    Hashing the raw bytes made the committed artifact match the machine that
    built it and fail on two of the three runners, for a reason that had
    nothing to do with the data.
    """
    monkeypatch.setattr(artifact_store, "PROJECT_ROOT", tmp_path)
    (tmp_path / "lf.json").write_bytes(b'{"metric": 1}\n')
    (tmp_path / "crlf.json").write_bytes(b'{"metric": 1}\r\n')

    assert artifact_store.input_sha256("lf.json") == artifact_store.input_sha256(
        "crlf.json"
    )


def test_binary_inputs_are_hashed_byte_for_byte(monkeypatch, tmp_path):
    """Normalisation is for text; a model blob's bytes are the model."""
    monkeypatch.setattr(artifact_store, "PROJECT_ROOT", tmp_path)
    (tmp_path / "model.bin").write_bytes(b"\r\n")
    assert artifact_store.input_sha256("model.bin") == artifact_store.sha256_file(
        tmp_path / "model.bin"
    )


def test_tracked_inputs_are_read_from_the_commit(monkeypatch, tmp_path):
    """A dirty work tree must not change what the artifact claims to describe.

    The repository's model blob is modified on a developer machine (retrained,
    not committed). Hashing the work tree made the committed artifact stale in
    CI while passing locally -- the exact inversion of a useful check.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "data.json").write_text('{"metric": 1}\n', encoding="utf-8")

    def git(*args):
        subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )

    git("init", "-q")
    git("add", "data.json")
    git(
        "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "-m", "fixture",
    )

    monkeypatch.setattr(artifact_store, "PROJECT_ROOT", repo)
    committed = artifact_store.input_sha256("data.json")

    (repo / "data.json").write_text('{"metric": 999}\n', encoding="utf-8")
    assert artifact_store.input_sha256("data.json") == committed
    assert artifact_store.sha256_file(repo / "data.json") != committed


def test_stale_input_is_detected(monkeypatch, tmp_path):
    """The detector must notice a changed input, not just a missing one."""
    (tmp_path / "input.txt").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(artifact_store, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(artifact_store, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(artifact_store, "_CACHE", {})
    artifact_store.write(
        "synthetic",
        {
            "schema_version": 1,
            "generator": "tests",
            "inputs": {"input.txt": {"sha256": "0" * 64}},
        },
    )

    drift = load("synthetic", use_cache=False).input_drift()
    assert [entry["input"] for entry in drift] == ["input.txt"]
    assert drift[0]["actual"] != drift[0]["expected"]


def test_missing_artifact_names_the_fix(monkeypatch, tmp_path):
    monkeypatch.setattr(artifact_store, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(artifact_store, "_CACHE", {})
    with pytest.raises(MissingArtifact) as excinfo:
        load("model_metrics", use_cache=False)
    message = str(excinfo.value)
    assert "build_metrics_artifact.py" in message
    assert "not committed" in message


def test_artifact_name_cannot_escape_the_directory():
    with pytest.raises(ArtifactError):
        artifact_store.artifact_path("../secrets")


def test_wrong_schema_version_is_refused(monkeypatch, tmp_path):
    path = tmp_path / "future.json"
    path.write_text('{"schema_version": 99, "generator": "x"}', encoding="utf-8")
    monkeypatch.setattr(artifact_store, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(artifact_store, "_CACHE", {})
    with pytest.raises(ArtifactError) as excinfo:
        load("future", use_cache=False)
    assert "schema_version" in str(excinfo.value)


# ── No route may reintroduce a literal ──────────────────────────────


def _numeric_literals(function: ast.AST) -> list[float | int]:
    """Numbers that could be a metric.

    HTTP status codes are the one int allowed: they are protocol, not
    measurement. A float is never allowed -- every float in these routes used
    to be a fabricated score -- and an int outside 100..599 is treated as a
    metric that lost its claim.
    """
    found: list[float | int] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                continue
            if isinstance(node.value, int) and 100 <= node.value <= 599:
                continue
            found.append(node.value)
    return found


def test_projection_module_has_no_numeric_literals():
    """views.py may only *read* numbers; a literal there is a fabricated metric."""
    tree = ast.parse((REPO_ROOT / "artifacts" / "views.py").read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            literals = _numeric_literals(node)
            if literals:
                offenders.append(f"{node.name}: {literals}")
    assert offenders == [], (
        "artifacts/views.py must project artifact values, not compute new ones: "
        + "; ".join(offenders)
    )


def test_metric_routes_contain_no_numeric_literals():
    tree = ast.parse((REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8"))
    wanted = {"model_health", "drift_status", "artifact_manifest"}
    seen = set()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name in wanted:
            seen.add(node.name)
            literals = _numeric_literals(node)
            if literals:
                offenders.append(f"{node.name}: {literals}")
    assert seen == wanted, f"routes renamed or removed: missing {wanted - seen}"
    assert offenders == [], (
        "these routes must return artifact projections, not literals: "
        + "; ".join(offenders)
    )


def test_no_hardcoded_metric_values_remain_in_the_api():
    """The specific numbers that used to be served, as a belt-and-braces check."""
    source = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    for literal in ("0.7885", "0.6196", '"v8"'):
        assert literal not in source, (
            f"{literal} is a stale claim; serve it from an artifact instead"
        )
