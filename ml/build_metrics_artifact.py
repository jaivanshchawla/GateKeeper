#!/usr/bin/env python3
"""Build the metric artifacts the API is allowed to serve.

The point of this script is that a number on the dashboard has a *provenance*:
some file, at some JSON pointer, whose bytes are hashed into the artifact. If a
figure cannot be traced this way it does not get served -- ``governance/``
claims are recorded as ``warnings`` instead of being copied into ``metrics``.

Two artifacts are produced:

``artifacts/model_metrics.json``
    Model identity (content digest, not a hand-written "v8"), the cross-repo
    LORO ROC-AUC from the last committed protocol run, and per-repo
    out-of-window ROC-AUC from the frozen evaluation files.

``artifacts/model_drift.json``
    A normalised view of the committed Evidently run, including a
    ``needs_retraining`` decision per repository and a warning when the source
    report contradicts itself (dataset drift flagged with zero drifted
    features).

Usage
-----
    python ml/build_metrics_artifact.py            # regenerate
    python ml/build_metrics_artifact.py --check    # fail if committed files differ

``--check`` is the CI invariant: it rebuilds from the committed inputs and
compares the result with the committed file, so a hand-edited artifact, or a
model retrained without regenerating, fails the build instead of shipping a
stale number. Comparison is on parsed content rather than bytes because a
Windows checkout rewrites line endings, which says nothing about the data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artifacts import SCHEMA_VERSION, digest_bytes, read_input, write

GENERATOR = "ml/build_metrics_artifact.py"

LORO_SOURCE = "data/p3_p4_results.json"
MANIFEST_SOURCE = "data/dataset_manifest.json"
CONFIG_SOURCE = "ml/config.yaml"
MODEL_SOURCE = "models/gatekeeper_risk_model.skops"
CARD_SOURCE = "governance/model_card.md"
DRIFT_SOURCE = "data/drift_results.json"

# Threshold at which a drift share is treated as actionable. 0.3 is the
# conventional "a third of features moved" cue; the committed run also reports
# dataset_drift directly, and either signal is enough.
DRIFT_SHARE_ALERT = 0.3

# A per-repo OOW sample below this size gets a warning: the AUC interval is
# wide enough that a two-point move is not a signal.
MIN_OOW_N = 200

_CARD_LORO = re.compile(
    r"ROC-AUC \(cross-repo LORO\)\s*\|\s*\*{0,2}([0-9]+(?:\.[0-9]+)?)"
)
_CARD_OOW = re.compile(
    r"ROC-AUC \(out-of-window[^|]*\)\s*\|\s*\*{0,2}([0-9]+(?:\.[0-9]+)?)"
)


# ── Inputs ──────────────────────────────────────────────────────────


def _read_bytes(relative: str) -> bytes:
    """Read an input as the *repository* holds it, not as this work tree does.

    `artifacts.read_input` prefers HEAD and normalises text line endings, so
    the artifact is a function of the commit: CI on Linux, a Windows checkout
    with CRLF files and a laptop with an uncommitted model rebuild the same
    bytes. Reading the work tree made all three disagree.
    """
    try:
        return read_input(relative)
    except FileNotFoundError:
        raise SystemExit(
            f"error: {relative} is required to build the metric artifact but is "
            "neither committed at HEAD nor present in the work tree. This file "
            "is normally committed; run `git status` to see whether it was "
            "deleted."
        ) from None


def _sha256(relative: str) -> str:
    return digest_bytes(relative, _read_bytes(relative))


def _json(relative: str) -> Any:
    try:
        return json.loads(_read_bytes(relative))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: {relative} is not valid JSON: {exc}") from exc


def _inputs(*relatives: str) -> dict[str, dict]:
    return {
        relative: {
            "sha256": _sha256(relative),
            "bytes": len(_read_bytes(relative)),
        }
        for relative in relatives
    }


# ── Model metrics ───────────────────────────────────────────────────


def _claimed_metrics() -> dict[str, float]:
    """Numbers the model card asserts, parsed so a mismatch can be reported."""
    text = _read_bytes(CARD_SOURCE).decode("utf-8", errors="replace")
    claims: dict[str, float] = {}
    match = _CARD_LORO.search(text)
    if match:
        claims["loro_roc_auc"] = float(match.group(1))
    match = _CARD_OOW.search(text)
    if match:
        claims["oow_auc_mean_excl_react"] = float(match.group(1))
    return claims


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_model_metrics() -> dict:
    manifest = _json(MANIFEST_SOURCE)
    repos = sorted((manifest.get("repos") or {}).keys())
    if not repos:
        raise SystemExit(f"error: {MANIFEST_SOURCE} lists no repositories")

    config = yaml.safe_load(_read_bytes(CONFIG_SOURCE))
    feature_columns = config.get("feature_columns") or []
    if not feature_columns:
        raise SystemExit(f"error: {CONFIG_SOURCE} has no feature_columns")

    per_repo_oow: dict[str, dict] = {}
    missing_oow: list[str] = []
    warnings: list[str] = []
    inputs = [LORO_SOURCE, MANIFEST_SOURCE, CONFIG_SOURCE, MODEL_SOURCE, CARD_SOURCE]

    for repo in repos:
        relative = f"data/w72_{repo}.json"
        if not (PROJECT_ROOT / relative).is_file():
            missing_oow.append(repo)
            continue
        payload = _json(relative)
        metrics = payload.get("metrics") or {}
        if "roc_auc" not in metrics:
            missing_oow.append(repo)
            continue
        inputs.append(relative)
        per_repo_oow[repo] = {
            "roc_auc": float(metrics["roc_auc"]),
            "n": int(metrics.get("n", payload.get("scored_count", 0))),
            "ci": [float(v) for v in (metrics.get("roc_auc_ci") or [])],
            "base_rate": float(metrics.get("base_rate", 0.0)),
        }

    if missing_oow:
        warnings.append(
            "No frozen out-of-window artifact for: "
            + ", ".join(missing_oow)
            + ". They are excluded from the OOW means; run the frozen evaluation "
            "for them before quoting OOW numbers."
        )
    if not per_repo_oow:
        raise SystemExit(
            "error: no data/w72_<repo>.json artifacts were readable, so there is "
            "no out-of-window metric to serve. Refusing to write an artifact "
            "with empty metrics."
        )

    for repo, metrics in per_repo_oow.items():
        if metrics["n"] and metrics["n"] < MIN_OOW_N:
            low, high = (metrics["ci"] + [None, None])[:2]
            warnings.append(
                f"{repo}: OOW n={metrics['n']} (<{MIN_OOW_N}); "
                f"AUC {metrics['roc_auc']:.3f} has a wide interval"
                + (f" [{low:.3f}, {high:.3f}]" if low is not None else "")
                + " — treat as directional, not a point estimate."
            )

    all_aucs = [m["roc_auc"] for m in per_repo_oow.values()]
    oow_mean = _mean(all_aucs)
    excl_react = _mean(
        [m["roc_auc"] for repo, m in per_repo_oow.items() if repo != "react"]
    )

    loro_payload = _json(LORO_SOURCE)
    loro = loro_payload.get("v7_loro_roc_auc")
    if loro is None:
        raise SystemExit(
            f"error: {LORO_SOURCE} has no v7_loro_roc_auc; cannot source the "
            "LORO metric. Update this generator when the protocol file changes."
        )

    claims = _claimed_metrics()
    for key, served in (
        ("loro_roc_auc", float(loro)),
        ("oow_auc_mean_excl_react", excl_react),
    ):
        claimed = claims.get(key)
        if claimed is not None and served is not None and abs(claimed - served) > 5e-4:
            warnings.append(
                f"{CARD_SOURCE} claims {key}={claimed:.4f}; {LORO_SOURCE} / frozen "
                f"OOW artifacts give {served:.4f} ({served - claimed:+.4f}). "
                "Serving the artifact value — update the card or commit the run "
                "that produced the claimed number."
            )

    model_bytes = _read_bytes(MODEL_SOURCE)
    model_sha = digest_bytes(MODEL_SOURCE, model_bytes)

    training_commits = sum(
        int((repo_data or {}).get("sampled_rows", 0))
        for repo_data in (manifest.get("repos") or {}).values()
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "model": {
            # A content digest cannot drift from the bytes it describes, which
            # is the entire reason "v8" was replaced.
            "version": f"skops-{model_sha[:12]}",
            "sha256": model_sha,
            "bytes": len(model_bytes),
            "path": MODEL_SOURCE,
        },
        "metrics": {
            "loro_roc_auc": float(loro),
            "loro_roc_auc_claimed": claims.get("loro_roc_auc"),
            "oow_auc_mean": oow_mean,
            "oow_auc_mean_excl_react": excl_react,
            "oow_auc_claimed": claims.get("oow_auc_mean_excl_react"),
            "n_features": len(feature_columns),
            "training_repos": len(repos),
            "training_commits": training_commits,
        },
        "per_repo_oow": per_repo_oow,
        "sources": {
            "/model/version": {
                "path": MODEL_SOURCE,
                "method": "sha256",
                "note": "first 12 hex characters of the model digest",
            },
            "/model/sha256": {"path": MODEL_SOURCE, "method": "sha256"},
            "/metrics/loro_roc_auc": {
                "path": LORO_SOURCE,
                "pointer": "/v7_loro_roc_auc",
            },
            "/metrics/loro_roc_auc_claimed": {
                "path": CARD_SOURCE,
                "method": "regex",
                "note": "headline metrics table",
            },
            "/metrics/oow_auc_mean": {
                "path": "data/w72_<repo>.json",
                "pointer": "/metrics/roc_auc",
                "method": "mean over per_repo_oow",
            },
            "/metrics/oow_auc_mean_excl_react": {
                "path": "data/w72_<repo>.json",
                "pointer": "/metrics/roc_auc",
                "method": "mean over per_repo_oow excluding react",
            },
            "/metrics/n_features": {
                "path": CONFIG_SOURCE,
                "pointer": "/feature_columns",
                "method": "length",
            },
            "/metrics/training_repos": {
                "path": MANIFEST_SOURCE,
                "pointer": "/repos",
                "method": "count",
            },
            "/metrics/training_commits": {
                "path": MANIFEST_SOURCE,
                "pointer": "/repos/<repo>/sampled_rows",
                "method": "sum",
            },
            "/per_repo_oow": {
                "path": "data/w72_<repo>.json",
                "pointer": "/metrics",
            },
        },
        "inputs": _inputs(*sorted(set(inputs))),
        "warnings": sorted(warnings),
    }
    return payload


# ── Drift ───────────────────────────────────────────────────────────


def build_model_drift() -> dict:
    source = _json(DRIFT_SOURCE)
    repos_in = source.get("repos") or {}
    if not repos_in:
        raise SystemExit(f"error: {DRIFT_SOURCE} has no per-repo results")

    repos_out: dict[str, dict] = {}
    contradictory: list[str] = []
    warnings: list[str] = []

    for repo in sorted(repos_in):
        entry = repos_in[repo] or {}
        drifted_features = list(entry.get("drifted_features") or [])
        drifted_count = int(entry.get("drifted_count", len(drifted_features)))
        drift_share = float(entry.get("drift_share", 0.0))
        dataset_drift = bool(entry.get("dataset_drift", False))

        report = entry.get("report_html")
        if report:
            relative = str(report).replace("\\", "/")
            # The source artifact was written on Windows; normalise so the API
            # cannot hand a backslash path to a Linux container.
            if not (PROJECT_ROOT / relative).is_file():
                relative = None
        else:
            relative = None

        if dataset_drift and drifted_count == 0 and drift_share > 0:
            contradictory.append(repo)

        repos_out[repo] = {
            "reference_rows": int(entry.get("reference_rows", 0)),
            "current_rows": int(entry.get("current_rows", 0)),
            "dataset_drift": dataset_drift,
            "drift_share": drift_share,
            "drifted_features": drifted_features,
            "drifted_count": drifted_count,
            "total_features": int(entry.get("total_features", 0)),
            "needs_retraining": bool(
                dataset_drift
                or drifted_count > 0
                or drift_share >= DRIFT_SHARE_ALERT
            ),
            "report": relative,
        }

    if contradictory:
        warnings.append(
            "The committed Evidently run flags dataset drift for "
            + ", ".join(contradictory)
            + " while reporting zero drifted features. The flag is served as "
            "reported (needs_retraining=true), but the source report is "
            "self-contradictory: regenerate it with "
            "`python systems/drift_detection.py` before treating drift as "
            "measured."
        )

    summary = {
        "repos_analyzed": len(repos_out),
        "repos_with_drift": sum(1 for r in repos_out.values() if r["dataset_drift"]),
        "needs_retraining": any(r["needs_retraining"] for r in repos_out.values()),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "source": {
            "path": DRIFT_SOURCE,
            "sha256": _sha256(DRIFT_SOURCE),
        },
        "evidently_generated_at": source.get("generated_at"),
        "repos": repos_out,
        "summary": summary,
        "inputs": _inputs(DRIFT_SOURCE),
        "warnings": warnings,
    }


# ── CLI ─────────────────────────────────────────────────────────────


BUILDERS = {
    "model_metrics": build_model_metrics,
    "model_drift": build_model_drift,
}


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _diff_paths(committed: Any, rebuilt: Any, prefix: str = "") -> list[str]:
    """Leaf paths that differ, so --check says *what* went stale."""
    if isinstance(committed, dict) and isinstance(rebuilt, dict):
        out: list[str] = []
        for key in sorted(set(committed) | set(rebuilt)):
            if key not in committed or key not in rebuilt:
                out.append(f"{prefix}/{key}")
            else:
                out.extend(_diff_paths(committed[key], rebuilt[key], f"{prefix}/{key}"))
        return out
    if committed != rebuilt:
        return [prefix or "/"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare rebuilt artifacts with the committed bytes; exit 1 if stale.",
    )
    args = parser.parse_args(argv)

    status = 0
    for name, builder in BUILDERS.items():
        payload = builder()
        path = PROJECT_ROOT / "artifacts" / f"{name}.json"

        if args.check:
            if not path.is_file():
                print(f"STALE {path.name}: not committed")
                status = 1
                continue
            committed = json.loads(path.read_text(encoding="utf-8"))
            if committed == payload:
                print(f"ok    {path.relative_to(PROJECT_ROOT).as_posix()}")
                continue
            status = 1
            differing = _diff_paths(committed, payload)
            print(
                f"STALE {path.relative_to(PROJECT_ROOT).as_posix()}: "
                f"{len(differing)} field(s) differ"
            )
            for field in differing[:15]:
                print(f"        {field}")
            if len(differing) > 15:
                print(f"        ... and {len(differing) - 15} more")
            continue

        write(name, payload)
        metrics = payload.get("metrics") or {}
        summary = (
            f"loro={metrics['loro_roc_auc']:.4f} oow={metrics['oow_auc_mean']:.4f}"
            if metrics
            else f"repos={len(payload.get('repos') or {})}"
        )
        print(
            f"wrote {path.relative_to(PROJECT_ROOT).as_posix()} "
            f"({payload['schema_version']}, {summary}, "
            f"{len(payload.get('warnings') or [])} warning(s))"
        )

    return status


if __name__ == "__main__":
    raise SystemExit(main())
