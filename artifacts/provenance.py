"""Prove that a payload's numbers came from artifacts, not from thin air.

A comment cannot stop the next person from writing ``"roc_auc": 0.7885`` into a
route handler. A *proof* can: :func:`prove` walks every leaf of a response and
refuses to accept a value that was not declared against a source, and refuses a
declared value that disagrees with the artifact it claims to come from.

Declarations are deliberately separate from the route code (see
``artifacts/claims.py``) so a reviewer can read the served surface in one
place, and so a test can assert the surface without importing FastAPI.

Path syntax
-----------
Paths use JSON-pointer style segments: ``/metrics/oow_auc_mean``,
``/per_repo_oow/django``, ``/0/id`` for a list element. A ``*`` segment matches
exactly one segment and binds it. Bindings are positional: the *n*-th ``*`` in
a declaration's source pointer receives the *n*-th bound segment, so
``/per_repo_oow/*`` can be proven from the artifact's own
``/per_repo_oow/*/roc_auc``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .store import Artifact, ArtifactError, load

# ── Declarations ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    """A value that must equal a pointer inside a committed artifact."""

    artifact: str
    pointer: str
    note: str = ""


@dataclass(frozen=True)
class Literal:
    """A value that is not a metric, declared so it cannot hide a metric.

    Used for constants such as the artifact filename in a response. The reason
    is required because "it was easier" is not one.
    """

    reason: str


@dataclass(frozen=True)
class Claimed:
    """A path pattern and the authority for every value matching it."""

    pattern: str
    claim: Source | Literal

    def __post_init__(self) -> None:
        if not isinstance(self.claim, (Source, Literal)):
            raise TypeError(
                f"{self.pattern}: claim must be Source(...) or Literal(...), got "
                f"{type(self.claim).__name__}. A bare string loses the distinction "
                "between 'measured somewhere' and 'a constant'."
            )
        if not self.pattern.startswith("/"):
            raise ValueError(f"claim pattern must start with '/': {self.pattern!r}")


def _segments(path: str) -> list[str]:
    if path in ("", "/"):
        return []
    if not path.startswith("/"):
        raise ValueError(f"path must start with '/': {path!r}")
    return path[1:].split("/")


def flatten(payload: Any, prefix: Sequence[str] = ()) -> Iterator[tuple[str, Any]]:
    """Yield ``(path, leaf)`` for every non-container value, depth first.

    Empty containers are leaves too: an unprovenanced ``{}`` in a metric
    response is still a claim that there is nothing to report.
    """
    if isinstance(payload, dict):
        if not payload:
            yield "/" + "/".join(prefix), payload
            return
        for key, value in payload.items():
            yield from flatten(value, (*prefix, str(key)))
    elif isinstance(payload, (list, tuple)):
        if not payload:
            yield "/" + "/".join(prefix), list(payload)
            return
        for index, value in enumerate(payload):
            yield from flatten(value, (*prefix, str(index)))
    else:
        yield "/" + "/".join(prefix), payload


def _match(pattern: str, path: str) -> list[str] | None:
    """Bind wildcards, or return None when the pattern does not match."""
    pattern_segments = _segments(pattern)
    path_segments = _segments(path)
    if len(pattern_segments) != len(path_segments):
        return None
    bindings: list[str] = []
    for expected, actual in zip(pattern_segments, path_segments):
        if expected == "*":
            bindings.append(actual)
        elif expected != actual:
            return None
    return bindings


def _substitute(pointer: str, bindings: Sequence[str]) -> str:
    """Replace the *n*-th ``*`` segment of a pointer with the *n*-th binding."""
    if not bindings:
        return pointer
    remaining = list(bindings)
    out: list[str] = []
    for segment in _segments(pointer):
        if segment == "*":
            if not remaining:
                raise ProveError(
                    f"pointer {pointer!r} has more wildcards than the path it "
                    f"was matched by ({len(bindings)})"
                )
            out.append(remaining.pop(0))
        else:
            out.append(segment)
    return "/" + "/".join(out)


# ── Problems ────────────────────────────────────────────────────────


class ProveError(Exception):
    """A declaration or pointer could not be used at all."""


@dataclass(frozen=True)
class UnprovenancedMetric:
    """A response leaf with no declared source."""

    endpoint: str
    path: str
    value: Any

    def __str__(self) -> str:
        return (
            f"{self.endpoint}{self.path} = {self.value!r} has no declared source. "
            "Add a Claimed(...) entry in artifacts/claims.py pointing at a "
            "committed artifact, or delete the field."
        )


@dataclass(frozen=True)
class MismatchedProvenance:
    """A declared value that disagrees with the artifact it cites."""

    endpoint: str
    path: str
    served: Any
    expected: Any
    source: Source

    def __str__(self) -> str:
        return (
            f"{self.endpoint}{self.path} serves {self.served!r} but "
            f"{self.source.artifact}{self.source.pointer} says {self.expected!r}"
        )


@dataclass(frozen=True)
class MissingSource:
    """The cited artifact or pointer cannot be read."""

    endpoint: str
    path: str
    source: Source
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.endpoint}{self.path} cites "
            f"{self.source.artifact}{self.source.pointer}, which cannot be read: "
            f"{self.detail}"
        )


Problem = UnprovenancedMetric | MismatchedProvenance | MissingSource


class ProvenanceError(Exception):
    """One or more leaves failed proof. Carries every problem, not just the first."""

    def __init__(self, endpoint: str, problems: Iterable[Problem]):
        self.endpoint = endpoint
        self.problems = list(problems)
        lines = [f"{len(self.problems)} provenance problem(s) in {endpoint}:"]
        lines.extend(f"  - {problem}" for problem in self.problems[:12])
        if len(self.problems) > 12:
            lines.append(f"  ... and {len(self.problems) - 12} more")
        super().__init__("\n".join(lines))


# ── Values ──────────────────────────────────────────────────────────


def _same(served: Any, expected: Any) -> bool:
    """Compare a served value with its source, tolerating float round-trips."""
    if isinstance(served, bool) or isinstance(expected, bool):
        return served is expected
    if isinstance(served, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(served, float) or isinstance(expected, float):
            return math.isclose(float(served), float(expected), rel_tol=1e-9, abs_tol=1e-12)
        return served == expected
    if isinstance(served, list) and isinstance(expected, list):
        return len(served) == len(expected) and all(
            _same(a, b) for a, b in zip(served, expected)
        )
    return served == expected


@dataclass(frozen=True)
class Proof:
    """What a successful proof covered -- useful in a test failure message."""

    endpoint: str
    leaves: int
    artifacts: tuple[str, ...]


def prove(
    endpoint: str,
    payload: Any,
    claims: Sequence[Claimed],
    *,
    artifacts: Mapping[str, Artifact] | None = None,
) -> Proof:
    """Assert every leaf in `payload` is declared and matches its artifact.

    Raises :class:`ProvenanceError` listing *all* problems. Collecting them
    beats raising on the first: a new endpoint usually has several unprovenanced
    fields at once, and fixing them one push at a time is how people give up
    and hardcode the numbers again.
    """
    loaded: dict[str, Artifact] = dict(artifacts or {})
    problems: list[Problem] = []
    leaves = 0

    for path, value in flatten(payload):
        leaves += 1
        binding: list[str] | None = None
        claim: Source | Literal | None = None

        for candidate in claims:
            matched = _match(candidate.pattern, path)
            if matched is not None:
                claim = candidate.claim
                binding = matched
                break

        if claim is None:
            problems.append(UnprovenancedMetric(endpoint, path, value))
            continue
        if isinstance(claim, Literal):
            continue

        if claim.artifact not in loaded:
            try:
                loaded[claim.artifact] = load(claim.artifact)
            except ArtifactError as exc:
                problems.append(MissingSource(endpoint, path, claim, str(exc)))
                continue

        try:
            pointer = _substitute(claim.pointer, binding or [])
            expected = loaded[claim.artifact].get(pointer)
        except (ArtifactError, ProveError) as exc:
            problems.append(MissingSource(endpoint, path, claim, str(exc)))
            continue

        if not _same(value, expected):
            problems.append(
                MismatchedProvenance(endpoint, path, value, expected, claim)
            )

    # Every cited artifact must exist even when its branch produced no leaves
    # (an empty per-repo map still claims those files are readable).
    for artifact_name in {c.claim.artifact for c in claims if isinstance(c.claim, Source)}:
        if artifact_name in loaded:
            continue
        try:
            loaded[artifact_name] = load(artifact_name)
        except ArtifactError as exc:
            problems.append(
                MissingSource(
                    endpoint,
                    "(declared but unused)",
                    Source(artifact_name, ""),
                    str(exc),
                )
            )

    if problems:
        raise ProvenanceError(endpoint, problems)

    return Proof(
        endpoint=endpoint,
        leaves=leaves,
        artifacts=tuple(sorted({c.claim.artifact for c in claims if isinstance(c.claim, Source)})),
    )


def prove_all(
    endpoint: str,
    payloads: Iterable[Any],
    claims: Sequence[Claimed],
) -> list[Proof]:
    """Prove several responses for one endpoint (list shapes vary by data)."""
    return [prove(endpoint, payload, claims) for payload in payloads]
