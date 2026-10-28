"""Artifact store: every metric the API serves must come from a committed file.

Why this exists
---------------
The dashboard used to report ``roc_auc: 0.7885``, ``oow_auc: 0.6196`` and
``training_repos: 5`` as literals inside the route handlers, and ``/drift``
returned ``dataset_drift: false`` for every repository while a committed drift
report said otherwise. Numbers like that cannot be checked, cannot be
regenerated, and silently survive a retrain that changes the model.

An *artifact* here is a JSON file committed next to the code, written by a
generator that records:

* ``sources`` -- for each served number, the file and JSON pointer it came from
* ``inputs``  -- the sha256 of every file the generator read
* ``warnings`` -- claims that could not be reconciled with an artifact

Reading through this module gives three guarantees the literals could not:

1. a missing artifact is an error, never a silent fallback to a default;
2. :func:`input_drift` names any input that changed since the artifact was
   built, so staleness is detectable without trusting the generator;
3. the model *file digest* is the version, so "v8" cannot outlive the bytes it
   was supposed to describe.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SCHEMA_VERSION = 1


class ArtifactError(Exception):
    """Base class for every failure to read a trustworthy artifact."""


class MissingArtifact(ArtifactError):
    """The artifact file is not committed, or the name is wrong."""


class InvalidArtifact(ArtifactError):
    """The file exists but is not a well-formed artifact of a known schema."""


class InputChanged(ArtifactError):
    """An input the artifact was built from no longer matches its recorded hash."""


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file's working-tree bytes, streamed for large blobs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Text inputs are hashed with normalised line endings. Git checks these files
# out with the platform's line endings, so hashing the raw bytes made an
# artifact valid on the machine that built it and stale on the other two CI
# runners -- a check that fails for the operating system, not the data.
TEXT_SUFFIXES = frozenset({".csv", ".html", ".json", ".md", ".txt", ".yaml", ".yml"})


def is_text_input(relative: str | Path) -> bool:
    """Whether an input is a text file whose line endings are incidental."""
    return Path(relative).suffix.lower() in TEXT_SUFFIXES


def canonical_text(data: bytes) -> bytes:
    """LF line endings, so a Windows and a Linux checkout hash identically."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def committed_bytes(relative: str) -> bytes | None:
    """A tracked input's content at HEAD, or None when it is not tracked there.

    An artifact is committed, so it has to describe the *committed* inputs. The
    first version read the working tree, which meant a locally modified model
    blob, or a file git had just checked out with CRLF, made the same artifact
    look stale everywhere except the machine that generated it. Reading HEAD
    makes the artifact a function of the commit, which is what a fresh
    checkout -- and therefore CI -- can reproduce.
    """
    result = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{relative}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def read_input(relative: str) -> bytes:
    """An input's content: committed at HEAD when tracked, else the work tree."""
    committed = committed_bytes(relative)
    if committed is not None:
        return committed
    return (PROJECT_ROOT / relative).read_bytes()


def digest_bytes(relative: str, data: bytes) -> str:
    """Hex sha256 of an input, with text line endings normalised."""
    if is_text_input(relative):
        data = canonical_text(data)
    return hashlib.sha256(data).hexdigest()


def input_sha256(relative: str) -> str:
    """The digest an artifact records for an input (see :func:`read_input`)."""
    return digest_bytes(relative, read_input(relative))


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def parse_pointer(pointer: str) -> list[str]:
    """Split an RFC 6901-ish JSON pointer into tokens.

    ``""`` and ``"/"`` both mean "the whole document". A trailing slash is
    rejected rather than silently ignored: ``/metrics/`` is a typo, and
    returning the metrics object for it would hide the typo.
    """
    if pointer in ("", "/"):
        return []
    if not pointer.startswith("/"):
        raise InvalidArtifact(
            f"JSON pointer must start with '/': {pointer!r}. "
            "Use '/metrics/roc_auc' style; an empty string means the document."
        )
    if pointer.endswith("/"):
        raise InvalidArtifact(f"JSON pointer has a trailing slash: {pointer!r}")
    return [_unescape(token) for token in pointer[1:].split("/")]


@dataclass(frozen=True)
class Artifact:
    """A loaded, schema-checked artifact file."""

    name: str
    path: Path
    payload: dict
    sha256: str

    def get(self, pointer: str = "") -> Any:
        """Resolve a JSON pointer inside the payload.

        Raises :class:`InvalidArtifact` with the failing token instead of
        returning ``None``: a mistyped pointer in a route handler must be a
        loud error, not a dash on the dashboard.
        """
        current: Any = self.payload
        walked: list[str] = []
        for token in parse_pointer(pointer):
            walked.append(token)
            if isinstance(current, list):
                try:
                    index = int(token)
                except ValueError as exc:
                    raise InvalidArtifact(
                        f"{self.name}: {token!r} is not a list index in "
                        f"/{'/'.join(walked[:-1])}"
                    ) from exc
                if not 0 <= index < len(current):
                    raise InvalidArtifact(
                        f"{self.name}: index {index} out of range in "
                        f"/{'/'.join(walked[:-1])} (length {len(current)})"
                    )
                current = current[index]
            elif isinstance(current, dict):
                if token not in current:
                    raise InvalidArtifact(
                        f"{self.name}: no key {token!r} under "
                        f"/{'/'.join(walked[:-1])}; available: "
                        f"{', '.join(sorted(current)[:8]) or '(empty)'}"
                    )
                current = current[token]
            else:
                raise InvalidArtifact(
                    f"{self.name}: /{'/'.join(walked[:-1])} is a "
                    f"{type(current).__name__}, cannot descend into {token!r}"
                )
        return current

    def source_of(self, pointer: str) -> dict:
        """Declared provenance for a served path, if the generator recorded one."""
        sources = self.payload.get("sources") or {}
        return sources.get(pointer, {})

    def input_drift(self) -> list[dict]:
        return input_drift(self)


_CACHE: dict[str, tuple[int, int, Artifact]] = {}
_CACHE_LOCK = threading.Lock()


def artifact_path(name: str) -> Path:
    """Path of an artifact, rejecting names that try to leave the directory."""
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise InvalidArtifact(
            f"artifact name must be a bare file stem, got {name!r} "
            "(e.g. 'model_metrics' for artifacts/model_metrics.json)"
        )
    return ARTIFACT_DIR / f"{name}.json"


def load(name: str, *, use_cache: bool = True) -> Artifact:
    """Load and schema-check an artifact.

    The cache key includes mtime and size, so a regenerated artifact is picked
    up without a process restart -- the failure mode a plain ``lru_cache`` on
    the name would introduce during development.
    """
    path = artifact_path(name)
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        available = ", ".join(sorted(p.stem for p in ARTIFACT_DIR.glob("*.json")))
        raise MissingArtifact(
            f"artifact {name!r} is not committed at {path}. "
            f"Run `python ml/build_metrics_artifact.py` to generate it. "
            f"Available artifacts: {available or '(none)'}"
        ) from exc

    key = str(path)
    signature = (stat.st_mtime_ns, stat.st_size)
    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached and cached[0] == signature[0] and cached[1] == signature[1]:
            return cached[2]

    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidArtifact(f"artifact {name!r} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise InvalidArtifact(
            f"artifact {name!r} must be a JSON object, got {type(payload).__name__}"
        )
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise InvalidArtifact(
            f"artifact {name!r} has schema_version {version!r}, "
            f"this code reads {SCHEMA_VERSION}. Regenerate it with "
            "`python ml/build_metrics_artifact.py`."
        )
    if not payload.get("generator"):
        raise InvalidArtifact(
            f"artifact {name!r} does not name its generator; an artifact whose "
            "origin is unknown cannot be trusted"
        )

    artifact = Artifact(
        name=name,
        path=path,
        payload=payload,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    if use_cache:
        with _CACHE_LOCK:
            _CACHE[key] = (signature[0], signature[1], artifact)
    return artifact


def write(name: str, payload: dict) -> Path:
    """Write an artifact deterministically.

    No timestamps are inserted here on purpose: regenerating from unchanged
    inputs must produce identical bytes, which is what makes
    ``build_metrics_artifact.py --check`` a usable CI invariant.
    """
    path = artifact_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    with _CACHE_LOCK:
        _CACHE.pop(str(path), None)
    return path


def input_drift(artifact: Artifact) -> list[dict]:
    """Inputs whose bytes no longer match what the artifact was built from.

    Empty means the artifact still describes the files on disk. A missing input
    is reported too -- a generator that read a file which has since been deleted
    produced a number nobody can reproduce.
    """
    problems: list[dict] = []
    for relative, recorded in (artifact.payload.get("inputs") or {}).items():
        expected = recorded.get("sha256") if isinstance(recorded, dict) else recorded
        try:
            actual = input_sha256(relative)
        except FileNotFoundError:
            actual = None
        if expected and actual != expected:
            problems.append({"input": relative, "expected": expected, "actual": actual})
    return problems


def list_artifacts() -> list[dict]:
    """Describe every committed artifact, for /meta/artifacts."""
    described = []
    for path in sorted(ARTIFACT_DIR.glob("*.json")):
        artifact = load(path.stem)
        described.append(
            {
                "name": artifact.name,
                "path": str(artifact.path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": artifact.sha256,
                "generator": artifact.payload.get("generator"),
                "inputs": sorted((artifact.payload.get("inputs") or {}).keys()),
                "stale_inputs": artifact.input_drift(),
                "warnings": artifact.payload.get("warnings") or [],
            }
        )
    return described
