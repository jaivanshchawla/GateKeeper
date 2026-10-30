#!/usr/bin/env python3
"""Add a repository to the corpus from nothing but its clone URL.

The registry made a repository one line of YAML, but *getting* that line was
still an afternoon: clone it, mine it, eyeball the class balance, remember
which file the entry goes in, and hope the pipeline does not choke on the new
history later. This module is that afternoon as one command::

    python -m pipelines.onboard https://github.com/owner/repo.git

It runs five stages and stops at the first one that fails:

1. **resolve**  -- derive a registry name and clone directory from the URL.
2. **clone**    -- check the repository out (or reuse an existing checkout).
3. **score**    -- score a sample of real commits with the production model,
                   and measure each one against the outcome it actually had
                   (was a touched file changed again, or the commit reverted,
                   within the label window?).
4. **gate**     -- decide from those measurements whether the history is
                   usable as corpus. A gate failure is a refusal to register,
                   not a warning: a repository whose commits all score the
                   same, or whose outcomes cannot be observed, teaches the
                   next retrain nothing.
5. **register** -- append the entry to ``registry/repos.yaml``, preserving the
                   file's comments, and only after every gate passed.

``--dry-run`` runs stages 1-4 and writes the report without touching the
registry, which is what you want when evaluating a candidate repository.

The report is written to ``artifacts/onboarding/<name>.json`` in both cases,
so the decision is a committed artifact rather than a paragraph in chat.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

import registry
from registry import RegistryError

REPORT_DIR = PROJECT_ROOT / "artifacts" / "onboarding"

# ── Gates ───────────────────────────────────────────────────────────────────
# Every threshold is a claim about what makes history useful as corpus, and
# each one is here because a repository that failed it produced a model that
# could not be trusted on that repository. They are deliberately low: a gate
# that only large projects pass would refuse the small repositories this
# pipeline exists to onboard.
MIN_COMMITS_SCORED = 25
MIN_DISTINCT_SCORES = 5
MIN_SCORE_SPREAD = 0.02
MIN_OBSERVABLE_OUTCOMES = 10
MIN_LABEL_RATE = 0.02
MAX_LABEL_RATE = 0.95

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class OnboardError(RuntimeError):
    """A stage failed in a way the caller should see, not catch."""


# ── Stage 1: resolve ────────────────────────────────────────────────────────


def derive_name(url: str, override: str | None = None) -> str:
    """The registry name for a clone URL.

    ``https://github.com/owner/Some.Repo.git`` becomes ``some.repo``. Owner
    is dropped on purpose: the five declared repositories are known by their
    project name, and two owners of the same name would be a registry
    collision the caller has to resolve with ``--name``.
    """
    if override:
        name = override.strip()
    else:
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        name = tail.removesuffix(".git").lower()
    if not NAME_RE.match(name):
        raise OnboardError(
            f"cannot derive a registry name from {url!r} (got {name!r}); "
            f"names must match {NAME_RE.pattern}, pass --name"
        )
    return name


def check_remote(url: str) -> str:
    """Fail early on an unusable URL, before cloning anything.

    ``git ls-remote`` is the cheapest possible check of both the URL and the
    network, and it is what tells a typo apart from a private repository.
    """
    if not url.startswith("https://"):
        raise OnboardError(f"{url}: the registry requires an https:// clone URL")
    try:
        result = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            check=False,
            capture_output=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OnboardError(f"{url}: could not reach the remote ({exc})") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise OnboardError(
            f"{url}: not a reachable public repository"
            + (f" ({detail[-1]})" if detail else "")
        )
    return result.stdout.split("\t")[0].strip()


# ── Stage 2: clone ──────────────────────────────────────────────────────────


def clone(url: str, destination: Path, head: str = "") -> Path:
    """Check the repository out at ``destination``, reusing an existing clone.

    A full clone rather than a shallow one: the label is "a touched file was
    changed again within seven days", which cannot be computed at the tip of
    a truncated history.
    """
    if (destination / ".git").exists():
        _git(["fetch", "--quiet", "--prune", "origin"], destination)
        return destination

    if destination.exists() and any(destination.iterdir()):
        raise OnboardError(
            f"{destination} exists and is not a git checkout; move it aside first"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--quiet", url, str(destination)],
        check=False,
        capture_output=True,
        timeout=1800,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise OnboardError(
            f"git clone {url} failed: {(result.stderr or '').strip().splitlines()[-1:]}"
        )
    return destination


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        cwd=cwd,
        capture_output=True,
        timeout=300,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise OnboardError(f"git {' '.join(args)} failed in {cwd}: {result.stderr.strip()}")
    return result.stdout


def head_sha(checkout: Path) -> str:
    return _git(["rev-parse", "HEAD"], checkout).strip()


def sample_commits(checkout: Path, since: str, limit: int) -> list[tuple[str, int]]:
    """Non-merge commits to score, newest first, as (sha, unix timestamp)."""
    raw = _git(
        [
            "log",
            f"--since={since}",
            "--no-merges",
            "--format=%H|%ct",
            f"--max-count={limit}",
        ],
        checkout,
    )
    commits: list[tuple[str, int]] = []
    for line in raw.strip().splitlines():
        sha, _, ts = line.partition("|")
        try:
            commits.append((sha.strip(), int(ts)))
        except ValueError:
            continue
    return commits


# ── Stage 3: score ──────────────────────────────────────────────────────────


def _load_model_and_columns():
    """The production model and the feature order it was trained on."""
    import skops.io as sio

    model_path = PROJECT_ROOT / "models" / "gatekeeper_risk_model.skops"
    if not model_path.exists():
        raise OnboardError(f"{model_path} is missing; cannot score the candidate")
    trusted = [
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
        "numpy.dtype",
        "numpy.ndarray",
        "pandas.core.frame.DataFrame",
        "pandas.core.series.Series",
    ]
    model = sio.loads(model_path.read_bytes(), trusted=trusted)

    config = yaml.safe_load((PROJECT_ROOT / "ml" / "config.yaml").read_text(encoding="utf-8"))
    columns = config.get("feature_columns") or []
    if not columns:
        raise OnboardError("ml/config.yaml declares no feature_columns")
    return model, columns


def _file_commit_index(checkout: Path, since: str, until: str) -> dict[str, list[tuple[int, str]]]:
    """Every file's commits in a date range, for the retouch outcome.

    One git call for the whole window instead of a per-commit diff, which is
    the difference between minutes and hours on a real history.
    """
    raw = _git(
        [
            "log",
            f"--since={since}",
            f"--until={until}",
            "--no-merges",
            "--format=COMMIT|%H|%ct",
            "--name-only",
        ],
        checkout,
    )
    index: dict[str, list[tuple[int, str]]] = {}
    sha, timestamp = "", 0
    for line in raw.splitlines():
        if line.startswith("COMMIT|"):
            _, sha, timestamp_text = line.split("|", 2)
            timestamp = int(timestamp_text)
        elif line.strip() and sha:
            index.setdefault(line.strip(), []).append((timestamp, sha))
    for touches in index.values():
        touches.sort()
    return index


def _was_retouched(
    sha: str, timestamp: int, files: list[str], index: dict, window_days: int
) -> bool:
    """Did any file this commit touched change again inside the window?"""
    end = timestamp + window_days * 86400
    short = sha[:8]
    for path in files:
        for when, other in index.get(path, ()):
            if when > timestamp and when <= end and other[:8] != short:
                return True
    return False


def _was_reverted(sha: str, when: datetime, window_days: int, checkout: Path) -> bool:
    """Was this commit named in a revert inside the window?"""
    raw = _git(
        [
            "log",
            f"--since={when.isoformat()}",
            f"--until={(when + timedelta(days=window_days)).isoformat()}",
            "-i",
            "--grep=revert",
            "--format=%H|%s",
            "--max-count=50",
        ],
        checkout,
    )
    short = sha[:8]
    return any(short in line.split("|", 1)[-1] for line in raw.splitlines() if "|" in line)


def score_commits(
    checkout: Path,
    commits: list[tuple[str, int]],
    repo_name: str,
    window_days: int = 7,
) -> dict:
    """Score a sample and measure each score against the outcome that followed.

    The outcome is the same one the training label encodes -- a touched file
    changed again, or the commit reverted, within ``window_days`` -- computed
    forward from the commit instead of backwards from the tip.
    """
    from ml.extract_features import CommitFeatureExtractor
    from ml.scoring import _determine_band

    model, columns = _load_model_and_columns()

    if not commits:
        return {
            "commits_scored": 0,
            "band_counts": {},
            "scores": [],
            "observable_outcomes": 0,
            "label_rate": None,
            "precision_at_high": None,
            "outcome_window_days": window_days,
        }

    oldest = datetime.fromtimestamp(min(ts for _, ts in commits), tz=timezone.utc)
    newest = datetime.fromtimestamp(max(ts for _, ts in commits), tz=timezone.utc)
    since = (oldest - timedelta(days=1)).date().isoformat()
    until = (newest + timedelta(days=window_days + 1)).date().isoformat()
    index = _file_commit_index(checkout, since, until)

    extractor = CommitFeatureExtractor(
        repo_path=str(checkout),
        since=oldest.date().isoformat(),
        label_window_days=window_days,
    )

    records: list[dict] = []
    failures = 0
    for sha, timestamp in commits:
        try:
            features = extractor.extract_single_commit(str(checkout), sha)
            vector = [float(features.get(column, 0) or 0) for column in columns]
            score = float(model.predict_proba(np.array([vector]))[0][1])
        except Exception as exc:  # noqa: BLE001 - one unscoreable commit is data, not a crash
            failures += 1
            print(f"  ! {sha[:8]} could not be scored: {exc}", file=sys.stderr)
            continue

        files = [
            name for name in str(features.get("touched_files", "")).split("|") if name
        ][:10]
        when = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        # The tip of the history has no future to observe, so its outcome is
        # unknown rather than negative.
        observable = when + timedelta(days=window_days) <= datetime.now(timezone.utc)
        risky = None
        if observable:
            risky = _was_retouched(sha, timestamp, files, index, window_days) or _was_reverted(
                sha, when, window_days, checkout
            )
        records.append(
            {
                "sha": sha,
                "score": score,
                "band": _determine_band(score, repo_name),
                "risky": risky,
                "files": files,
            }
        )

    scores = [record["score"] for record in records]
    bands = Counter(record["band"] for record in records)
    observed = [record for record in records if record["risky"] is not None]
    high = [record for record in observed if record["band"] == "high"]

    return {
        "commits_scored": len(records),
        "commit_failures": failures,
        "band_counts": dict(sorted(bands.items())),
        "scores": scores,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_median": float(np.median(scores)) if scores else None,
        "distinct_scores": len({round(score, 6) for score in scores}),
        "observable_outcomes": len(observed),
        "label_rate": (
            sum(1 for record in observed if record["risky"]) / len(observed)
            if observed
            else None
        ),
        # Precision of the high band: of the commits it called high, how many
        # actually had a bad outcome. A band that never fires, or fires on
        # everything, is what these gates protect against.
        "precision_at_high": (
            sum(1 for record in high if record["risky"]) / len(high) if high else None
        ),
        "high_band_count": len(high),
        "outcome_window_days": window_days,
        # Every scored commit, not a truncation: precision is a rate over the
        # whole sample, and a top-five excerpt would quietly change it.
        "commits": sorted(records, key=lambda record: record["score"], reverse=True),
    }


# ── Stage 4: gate ───────────────────────────────────────────────────────────


def calibrate(scores: list[float]) -> dict:
    """Per-repo band cut-offs from this repository's own score distribution.

    ``ml/config.yaml`` gives the five corpus repositories ``high`` at the 90th
    percentile and ``medium`` at the 75th, which is what makes a band mean "top
    10% of this repository" rather than "above some absolute number". Without
    an entry a repository falls back to ``_global``, and on a new history that
    fallback labels everything low -- so a freshly onboarded repository has
    working bands only if onboarding computes these.
    """
    if not scores:
        return {}
    high, medium = (float(value) for value in np.percentile(scores, [90, 75]))
    # A flat or nearly flat distribution can collapse the two cut-offs onto
    # each other; keep them ordered so no band is unreachable.
    if high <= medium:
        high = min(1.0, medium + 0.01)
    return {"high": round(high, 4), "medium": round(medium, 4)}


def rebucket(scores: list[float], thresholds: dict) -> dict:
    """Band counts for a score list under given cut-offs."""
    counts = Counter()
    for score in scores:
        if thresholds and score >= thresholds["high"]:
            counts["high"] += 1
        elif thresholds and score >= thresholds["medium"]:
            counts["medium"] += 1
        else:
            counts["low"] += 1
    return dict(sorted(counts.items()))


def write_thresholds(name: str, thresholds: dict, path: Path | None = None) -> dict:
    """Record this repository's cut-offs in ``ml/config.yaml``.

    Appended as text so the existing comments survive, then verified by
    reloading: if the file no longer yields the entry under ``thresholds``, it
    is restored. A repository whose bands silently fall back to another
    repository's cut-offs is worse than one that fails loudly here.
    """
    if not thresholds:
        return {"written": False, "reason": "no scores to calibrate from"}

    config_path = Path(path) if path is not None else PROJECT_ROOT / "ml" / "config.yaml"
    original = config_path.read_text(encoding="utf-8")
    block = f"  {name}:\n    high: {thresholds['high']}\n    medium: {thresholds['medium']}\n"
    updated = original + ("" if original.endswith("\n") else "\n") + block
    config_path.write_text(updated, encoding="utf-8")

    try:
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        recorded = (parsed.get("thresholds") or {}).get(name)
    except yaml.YAMLError:
        recorded = None

    if recorded != {"high": thresholds["high"], "medium": thresholds["medium"]}:
        config_path.write_text(original, encoding="utf-8")
        return {"written": False, "reason": "the file did not reload with the entry"}
    return {"written": True, "path": str(config_path), "thresholds": thresholds}


def _gate(name: str, passed: bool, value, requirement: str, detail: str = "") -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "requirement": requirement,
        "detail": detail,
    }


def evaluate_gates(scored: dict, min_commits: int = MIN_COMMITS_SCORED) -> tuple[bool, list[dict]]:
    """Decide whether this history can be corpus, from measurements alone."""
    count = scored["commits_scored"]
    labels = scored.get("label_rate")
    spread = (
        scored["score_max"] - scored["score_min"]
        if scored.get("score_min") is not None and scored.get("score_max") is not None
        else 0.0
    )

    gates = [
        _gate(
            "enough_commits_scored",
            count >= min_commits,
            count,
            f">= {min_commits} commits scored",
            "a handful of commits cannot show a distribution",
        ),
        _gate(
            "model_discriminates",
            scored.get("distinct_scores", 0) >= MIN_DISTINCT_SCORES
            and spread >= MIN_SCORE_SPREAD,
            {"distinct": scored.get("distinct_scores", 0), "spread": round(spread, 4)},
            f">= {MIN_DISTINCT_SCORES} distinct scores and spread >= {MIN_SCORE_SPREAD}",
            "identical scores mean the model sees nothing it can rank here",
        ),
        _gate(
            "outcomes_are_observable",
            scored.get("observable_outcomes", 0) >= MIN_OBSERVABLE_OUTCOMES,
            scored.get("observable_outcomes", 0),
            f">= {MIN_OBSERVABLE_OUTCOMES} commits with a full outcome window",
            "a history younger than the label window cannot be labelled",
        ),
        _gate(
            "label_rate_is_usable",
            labels is not None and MIN_LABEL_RATE <= labels <= MAX_LABEL_RATE,
            None if labels is None else round(labels, 4),
            f"{MIN_LABEL_RATE} <= risky rate <= {MAX_LABEL_RATE}",
            "0% or 100% risky is a broken label, not an easy corpus",
        ),
    ]
    return all(gate["passed"] for gate in gates), gates


# ── Stage 5: register ───────────────────────────────────────────────────────


def build_entry(name: str, url: str, clone_dir: str, role: str, description: str) -> str:
    """One YAML entry, indented to match the file it is appended to."""
    lines = [
        f"  - name: {name}",
        f"    url: {url}",
        f"    clone_dir: {clone_dir}",
        f"    role: {role}",
    ]
    if description:
        lines.append(f"    description: {description}")
    return "\n".join(lines) + "\n"


def register(
    name: str,
    url: str,
    clone_dir: str,
    role: str,
    description: str = "",
    path: Path | None = None,
) -> dict:
    """Append the repository to the registry and prove the file still loads.

    Appending text rather than re-serialising the YAML keeps the file's
    comments, which are the only explanation of the format a future
    contributor gets. The write is rolled back if the result does not parse,
    so a bug here cannot leave the registry unloadable.
    """
    registry_path = Path(path) if path is not None else registry.DEFAULT_PATH
    original = registry_path.read_text(encoding="utf-8")

    for existing in registry.load(registry_path):
        if existing.name == name:
            if existing.url == url:
                return {"added": False, "reason": "already declared", "path": str(registry_path)}
            raise OnboardError(
                f"{name!r} is already declared as {existing.url}; pass --name to add this "
                f"repository under a different name"
            )

    separator = "" if original.endswith("\n") else "\n"
    updated = original + separator + build_entry(name, url, clone_dir, role, description)
    registry_path.write_text(updated, encoding="utf-8")
    registry.clear_cache()

    try:
        declared = registry.load(registry_path)
    except RegistryError as exc:
        registry_path.write_text(original, encoding="utf-8")
        registry.clear_cache()
        raise OnboardError(f"registry rejected the new entry, rolled back: {exc}") from exc

    if not any(repo.name == name for repo in declared):
        registry_path.write_text(original, encoding="utf-8")
        registry.clear_cache()
        raise OnboardError("the new entry did not survive a reload, rolled back")

    return {
        "added": True,
        "path": str(registry_path),
        "declared_repositories": len(declared),
    }


# ── Orchestration ───────────────────────────────────────────────────────────


def onboard(
    url: str,
    *,
    name: str | None = None,
    role: str = "validate",
    since: str = "2023-01-01",
    sample: int = 40,
    window_days: int = 7,
    clone_dir: str | None = None,
    local: Path | None = None,
    registry_path: Path | None = None,
    report_dir: Path | None = None,
    dry_run: bool = False,
    min_commits: int = MIN_COMMITS_SCORED,
) -> dict:
    """Run every stage and return the report.

    With ``local`` the clone stage is skipped and that checkout is scored,
    which is how a history is evaluated before it is ever published -- and how
    the pipeline runs offline. The remote check is skipped with it, because
    the checkout on disk is the evidence and a private or not-yet-pushed
    repository has no reachable URL yet.

    Raises OnboardError when a stage fails. A gate failure is not an
    exception: it is a report with ``passed: false`` and a registry that was
    left alone.
    """
    if role not in registry.ROLES:
        raise OnboardError(f"role must be one of {', '.join(registry.ROLES)}, got {role!r}")

    started = time.time()
    resolved = derive_name(url, name)
    clone_relative = clone_dir or f"repos/{resolved}"
    destination = Path(local) if local is not None else PROJECT_ROOT / clone_relative

    print(f"[1/5] resolve  {url} -> {resolved}")
    if local is not None:
        print(f"      local checkout {destination} (remote check skipped)")
    else:
        head = check_remote(url)
        print(f"      remote HEAD {head[:12]}")

    print(f"[2/5] clone    {destination}")
    if local is None:
        clone(url, destination, head)
    elif not (destination / ".git").exists():
        raise OnboardError(f"{destination} is not a git checkout")
    tip = head_sha(destination)

    print(f"[3/5] score    up to {sample} commits since {since} at {tip[:12]}")
    commits = sample_commits(destination, since, sample)
    scored = score_commits(destination, commits, resolved, window_days=window_days)

    print("[4/5] gate")
    passed, gates = evaluate_gates(scored, min_commits=min_commits)
    for gate in gates:
        mark = "pass" if gate["passed"] else "FAIL"
        print(f"      {mark}  {gate['name']}: {gate['value']} ({gate['requirement']})")

    # Bands are only meaningful once this repository has its own cut-offs, so
    # calibrate from the sample and report the distribution they produce.
    thresholds = calibrate(scored["scores"]) if passed else {}
    if thresholds:
        scored["band_counts_default"] = scored["band_counts"]
        scored["band_counts"] = rebucket(scored["scores"], thresholds)
        high_cut = thresholds["high"]
        in_band = [record for record in scored["commits"] if record["score"] >= high_cut]
        observed = [record for record in in_band if record["risky"] is not None]
        scored["high_band_count"] = len(in_band)
        scored["precision_at_high"] = (
            sum(1 for record in observed if record["risky"]) / len(observed)
            if observed
            else None
        )
        print(f"      thresholds {thresholds} from {scored['commits_scored']} scores")

    registration: dict = {"added": False, "reason": "dry run" if dry_run else "gates failed"}
    calibration: dict = {"written": False, "reason": "dry run" if dry_run else "gates failed"}
    if passed and not dry_run:
        print("[5/5] register")
        registration = register(
            resolved,
            url,
            clone_relative,
            role,
            description=f"{resolved} corpus, onboarded {datetime.now(timezone.utc):%Y-%m-%d}.",
            path=registry_path,
        )
        print(f"      added to {registration['path']}")
        calibration = write_thresholds(resolved, thresholds)
        if calibration.get("written"):
            print(f"      thresholds recorded in {calibration['path']}")
        else:
            print(f"      thresholds NOT recorded: {calibration.get('reason')}")
    else:
        print(f"[5/5] register skipped ({registration['reason']})")

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "name": resolved,
            "url": url,
            "clone_dir": clone_relative,
            "role": role,
            "head": tip,
            "since": since,
        },
        "scored": scored,
        "thresholds": thresholds,
        "gates": gates,
        "passed": passed,
        "registry": registration,
        "calibration": calibration,
        "dry_run": dry_run,
        "duration_seconds": round(time.time() - started, 1),
    }

    out_dir = report_dir or REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{resolved}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    print(f"report written to {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Onboard a repository into the Gatekeeper corpus from its clone URL."
    )
    parser.add_argument("url", help="https:// clone URL of the repository")
    parser.add_argument("--name", help="registry name (default: derived from the URL)")
    parser.add_argument(
        "--role",
        default="validate",
        choices=list(registry.ROLES),
        help="how the corpus uses it: train builds the model, validate only scores "
        "(default), holdout is reserved for final evaluation",
    )
    parser.add_argument("--since", default="2023-01-01", help="mine commits after this date")
    parser.add_argument("--sample", type=int, default=40, help="commits to score")
    parser.add_argument("--window-days", type=int, default=7, help="outcome window")
    parser.add_argument("--clone-dir", help="checkout path relative to the project root")
    parser.add_argument(
        "--local",
        type=Path,
        help="score this existing checkout instead of cloning, and skip the remote check",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="registry file to append to (default: registry/repos.yaml)",
    )
    parser.add_argument("--min-commits", type=int, default=MIN_COMMITS_SCORED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="score and gate the repository, but never write the registry",
    )
    args = parser.parse_args(argv)

    if shutil.which("git") is None:
        print("git is not on PATH", file=sys.stderr)
        return 2

    try:
        report = onboard(
            args.url,
            name=args.name,
            role=args.role,
            since=args.since,
            sample=args.sample,
            window_days=args.window_days,
            clone_dir=args.clone_dir,
            local=args.local,
            registry_path=args.registry,
            dry_run=args.dry_run,
            min_commits=args.min_commits,
        )
    except OnboardError as exc:
        print(f"onboarding failed: {exc}", file=sys.stderr)
        return 1

    scored = report["scored"]
    print("\n" + "=" * 62)
    print(f"{report['repo']['name']}: {'ACCEPTED' if report['passed'] else 'REJECTED'}")
    print(f"  commits scored     {scored['commits_scored']}")
    print(f"  band distribution  {scored['band_counts']}")
    if scored.get("precision_at_high") is not None:
        print(f"  precision @ high   {scored['precision_at_high']:.2f}")
    print(f"  registered         {report['registry'].get('added', False)}")
    print("=" * 62)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
