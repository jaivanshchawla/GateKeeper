#!/usr/bin/env python3
"""Shared runtime for Gatekeeper husky hooks.

Husky only guarantees that a POSIX `sh` and Node are present; it says
nothing about which Python is on PATH. This module resolves a usable
interpreter and the repo root once, so every hook script can stay short
and behave the same on Windows, macOS, Linux and CI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# scripts/hooks/_runtime.py -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# Env vars that short-circuit a hook, mirroring the pre-commit convention
# already used by .pre-commit-config.yaml (SKIP=gatekeeper-score).
SKIP_ENV = "GATEKEEPER_SKIP_HOOKS"

BANNER = "\033[36m[gatekeeper-hooks]\033[0m"


def _colour_supported() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stderr.isatty()


def log(message: str) -> None:
    """Print a prefixed line to stderr (github/husky style)."""
    text = message if _colour_supported() else message.replace(BANNER, "[gatekeeper-hooks]")
    print(text, file=sys.stderr)


def skipped(hook_name: str) -> bool:
    """True when the hook was explicitly disabled for this run.

    `HUSKY=0` is handled by husky itself; these are the escape hatches the
    hook scripts own.
    """
    if os.environ.get("HUSKY") == "0":
        return True
    raw = os.environ.get(SKIP_ENV, "")
    requested = {part.strip() for part in raw.split(",") if part.strip()}
    return "all" in requested or hook_name in requested


def project_python() -> str:
    """Return the interpreter best suited to run project tooling.

    Preference order:
      1. ``GATEKEEPER_PYTHON`` (explicit override, used by CI)
      2. the in-repo virtualenv
      3. ``python3`` then ``python`` on PATH
      4. the interpreter already running this file
    """
    override = os.environ.get("GATEKEEPER_PYTHON")
    if override:
        return override

    for candidate in (
        REPO_ROOT / "venv" / "Scripts" / "python.exe",  # Windows
        REPO_ROOT / "venv" / "bin" / "python",  # POSIX
    ):
        if candidate.exists():
            return str(candidate)

    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return found

    return sys.executable


def git(*args: str, check: bool = False) -> tuple[int, str]:
    """Run a git command in the repo root and return (code, stripped stdout)."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.returncode, result.stdout.strip()


def staged_files(*paths: str, statuses: str = "ACMR") -> list[str]:
    """Files staged for the current commit, optionally limited to globs."""
    args = ["diff", "--cached", "--name-only", f"--diff-filter={statuses}", "--"]
    args.extend(paths or ["."])
    code, out = git(*args)
    if code != 0 or not out:
        return []
    return [line for line in out.splitlines() if line.strip()]


def run(command: list[str], *, label: str, timeout: int = 900) -> int:
    """Run a subprocess in the repo root, streaming output. Returns its exit code."""
    log(f"{BANNER} {label}")
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, timeout=timeout, check=False)
    except FileNotFoundError:
        log(f"{BANNER} skip: {command[0]} not found on PATH")
        return 0
    except subprocess.TimeoutExpired:
        log(f"{BANNER} FAIL: {label} exceeded {timeout}s")
        return 1
    return completed.returncode


def python_module(module: str, *args: str) -> list[str]:
    """Command that runs a module with the resolved project interpreter."""
    return [project_python(), "-m", module, *args]


# ── pre-push ref parsing ─────────────────────────────────────────────

ZERO_SHA = "0" * 40


def parse_push_refs(stdin_data: str) -> list[tuple[str, str]]:
    """Parse git's pre-push stdin into (remote_sha, local_sha) pairs.

    Each stdin line is: <local ref> <local sha> <remote ref> <remote sha>
    A local sha of all zeros means a delete, which has nothing to lint.
    """
    refs: list[tuple[str, str]] = []
    for line in stdin_data.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        local_sha, remote_sha = parts[1], parts[3]
        if local_sha == ZERO_SHA:
            continue
        refs.append((remote_sha, local_sha))
    return refs


def _fallback_base() -> str | None:
    """Best guess at the branch a new local branch forked from."""
    for candidate in ("origin/main", "origin/master", "main", "master"):
        code, _ = git("rev-parse", "--verify", "--quiet", candidate)
        if code == 0:
            return candidate
    return None


def changed_files(stdin_data: str, *globs: str) -> list[str]:
    """Files changed by the refs being pushed, limited to ``globs``.

    Mirrors pre-commit's behaviour of linting only what changed. This repo
    has pre-existing lint debt, so a whole-tree lint here would block every
    push; scoping to the outgoing diff keeps the gate meaningful.
    """
    patterns = list(globs) or ["."]
    seen: list[str] = []
    for remote_sha, local_sha in parse_push_refs(stdin_data):
        if remote_sha == ZERO_SHA:
            base = _fallback_base()
            spec = f"{base}...{local_sha}" if base else f"{local_sha}~1..{local_sha}"
        else:
            spec = f"{remote_sha}..{local_sha}"
        code, out = git("diff", "--name-only", spec, "--", *patterns)
        if code != 0:
            continue
        for path in out.splitlines():
            if path and path not in seen:
                seen.append(path)
    return seen
