#!/usr/bin/env python3
"""Shared runtime for Gatekeeper husky hooks.

Husky only guarantees that a POSIX `sh` and Node are present; it says
nothing about which Python is on PATH. This module resolves a usable
interpreter and the repo root once, so every hook script can stay short
and behave the same on Windows, macOS, Linux and CI.

It also owns three things every hook would otherwise duplicate: the list of
hooks this project ships, the verbose-tracing switch, and the gate-run marker
that keeps background housekeeping out of the way while a gate is working.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

# scripts/hooks/_runtime.py -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# Env vars that short-circuit a hook, mirroring the pre-commit convention
# already used by .pre-commit-config.yaml (SKIP=gatekeeper-score).
SKIP_ENV = "GATEKEEPER_SKIP_HOOKS"

# Verbose tracing. `HUSKY=2` is husky's own switch: its `h` dispatcher runs
# `set -x` for the sh layer. We honour the same value in Python so one
# variable traces the whole chain, and accept GATEKEEPER_TRACE=1 for when
# husky is bypassed entirely (tests, `python -m scripts.hooks.x`).
TRACE_ENV = "GATEKEEPER_TRACE"

# Hooks this project ships, in githooks(5) execution order. This is the
# single owner of the list: the doctor, the tests and the docs all read it
# rather than each repeating the hook names.
HOOKS = (
    "applypatch-msg",
    "pre-applypatch",
    "post-applypatch",
    "pre-commit",
    "pre-merge-commit",
    "prepare-commit-msg",
    "commit-msg",
    "post-commit",
    "pre-rebase",
    "post-checkout",
    "post-rewrite",
    "post-merge",
    "pre-push",
    "pre-auto-gc",
)

# The 14 hooks husky generates a dispatcher for. We ship a script for every
# one of them, so `HOOKS` and `HUSKY_HOOKS` are the same set; the doctor
# asserts that, because a hook husky does not dispatch is a file that can
# never run.
HUSKY_HOOKS = (
    "pre-commit",
    "pre-merge-commit",
    "prepare-commit-msg",
    "commit-msg",
    "post-commit",
    "applypatch-msg",
    "pre-applypatch",
    "post-applypatch",
    "pre-rebase",
    "post-rewrite",
    "post-checkout",
    "post-merge",
    "pre-push",
    "pre-auto-gc",
)

# Hooks whose non-zero exit stops the git command they hang off. Everything
# else exists to inform and must never break a commit, merge or checkout.
# `pre-auto-gc` is unusual: a non-zero exit is not an error, it simply tells
# git not to bother with the background garbage collection.
BLOCKING_HOOKS = (
    "applypatch-msg",
    "pre-applypatch",
    "pre-commit",
    "pre-merge-commit",
    "commit-msg",
    "pre-rebase",
    "pre-push",
    "pre-auto-gc",
)

# Every hook name git knows about. Anything else in .husky/ is not a hook
# and will never run -- the single most common "hooks not running" cause.
KNOWN_GIT_HOOKS = (
    "applypatch-msg",
    "pre-applypatch",
    "post-applypatch",
    "pre-commit",
    "pre-merge-commit",
    "prepare-commit-msg",
    "commit-msg",
    "post-commit",
    "pre-rebase",
    "post-checkout",
    "post-merge",
    "pre-push",
    "pre-receive",
    "update",
    "proc-receive",
    "post-receive",
    "post-update",
    "reference-transaction",
    "push-to-checkout",
    "pre-auto-gc",
    "post-rewrite",
    "sendemail-validate",
    "fsmonitor-watchman",
    "p4-changelist",
    "p4-prepare-changelist",
    "p4-post-changelist",
    "p4-pre-submit",
    "post-index-change",
)


def module_for(hook: str) -> str:
    """Module name for a hook, e.g. pre-commit -> scripts.hooks.pre_commit."""
    return "scripts.hooks." + hook.replace("-", "_")


BANNER = "\033[36m[gatekeeper-hooks]\033[0m"


def _harden_console() -> None:
    """Never let non-ASCII output crash a hook.

    Windows terminals default to a legacy code page, where printing a
    character outside it raises UnicodeEncodeError. A hook that dies for a
    cosmetic reason blocks a commit, so degrade to escapes instead.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass


_harden_console()


def _colour_supported() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stderr.isatty()


def log(message: str) -> None:
    """Print a prefixed line to stderr (github/husky style)."""
    text = message if _colour_supported() else message.replace(BANNER, "[gatekeeper-hooks]")
    print(text, file=sys.stderr)


def tracing_enabled() -> bool:
    """True when the caller asked for verbose hook tracing."""
    return os.environ.get("HUSKY") == "2" or os.environ.get(TRACE_ENV) == "1"


def trace(message: str) -> None:
    """Explain what a hook is doing, when tracing is on.

    husky traces its own sh dispatcher with `set -x` on `HUSKY=2`, which
    stops the moment the hook hands over to Python. This continues the same
    trace through the layer that actually makes the decisions.
    """
    if tracing_enabled():
        print(f"+ {message}", file=sys.stderr)


def enter(hook: str, argv: Sequence[str] = ()) -> None:
    """Announce which hook is running, and under which interpreter.

    "Which Python did the hook actually get?" is the first question when a
    hook behaves differently in two terminals, and it is invisible without
    this line -- bootstrap.sh picks the interpreter before Python starts.
    """
    if not tracing_enabled():
        return
    trace(f"{hook}: repo={REPO_ROOT}")
    trace(f"{hook}: interpreter={sys.executable}")
    trace(f"{hook}: argv={[str(arg) for arg in argv]}")


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


def advisory(hook: str, action) -> int:
    """Run a non-blocking hook, never failing the git command it hangs off.

    `post-*`, `prepare-commit-msg` and reporting hooks exist to inform, not
    to gate. They must not be able to break a commit, merge or checkout, so
    any failure is reported and swallowed.
    """
    if skipped(hook):
        log(f"{BANNER} {hook} skipped via {SKIP_ENV}")
        return 0
    try:
        action()
    except Exception as exc:
        log(f"{BANNER} {hook}: {type(exc).__name__}: {exc} (continuing)")
    return 0


def project_python() -> str:
    """Return the interpreter best suited to run project tooling.

    Preference order:
      1. ``GATEKEEPER_PYTHON`` (explicit override, used by CI)
      2. the in-repo virtualenv
      3. the interpreter already running this hook
      4. ``python3`` then ``python`` on PATH

    The running interpreter outranks a PATH lookup on purpose: bootstrap.sh
    has already proved it starts, whereas ``python3`` on Windows is often the
    Microsoft Store alias stub -- which is on PATH, exits non-zero, and would
    turn every tool invocation into "ruff reported lint errors".
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

    if sys.executable:
        return sys.executable

    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return found

    return "python3"


def git(*args: str, check: bool = False) -> tuple[int, str]:
    """Run a git command in the repo root and return (code, stripped stdout).

    Returns 127 rather than raising when git cannot be started at all -- for
    instance when the working directory has been deleted underneath a hook.
    A hook must degrade to "no answer", not to a traceback.
    """
    trace(f"git {' '.join(args)} (in {REPO_ROOT})")
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        trace(f"  -> could not run git: {exc}")
        if check:
            raise RuntimeError(f"git {' '.join(args)} could not run: {exc}") from exc
        return 127, ""
    if tracing_enabled() and result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        if detail:
            trace(f"  -> exit {result.returncode}: {detail[0]}")
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
    trace(f"run ({label}): {' '.join(command)}")
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, timeout=timeout, check=False)
    except FileNotFoundError:
        log(f"{BANNER} skip: {command[0]} not found on PATH")
        return 0
    except subprocess.TimeoutExpired:
        log(f"{BANNER} FAIL: {label} exceeded {timeout}s")
        return 1
    trace(f"  -> exit {completed.returncode} after {time.monotonic() - started:.2f}s")
    return completed.returncode


def python_module(module: str, *args: str) -> list[str]:
    """Command that runs a module with the resolved project interpreter."""
    return [project_python(), "-m", module, *args]


# -- git directory, event log and the gate-run marker -----------------

# Local state lives inside the git directory on purpose: a hook that writes
# into the work tree would show up as untracked noise in every `git status`.
STATE_DIR = "gatekeeper"

# Marker a long-running hook drops so `pre-auto-gc` can stay out of its way.
GATE_RUN_FILE = "gate_run.json"

# A marker older than this is treated as abandoned. Without it a hook killed
# mid-run (Ctrl-C, closed terminal, SIGKILL on CI) would block garbage
# collection forever, which is a worse failure than the one it prevents.
GATE_RUN_STALE_SECONDS = 1800


def git_dir() -> Path | None:
    """Absolute path to the real git directory, or None outside a repo.

    Uses the command rather than assuming ``.git``: in a linked worktree the
    git directory lives elsewhere entirely.
    """
    code, out = git("rev-parse", "--git-dir")
    if code != 0 or not out:
        return None
    path = Path(out.strip())
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.is_dir() else None


def state_dir() -> Path | None:
    """Directory hooks may write local state into, created on demand."""
    directory = git_dir()
    if directory is None:
        return None
    target = directory / STATE_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def begin_gate_run(label: str) -> bool:
    """Mark that a long-running gate hook has started.

    Returns False when there is nowhere to write the marker, which callers
    treat as "no marker" rather than as an error -- the marker is an
    optimisation, never a requirement.
    """
    directory = state_dir()
    if directory is None:
        return False
    marker = {
        "label": label,
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    try:
        (directory / GATE_RUN_FILE).write_text(json.dumps(marker), encoding="utf-8")
    except OSError:
        return False
    trace(f"gate-run marker written ({label}, pid {os.getpid()})")
    return True


def end_gate_run() -> None:
    """Remove the marker written by :func:`begin_gate_run`, if any."""
    directory = state_dir()
    if directory is None:
        return
    try:
        (directory / GATE_RUN_FILE).unlink(missing_ok=True)
    except OSError:
        return
    trace("gate-run marker removed")


def gate_run_active(max_age: float = GATE_RUN_STALE_SECONDS) -> bool:
    """True when a gate hook is mid-run and recently started."""
    directory = state_dir()
    if directory is None:
        return False
    marker = directory / GATE_RUN_FILE
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        started = float(payload.get("started_at", 0))
    except (OSError, ValueError, TypeError):
        return False
    return 0 <= (time.time() - started) < max_age


# -- pre-push ref parsing ---------------------------------------------

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
