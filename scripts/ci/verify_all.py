#!/usr/bin/env python3
"""The verification matrix, in one command.

Why this file exists
--------------------
Two of the three jobs in .github/workflows/hooks.yml had never passed, and both
failed for a reason a local run would have shown immediately: one test needed a
library CI did not install (PyYAML), and Gate 3's docker build needed a
directory that is gitignored (dashboard/dist). Neither was visible from a
developer machine, because a developer machine has both. The checks therefore
live here, once, so "it passed locally" and "it passed in CI" are the same
sentence instead of two lists that drift apart.

`python -m scripts.ci.verify_all` runs them here; the `verification` job in
hooks.yml runs the same list on Ubuntu, macOS and Windows. Every step names the
CI step that performs its equivalent (`ci_step`), and
tests/test_ci_workflows.py asserts those names still exist in the workflows --
so a step cannot be dropped from one side without the other noticing.

Usage:
    python -m scripts.ci.verify_all                # the local matrix
    python -m scripts.ci.verify_all --list        # what would run, and why
    python -m scripts.ci.verify_all --only trace  # repeatable
    python -m scripts.ci.verify_all --docker      # + the Gate 3 rehearsal

Output goes to the console and to verify_all.log, which the workflows upload
when they fail: a red run whose log only exists behind a signed-in browser is a
red run nobody can fix.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# scripts/ci/verify_all.py -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "verify_all.log"
PYTHON = sys.executable
IS_WINDOWS = os.name == "nt"

# The hook layer under test. Three CI steps and three steps here run this exact
# set, which is why it is a constant rather than three file lists.
HOOK_TEST_FILES = (
    "tests/test_hooks.py",
    "tests/test_hook_integration.py",
    "tests/test_husky_fidelity.py",
)

HEALTH_URL = "http://localhost:8000/health"


class _Log:
    """Stdout plus an append-only file, so a human and an artifact both see it."""

    def __init__(self, path: Path):
        self.handle = path.open("a", encoding="utf-8", newline="")

    def emit(self, text: str = "") -> None:
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


@dataclass(frozen=True)
class Step:
    """One check: either a command, or a function for a multi-part assertion."""

    name: str
    why: str
    ci_step: str
    command: tuple[str, ...] = ()
    run: Callable[[_Log], bool] | None = None
    expect: str = ""
    env: tuple[tuple[str, str], ...] = ()
    requires: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("local", "ci")
    posix_only: bool = False
    gate_run_marker: bool = False


def _tool(name: str) -> str | None:
    """Resolve a tool once, so a missing one becomes a skip instead of a traceback."""
    return PYTHON if name == "python" else shutil.which(name)


def _py(*args: str) -> tuple[str, ...]:
    return (PYTHON, "-m", *args)


def _pytest(*paths: str) -> tuple[str, ...]:
    # -rs prints every skip with its reason: without it, "the data is not here"
    # and "the test never ran" look identical in a CI log.
    return (PYTHON, "-m", "pytest", *paths, "-q", "-rs")


def _sh(script: str) -> tuple[str, ...]:
    """POSIX sh, explicitly: the hook layer is written for sh, not for bash."""
    shell = shutil.which("sh") or shutil.which("bash") or "sh"
    return (shell, "-c", script)


def _run(
    command: tuple[str, ...],
    log: _Log,
    env: dict[str, str] | None = None,
    expect: str = "",
    cwd: Path | None = None,
) -> bool:
    """Run a command, streaming its output to the console and the log."""
    log.emit(f"$ {' '.join(str(part) for part in command)}")
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd or REPO_ROOT),
        env=dict(os.environ if env is None else env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    captured: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        captured.append(line)
        log.emit(line.rstrip("\n"))
    code = process.wait()

    if expect and expect not in "".join(captured):
        log.emit(f"!! expected output never appeared: {expect!r}")
        return False
    if code != 0:
        log.emit(f"!! exit status {code}")
        return False
    return True


# --------------------------------------------------------------------------
# Steps that are more than a command
# --------------------------------------------------------------------------


def _git_config(key: str) -> str:
    result = subprocess.run(
        ["git", "config", "--get", key],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _install_round_trip(log: _Log) -> bool:
    """Install, uninstall and reinstall husky, asserting the observable state.

    `core.hooksPath` and `.husky/_` are the only two things an install may
    change, so both are checked at every stage. An uninstall that leaves the
    generated shims behind is the failure this catches.
    """
    node = shutil.which("node")
    if not node:
        log.emit("!! node is not installed")
        return False

    env = {**os.environ, "HUSKY": "1"}

    def stage(script: str) -> bool:
        return _run((node, script), log, env=env)

    if not stage(".husky/install.mjs"):
        return False
    if _git_config("core.hooksPath") != ".husky/_":
        log.emit("!! core.hooksPath was not set to .husky/_")
        return False
    if not (REPO_ROOT / ".husky" / "_" / "h").is_file():
        log.emit("!! .husky/_/h is missing after an install")
        return False

    if not stage(".husky/uninstall.mjs"):
        return False
    if _git_config("core.hooksPath"):
        log.emit("!! core.hooksPath survived an uninstall")
        return False
    if (REPO_ROOT / ".husky" / "_").exists():
        log.emit("!! .husky/_ survived an uninstall")
        return False

    if not stage(".husky/install.mjs"):
        return False
    return _git_config("core.hooksPath") == ".husky/_"


def _doctor_rejects_flag_shaped_hookspath(log: _Log) -> bool:
    """A flag written into core.hooksPath must be caught, not tolerated.

    `npx husky --version` does exactly that damage. The setting is restored in
    a finally block because this runs against the real checkout, and leaving it
    broken would break the next commit -- including the one this gate is part of.
    """
    previous = _git_config("core.hooksPath")
    try:
        subprocess.run(
            ["git", "config", "core.hooksPath", "--version/_"],
            cwd=str(REPO_ROOT),
            check=True,
        )
        log.emit("$ git config core.hooksPath --version/_")
        if _run(_py("scripts.hooks.verify"), log, expect=""):
            log.emit("!! the doctor accepted a flag-shaped core.hooksPath")
            return False
        log.emit("   the doctor rejected it, as it should")
        return True
    finally:
        subprocess.run(
            ["git", "config", "core.hooksPath", previous or ".husky/_"],
            cwd=str(REPO_ROOT),
            check=False,
        )


def _shellcheck(log: _Log) -> bool:
    """Shellcheck the bootstrap, the helpers and every generated shim.

    The shim list comes from the hook layer itself, so adding a hook cannot
    quietly skip this check.
    """
    from scripts.hooks import _runtime as rt

    shellcheck = shutil.which("shellcheck")
    if not shellcheck:
        log.emit("!! shellcheck is not installed")
        return False
    scripts = [
        ".husky/lib/bootstrap.sh",
        ".husky/lib/common.sh",
        "tools/husky-conformance/run.sh",
        *(f".husky/{hook}" for hook in rt.HOOKS),
    ]
    return _run((shellcheck, "--shell=sh", "--severity=error", *scripts), log)


def _wait_for_health(log: _Log, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as response:
                body = response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError):
            time.sleep(2)
            continue
        if "healthy" in body:
            log.emit(f"   {HEALTH_URL} reports healthy")
            return True
        time.sleep(2)
    log.emit(f"!! {HEALTH_URL} never reported healthy within {timeout:.0f}s")
    return False


def _docker_rehearsal(log: _Log) -> bool:
    """Run Gate 3's steps here, so a push to main is not the first time.

    Gate 3 failed 95 runs in a row for a missing directory whose only evidence
    was a CI log that needs a signed-in browser. This is the same sequence --
    build the dashboard, build the images, wait for health, run the smoke tests,
    tear down -- with a readable log on disk either way.
    """
    docker = shutil.which("docker")
    if not docker:
        log.emit("!! docker is not installed")
        return False

    npm = shutil.which("npm")
    if npm:
        # The same two commands the Gate 3 workflow runs, in the same order: the
        # image COPYs dashboard/dist/, so it has to exist before the build.
        if not _run(
            (npm, "--prefix", "dashboard", "install", "--no-audit", "--no-fund"),
            log,
        ):
            log.emit("!! the dashboard dependencies could not be installed")
            return False
        if not _run((npm, "--prefix", "dashboard", "run", "build"), log):
            log.emit("!! the dashboard build failed (api/Dockerfile COPYs dist/)")
            return False
    else:
        log.emit("   npm is absent; relying on the tracked dashboard/dist/.gitkeep")

    try:
        if not _run((docker, "compose", "build"), log):
            return False
        if not _run((docker, "compose", "up", "-d"), log):
            return False
        if not _wait_for_health(log):
            return False
        return _run(
            (PYTHON, "-m", "pytest", "smoke_tests/", "-v", "--tb=short"),
            log,
        )
    finally:
        # Unconditional: leftover containers poison the next run, and these logs
        # are the only description of a failure that happened inside them.
        _run((docker, "compose", "logs", "--no-color"), log)
        _run((docker, "compose", "down"), log)


STEPS: tuple[Step, ...] = (
    # First, because everything below assumes the hooks are wired: a fresh
    # clone has no core.hooksPath and no generated .husky/_/, so `doctor
    # --strict` cannot pass before this runs. It did run first here, which is
    # why the verification job failed on all three platforms while passing on
    # every developer machine -- those already had husky installed.
    Step(
        name="install",
        why="install, uninstall and reinstall leave exactly the state they should",
        ci_step="Install, uninstall and reinstall",
        run=_install_round_trip,
        requires=("git", "python", "node"),
    ),
    Step(
        name="doctor",
        why="the hook wiring is intact, strictly (a missing piece is a failure here)",
        ci_step="Verify the hook wiring",
        command=_py("scripts.hooks.verify", "--strict"),
        requires=("git", "python"),
    ),
    Step(
        name="inventory",
        why="all 14 husky hooks are wired and dispatched",
        ci_step="List the hook inventory",
        command=_py("scripts.hooks.verify", "--list"),
        expect="14/14 husky hooks",
        requires=("git", "python"),
    ),
    Step(
        name="hook-tests",
        why=(
            "the hook layer's tests pass with only pytest, PyYAML and ruff "
            "installed -- ruff is not imported, it is run, which is why it "
            "was missing from every job that ran these tests"
        ),
        ci_step="Run the hook tests",
        command=_pytest(*HOOK_TEST_FILES),
        requires=("git", "python"),
    ),
    Step(
        name="hostile-environment",
        why="the same tests pass with every hook switch exported against them",
        ci_step="Re-run the hook tests against a hostile environment",
        command=_pytest(*HOOK_TEST_FILES),
        env=(
            ("GATEKEEPER_TRACE", "1"),
            ("HUSKY", "2"),
            ("GATEKEEPER_SKIP_HOOKS", "all"),
            ("GATEKEEPER_HOOK_TESTS", "0"),
            ("GATE1_BLOCK", "1"),
            ("GATEKEEPER_ALLOW_PROTECTED", "1"),
        ),
        gate_run_marker=True,
        requires=("git", "python"),
    ),
    Step(
        name="trace",
        why="HUSKY=2 reaches the Python layer and names the interpreter it chose",
        ci_step="Trace a hook end to end",
        command=_sh("HUSKY=2 sh .husky/_/pre-auto-gc"),
        expect="pre-auto-gc: interpreter=",
        requires=("git", "sh"),
        posix_only=True,
    ),
    Step(
        name="real-shims",
        why="the real .husky/_ -> .husky/<hook> chain exits 0 through the shell git uses",
        ci_step="Run the real pre-commit shim",
        command=_sh(
            "sh .husky/_/pre-commit && sh .husky/_/pre-applypatch && "
            "sh .husky/_/post-applypatch && sh .husky/_/pre-auto-gc"
        ),
        requires=("git", "sh"),
        posix_only=True,
    ),
    Step(
        name="commit-msg-shim",
        why="an invalid subject is rejected by the real commit-msg shim",
        ci_step="Reject a bad commit subject through the real commit-msg shim",
        command=_sh(
            'msg="${TMPDIR:-/tmp}/gatekeeper-commit-msg.txt" && '
            'printf "this is not conventional\\n" > "$msg" && '
            '! sh .husky/_/commit-msg "$msg"'
        ),
        requires=("git", "sh"),
        posix_only=True,
    ),
    Step(
        name="bootstrap",
        why="the bootstrap resolves an interpreter on the platform that needs it most",
        ci_step="Resolve the interpreter through the real bootstrap",
        command=_sh(
            ". .husky/lib/bootstrap.sh && gatekeeper_run_hook post-checkout 0 0 0"
        ),
        requires=("git", "sh"),
    ),
    Step(
        name="doctor-rejects-flag",
        why="a flag-shaped core.hooksPath is caught rather than tolerated",
        ci_step="The doctor catches a flag-shaped hooksPath",
        run=_doctor_rejects_flag_shaped_hookspath,
        requires=("git", "python"),
    ),
    Step(
        name="conformance",
        why="husky's own vendored suite passes against the husky we install",
        ci_step="Run husky's upstream conformance suite",
        command=_sh("sh tools/husky-conformance/run.sh --require"),
        requires=("git", "node", "sh"),
        posix_only=True,
    ),
    Step(
        name="shellcheck",
        why="no shell-level syntax error can take every hook down silently",
        ci_step="Shellcheck every hook shell script",
        run=_shellcheck,
        requires=("git", "python", "shellcheck"),
    ),
    Step(
        name="project-suite",
        why="the whole project suite still passes with the hooks installed",
        ci_step="Run the test suite",
        command=_pytest("tests/"),
        profiles=("local",),
        requires=("git", "python"),
    ),
    Step(
        name="workflow-invariants",
        why="the CI fixes themselves are asserted, not only the code they cover",
        ci_step="Run the test suite",
        command=_pytest("tests/test_ci_workflows.py"),
        requires=("git", "python"),
    ),
    Step(
        name="docker-rehearsal",
        why="Gate 3's steps, run locally before a push to main runs them for real",
        ci_step="Build Docker containers",
        run=_docker_rehearsal,
        profiles=("local",),
        requires=("docker",),
    ),
)


def _skip_reason(step: Step, profile: str) -> str:
    if profile not in step.profiles:
        return f"not part of the {profile} profile"
    if step.posix_only and IS_WINDOWS:
        return "POSIX-only step"
    missing = [tool for tool in step.requires if _tool(tool) is None]
    if missing:
        return f"missing tool: {', '.join(missing)}"
    return ""


def _execute(step: Step, log: _Log) -> bool:
    env = {**os.environ, **dict(step.env)} if step.env else None
    marker = REPO_ROOT / ".git" / "gatekeeper" / "gate_run.json"
    if step.gate_run_marker:
        # The pre-push hook writes this marker and pre-auto-gc declines while it
        # is there, so a test that reads it has to be handed the real thing.
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f'{{"label":"pre-push","pid":1,"started_at":{int(time.time())}}}\n',
            encoding="utf-8",
        )
    try:
        if step.run is not None:
            return bool(step.run(log))
        return _run(step.command, log, env=env, expect=step.expect)
    finally:
        if step.gate_run_marker:
            marker.unlink(missing_ok=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.ci.verify_all",
        description="Run the verification matrix this project's CI runs.",
    )
    parser.add_argument(
        "--profile",
        choices=("local", "ci"),
        default="ci" if os.environ.get("CI") else "local",
        help="which steps apply (default: ci when $CI is set, else local)",
    )
    parser.add_argument("--only", action="append", default=[], metavar="NAME")
    parser.add_argument("--docker", action="store_true", help="include the Gate 3 rehearsal")
    parser.add_argument("--list", action="store_true", help="show the matrix and exit")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failure instead of running every step",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    selected = [
        step
        for step in STEPS
        if (args.docker or step.name != "docker-rehearsal")
        and (not args.only or step.name in args.only)
    ]

    log = _Log(LOG_PATH)
    try:
        log.emit("")
        log.emit("=" * 72)
        log.emit(
            f"verify_all: profile={args.profile} python={sys.version.split()[0]} "
            f"platform={sys.platform}"
        )
        log.emit("=" * 72)

        if args.list:
            for step in selected:
                log.emit(f"  {step.name:22} {step.why}")
                log.emit(f"  {'':22} CI: {step.ci_step}")
            return 0

        results: list[tuple[str, str]] = []
        for step in selected:
            reason = _skip_reason(step, args.profile)
            log.emit("")
            log.emit(f"--- {step.name}: {step.why}")
            if reason:
                log.emit(f"skip: {reason}")
                results.append((step.name, f"skip ({reason})"))
                continue
            passed = _execute(step, log)
            results.append((step.name, "pass" if passed else "FAIL"))
            if not passed:
                log.emit(f"!! {step.name} failed")
                if args.fail_fast:
                    break

        failed = [name for name, status in results if status == "FAIL"]
        log.emit("")
        log.emit("=" * 72)
        for name, status in results:
            log.emit(f"  {status:6} {name}")
        log.emit("=" * 72)
        log.emit(f"verify_all: {len(results) - len(failed)}/{len(results)} steps ok")
        if failed:
            log.emit(f"verify_all: FAILED -> {', '.join(failed)}")
            return 1
        log.emit("verify_all: all steps passed")
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
