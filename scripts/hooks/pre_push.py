#!/usr/bin/env python3
"""pre-push hook: the local quality gate before code leaves the machine.

Stages, in order:

1. **Gate 1** — ``scripts/pre_push_score.py`` scores the outgoing commits
   and prints the band plus the top risk reasons. It warns by default;
   set ``GATE1_BLOCK=1`` to make it blocking. This is Gatekeeper's own
   product surface, so the hook runs it rather than duplicating it.
2. **ruff** — linted over the *outgoing diff only*, matching pre-commit's
   changed-files semantics. This repo carries pre-existing lint debt, so
   a whole-tree lint at pre-push would block every push and teach people
   to bypass the gate.
3. **pytest** — the test suite. Disable with ``GATEKEEPER_HOOK_TESTS=0``
   (useful for WIP branches); enable in CI by leaving it unset.

The pushed refs arrive on stdin in git's pre-push format; they are read
once here and forwarded verbatim to the Gate 1 script, which expects them.

Bypass: ``GATEKEEPER_SKIP_HOOKS=pre-push git push ...``
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a file by husky
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.hooks import _runtime as rt
else:
    from scripts.hooks import _runtime as rt

TESTS_TIMEOUT = 900
RUFF_TIMEOUT = 300


def _gate1(stdin_data: str) -> int:
    """Run the Gate 1 commit scorer, forwarding the pushed refs."""
    script = rt.REPO_ROOT / "scripts" / "pre_push_score.py"
    if not script.exists():
        rt.log(f"{rt.BANNER} skip: {script.name} not found")
        return 0

    rt.log(f"{rt.BANNER} Gate 1 — scoring outgoing commits")
    try:
        completed = subprocess.run(
            [rt.project_python(), str(script)],
            cwd=rt.REPO_ROOT,
            input=stdin_data,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        rt.log(f"{rt.BANNER} Gate 1 timed out; continuing (Gate 1 never blocks on time)")
        return 0
    # Gate 1 only returns non-zero when GATE1_BLOCK=1 escalated a high band.
    return completed.returncode


def _tests_enabled() -> bool:
    return os.environ.get("GATEKEEPER_HOOK_TESTS", "1") not in {"0", "false", "no"}


def main() -> int:
    # Read stdin first: it can only be consumed once, and only the
    # pre-push stage ever populates it.
    stdin_data = ""
    if not sys.stdin.isatty():
        stdin_data = sys.stdin.read()

    if rt.skipped("pre-push"):
        rt.log(f"{rt.BANNER} pre-push skipped via {rt.SKIP_ENV}")
        return 0

    failures: list[str] = []

    if _gate1(stdin_data) != 0:
        failures.append("Gate 1 flagged a blocking high-risk commit (GATE1_BLOCK=1)")

    changed = [f for f in rt.changed_files(stdin_data, "*.py") if (rt.REPO_ROOT / f).exists()]
    if changed:
        code = rt.run(
            rt.python_module("ruff", "check", *changed),
            label=f"ruff check ({len(changed)} changed file(s))",
            timeout=RUFF_TIMEOUT,
        )
        if code != 0:
            failures.append("ruff reported lint errors in changed Python files")
    else:
        rt.log(f"{rt.BANNER} ruff: no changed Python files in this push")

    if _tests_enabled():
        if rt.run(rt.python_module("pytest", "tests/", "-q"), label="pytest tests/", timeout=TESTS_TIMEOUT) != 0:
            failures.append("test suite failed")
    else:
        rt.log(f"{rt.BANNER} pytest skipped (GATEKEEPER_HOOK_TESTS=0)")

    if failures:
        rt.log(f"{rt.BANNER} pre-push blocked:")
        for failure in failures:
            rt.log(f"  - {failure}")
        rt.log(f"{rt.BANNER} bypass with: GATEKEEPER_SKIP_HOOKS=pre-push git push ...")
        return 1

    rt.log(f"{rt.BANNER} pre-push gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
