#!/usr/bin/env python3
"""pre-commit hook: lint what is actually being committed.

Two checks, both deliberately cheap (this runs on every commit):

1. ruff on the staged Python files only — not the whole tree, so the hook
   scales with the size of the change rather than the size of the repo.
2. leftover conflict markers in the staged content, using git's own
   ``diff --check`` instead of re-implementing a scanner.

Style-only problems (trailing whitespace) are reported but never block.

Bypass: ``GATEKEEPER_SKIP_HOOKS=pre-commit git commit ...``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a file by husky
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.hooks import _runtime as rt
else:
    from scripts.hooks import _runtime as rt

RUFF_TIMEOUT = 120


def _staged_conflict_markers() -> list[str]:
    """Paths whose staged content still contains a merge conflict marker."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--check", "--"],
        cwd=rt.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    offending: list[str] = []
    for line in result.stdout.splitlines():
        if "conflict marker" not in line:
            continue
        # Format: "<path>:<line>: leftover conflict marker"
        path = line.split(":", 1)[0]
        if path not in offending:
            offending.append(path)
    return offending


def main() -> int:
    if rt.skipped("pre-commit"):
        rt.log(f"{rt.BANNER} pre-commit skipped via {rt.SKIP_ENV}")
        return 0

    failures: list[str] = []

    python_files = [f for f in rt.staged_files("*.py") if (rt.REPO_ROOT / f).exists()]
    if python_files:
        code = rt.run(
            rt.python_module("ruff", "check", *python_files),
            label=f"ruff check ({len(python_files)} staged file(s))",
            timeout=RUFF_TIMEOUT,
        )
        if code != 0:
            failures.append("ruff reported lint errors in staged Python files")

    conflicts = _staged_conflict_markers()
    if conflicts:
        rt.log(f"{rt.BANNER} conflict markers staged in:")
        for path in conflicts:
            rt.log(f"  - {path}")
        failures.append("unresolved merge conflict markers")

    if failures:
        rt.log(f"{rt.BANNER} pre-commit blocked:")
        for failure in failures:
            rt.log(f"  - {failure}")
        rt.log(f"{rt.BANNER} bypass with: GATEKEEPER_SKIP_HOOKS=pre-commit git commit ...")
        return 1

    if python_files:
        rt.log(f"{rt.BANNER} pre-commit clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
