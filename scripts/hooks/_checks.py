#!/usr/bin/env python3
"""Content checks shared by the hooks that guard a commit.

Two hooks gate the *content* of a commit rather than its message:

* ``pre-commit``      -- runs when you `git commit`
* ``pre-applypatch``  -- runs when `git am` applies a patch

Git only calls one of them per command, so a patch series would otherwise
skip the checks a hand-made commit goes through. They are the same checks,
so they live here once and both hooks call in.

Both are deliberately cheap, because they run on every commit:

1. ``ruff`` on the staged Python files only -- not the whole tree, so the
   hook scales with the size of the change rather than the size of the
   repository.
2. Leftover conflict markers in the staged content, using git's own
   ``diff --check`` rather than re-implementing a scanner.

Style-only complaints (trailing whitespace) are reported but never block.
"""

from __future__ import annotations

import subprocess

from scripts.hooks import _runtime as rt

RUFF_TIMEOUT = 120


def conflict_markers() -> list[str]:
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


def staged_python_files() -> list[str]:
    """Staged Python files that still exist in the work tree.

    A staged file can be gone from disk (renamed away, deleted then
    re-added); passing the path to ruff would then be a usage error rather
    than a lint result.
    """
    return [f for f in rt.staged_files("*.py") if (rt.REPO_ROOT / f).exists()]


def run(*, hook: str, staged: list[str]) -> list[str]:
    """Run the content checks and return human-readable failures ([] = clean).

    ``hook`` only labels the trace and the ruff invocation, so that output
    says which git hook is complaining.
    """
    failures: list[str] = []

    if staged:
        rt.trace(f"{hook}: linting {len(staged)} staged Python file(s)")
        code = rt.run(
            rt.python_module("ruff", "check", *staged),
            label=f"ruff check ({len(staged)} staged file(s))",
            timeout=RUFF_TIMEOUT,
        )
        if code != 0:
            failures.append("ruff reported lint errors in staged Python files")

    offending = conflict_markers()
    if offending:
        rt.log(f"{rt.BANNER} conflict markers staged in:")
        for path in offending:
            rt.log(f"  - {path}")
        failures.append("unresolved merge conflict markers")

    return failures


def block(hook: str, failures: list[str], *, bypass: str) -> int:
    """Report failures for a blocking hook and return its exit code."""
    if not failures:
        return 0
    rt.log(f"{rt.BANNER} {hook} blocked:")
    for failure in failures:
        rt.log(f"  - {failure}")
    rt.log(f"{rt.BANNER} bypass with: {bypass}")
    return 1
