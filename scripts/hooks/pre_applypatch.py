#!/usr/bin/env python3
"""pre-applypatch hook: give `git am` the content checks it would otherwise skip.

``git am`` applies each patch and commits it itself. That path runs three
hooks -- ``applypatch-msg``, this one and ``post-applypatch`` -- and *none* of
the commit hooks. So without this hook, `git am` is a way to land content
that never passed ``pre-commit``: whitespace, conflict markers left in a
patch, files that do not survive the project's linter.

Git calls this with no arguments, after the patch is applied and before the
commit is created. A non-zero exit aborts the `am` and leaves the patch
applied but uncommitted, which is the same state `git am --abort` expects,
so the failure is recoverable.

The checks are not reimplemented: they are the ones ``pre-commit`` runs.

Bypass: ``GATEKEEPER_SKIP_HOOKS=pre-applypatch git am ...``
"""

from __future__ import annotations

import sys

from scripts.hooks import _checks
from scripts.hooks import _runtime as rt

BYPASS = f"{rt.SKIP_ENV}=pre-applypatch git am ... (or: git am --abort)"


def main() -> int:
    rt.enter("pre-applypatch")
    if rt.skipped("pre-applypatch"):
        rt.log(f"{rt.BANNER} pre-applypatch skipped via {rt.SKIP_ENV}")
        return 0

    staged = _checks.staged_python_files()
    failures = _checks.run(hook="pre-applypatch", staged=staged)

    code = _checks.block("pre-applypatch", failures, bypass=BYPASS)
    if code == 0 and staged:
        rt.log(f"{rt.BANNER} pre-applypatch clean")
    return code


if __name__ == "__main__":
    sys.exit(main())
