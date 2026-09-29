#!/usr/bin/env python3
"""pre-commit hook: lint what is actually being committed.

The checks themselves live in ``_checks.py``, because ``git am`` applies
patches without ever running this hook and ``pre-applypatch`` has to run the
same ones. What is left here is the hook's own policy: run the checks on the
staged content, report, and block.

Bypass: ``GATEKEEPER_SKIP_HOOKS=pre-commit git commit ...``
"""

from __future__ import annotations

import sys

from scripts.hooks import _checks
from scripts.hooks import _runtime as rt

BYPASS = f"{rt.SKIP_ENV}=pre-commit git commit ..."


def main() -> int:
    rt.enter("pre-commit")
    if rt.skipped("pre-commit"):
        rt.log(f"{rt.BANNER} pre-commit skipped via {rt.SKIP_ENV}")
        return 0

    staged = _checks.staged_python_files()
    failures = _checks.run(hook="pre-commit", staged=staged)

    code = _checks.block("pre-commit", failures, bypass=BYPASS)
    if code == 0 and staged:
        rt.log(f"{rt.BANNER} pre-commit clean")
    return code


if __name__ == "__main__":
    sys.exit(main())
