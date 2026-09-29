#!/usr/bin/env python3
"""applypatch-msg hook: apply the commit message rules to `git am` patches.

Git calls this as ``applypatch-msg <msgfile>`` while applying a mailbox with
``git am``. Those commits never go through ``commit-msg``, so without this
hook a patch series can land with subjects the rest of the gate would have
rejected -- the same rule, two entry points.

The rules themselves are not reimplemented: this calls the same
``lint_subject`` the commit-msg hook uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.hooks import _runtime as rt
from scripts.hooks.commit_msg import first_meaningful_line, lint_subject


def main(argv: list[str]) -> int:
    rt.enter("applypatch-msg", argv)
    if rt.skipped("applypatch-msg"):
        rt.log(f"{rt.BANNER} applypatch-msg skipped via {rt.SKIP_ENV}")
        return 0

    if not argv:
        return 0

    message_path = Path(argv[0])
    if not message_path.exists():
        # Nothing to check; git will fail on its own terms if that matters.
        return 0

    subject = first_meaningful_line(message_path.read_text(encoding="utf-8"))
    problems = lint_subject(subject)
    if not problems:
        return 0

    rt.log(f"{rt.BANNER} patch message rejected:")
    for problem in problems:
        rt.log(f"  - {problem}")
    rt.log(f"{rt.BANNER} fix the patch with: git am --abort")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
