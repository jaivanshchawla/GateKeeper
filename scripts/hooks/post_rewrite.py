#!/usr/bin/env python3
"""post-rewrite hook: report history that was rewritten.

Git calls this as ``post-rewrite <command>`` where command is ``amend`` or
``rebase``, with the rewritten sha pairs on stdin as ``<old> <new>`` lines.

Rewritten commits keep their content but lose their identity, so anything
already scored against the old shas -- the Gate 2 status check, the
dashboard timeline, the local outcome log -- now refers to commits that no
longer exist. The hook flags that rather than silently letting the records
drift.
"""

from __future__ import annotations

import sys

from scripts.hooks import _runtime as rt


def report(stdin_data: str) -> None:
    pairs = [line.split() for line in stdin_data.splitlines() if len(line.split()) >= 2]
    if not pairs:
        return
    rt.log(f"{rt.BANNER} {len(pairs)} commit(s) were rewritten")
    rt.log(f"{rt.BANNER} previously scored shas are now stale; re-run: gatekeeper check")


def main(argv: list[str]) -> int:
    # Read stdin before anything else: it carries the old/new sha pairs and
    # can only be consumed once.
    stdin_data = "" if sys.stdin.isatty() else sys.stdin.read()
    return rt.advisory("post-rewrite", lambda: report(stdin_data))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
