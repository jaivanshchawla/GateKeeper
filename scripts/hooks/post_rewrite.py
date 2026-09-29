#!/usr/bin/env python3
"""post-rewrite hook: report history that was rewritten.

Git calls this as ``post-rewrite <command>`` where command is ``amend`` or
``rebase``, with the rewritten sha pairs on stdin as ``<old> <new>`` lines.

Rewritten commits keep their content but lose their identity, so anything
already scored against the old shas -- the Gate 2 status check, the
dashboard timeline, the local outcome log -- now refers to commits that no
longer exist. The hook flags that rather than silently letting the records
drift, and writes the old -> new mapping into the event log so the stale
records can actually be corrected instead of merely noticed.
"""

from __future__ import annotations

import sys

from scripts.hooks import _events
from scripts.hooks import _runtime as rt


def parse_pairs(stdin_data: str) -> list[tuple[str, str]]:
    """Parse git's ``<old> <new>`` rewrite lines."""
    pairs: list[tuple[str, str]] = []
    for line in stdin_data.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def report(stdin_data: str, command: str = "") -> None:
    pairs = parse_pairs(stdin_data)
    if not pairs:
        return

    _events.write_event(
        "rewrite",
        command=command,
        rewrites=[{"old": old, "new": new} for old, new in pairs],
    )

    rt.log(f"{rt.BANNER} {len(pairs)} commit(s) were rewritten")
    rt.log(f"{rt.BANNER} previously scored shas are now stale; re-run: gatekeeper check")


def main(argv: list[str]) -> int:
    # Read stdin before anything else: it carries the old/new sha pairs and
    # can only be consumed once.
    stdin_data = "" if sys.stdin.isatty() else sys.stdin.read()
    rt.enter("post-rewrite", argv)
    # argv[0] is the operation, "amend" or "rebase".
    command = argv[0].strip() if argv else ""
    return rt.advisory("post-rewrite", lambda: report(stdin_data, command))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
