#!/usr/bin/env python3
"""post-commit hook: record the commit for local outcome tracking.

The commit is the only moment at which "this change was made, by this
person, on this branch" can be captured by something other than the log
itself, so it is written down here.

The file and its format are owned by ``_events.py``, which the other
reporting hooks share; this module is the commit-shaped caller.

Bypass: ``GATEKEEPER_SKIP_HOOKS=post-commit git commit ...``
"""

from __future__ import annotations

import sys

from scripts.hooks import _events
from scripts.hooks import _runtime as rt

# Re-exported so the stored location has one obvious name regardless of
# which hook is being read.
EVENTS_DIR = _events.EVENTS_DIR
EVENTS_FILE = _events.EVENTS_FILE


def record() -> None:
    record = _events.record_commit("commit", gate=1)
    if record is None:
        return
    rt.log(
        f"{rt.BANNER} recorded {record['sha'][:8]} for local outcome tracking"
    )


def main() -> int:
    rt.enter("post-commit")
    return rt.advisory("post-commit", record)


if __name__ == "__main__":
    sys.exit(main())
