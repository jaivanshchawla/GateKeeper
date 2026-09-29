#!/usr/bin/env python3
"""post-applypatch hook: record commits that `git am` created.

``git am`` never runs ``post-commit``, so a patch series used to land with no
trace in the local outcome log at all -- the log looked as though the commits
had never been made. This closes that hole by recording the same event from
the other end of the same operation.

Git calls this with no arguments, after the commit has been created. Nothing
can abort at this point, so it is a reporting hook: it must never fail the
`am` it is attached to.

Bypass: ``GATEKEEPER_SKIP_HOOKS=post-applypatch git am ...``
"""

from __future__ import annotations

import sys

from scripts.hooks import _events
from scripts.hooks import _runtime as rt


def record() -> None:
    record = _events.record_commit("applypatch", gate=1)
    if record is None:
        return
    rt.log(
        f"{rt.BANNER} recorded patch commit {record['sha'][:8]} for local outcome tracking"
    )


def main() -> int:
    rt.enter("post-applypatch")
    return rt.advisory("post-applypatch", record)


if __name__ == "__main__":
    sys.exit(main())
