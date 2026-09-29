#!/usr/bin/env python3
"""pre-auto-gc hook: keep background housekeeping out of a running gate.

Git runs ``git gc --auto`` on its own once enough loose objects accumulate --
typically right after a commit, a merge or a fetch. The gate is the one part
of this repository that reads large amounts of history: the pre-push hook
diffs the outgoing range, scores each commit with the risk model and runs the
test suite, which can take minutes.

Garbage collection repacking the object store underneath a long read is
contention at best, and a lock conflict at worst, so the gate drops a marker
when it starts and this hook declines the collection while that marker is
fresh. A non-zero exit here is not an error and does not fail anything: it
tells git to skip the automatic collection for now. The next git command
after the gate finishes will collect as usual.

The marker is ignored once it is older than ``GATE_RUN_STALE_SECONDS``, so a
gate killed mid-run cannot wedge garbage collection forever.

Bypass: ``GATEKEEPER_SKIP_HOOKS=pre-auto-gc git gc ...``
"""

from __future__ import annotations

import sys

from scripts.hooks import _runtime as rt


def decide() -> int:
    if rt.skipped("pre-auto-gc"):
        rt.trace("pre-auto-gc: skipped, collection allowed")
        return 0

    if rt.gate_run_active():
        rt.log(f"{rt.BANNER} skipping background gc: a gate hook is running")
        return 1

    rt.trace("pre-auto-gc: no gate running, collection allowed")
    return 0


def main() -> int:
    rt.enter("pre-auto-gc")
    return decide()


if __name__ == "__main__":
    sys.exit(main())
