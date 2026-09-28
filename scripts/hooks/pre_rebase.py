#!/usr/bin/env python3
"""pre-rebase hook: refuse to rebase a protected branch.

Rewriting the history of a shared branch breaks every clone and every
open pull request, and it silently invalidates the commit shas Gate 2 and
the dashboard have already scored -- so the gate blocks it locally.

Git calls this as: ``pre-rebase <upstream> [<branch>]``. When no branch is
given the current branch is being rebased.

Bypass: ``GATEKEEPER_ALLOW_PROTECTED=1 git rebase ...``
"""

from __future__ import annotations

import os
import sys

from scripts.hooks import _policy as policy
from scripts.hooks import _runtime as rt

OVERRIDE_ENV = "GATEKEEPER_ALLOW_PROTECTED"


def main(argv: list[str]) -> int:
    if rt.skipped("pre-rebase"):
        rt.log(f"{rt.BANNER} pre-rebase skipped via {rt.SKIP_ENV}")
        return 0

    # argv[0] is the upstream; argv[1], when present, is the branch being
    # rebased. Without it, it is whatever is checked out.
    branch = argv[1].strip() if len(argv) > 1 and argv[1].strip() else policy.current_branch()
    if not policy.is_protected(branch):
        return 0

    if os.environ.get(OVERRIDE_ENV) == "1":
        rt.log(f"{rt.BANNER} rebasing protected branch '{branch}' (overridden)")
        return 0

    rt.log(f"{rt.BANNER} refusing to rebase protected branch '{branch}'")
    rt.log(f"{rt.BANNER} rewriting it would invalidate every commit already scored by Gate 2")
    rt.log(f"{rt.BANNER} protected by 'protected_branches' in {policy.CONFIG_FILENAME}")
    rt.log(f"{rt.BANNER} override with: {OVERRIDE_ENV}=1 git rebase ...")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
