#!/usr/bin/env python3
"""post-checkout hook: report the gate context of the branch you landed on.

Git calls this as ``post-checkout <prev-head> <new-head> <branch-flag>``.
The flag is 1 for a branch checkout and 0 for a file checkout, so file
checkouts stay silent.

Knowing you have just landed on a protected branch, or on a branch whose
Gate 2 status is unknown, is cheaper to learn here than at push time.
"""

from __future__ import annotations

import sys

from scripts.hooks import _policy as policy
from scripts.hooks import _runtime as rt


def report() -> None:
    branch = policy.current_branch()
    if not branch:
        rt.log(f"{rt.BANNER} checked out a detached HEAD; no branch context")
        return

    if policy.is_protected(branch):
        rt.log(f"{rt.BANNER} on protected branch '{branch}': direct commits, rebases and")
        rt.log(f"{rt.BANNER} merges are refused locally (see {policy.CONFIG_FILENAME})")
    else:
        rt.log(f"{rt.BANNER} checked out '{branch}'")


def main(argv: list[str]) -> int:
    # argv[2] is the branch flag; anything but "1" is a file checkout.
    if len(argv) < 3 or argv[2] != "1":
        return 0
    return rt.advisory("post-checkout", report)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
