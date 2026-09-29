#!/usr/bin/env python3
"""pre-merge-commit hook: refuse to merge into a protected branch.

Git runs this while it is creating a merge commit. Merging straight into
`main` is the most common way an unreviewed change reaches the default
branch, and it is exactly what Gatekeeper's `direct_to_main` rule is about
-- so it is blocked locally rather than only reported after the fact.

Bypass: ``GATEKEEPER_ALLOW_PROTECTED=1 git merge ...``
"""

from __future__ import annotations

import os
import sys

from scripts.hooks import _policy as policy
from scripts.hooks import _runtime as rt

OVERRIDE_ENV = "GATEKEEPER_ALLOW_PROTECTED"


def main() -> int:
    rt.enter("pre-merge-commit")
    if rt.skipped("pre-merge-commit"):
        rt.log(f"{rt.BANNER} pre-merge-commit skipped via {rt.SKIP_ENV}")
        return 0

    branch = policy.current_branch()
    if not policy.is_protected(branch):
        return 0

    incoming = ""
    code, out = rt.git("rev-parse", "--abbrev-ref", "MERGE_HEAD")
    if code == 0:
        incoming = out.strip()

    if os.environ.get(OVERRIDE_ENV) == "1":
        rt.log(f"{rt.BANNER} merging into protected branch '{branch}' (overridden)")
        return 0

    rt.log(f"{rt.BANNER} refusing to merge into protected branch '{branch}'")
    if incoming:
        rt.log(f"{rt.BANNER}   incoming: {incoming}")
    rt.log(f"{rt.BANNER} protected by 'protected_branches' in {policy.CONFIG_FILENAME}")
    rt.log(f"{rt.BANNER} merge through a pull request, or override with:")
    rt.log(f"{rt.BANNER}   {OVERRIDE_ENV}=1 git merge ...")

    severity = policy.rule_severity("direct_to_main")
    if severity:
        rt.log(f"{rt.BANNER}  ({policy.CONFIG_FILENAME}: direct_to_main severity={severity})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
