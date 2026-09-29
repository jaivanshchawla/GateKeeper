#!/usr/bin/env python3
"""post-merge hook: flag merges that changed the gate itself.

After a merge or pull the local gate can be stale: the model may have been
re-exported, the rule configuration may have changed, or the hook scripts
themselves may have moved on.

Re-running the suite on every pull would make pulling painful, so this
reports only when one of those files actually changed in the merge.

The merge is also recorded in the shared event log, because a pull that
brings work in is exactly the kind of event the outcome tracker otherwise
cannot see.
"""

from __future__ import annotations

import sys

from scripts.hooks import _events
from scripts.hooks import _policy as policy
from scripts.hooks import _runtime as rt

# Files whose change means the local gate no longer matches the checkout.
GATE_RELEVANT = (
    "models/gatekeeper_risk_model.skops",
    "ml/config.yaml",
    ".gatekeeper.yml",
    ".husky/",
    "scripts/hooks/",
    "ruff.toml",
)


def _relevant(changed: list[str]) -> list[str]:
    return [
        path
        for path in changed
        if any(path == prefix or path.startswith(prefix) for prefix in GATE_RELEVANT)
    ]


def _record(changed: list[str], touched: list[str]) -> None:
    """Append a merge event, so the outcome log sees pulls as well."""
    _events.record_commit(
        "merge",
        gate=None,
        files_changed=len(changed),
        gate_files_changed=len(touched),
    )


def check() -> None:
    changed = policy.changed_files_between("ORIG_HEAD")
    touched = _relevant(changed)

    # Recorded even when the outgoing diff cannot be resolved (a fast-forward
    # leaves ORIG_HEAD pointing somewhere unexpected): the merge happening at
    # all is the fact worth keeping.
    _record(changed, touched)

    if not changed:
        return

    if not touched:
        rt.log(f"{rt.BANNER} merged {len(changed)} file(s); gate unchanged")
        return

    rt.log(f"{rt.BANNER} this merge changed the gate itself:")
    for path in sorted(touched)[:10]:
        rt.log(f"{rt.BANNER}   {path}")
    rt.log(f"{rt.BANNER} re-check the local gate with: npm run hooks:verify")
    rt.log(f"{rt.BANNER} re-score the current config with: gatekeeper config")


def main() -> int:
    rt.enter("post-merge")
    return rt.advisory("post-merge", check)


if __name__ == "__main__":
    sys.exit(main())
