#!/usr/bin/env python3
"""post-commit hook: record the commit for local outcome tracking.

Writes one JSON line per commit to ``<git-dir>/gatekeeper/hook_events.jsonl``.
It lives inside the git directory on purpose: a hook that litters the work
tree with an untracked file would show up in every ``git status``.

This is the local half of Gatekeeper's outcome tracking (see
``ml/outcomes.py``): the commit is recorded at creation time so it can later
be joined against what actually happened to it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from scripts.hooks import _policy as policy
from scripts.hooks import _runtime as rt

EVENTS_DIR = "gatekeeper"
EVENTS_FILE = "hook_events.jsonl"


def _git_dir():
    code, out = rt.git("rev-parse", "--git-dir")
    if code != 0 or not out:
        return None
    path = rt.REPO_ROOT / out.strip()
    return path if path.is_dir() else None


def record() -> None:
    git_dir = _git_dir()
    if git_dir is None:
        return

    code, sha = rt.git("rev-parse", "HEAD")
    if code != 0 or not sha:
        return
    _, subject = rt.git("log", "-1", "--format=%s")
    _, author = rt.git("log", "-1", "--format=%an")

    event = {
        "event": "commit",
        "sha": sha.strip(),
        "branch": policy.current_branch(),
        "author": author.strip(),
        "subject": subject.strip(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "gate": 1,
    }

    directory = git_dir / EVENTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / EVENTS_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")

    rt.log(f"{rt.BANNER} recorded {policy.short_sha(sha)} for local outcome tracking")


def main() -> int:
    return rt.advisory("post-commit", record)


if __name__ == "__main__":
    sys.exit(main())
