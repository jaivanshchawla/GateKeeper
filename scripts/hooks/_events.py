#!/usr/bin/env python3
"""The local hook event log, written by every reporting hook.

Gatekeeper's outcome tracking (``ml/outcomes.py``) joins a commit against
what later happened to it -- was it reverted, did it get flagged, how did
the scored risk compare with reality. That join needs a record of the
commit *at the time it was created*, which is the one moment only a hook can
observe.

Several hooks have something worth recording, and they all append to one
newline-delimited JSON file inside the git directory:

* ``post-commit``      -- a commit was created
* ``post-applypatch``  -- a ``git am`` patch created a commit
* ``post-merge``       -- a merge or pull landed
* ``post-rewrite``     -- commits were amended or rebased out of existence

One writer, one file format, so a reader never has to know which hook wrote
a line. The file lives under the git directory rather than the work tree: a
hook that litters the checkout with an untracked file would show up in every
``git status``, including the ones this project's own hooks read.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.hooks import _policy as policy
from scripts.hooks import _runtime as rt

# Kept as module constants, and also re-exported by post_commit, because the
# path is part of the contract with whatever reads the log.
EVENTS_DIR = rt.STATE_DIR
EVENTS_FILE = "hook_events.jsonl"

# How many lines `verify --events` shows by default.
DEFAULT_TAIL = 10


def events_path():
    """Path to the event log, or None outside a git repository."""
    directory = rt.git_dir()
    return None if directory is None else directory / EVENTS_DIR / EVENTS_FILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def head_details() -> dict[str, str]:
    """``sha``, ``author`` and ``subject`` of HEAD, each '' when unknown."""
    code, sha = rt.git("rev-parse", "HEAD")
    _, subject = rt.git("log", "-1", "--format=%s")
    _, author = rt.git("log", "-1", "--format=%an")
    return {
        "sha": sha.strip() if code == 0 else "",
        "subject": subject.strip(),
        "author": author.strip(),
    }


def write_event(event: str, **fields) -> bool:
    """Append one event to the log. Returns False when it cannot be written.

    Never raises: every caller is a reporting hook, and a hook that cannot
    write its note must not fail the git command it hangs off.
    """
    path = events_path()
    if path is None:
        return False

    record = {"event": event, "recorded_at": _now()}
    if event != "rewrite":
        record["branch"] = policy.current_branch()
    record.update({key: value for key, value in fields.items() if value not in (None, "")})

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        return False
    return True


def record_commit(event: str = "commit", gate: int | None = 1, **fields) -> dict | None:
    """Record a newly created commit. Returns the record, or None on failure."""
    details = head_details()
    if not details["sha"]:
        return None
    if gate is not None:
        fields.setdefault("gate", gate)
    record = {"event": event, **details, **fields}
    return record if write_event(**record) else None


def read_events(limit: int = DEFAULT_TAIL) -> list[dict]:
    """The most recent events, oldest first. [] when there is no log."""
    path = events_path()
    if path is None or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    events: list[dict] = []
    for line in lines[-limit:] if limit > 0 else lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            # A torn last line (a hook killed mid-write) must not hide the
            # records around it.
            continue
    return events
