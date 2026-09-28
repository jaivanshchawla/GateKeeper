#!/usr/bin/env python3
"""commit-msg hook: enforce a conventional commit subject.

Run by husky as:  sh .husky/commit-msg "$1"

The type allow-list is taken from this repository's own history (git log
shows feat/fix/docs/test/perf/style/eval/analysis/meas/chore) rather than
invented, so the gate describes how the project already commits.

Exemptions: merge, revert, fixup!/squash! and WIP commits pass through,
and `GATEKEEPER_SKIP_HOOKS=commit-msg git commit ...` bypasses the gate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a file by husky
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.hooks import _runtime as rt
else:
    from scripts.hooks import _runtime as rt

# Conventional Commits types, plus the vocabulary this repo actually uses.
ALLOWED_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
    "meas",
    "eval",
    "analysis",
    "backfill",
)

SUBJECT_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^()\s]+)\))?(?P<breaking>!)?: (?P<subject>\S.*)$"
)

MAX_SUBJECT_LENGTH = 100

# Prefixes Git itself generates, which must never be linted.
GENERATED_PREFIXES = ("Merge ", "Revert ", "fixup!", "squash!", "WIP")


def lint_subject(subject: str) -> list[str]:
    """Return a list of human-readable problems with ``subject`` (empty = ok)."""
    subject = subject.strip()
    if not subject:
        return ["commit subject is empty"]

    if subject.startswith(GENERATED_PREFIXES):
        return []

    match = SUBJECT_RE.match(subject)
    if not match:
        return [
            f"'{subject}' is not a conventional commit subject",
            "expected: <type>(<scope>): <subject>",
            f"allowed types: {', '.join(ALLOWED_TYPES)}",
        ]

    problems: list[str] = []
    commit_type = match.group("type")
    if commit_type not in ALLOWED_TYPES:
        problems.append(
            f"unknown type '{commit_type}' - use one of: {', '.join(ALLOWED_TYPES)}"
        )

    body = match.group("subject")
    if body.endswith("."):
        problems.append("subject should not end with a period")
    if len(subject) > MAX_SUBJECT_LENGTH:
        problems.append(
            f"subject is {len(subject)} chars, keep it under {MAX_SUBJECT_LENGTH}"
        )
    return problems


def _first_meaningful_line(message: str) -> str:
    """First non-empty, non-comment line of a commit message."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def main(argv: list[str]) -> int:
    if rt.skipped("commit-msg"):
        rt.log(f"{rt.BANNER} commit-msg skipped via {rt.SKIP_ENV}")
        return 0

    # `--lint "<subject>"` is used by tests; git passes a message file path.
    if len(argv) >= 2 and argv[0] == "--lint":
        problems = lint_subject(argv[1])
    else:
        message_path = Path(argv[0]) if argv else rt.REPO_ROOT / ".git" / "COMMIT_EDITMSG"
        if not message_path.exists():
            rt.log(f"{rt.BANNER} commit-msg: no message at {message_path}, skipping")
            return 0
        problems = lint_subject(_first_meaningful_line(message_path.read_text(encoding="utf-8")))

    if not problems:
        return 0

    rt.log(f"{rt.BANNER} commit message rejected:")
    for problem in problems:
        rt.log(f"  - {problem}")
    rt.log(
        f"{rt.BANNER} bypass with: GATEKEEPER_SKIP_HOOKS=commit-msg git commit ..."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
