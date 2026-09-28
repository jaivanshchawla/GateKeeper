#!/usr/bin/env python3
"""Branch and configuration policy shared by the Gatekeeper hooks.

Several hooks need to answer the same two questions -- "what does
.gatekeeper.yml say?" and "which branch are we on, and does it matter?" --
so both live here rather than being re-derived per hook.

PyYAML is optional on purpose. A hook may be started by whatever Python the
developer has on PATH, which is not guaranteed to be the project venv, so a
missing parser degrades to built-in defaults instead of breaking the hook.
The project interpreter (used for ruff/pytest/Gate 1) is resolved separately
in _runtime.py.
"""

from __future__ import annotations

from fnmatch import fnmatch

from scripts.hooks import _runtime as rt

CONFIG_FILENAME = ".gatekeeper.yml"

# Used when .gatekeeper.yml is missing, unreadable, or PyYAML is unavailable.
DEFAULT_PROTECTED_BRANCHES = ("main", "master", "release/*", "hotfix/*")

# Branches that are not really branches (detached HEAD, fresh repo).
_NON_BRANCHES = ("HEAD", "")


def gatekeeper_config() -> dict:
    """Load .gatekeeper.yml, or {} when it cannot be read."""
    path = rt.REPO_ROOT / CONFIG_FILENAME
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        rt.log(f"{rt.BANNER} PyYAML not available; using built-in defaults")
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except Exception as exc:
        rt.log(f"{rt.BANNER} could not parse {CONFIG_FILENAME}: {exc}")
        return {}
    return loaded if isinstance(loaded, dict) else {}


def protected_branches() -> tuple[str, ...]:
    """Branch names or glob patterns that hooks must not mutate."""
    configured = gatekeeper_config().get("protected_branches")
    if isinstance(configured, list):
        patterns = tuple(str(item) for item in configured if str(item).strip())
        if patterns:
            return patterns
    return DEFAULT_PROTECTED_BRANCHES


def current_branch() -> str:
    """Short name of the checked-out branch, or '' when detached."""
    code, out = rt.git("rev-parse", "--abbrev-ref", "HEAD")
    if code != 0:
        return ""
    branch = out.strip()
    return "" if branch in _NON_BRANCHES else branch


def branch_matches(branch: str, pattern: str) -> bool:
    """True when ``branch`` is ``pattern``, honouring ``*`` globs."""
    if not branch or not pattern:
        return False
    return branch == pattern or fnmatch(branch, pattern)


def is_protected(branch: str) -> bool:
    """True when ``branch`` matches any protected pattern."""
    return any(branch_matches(branch, pattern) for pattern in protected_branches())


def rule_severity(rule_name: str) -> str | None:
    """Configured severity for a rule, or None when it is not configured."""
    rules = gatekeeper_config().get("rules")
    if not isinstance(rules, dict):
        return None
    rule = rules.get(rule_name)
    if isinstance(rule, dict):
        severity = rule.get("severity")
        return str(severity) if severity else None
    return None


def short_sha(sha: str, length: int = 8) -> str:
    """Abbreviate a commit sha for display."""
    return sha[:length] if sha else ""


def changed_files_between(base: str, head: str = "HEAD") -> list[str]:
    """Files changed in ``base..head``, or [] when either ref is unknown."""
    if not base:
        return []
    code, out = rt.git("diff", "--name-only", f"{base}..{head}")
    if code != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]
