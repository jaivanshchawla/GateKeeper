#!/usr/bin/env python3
"""
U.5b: Scoped policy engine.

Extends the rule engine with:
- Branch-aware: different rule severities per target branch
- Path-scoped: stricter rules on critical paths
- Time-based: freeze windows blocking high-band merges
"""
from __future__ import annotations

import fnmatch
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from rules.base import CommitContext, Severity


@dataclass
class ScopedPolicy:
    """Policy that applies different severities based on context."""
    branch_rules: dict[str, dict[str, str]] = field(default_factory=dict)
    path_rules: dict[str, dict[str, str]] = field(default_factory=dict)
    freeze_windows: list[dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "branch_rules": self.branch_rules,
            "path_rules": self.path_rules,
            "freeze_windows": self.freeze_windows,
        }


DEFAULT_SCOPED_CONFIG = {
    "branch_rules": {
        "main": {"large_change": "block", "config_and_code": "block", "revert_hotspot": "block"},
        "develop": {"large_change": "warn", "config_and_code": "warn"},
        "release/*": {"*": "block"},  # all rules block on release branches
    },
    "path_rules": {
        "auth/**": {"severity_boost": "block"},
        "payments/**": {"severity_boost": "block"},
        "migrations/**": {"severity_boost": "warn"},
        "**/Dockerfile": {"dependency_change": "block"},
        "**/*.lock": {"dependency_change": "block"},
    },
    "freeze_windows": [],
}


class ScopedPolicyEngine:
    """Evaluates policies with branch/path/time awareness."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**DEFAULT_SCOPED_CONFIG, **(config or {})}

    def resolve_severity(
        self,
        rule_name: str,
        base_severity: Severity,
        ctx: CommitContext,
    ) -> Severity:
        """Resolve the effective severity for a rule given context."""
        severity = base_severity

        # Branch override
        branch_rules = self.config.get("branch_rules", {})
        target = getattr(ctx, "target_branch", ctx.default_branch)
        if target in branch_rules:
            br = branch_rules[target]
            if rule_name in br:
                severity = Severity(br[rule_name])
            elif "*" in br:
                severity = Severity(br["*"])

        # Path override (boost severity for critical paths)
        path_rules = self.config.get("path_rules", {})
        for path_pattern, overrides in path_rules.items():
            for f in ctx.files:
                if fnmatch.fnmatch(f, path_pattern):
                    if "severity_boost" in overrides:
                        boost = Severity(overrides["severity_boost"])
                        if self._severity_rank(boost) > self._severity_rank(severity):
                            severity = boost
                    if rule_name in overrides:
                        override = Severity(overrides[rule_name])
                        if self._severity_rank(override) > self._severity_rank(severity):
                            severity = override

        return severity

    def check_freeze(self, ctx: CommitContext) -> str | None:
        """Check if the current time falls in a freeze window.

        Returns the freeze name if blocked, None if clear.
        """
        now = datetime.now(timezone.utc)
        for window in self.config.get("freeze_windows", []):
            start = datetime.fromisoformat(window["start"])
            end = datetime.fromisoformat(window["end"])
            if start <= now <= end:
                # Check if this window applies to the target branch
                branches = window.get("branches", [])
                target = getattr(ctx, "target_branch", ctx.default_branch)
                if not branches or target in branches:
                    return window.get("name", "freeze")
        return None

    def format_scoped_results(
        self,
        rule_results: list,
        ctx: CommitContext,
    ) -> str:
        """Format results with branch/path context."""
        lines = []
        target = getattr(ctx, "target_branch", ctx.default_branch)
        lines.append(f"**Branch:** `{target}`")

        freeze = self.check_freeze(ctx)
        if freeze:
            lines.append(f"🚨 **FREEZE ACTIVE:** `{freeze}` — high-risk merges blocked")

        critical_paths = []
        path_rules = self.config.get("path_rules", {})
        for f in ctx.files:
            for pattern in path_rules:
                if fnmatch.fnmatch(f, pattern):
                    critical_paths.append(f)
        if critical_paths:
            lines.append(f"**Critical paths:** {', '.join(set(critical_paths)[:5])}")

        return "\n".join(lines)

    @staticmethod
    def _severity_rank(s: Severity) -> int:
        return {"info": 0, "warn": 1, "block": 2}.get(s.value, 0)


def load_scoped_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract scoped policy config from .gatekeeper.yml."""
    return {
        "branch_rules": config.get("branch_rules", DEFAULT_SCOPED_CONFIG["branch_rules"]),
        "path_rules": config.get("path_rules", DEFAULT_SCOPED_CONFIG["path_rules"]),
        "freeze_windows": config.get("freeze_windows", []),
    }
