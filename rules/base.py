#!/usr/bin/env python3
"""
Base classes for the Gatekeeper rule engine.

Each rule implements BaseRule.evaluate(commit_context) -> RuleResult.
Severity ladder: info (report only), warn (comment, do not fail), block (fail the gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class RuleResult:
    """Output of a single rule evaluation."""
    rule_name: str
    severity: Severity
    passed: bool  # True = no issue, False = rule triggered
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class CommitContext:
    """All information a rule needs about a commit.

    Populated by the rule engine from score_pr.py or the API.
    Not every field is available in every context (e.g. direct_to_main
    is only available in a push context).
    """
    # Core commit metadata
    hash: str = ""
    author: str = ""
    message: str = ""
    files: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0
    files_touched: int = 0
    dirs_touched: int = 0
    is_merge: bool = False
    hour_of_day: int = 12
    day_of_week: int = 0  # 0=Monday

    # Author history
    author_prior_commits: int = 0

    # File history (from model features)
    file_revert_count_max: int = 0
    file_prior_changes_max: int = 0
    file_prior_risky_max: int = 0

    # Repo context
    repo_name: str = ""
    default_branch: str = "main"
    is_direct_push: bool = False  # True if pushing directly to default branch

    # ML model output (optional, filled by engine after scoring)
    risk_score: float = 0.0
    risk_label: str = ""


class BaseRule:
    """Abstract base for all Gatekeeper rules.

    Subclasses must implement evaluate(). The constructor receives
    per-repo overrides from .gatekeeper.yml.
    """

    name: str = "base"
    default_config: dict[str, Any] = {}
    default_severity: Severity = Severity.WARN

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**self.default_config, **(config or {})}
        self.severity = Severity(self.config.get("severity", self.default_severity.value))
        self.enabled = self.config.get("enabled", True)

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        """Evaluate the rule against a commit context.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def _result(self, passed: bool, message: str, evidence: dict | None = None) -> RuleResult:
        return RuleResult(
            rule_name=self.name,
            severity=self.severity,
            passed=passed,
            message=message,
            evidence=evidence or {},
        )
