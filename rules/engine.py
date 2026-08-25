#!/usr/bin/env python3
"""
Rule engine: loads .gatekeeper.yml, instantiates rules, evaluates a commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rules.base import CommitContext, RuleResult, Severity
from rules.rules import ALL_RULES


DEFAULT_CONFIG: dict[str, Any] = {
    "rules": {
        "large_change": {"max_lines": 500, "severity": "warn"},
        "too_many_files": {"max_files": 20, "severity": "warn"},
        "no_tests": {"severity": "warn", "exempt_paths": ["docs/**", "*.md"]},
        "config_and_code": {"severity": "warn"},
        "revert_hotspot": {"revert_count": 3, "window_days": 60, "severity": "block"},
        "first_touch": {"severity": "info"},
        "weekend_deploy": {"severity": "info"},
        "stale_file": {"days": 180, "severity": "info"},
        "direct_to_main": {"severity": "warn"},
    },
    "ml_scoring": {
        "enabled": True,
        "band_thresholds": {"high": 0.90, "medium": 0.75},
    },
    "fail_on": ["block"],
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load .gatekeeper.yml, falling back to defaults."""
    if config_path is None:
        config_path = Path(".gatekeeper.yml")
    else:
        config_path = Path(config_path)

    config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        # Deep merge: user overrides default per-rule
        if "rules" in user_config:
            for rule_name, rule_cfg in user_config["rules"].items():
                if rule_name in config["rules"]:
                    config["rules"][rule_name] = {**config["rules"][rule_name], **rule_cfg}
                else:
                    config["rules"][rule_name] = rule_cfg
        for key in ("ml_scoring", "fail_on"):
            if key in user_config:
                config[key] = user_config[key]

    return config


class RuleEngine:
    """Evaluates all configured rules against a commit."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or DEFAULT_CONFIG
        self.rules = []
        for rule_name, rule_cfg in self.config.get("rules", {}).items():
            if rule_name in ALL_RULES:
                # Check if explicitly disabled
                if rule_cfg and not rule_cfg.get("enabled", True):
                    continue
                rule_cls = ALL_RULES[rule_name]
                self.rules.append(rule_cls(rule_cfg))

    def evaluate(self, ctx: CommitContext) -> list[RuleResult]:
        """Run all rules and return results."""
        results = []
        for rule in self.rules:
            result = rule.evaluate(ctx)
            results.append(result)
        return results

    def should_block(self, results: list[RuleResult]) -> bool:
        """Check if any block-severity rule failed."""
        fail_severities = set(self.config.get("fail_on", ["block"]))
        for r in results:
            if not r.passed and r.severity.value in fail_severities:
                return True
        return False

    def format_results(self, results: list[RuleResult]) -> str:
        """Format rule results as markdown for PR comments."""
        lines = []
        # Group by severity
        blocked = [r for r in results if not r.passed and r.severity == Severity.BLOCK]
        warned = [r for r in results if not r.passed and r.severity == Severity.WARN]
        info = [r for r in results if not r.passed and r.severity == Severity.INFO]

        if blocked:
            lines.append("### Blocked")
            for r in blocked:
                lines.append(f"- **{r.rule_name}**: {r.message}")
        if warned:
            lines.append("### Warnings")
            for r in warned:
                lines.append(f"- **{r.rule_name}**: {r.message}")
        if info:
            lines.append("### Info")
            for r in info:
                lines.append(f"- **{r.rule_name}**: {r.message}")

        if not any([blocked, warned, info]):
            lines.append("All rules passed.")

        return "\n".join(lines)


def evaluate_commit(ctx: CommitContext, config_path: str | Path | None = None) -> tuple[list[RuleResult], bool]:
    """Convenience function: load config, evaluate, return results and should_block."""
    config = load_config(config_path)
    engine = RuleEngine(config)
    results = engine.evaluate(ctx)
    return results, engine.should_block(results)
