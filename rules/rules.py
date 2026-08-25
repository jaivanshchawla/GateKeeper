#!/usr/bin/env python3
"""
Gatekeeper rule implementations.

Each rule class follows the BaseRule interface and evaluates one specific
risk condition. Rules are individually enable/disable/configurable per
repo via .gatekeeper.yml.
"""

from __future__ import annotations

import fnmatch

from rules.base import BaseRule, CommitContext, RuleResult


class LargeChangeRule(BaseRule):
    """Flag commits that touch too many lines."""
    name = "large_change"
    default_config = {"max_lines": 500, "severity": "warn"}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        total = ctx.lines_added + ctx.lines_deleted
        max_lines = self.config.get("max_lines", 500)
        passed = total <= max_lines
        return self._result(
            passed,
            f"{total} lines changed (threshold: {max_lines})" if not passed else f"{total} lines changed (within {max_lines} limit)",
            {"total_lines": total, "max_lines": max_lines},
        )


class TooManyFilesRule(BaseRule):
    """Flag commits that touch too many files."""
    name = "too_many_files"
    default_config = {"max_files": 20, "severity": "warn"}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        max_files = self.config.get("max_files", 20)
        passed = ctx.files_touched <= max_files
        return self._result(
            passed,
            f"{ctx.files_touched} files touched (threshold: {max_files})" if not passed else f"{ctx.files_touched} files touched (within {max_files} limit)",
            {"files_touched": ctx.files_touched, "max_files": max_files},
        )


class NoTestsRule(BaseRule):
    """Warn when a code change doesn't include test files."""
    name = "no_tests"
    default_config = {"severity": "warn", "exempt_paths": ["docs/**", "*.md"]}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        exempt = self.config.get("exempt_paths", [])

        # Check if ALL files are exempt
        all_exempt = all(
            any(fnmatch.fnmatch(f, pat) for pat in exempt)
            for f in ctx.files
        ) if ctx.files else False

        if all_exempt:
            return self._result(True, "All files are exempt (docs/markdown)")

        # Check if any test file is included
        test_patterns = ("test", "spec", "_test.", "_spec.", "tests/", "test_", "__tests__")
        has_tests = any(
            any(pat in f.lower() for pat in test_patterns)
            for f in ctx.files
        )

        # Only flag if there are code changes (not test-only or config-only)
        has_code = ctx.lines_added > 0 or ctx.lines_deleted > 0

        passed = has_tests or not has_code
        return self._result(
            passed,
            "No test files included in code change" if not passed else "Tests included or no code changes",
            {"has_tests": has_tests, "has_code": has_code, "files": ctx.files[:5]},
        )


class ConfigAndCodeRule(BaseRule):
    """Flag commits that touch both config/CI and source code."""
    name = "config_and_code"
    default_config = {"severity": "warn"}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        config_patterns = (".yaml", ".yml", ".toml", ".lock", "dockerfile", ".github/",
                           "docker-compose", "makefile", ".env", ".ini", ".cfg", "setup.py",
                           "setup.cfg", "pyproject.toml", "package.json", "cargo.toml")
        code_extensions = (".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".c", ".cpp", ".h")

        has_config = any(
            any(pat in f.lower() for pat in config_patterns)
            for f in ctx.files
        )
        has_code = any(
            f.endswith(code_extensions)
            for f in ctx.files
        )

        passed = not (has_config and has_code)
        return self._result(
            passed,
            "Touches both config/CI and source code" if not passed else "Config-only or code-only change",
            {"has_config": has_config, "has_code": has_code},
        )


class RevertHotspotRule(BaseRule):
    """Block commits to files with high revert history."""
    name = "revert_hotspot"
    default_config = {"revert_count": 3, "window_days": 60, "severity": "block"}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        max_reverts = self.config.get("revert_count", 3)
        passed = ctx.file_revert_count_max < max_reverts
        return self._result(
            passed,
            f"File reverted {ctx.file_revert_count_max}x (threshold: {max_reverts})" if not passed else f"File revert count: {ctx.file_revert_count_max} (within {max_reverts} limit)",
            {"revert_count": ctx.file_revert_count_max, "threshold": max_reverts},
        )


class FirstTouchRule(BaseRule):
    """Inform when a contributor touches a file/dir for the first time."""
    name = "first_touch"
    default_config = {"severity": "info"}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        # This is informational — always passes
        is_first = ctx.author_prior_commits <= 1
        return self._result(
            True,  # never blocks
            f"First-time contributor ({ctx.author_prior_commits} prior commits)" if is_first else f"Experienced contributor ({ctx.author_prior_commits} prior commits)",
            {"author_prior_commits": ctx.author_prior_commits},
        )


class WeekendDeployRule(BaseRule):
    """Inform when commits happen on weekends."""
    name = "weekend_deploy"
    default_config = {"severity": "info"}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        is_weekend = ctx.day_of_week >= 5  # 5=Saturday, 6=Sunday
        return self._result(
            True,  # informational only
            f"Committed on {'Saturday' if ctx.day_of_week == 5 else 'Sunday'}" if is_weekend else f"Committed on {'Mon Tue Wed Thu Fri'.split()[ctx.day_of_week]}",
            {"day_of_week": ctx.day_of_week, "is_weekend": is_weekend},
        )


class StaleFileRule(BaseRule):
    """Inform when a commit touches files not changed in N days."""
    name = "stale_file"
    default_config = {"days": 180, "severity": "info"}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        threshold = self.config.get("days", 180)
        # Use file_prior_changes_max as a proxy — if 0, the file is effectively new/stale
        # The real staleness check uses days_since_last_change from the feature set
        return self._result(
            True,  # informational only
            f"File history: {ctx.file_prior_changes_max} prior changes",
            {"file_prior_changes_max": ctx.file_prior_changes_max, "threshold": threshold},
        )


class DirectToMainRule(BaseRule):
    """Warn when pushing directly to the default branch."""
    name = "direct_to_main"
    default_config = {"severity": "warn"}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        passed = not ctx.is_direct_push
        return self._result(
            passed,
            f"Direct push to {ctx.default_branch}" if not passed else f"Not a direct push to {ctx.default_branch}",
            {"is_direct_push": ctx.is_direct_push, "branch": ctx.default_branch},
        )


# Registry of all rules
ALL_RULES: dict[str, type[BaseRule]] = {
    "large_change": LargeChangeRule,
    "too_many_files": TooManyFilesRule,
    "no_tests": NoTestsRule,
    "config_and_code": ConfigAndCodeRule,
    "revert_hotspot": RevertHotspotRule,
    "first_touch": FirstTouchRule,
    "weekend_deploy": WeekendDeployRule,
    "stale_file": StaleFileRule,
    "direct_to_main": DirectToMainRule,
}
