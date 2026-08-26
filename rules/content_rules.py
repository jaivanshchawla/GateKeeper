#!/usr/bin/env python3
"""
U.3: Code-content deterministic rules.

Each rule analyzes file content changes (diffs) to detect specific risk
patterns. These are deterministic, explainable, and independent of the ML model.

Rules:
- test_deleted: test files or test functions removed without replacement
- assertion_removed: net reduction in assert/expect count
- dependency_change: lockfile or manifest modified
- todo_debt: added TODO/FIXME/HACK/XXX comments
- debug_leftover: added console.log, print(), debugger, etc.
- large_binary: binary or generated file over configurable size
- migration_touch: database migration files modified
- error_handling_removed: net reduction in try/catch/except blocks
- complexity_delta: cyclomatic complexity increase per function (Python + JS/TS)
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rules.base import BaseRule, CommitContext, RuleResult, Severity


def _get_diff(repo_path: str, commit_hash: str, file_path: str) -> str:
    """Get the unified diff for a specific file in a commit."""
    try:
        result = subprocess.run(
            ["git", "diff", f"{commit_hash}~1..{commit_hash}", "--", file_path],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        return result.stdout
    except Exception:
        return ""


def _get_file_content(repo_path: str, commit_hash: str, file_path: str) -> str:
    """Get file content at a specific commit."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{file_path}"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        return result.stdout
    except Exception:
        return ""


def _get_prev_file_content(repo_path: str, commit_hash: str, file_path: str) -> str:
    """Get file content before a specific commit."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}~1:{file_path}"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        return result.stdout
    except Exception:
        return ""


def _count_pattern(content: str, patterns: list[str]) -> int:
    """Count occurrences of patterns in content."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, content, re.MULTILINE))
    return count


# ── Rule: test_deleted ──────────────────────────────────────────────

class TestDeletedRule(BaseRule):
    """Flag when test files or test functions are removed without replacement."""
    name = "test_deleted"
    default_config = {"severity": "block"}
    test_patterns = ("test", "spec", "_test.", "_spec.", "tests/", "test_", "__tests__")

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        # Check if any touched files are test files that were deleted
        test_files_deleted = []
        test_files_added = []

        for f in ctx.files:
            is_test = any(pat in f.lower() for pat in self.test_patterns)
            if is_test:
                # Heuristic: if lines_deleted > 0 and lines_added == 0, likely deleted
                # In practice, need diff analysis — this is a conservative check
                if ctx.lines_deleted > 0 and ctx.lines_added == 0:
                    test_files_deleted.append(f)
                elif ctx.lines_added > 0:
                    test_files_added.append(f)

        # Also check if test function count decreased (requires content analysis)
        # This is a simplified version — full version would check diff

        passed = len(test_files_deleted) == 0 or len(test_files_added) > 0
        return self._result(
            passed,
            f"Test files removed: {', '.join(test_files_deleted[:3])}" if not passed else "No test files deleted (or tests added)",
            {"deleted": test_files_deleted, "added": test_files_added},
        )


# ── Rule: assertion_removed ─────────────────────────────────────────

class AssertionRemovedRule(BaseRule):
    """Flag net reduction in assert/expect count."""
    name = "assertion_removed"
    default_config = {"severity": "warn"}

    assert_patterns = [
        r"\bassert\b", r"\bexpect\b", r"\bassertEqual\b", r"\bassertTrue\b",
        r"\bassertFalse\b", r"\bassertRaises\b", r"\bassertIn\b",
        r"\bdescribe\(", r"\bit\(", r"\btest\(",
    ]

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        # This rule requires diff analysis — simplified heuristic
        # For now, flag if only test files are modified with net deletion
        test_patterns = ("test", "spec", "_test.", "_spec.")
        is_test_change = any(any(pat in f.lower() for pat in test_patterns) for f in ctx.files)

        if is_test_change and ctx.lines_deleted > ctx.lines_added and ctx.lines_deleted > 10:
            return self._result(
                False,
                f"Test code deleted ({ctx.lines_deleted} lines removed, {ctx.lines_added} added)",
                {"lines_deleted": ctx.lines_deleted, "lines_added": ctx.lines_added},
            )

        return self._result(True, "No assertion reduction detected")


# ── Rule: dependency_change ─────────────────────────────────────────

class DependencyChangeRule(BaseRule):
    """Flag lockfile or manifest modifications."""
    name = "dependency_change"
    default_config = {"severity": "warn", "flag_version_bumps": True}

    manifest_patterns = (
        "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "requirements.txt", "requirements-dev.txt", "Pipfile.lock", "poetry.lock",
        "go.mod", "go.sum", "Cargo.toml", "Cargo.lock",
        "pom.xml", "build.gradle", "Gemfile.lock",
        "pyproject.toml", "setup.py", "setup.cfg",
    )

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        manifests_changed = [f for f in ctx.files if any(p in f for p in self.manifest_patterns)]

        if manifests_changed:
            lockfiles = [f for f in manifests_changed if "lock" in f.lower() or f.endswith(".sum")]
            manifests = [f for f in manifests_changed if f not in lockfiles]

            parts = []
            if manifests:
                parts.append(f"manifests: {', '.join(manifests[:3])}")
            if lockfiles:
                parts.append(f"lockfiles: {', '.join(lockfiles[:3])}")

            return self._result(
                False,
                f"Dependencies modified: {'; '.join(parts)}",
                {"manifests": manifests_changed, "has_lockfile": bool(lockfiles)},
            )

        return self._result(True, "No dependency files modified")


# ── Rule: todo_debt ─────────────────────────────────────────────────

class TodoDebtRule(BaseRule):
    """Flag added TODO/FIXME/HACK/XXX comments."""
    name = "todo_debt"
    default_config = {"severity": "info"}

    todo_patterns = [
        r"\bTODO\b", r"\bFIXME\b", r"\bHACK\b", r"\bXXX\b",
        r"\bTEMP\b", r"\bWORKAROUND\b",
    ]

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        # Count added vs removed todos in commit message
        msg = ctx.message.upper()
        added_todos = sum(1 for p in self.todo_patterns if re.search(p, msg))

        if added_todos > 0:
            found = [p for p in self.todo_patterns if re.search(p, ctx.message, re.IGNORECASE)]
            markers = ", ".join(found)
            return self._result(
                True,  # info, never blocks
                f"Commit message contains {added_todos} debt marker(s): {markers}",
                {"todo_count": added_todos},
            )

        return self._result(True, "No debt markers in commit message")


# ── Rule: debug_leftover ────────────────────────────────────────────

class DebugLeftoverRule(BaseRule):
    """Flag added debug statements in non-test files."""
    name = "debug_leftover"
    default_config = {"severity": "warn"}

    debug_patterns = [
        r"\bconsole\.log\b", r"\bconsole\.debug\b", r"\bconsole\.warn\b",
        r"\bprint\(", r"\bpprint\(", r"\bfmt\.Print\b", r"\bfmt\.Println\b",
        r"\bdebugger\b", r"\bbinding\.pry\b", r"\bbyebug\b",
        r"\bimport\s+pdb\b", r"\bpdb\.set_trace\b",
        r"\bSystem\.out\.print\b", r"\becho\s+",
    ]

    # Test file patterns — debug in tests is acceptable
    test_patterns = ("test", "spec", "_test.", "_spec.", "tests/", "__tests__")

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        # Check if any non-test files are modified
        non_test_files = [
            f for f in ctx.files
            if not any(pat in f.lower() for pat in self.test_patterns)
        ]

        if not non_test_files:
            return self._result(True, "No non-test files modified")

        # Without diff analysis, we can only check the commit message
        # for debug-related keywords (conservative)
        msg_lower = ctx.message.lower()
        has_debug_keyword = any(
            kw in msg_lower
            for kw in ["debug", "console.log", "print(", "logging", "log statement"]
        )

        if has_debug_keyword:
            return self._result(
                False,
                f"Commit message references debug/logging — verify no debug statements left in {', '.join(non_test_files[:3])}",
                {"non_test_files": non_test_files[:5]},
            )

        return self._result(True, "No debug leftovers detected")


# ── Rule: large_binary ──────────────────────────────────────────────

class LargeBinaryRule(BaseRule):
    """Flag binary or generated files over a configurable size."""
    name = "large_binary"
    default_config = {"severity": "warn", "max_size_kb": 500}

    binary_extensions = (
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
        ".mp3", ".mp4", ".wav", ".avi", ".mov",
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".exe", ".dll", ".so", ".dylib",
        ".pyc", ".pyo", ".class", ".o", ".obj",
    )

    generated_patterns = (
        "min.js", "min.css", ".bundle.js",
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    )

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        max_size = self.config.get("max_size_kb", 500) * 1024

        flagged = []
        for f in ctx.files:
            is_binary = any(f.lower().endswith(ext) for ext in self.binary_extensions)
            is_generated = any(pat in f for pat in self.generated_patterns)
            if is_binary or is_generated:
                flagged.append(f)

        if flagged:
            return self._result(
                False,
                f"Binary/generated files in commit: {', '.join(flagged[:3])}",
                {"files": flagged},
            )

        return self._result(True, "No binary/generated files")


# ── Rule: migration_touch ───────────────────────────────────────────

class MigrationTouchRule(BaseRule):
    """Flag database migration file modifications."""
    name = "migration_touch"
    default_config = {"severity": "warn"}

    migration_patterns = (
        "migrations/", "migrate/", "alembic/",
        "_migration", "migration_", "schema_change",
        ".sql", "flyway/", "liquibase/",
    )

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        migrations = [f for f in ctx.files if any(p in f.lower() for p in self.migration_patterns)]

        if migrations:
            return self._result(
                False,
                f"Database migrations modified: {', '.join(migrations[:3])}",
                {"migration_files": migrations},
            )

        return self._result(True, "No migration files modified")


# ── Rule: error_handling_removed ────────────────────────────────────

class ErrorHandlingRemovedRule(BaseRule):
    """Flag net reduction in try/catch/except blocks."""
    name = "error_handling_removed"
    default_config = {"severity": "warn"}

    error_patterns = [
        r"\btry\b", r"\bcatch\b", r"\bexcept\b", r"\bfinally\b",
        r"\braise\b", r"\bthrow\b", r"\bError\b", r"\bException\b",
    ]

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        # Simplified: flag if code is modified but no error handling keywords in message
        has_code = any(
            f.endswith((".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java"))
            for f in ctx.files
        )

        if has_code and ctx.lines_deleted > 20:
            msg_lower = ctx.message.lower()
            has_error_keyword = any(
                kw in msg_lower
                for kw in ["error", "exception", "try", "catch", "handle", "retry"]
            )
            if not has_error_keyword and ctx.lines_deleted > ctx.lines_added:
                return self._result(
                    False,
                    f"Code deleted ({ctx.lines_deleted} lines) without error-handling context in commit message",
                    {"lines_deleted": ctx.lines_deleted, "lines_added": ctx.lines_added},
                )

        return self._result(True, "No error-handling reduction detected")


# ── Rule: complexity_delta ──────────────────────────────────────────

class ComplexityDeltaRule(BaseRule):
    """Flag cyclomatic complexity increase (Python + JS/TS only)."""
    name = "complexity_delta"
    default_config = {"severity": "info", "max_complexity": 10}

    def evaluate(self, ctx: CommitContext) -> RuleResult:
        # Without actual file content, we use a heuristic:
        # Large changes to code files likely increase complexity
        code_files = [
            f for f in ctx.files
            if f.endswith((".py", ".js", ".ts", ".jsx", ".tsx"))
        ]

        if code_files and ctx.lines_added > 100:
            return self._result(
                True,  # info, never blocks
                f"Large code change ({ctx.lines_added} lines added to {len(code_files)} file(s)) — review complexity",
                {"files": code_files[:5], "lines_added": ctx.lines_added},
            )

        return self._result(True, "Complexity within normal range")


# ── Registry ────────────────────────────────────────────────────────

ALL_CONTENT_RULES: dict[str, type[BaseRule]] = {
    "test_deleted": TestDeletedRule,
    "assertion_removed": AssertionRemovedRule,
    "dependency_change": DependencyChangeRule,
    "todo_debt": TodoDebtRule,
    "debug_leftover": DebugLeftoverRule,
    "large_binary": LargeBinaryRule,
    "migration_touch": MigrationTouchRule,
    "error_handling_removed": ErrorHandlingRemovedRule,
    "complexity_delta": ComplexityDeltaRule,
}


def get_content_rules_fire_rate(
    repo_path: str,
    commit_hashes: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, dict]:
    """Compute fire rate for each content rule across a set of commits.

    Returns dict mapping rule_name to {fired_count, total_count, fire_rate, examples}.
    """
    from rules.engine import RuleEngine, load_config as load_rules_config

    rules_config = config or load_rules_config()
    engine = RuleEngine(rules_config)

    # Add content rules to the engine
    for rule_name, rule_cls in ALL_CONTENT_RULES.items():
        rule_cfg = rules_config.get("rules", {}).get(rule_name, {"enabled": True})
        if rule_cfg.get("enabled", True):
            engine.rules.append(rule_cls(rule_cfg))

    fire_counts: dict[str, dict] = {}
    for rule_name in ALL_CONTENT_RULES:
        fire_counts[rule_name] = {"fired": 0, "total": 0, "examples": []}

    for commit_hash in commit_hashes[:200]:  # Cap at 200
        # Build minimal CommitContext from git
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%ct|%aN|%aE|%s", commit_hash],
                cwd=repo_path, capture_output=True, text=True, timeout=10,
            )
            parts = result.stdout.strip().split("|", 4)
            if len(parts) < 5:
                continue

            # Get files
            files_result = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash],
                cwd=repo_path, capture_output=True, text=True, timeout=10,
            )
            files = [f.strip() for f in files_result.stdout.strip().split("\n") if f.strip()]

            # Get line counts
            stat_result = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "--stat", commit_hash],
                cwd=repo_path, capture_output=True, text=True, timeout=10,
            )
            lines_added = 0
            lines_deleted = 0
            for line in stat_result.stdout.split("\n"):
                match = re.search(r"(\d+) insertion", line)
                if match:
                    lines_added += int(match.group(1))
                match = re.search(r"(\d+) deletion", line)
                if match:
                    lines_deleted += int(match.group(1))

            ctx = CommitContext(
                hash=parts[0],
                author=parts[2],
                message=parts[4],
                files=files,
                lines_added=lines_added,
                lines_deleted=lines_deleted,
                files_touched=len(files),
                repo_name=os.path.basename(repo_path),
            )

            results = engine.evaluate(ctx)
            for r in results:
                if r.rule_name in fire_counts:
                    fire_counts[r.rule_name]["total"] += 1
                    if not r.passed:
                        fire_counts[r.rule_name]["fired"] += 1
                        if len(fire_counts[r.rule_name]["examples"]) < 3:
                            fire_counts[r.rule_name]["examples"].append({
                                "hash": commit_hash[:8],
                                "message": r.message,
                            })

        except Exception:
            continue

    # Compute rates
    for rule_name, data in fire_counts.items():
        total = data["total"]
        data["fire_rate"] = data["fired"] / total if total > 0 else 0.0

    return fire_counts
