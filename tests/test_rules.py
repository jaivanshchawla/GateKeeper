#!/usr/bin/env python3
"""Unit tests for the Gatekeeper rule engine — one test per rule."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rules.base import CommitContext, Severity
from rules.rules import (
    LargeChangeRule,
    TooManyFilesRule,
    NoTestsRule,
    ConfigAndCodeRule,
    RevertHotspotRule,
    FirstTouchRule,
    WeekendDeployRule,
    StaleFileRule,
    DirectToMainRule,
    ALL_RULES,
)
from rules.engine import RuleEngine, load_config, DEFAULT_CONFIG


# ── Helper to build a CommitContext ──────────────────────────────────

def make_ctx(**kwargs) -> CommitContext:
    defaults = {
        "hash": "abc123",
        "author": "testuser",
        "message": "fix: update config",
        "files": ["src/app.py", "tests/test_app.py"],
        "lines_added": 10,
        "lines_deleted": 5,
        "files_touched": 2,
        "dirs_touched": 2,
        "is_merge": False,
        "hour_of_day": 14,
        "day_of_week": 2,  # Tuesday
        "author_prior_commits": 50,
        "file_revert_count_max": 0,
        "file_prior_changes_max": 5,
        "repo_name": "test-repo",
        "default_branch": "main",
        "is_direct_push": False,
    }
    defaults.update(kwargs)
    return CommitContext(**defaults)


# ── LargeChangeRule ──────────────────────────────────────────────────

class TestLargeChangeRule:
    def test_pass_small_change(self):
        rule = LargeChangeRule()
        ctx = make_ctx(lines_added=10, lines_deleted=5)
        r = rule.evaluate(ctx)
        assert r.passed
        assert r.severity == Severity.WARN

    def test_fail_large_change(self):
        rule = LargeChangeRule()
        ctx = make_ctx(lines_added=400, lines_deleted=200)
        r = rule.evaluate(ctx)
        assert not r.passed
        assert r.severity == Severity.WARN
        assert "600" in r.message

    def test_custom_threshold(self):
        rule = LargeChangeRule({"max_lines": 100})
        ctx = make_ctx(lines_added=50, lines_deleted=60)
        r = rule.evaluate(ctx)
        assert not r.passed

    def test_boundary(self):
        rule = LargeChangeRule({"max_lines": 100})
        ctx = make_ctx(lines_added=50, lines_deleted=50)
        r = rule.evaluate(ctx)
        assert r.passed  # exactly 100 <= 100


# ── TooManyFilesRule ─────────────────────────────────────────────────

class TestTooManyFilesRule:
    def test_pass_few_files(self):
        rule = TooManyFilesRule()
        ctx = make_ctx(files_touched=3)
        r = rule.evaluate(ctx)
        assert r.passed

    def test_fail_many_files(self):
        rule = TooManyFilesRule()
        ctx = make_ctx(files_touched=25)
        r = rule.evaluate(ctx)
        assert not r.passed

    def test_custom_threshold(self):
        rule = TooManyFilesRule({"max_files": 5})
        ctx = make_ctx(files_touched=6)
        r = rule.evaluate(ctx)
        assert not r.passed


# ── NoTestsRule ──────────────────────────────────────────────────────

class TestNoTestsRule:
    def test_pass_with_tests(self):
        rule = NoTestsRule()
        ctx = make_ctx(files=["src/app.py", "tests/test_app.py"])
        r = rule.evaluate(ctx)
        assert r.passed

    def test_fail_no_tests(self):
        rule = NoTestsRule()
        ctx = make_ctx(files=["src/app.py", "src/utils.py"], lines_added=10)
        r = rule.evaluate(ctx)
        assert not r.passed

    def test_exempt_docs(self):
        rule = NoTestsRule()
        ctx = make_ctx(files=["docs/README.md"], lines_added=10)
        r = rule.evaluate(ctx)
        assert r.passed

    def test_test_only_change(self):
        rule = NoTestsRule()
        ctx = make_ctx(files=["tests/test_app.py"], lines_added=10)
        r = rule.evaluate(ctx)
        assert r.passed  # test files are present

    def test_custom_exemptions(self):
        rule = NoTestsRule({"exempt_paths": ["vendor/**"]})
        ctx = make_ctx(files=["vendor/lib.py"], lines_added=10)
        r = rule.evaluate(ctx)
        assert r.passed

    def test_no_code_no_flag(self):
        rule = NoTestsRule()
        ctx = make_ctx(files=["README.md"], lines_added=0, lines_deleted=0)
        r = rule.evaluate(ctx)
        assert r.passed  # no code changes = no test required


# ── ConfigAndCodeRule ────────────────────────────────────────────────

class TestConfigAndCodeRule:
    def test_code_only(self):
        rule = ConfigAndCodeRule()
        ctx = make_ctx(files=["src/app.py"])
        r = rule.evaluate(ctx)
        assert r.passed

    def test_config_only(self):
        rule = ConfigAndCodeRule()
        ctx = make_ctx(files=["pyproject.toml"])
        r = rule.evaluate(ctx)
        assert r.passed

    def test_config_and_code(self):
        rule = ConfigAndCodeRule()
        ctx = make_ctx(files=["src/app.py", "pyproject.toml"])
        r = rule.evaluate(ctx)
        assert not r.passed

    def test_dockerfile_and_code(self):
        rule = ConfigAndCodeRule()
        ctx = make_ctx(files=["Dockerfile", "app.py"])
        r = rule.evaluate(ctx)
        assert not r.passed


# ── RevertHotspotRule ───────────────────────────────────────────────

class TestRevertHotspotRule:
    def test_pass_low_reverts(self):
        rule = RevertHotspotRule()
        ctx = make_ctx(file_revert_count_max=1)
        r = rule.evaluate(ctx)
        assert r.passed
        assert r.severity == Severity.BLOCK

    def test_fail_high_reverts(self):
        rule = RevertHotspotRule()
        ctx = make_ctx(file_revert_count_max=5)
        r = rule.evaluate(ctx)
        assert not r.passed
        assert r.severity == Severity.BLOCK

    def test_boundary(self):
        rule = RevertHotspotRule({"revert_count": 3})
        ctx = make_ctx(file_revert_count_max=3)
        r = rule.evaluate(ctx)
        assert not r.passed  # 3 >= 3 triggers


# ── FirstTouchRule ───────────────────────────────────────────────────

class TestFirstTouchRule:
    def test_first_time_contributor(self):
        rule = FirstTouchRule()
        ctx = make_ctx(author_prior_commits=0)
        r = rule.evaluate(ctx)
        assert r.passed  # info rules always pass
        assert r.severity == Severity.INFO
        assert "First-time" in r.message

    def test_experienced(self):
        rule = FirstTouchRule()
        ctx = make_ctx(author_prior_commits=100)
        r = rule.evaluate(ctx)
        assert r.passed
        assert "Experienced" in r.message


# ── WeekendDeployRule ────────────────────────────────────────────────

class TestWeekendDeployRule:
    def test_weekday(self):
        rule = WeekendDeployRule()
        ctx = make_ctx(day_of_week=2)  # Tuesday
        r = rule.evaluate(ctx)
        assert r.passed

    def test_saturday(self):
        rule = WeekendDeployRule()
        ctx = make_ctx(day_of_week=5)
        r = rule.evaluate(ctx)
        assert r.passed  # info only
        assert "Saturday" in r.message

    def test_sunday(self):
        rule = WeekendDeployRule()
        ctx = make_ctx(day_of_week=6)
        r = rule.evaluate(ctx)
        assert "Sunday" in r.message


# ── StaleFileRule ────────────────────────────────────────────────────

class TestStaleFileRule:
    def test_always_passes(self):
        rule = StaleFileRule()
        ctx = make_ctx(file_prior_changes_max=0)
        r = rule.evaluate(ctx)
        assert r.passed  # informational only


# ── DirectToMainRule ─────────────────────────────────────────────────

class TestDirectToMainRule:
    def test_not_direct(self):
        rule = DirectToMainRule()
        ctx = make_ctx(is_direct_push=False)
        r = rule.evaluate(ctx)
        assert r.passed

    def test_direct_push(self):
        rule = DirectToMainRule()
        ctx = make_ctx(is_direct_push=True)
        r = rule.evaluate(ctx)
        assert not r.passed


# ── RuleEngine integration ──────────────────────────────────────────

class TestRuleEngine:
    def test_all_rules_registered(self):
        assert len(ALL_RULES) == 9

    def test_engine_with_defaults(self):
        engine = RuleEngine()
        ctx = make_ctx()
        results = engine.evaluate(ctx)
        assert len(results) == 9  # all rules evaluated

    def test_engine_no_blocks(self):
        engine = RuleEngine()
        ctx = make_ctx(lines_added=10, lines_deleted=5, files_touched=2, file_revert_count_max=0)
        results = engine.evaluate(ctx)
        assert not engine.should_block(results)

    def test_engine_blocks_on_revert(self):
        engine = RuleEngine()
        ctx = make_ctx(file_revert_count_max=5)
        results = engine.evaluate(ctx)
        assert engine.should_block(results)

    def test_format_results(self):
        engine = RuleEngine()
        ctx = make_ctx(file_revert_count_max=5)
        results = engine.evaluate(ctx)
        md = engine.format_results(results)
        assert "Blocked" in md
        assert "revert_hotspot" in md

    def test_disabled_rule(self):
        config = DEFAULT_CONFIG.copy()
        config["rules"] = dict(config["rules"])
        config["rules"]["large_change"] = {"enabled": False}
        engine = RuleEngine(config)
        names = [r.rule_name for r in engine.evaluate(make_ctx())]
        assert "large_change" not in names

    def test_load_config(self):
        config = load_config()
        assert "rules" in config
        assert "fail_on" in config

    def test_20_real_commits(self):
        """Run engine on 20 synthetic commits with varied profiles."""
        engine = RuleEngine()
        profiles = [
            {"lines_added": 5, "lines_deleted": 0, "files_touched": 1, "author_prior_commits": 200},
            {"lines_added": 500, "lines_deleted": 300, "files_touched": 30, "author_prior_commits": 5},
            {"lines_added": 10, "lines_deleted": 10, "files_touched": 3, "file_revert_count_max": 5},
            {"lines_added": 20, "lines_deleted": 5, "files_touched": 2, "day_of_week": 6},
            {"lines_added": 100, "lines_deleted": 50, "files_touched": 10,
             "files": ["src/app.py", "pyproject.toml"]},
            {"lines_added": 0, "lines_deleted": 0, "files_touched": 0, "files": []},
            {"lines_added": 50, "lines_deleted": 10, "files_touched": 5, "is_direct_push": True},
            {"lines_added": 1000, "lines_deleted": 500, "files_touched": 50},
            {"lines_added": 3, "lines_deleted": 1, "files_touched": 1, "author_prior_commits": 0},
            {"lines_added": 200, "lines_deleted": 100, "files_touched": 15, "file_revert_count_max": 4},
            {"lines_added": 30, "lines_deleted": 10, "files_touched": 4, "day_of_week": 5},
            {"lines_added": 15, "lines_deleted": 5, "files_touched": 3,
             "files": ["docs/guide.md", "README.md"]},
            {"lines_added": 80, "lines_deleted": 40, "files_touched": 8, "hour_of_day": 3},
            {"lines_added": 500, "lines_deleted": 0, "files_touched": 1, "author_prior_commits": 1},
            {"lines_added": 40, "lines_deleted": 20, "files_touched": 6,
             "files": ["Dockerfile", "src/main.py", "tests/test_main.py"]},
            {"lines_added": 5, "lines_deleted": 5, "files_touched": 2, "file_revert_count_max": 2},
            {"lines_added": 200, "lines_deleted": 100, "files_touched": 12,
             "is_direct_push": True, "day_of_week": 6},
            {"lines_added": 10, "lines_deleted": 0, "files_touched": 1, "author_prior_commits": 50},
            {"lines_added": 300, "lines_deleted": 200, "files_touched": 25},
            {"lines_added": 20, "lines_deleted": 10, "files_touched": 3, "file_revert_count_max": 3},
        ]

        blocks = 0
        warns = 0
        for i, p in enumerate(profiles):
            ctx = make_ctx(hash=f"commit_{i}", **p)
            results = engine.evaluate(ctx)
            blocked = engine.should_block(results)
            if blocked:
                blocks += 1
            warns += sum(1 for r in results if not r.passed and r.severity == Severity.WARN)

        print(f"\n  20 commits: {blocks} blocked, {warns} warnings")
        assert blocks > 0, "At least one commit should be blocked"
        assert warns > 0, "At least one commit should have warnings"
