#!/usr/bin/env python3
"""Tests for the husky hook layer (scripts/hooks/).

These cover the pure logic the hooks rely on — commit message rules,
pre-push ref parsing, interpreter resolution and skip semantics — plus one
end-to-end run of the wiring doctor, so a broken configuration fails CI
instead of silently disabling every hook.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.hooks import _runtime as rt
from scripts.hooks import commit_msg


class TestCommitMessageRules:
    def test_accepts_conventional_subject(self):
        assert commit_msg.lint_subject("feat(husky): add pre-commit hook") == []

    def test_accepts_scope_less_subject(self):
        assert commit_msg.lint_subject("docs: explain the hook layout") == []

    def test_accepts_breaking_change_marker(self):
        assert commit_msg.lint_subject("feat!: drop the old hook format") == []

    def test_accepts_repo_specific_types(self):
        for commit_type in ("meas", "eval", "analysis", "backfill"):
            assert commit_msg.lint_subject(f"{commit_type}: record results") == []

    def test_rejects_missing_type(self):
        problems = commit_msg.lint_subject("random junk")
        assert problems
        assert "not a conventional commit subject" in problems[0]

    def test_rejects_unknown_type(self):
        problems = commit_msg.lint_subject("wobble: do a thing")
        assert any("unknown type" in p for p in problems)

    def test_rejects_empty_subject(self):
        assert commit_msg.lint_subject("   ") == ["commit subject is empty"]

    def test_rejects_trailing_period(self):
        problems = commit_msg.lint_subject("fix: correct the band cutoff.")
        assert any("period" in p for p in problems)

    def test_rejects_overlong_subject(self):
        problems = commit_msg.lint_subject("fix: " + "x" * commit_msg.MAX_SUBJECT_LENGTH)
        assert any("keep it under" in p for p in problems)

    @pytest.mark.parametrize(
        "subject",
        [
            "Merge branch 'main' into husky",
            "Revert \"feat: something\"",
            "fixup! feat: something",
            "squash! feat: something",
            "WIP",
        ],
    )
    def test_git_generated_subjects_pass(self, subject):
        assert commit_msg.lint_subject(subject) == []

    def test_lint_cli_exit_codes(self):
        script = os.path.join(REPO_ROOT, "scripts", "hooks", "commit_msg.py")
        good = subprocess.run(
            [sys.executable, script, "--lint", "feat: fine"],
            capture_output=True,
            text=True,
            check=False,
        )
        bad = subprocess.run(
            [sys.executable, script, "--lint", "nope"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert good.returncode == 0
        assert bad.returncode == 1


class TestPushRefParsing:
    ZERO = "0" * 40

    def test_parses_remote_and_local_sha(self):
        line = "refs/heads/husky abc123 refs/heads/husky def456"
        assert rt.parse_push_refs(line) == [("def456", "abc123")]

    def test_skips_branch_deletes(self):
        line = f"(delete) {self.ZERO} refs/heads/gone def456"
        assert rt.parse_push_refs(line) == []

    def test_skips_malformed_lines(self):
        assert rt.parse_push_refs("garbage\n\nrefs/heads/x sha") == []

    def test_reads_multiple_refs(self):
        data = "refs/heads/a 1 refs/heads/a 2\nrefs/heads/b 3 refs/heads/b 4"
        assert rt.parse_push_refs(data) == [("2", "1"), ("4", "3")]

    def test_new_branch_diff_still_resolves(self):
        # A zero remote sha means "brand new branch"; changed_files must not
        # crash and must fall back to a base ref for the diff spec.
        head = rt.git("rev-parse", "HEAD", check=True)[1]
        line = f"refs/heads/new {head} refs/heads/new {self.ZERO}"
        files = rt.changed_files(line, "*.py")
        assert isinstance(files, list)

    def test_existing_branch_diff_is_scoped_to_outgoing_commits(self):
        head = rt.git("rev-parse", "HEAD", check=True)[1]
        parent = rt.git("rev-parse", "HEAD~1", check=True)[1]
        line = f"refs/heads/husky {head} refs/heads/husky {parent}"
        files = rt.changed_files(line, "*.py")
        assert all(f.endswith(".py") for f in files)


class TestRuntimeResolution:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("GATEKEEPER_PYTHON", "/custom/python")
        assert rt.project_python() == "/custom/python"

    def test_prefers_repo_venv_when_present(self, monkeypatch):
        monkeypatch.delenv("GATEKEEPER_PYTHON", raising=False)
        has_venv = (rt.REPO_ROOT / "venv").exists()
        if not has_venv:
            pytest.skip("no in-repo venv to prefer")
        assert str(rt.REPO_ROOT / "venv") in rt.project_python()

    def test_repo_root_is_project_root(self):
        assert (rt.REPO_ROOT / "scripts" / "hooks").is_dir()
        assert (rt.REPO_ROOT / "package.json").is_file()

    @pytest.mark.parametrize(
        ("value", "hook", "expected"),
        [
            ("pre-commit", "pre-commit", True),
            ("commit-msg,pre-push", "commit-msg", True),
            ("commit-msg,pre-push", "pre-push", True),
            # a comma list must not skip hooks it does not name
            ("commit-msg,pre-push", "pre-commit", False),
            ("all", "anything", True),
            ("", "pre-commit", False),
        ],
    )
    def test_skip_env(self, monkeypatch, value, hook, expected):
        monkeypatch.setenv(rt.SKIP_ENV, value)
        assert rt.skipped(hook) is expected

    def test_husky_zero_disables_everything(self, monkeypatch):
        monkeypatch.setenv("HUSKY", "0")
        assert rt.skipped("pre-push") is True
        assert rt.skipped("anything") is True


class TestWiring:
    @pytest.mark.parametrize("hook", ["pre-commit", "commit-msg", "pre-push"])
    def test_shim_delegates_to_existing_script(self, hook):
        shim = os.path.join(REPO_ROOT, ".husky", hook)
        assert os.path.isfile(shim), f"missing shim {shim}"
        script = os.path.join(REPO_ROOT, "scripts", "hooks", f"{hook.replace('-', '_')}.py")
        assert os.path.isfile(script), f"shim {hook} points at a missing script"

    def test_doctor_passes(self):
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts", "hooks", "verify.py")],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, result.stderr
