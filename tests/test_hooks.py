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
from scripts.hooks import commit_msg, verify


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
        base = [sys.executable, "-m", "scripts.hooks.commit_msg"]
        good = subprocess.run(
            [*base, "--lint", "feat: fine"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        bad = subprocess.run(
            [*base, "--lint", "nope"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
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

    def test_every_husky_hook_is_implemented(self):
        """husky writes a dispatcher for all 14; we ship a script for each.

        A hook husky does not dispatch would be a file that can never run,
        and one it dispatches with nothing behind it is silent dead weight.
        """
        assert set(rt.HOOKS) == set(rt.HUSKY_HOOKS)
        assert len(rt.HOOKS) == 14

    def test_blocking_hooks_are_a_subset_of_the_shipped_hooks(self):
        assert set(rt.BLOCKING_HOOKS).issubset(set(rt.HOOKS))

    def test_hooks_are_listed_in_git_execution_order(self):
        # githooks(5) order. Reading a trace is much harder if the inventory
        # groups hooks by anything else.
        assert rt.HOOKS[0] == "applypatch-msg"
        assert rt.HOOKS[-1] == "pre-auto-gc"
        assert rt.HOOKS.index("pre-commit") < rt.HOOKS.index("commit-msg")
        assert rt.HOOKS.index("commit-msg") < rt.HOOKS.index("post-commit")


class TestTracing:
    """`HUSKY=2` traces husky's own sh layer; we continue it into Python."""

    def test_silent_by_default(self, monkeypatch, capsys):
        monkeypatch.delenv("HUSKY", raising=False)
        monkeypatch.delenv(rt.TRACE_ENV, raising=False)
        assert rt.tracing_enabled() is False
        rt.trace("should not appear")
        assert capsys.readouterr().err == ""

    def test_husky_two_enables_it(self, monkeypatch, capsys):
        monkeypatch.setenv("HUSKY", "2")
        assert rt.tracing_enabled() is True
        rt.trace("visible")
        assert "+ visible" in capsys.readouterr().err

    def test_trace_env_enables_it_without_husky(self, monkeypatch, capsys):
        monkeypatch.delenv("HUSKY", raising=False)
        monkeypatch.setenv(rt.TRACE_ENV, "1")
        rt.trace("visible")
        assert "+ visible" in capsys.readouterr().err

    def test_husky_zero_does_not_trace(self, monkeypatch):
        # 0 means "disabled", not "as loud as possible".
        #
        # GATEKEEPER_TRACE has to be cleared too: this asserts something about
        # one switch, so leaving the other switch to the ambient environment
        # makes it fail for whoever has `export GATEKEEPER_TRACE=1` in their
        # shell -- and pre-push runs this suite, so a debugging habit would
        # have blocked their own push.
        monkeypatch.setenv("HUSKY", "0")
        monkeypatch.delenv(rt.TRACE_ENV, raising=False)
        assert rt.tracing_enabled() is False

    def test_enter_reports_the_interpreter_and_arguments(self, monkeypatch, capsys):
        """The first question about a misbehaving hook is which Python it got."""
        monkeypatch.setenv(rt.TRACE_ENV, "1")
        rt.enter("pre-commit", ["one", "two"])
        err = capsys.readouterr().err
        assert "pre-commit: interpreter=" in err
        assert sys.executable in err
        assert "['one', 'two']" in err

    def test_enter_is_silent_when_not_tracing(self, monkeypatch, capsys):
        monkeypatch.delenv("HUSKY", raising=False)
        monkeypatch.delenv(rt.TRACE_ENV, raising=False)
        rt.enter("pre-commit")
        assert capsys.readouterr().err == ""


class TestWiring:
    @pytest.mark.parametrize("hook", list(rt.HOOKS))
    def test_every_shim_delegates_to_an_existing_script(self, hook):
        shim = os.path.join(REPO_ROOT, ".husky", hook)
        assert os.path.isfile(shim), f"missing shim {shim}"
        script = os.path.join(REPO_ROOT, "scripts", "hooks", f"{hook.replace('-', '_')}.py")
        assert os.path.isfile(script), f"shim {hook} points at a missing script"

    def test_structural_checks_pass_in_any_checkout(self):
        """The part of the doctor that must hold without a local install.

        Everything environmental -- whether *this* machine has run
        `npm install` and therefore set core.hooksPath, generated .husky/_, or
        has Node at all -- is deliberately excluded. Otherwise this test
        would fail on a fresh clone while telling us nothing about the
        repository.
        """
        checks = [
            verify._check_hooks_present(),
            verify._check_shim_shape(),
            ("unknown files", verify._check_unknown_hooks(), []),
            verify._check_generated_dir(),
            verify._check_git_dir_shadowing(),
        ]
        for name, failures, _warnings in checks:
            assert failures == [], f"{name}: {failures}"

    def test_doctor_passes_where_the_hooks_are_installed(self):
        code, value = rt.git("config", "--get", "core.hooksPath")
        if code != 0 or value != verify.EXPECTED_HOOKS_PATH:
            pytest.skip("hooks are not installed in this checkout")
        assert verify.main([]) == 0

    def test_doctor_module_runs_as_a_program(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.hooks.verify", "--list"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "14/14" in result.stderr

    def test_bootstrap_uses_module_invocation(self):
        """Hooks must be run as modules, not file paths.

        Running `python scripts/hooks/x.py` sets sys.path[0] to scripts/,
        which is what made the hook scripts need a sys.path shim each.
        `python -m scripts.hooks.x` removes that entirely.
        """
        bootstrap = os.path.join(REPO_ROOT, ".husky", "lib", "bootstrap.sh")
        with open(bootstrap, encoding="utf-8") as handle:
            text = handle.read()
        assert 'exec "$candidate" -m "scripts.hooks.$module"' in text
