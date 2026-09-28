#!/usr/bin/env python3
"""Integration tests for the hook layer against real git repositories.

Each test builds a throwaway repository, points the hook runtime at it, and
runs the hook's entry point the way git would. That exercises the real
decision paths -- protected branches, merge refusal, message seeding,
outcome recording -- without touching this repository's own git state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.hooks import _policy as policy
from scripts.hooks import _runtime as rt
from scripts.hooks import (
    applypatch_msg,
    post_checkout,
    post_commit,
    post_merge,
    post_rewrite,
    pre_merge_commit,
    pre_rebase,
    prepare_commit_msg,
    verify,
)


def _run(repo, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo with one commit, wired into the hook runtime."""
    path = tmp_path / "repo"
    path.mkdir()
    _run(path, "init", "-q", "-b", "main")
    _run(path, "config", "user.email", "test@example.com")
    _run(path, "config", "user.name", "Test")
    _run(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    _run(path, "add", "README.md")
    _run(path, "commit", "-q", "-m", "chore: initial commit")

    # Every hook reaches git through rt, so patching the root is enough.
    monkeypatch.setattr(rt, "REPO_ROOT", path)
    monkeypatch.delenv(rt.SKIP_ENV, raising=False)
    monkeypatch.delenv("HUSKY", raising=False)
    monkeypatch.delenv("GATEKEEPER_ALLOW_PROTECTED", raising=False)
    return path


def _branch(repo, name: str) -> None:
    assert _run(repo, "checkout", "-q", "-B", name).returncode == 0


def _commit(repo, name: str, message: str = "chore: work") -> str:
    (repo / name).write_text(f"{name}\n", encoding="utf-8")
    _run(repo, "add", name)
    assert _run(repo, "commit", "-q", "-m", message).returncode == 0
    return _run(repo, "rev-parse", "HEAD").stdout.strip()


class TestProtectedBranchPolicy:
    def test_defaults_include_main(self, repo):
        assert policy.is_protected("main")
        assert not policy.is_protected("feature/x")

    def test_globs_match(self, repo):
        assert policy.is_protected("release/2.0")
        assert policy.is_protected("hotfix/urgent")

    def test_reads_configured_list(self, repo):
        (repo / ".gatekeeper.yml").write_text(
            "protected_branches:\n  - trunk\n  - stable/*\n", encoding="utf-8"
        )
        assert policy.is_protected("trunk")
        assert policy.is_protected("stable/1")
        assert not policy.is_protected("main")

    def test_falls_back_when_yaml_is_broken(self, repo):
        (repo / ".gatekeeper.yml").write_text("::: not yaml ::\n\t", encoding="utf-8")
        assert policy.protected_branches()  # defaults survive a parse failure

    def test_current_branch(self, repo):
        assert policy.current_branch() == "main"
        _branch(repo, "feature/x")
        assert policy.current_branch() == "feature/x"

    def test_rule_severity_reads_config(self, repo):
        (repo / ".gatekeeper.yml").write_text(
            "rules:\n  direct_to_main:\n    severity: block\n", encoding="utf-8"
        )
        assert policy.rule_severity("direct_to_main") == "block"


class TestPreMergeCommit:
    def test_blocks_merge_into_protected_branch(self, repo):
        assert pre_merge_commit.main() == 1

    def test_allows_merge_on_feature_branch(self, repo):
        _branch(repo, "feature/x")
        assert pre_merge_commit.main() == 0

    def test_override_env_allows_it(self, repo, monkeypatch):
        monkeypatch.setenv(pre_merge_commit.OVERRIDE_ENV, "1")
        assert pre_merge_commit.main() == 0

    def test_skip_env_disables_the_hook(self, repo, monkeypatch):
        monkeypatch.setenv(rt.SKIP_ENV, "pre-merge-commit")
        assert pre_merge_commit.main() == 0


class TestPreRebase:
    def test_blocks_rebase_of_current_protected_branch(self, repo):
        assert pre_rebase.main(["origin/main"]) == 1

    def test_blocks_when_branch_argument_is_protected(self, repo):
        _branch(repo, "feature/x")
        assert pre_rebase.main(["origin/main", "main"]) == 1

    def test_allows_rebase_of_feature_branch(self, repo):
        _branch(repo, "feature/x")
        assert pre_rebase.main(["origin/main", "feature/x"]) == 0


class TestPrepareCommitMsg:
    def test_seeds_cheat_sheet_for_editor_invocation(self, repo, tmp_path):
        msg = tmp_path / "COMMIT_EDITMSG"
        msg.write_text("\n", encoding="utf-8")
        assert prepare_commit_msg.main([str(msg), "template"]) == 0
        text = msg.read_text(encoding="utf-8")
        assert "feat" in text and "types:" in text
        assert "protected branch" in text  # we are on main

    def test_does_not_touch_message_given_with_m(self, repo, tmp_path):
        msg = tmp_path / "COMMIT_EDITMSG"
        msg.write_text("chore: real message\n", encoding="utf-8")
        assert prepare_commit_msg.main([str(msg), "message"]) == 0
        assert msg.read_text(encoding="utf-8") == "chore: real message\n"

    def test_does_not_append_when_cleanup_keeps_comments(self, repo, tmp_path):
        _run(repo, "config", "commit.cleanup", "whitespace")
        msg = tmp_path / "COMMIT_EDITMSG"
        msg.write_text("\n", encoding="utf-8")
        assert prepare_commit_msg.main([str(msg), "template"]) == 0
        assert "#" not in msg.read_text(encoding="utf-8")

    def test_skips_merge_messages(self, repo, tmp_path):
        msg = tmp_path / "MERGE_MSG"
        msg.write_text("\n", encoding="utf-8")
        assert prepare_commit_msg.main([str(msg), "merge"]) == 0
        assert msg.read_text(encoding="utf-8") == "\n"

    def test_does_not_duplicate_on_amend(self, repo, tmp_path):
        msg = tmp_path / "COMMIT_EDITMSG"
        msg.write_text("chore: already written\n", encoding="utf-8")
        prepare_commit_msg.main([str(msg), "commit"])
        assert msg.read_text(encoding="utf-8") == "chore: already written\n"


class TestPostCommit:
    def test_records_the_commit(self, repo, capsys):
        sha = _commit(repo, "a.txt")
        assert post_commit.main() == 0
        events = repo / ".git" / post_commit.EVENTS_DIR / post_commit.EVENTS_FILE
        assert events.is_file()
        record = json.loads(events.read_text(encoding="utf-8").strip())
        assert record["sha"] == sha
        assert record["branch"] == "main"
        assert record["event"] == "commit"

    def test_appends_one_line_per_commit(self, repo):
        _commit(repo, "a.txt")
        post_commit.main()
        _commit(repo, "b.txt")
        post_commit.main()
        events = repo / ".git" / post_commit.EVENTS_DIR / post_commit.EVENTS_FILE
        assert len(events.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_never_fails_without_a_commit(self, repo, monkeypatch):
        monkeypatch.setattr(rt, "REPO_ROOT", repo / "does-not-exist")
        assert post_commit.main() == 0


class TestPostMerge:
    def test_reports_when_the_gate_changed(self, repo):
        base = _run(repo, "rev-parse", "HEAD").stdout.strip()
        (repo / ".gatekeeper.yml").write_text("rules: {}\n", encoding="utf-8")
        _run(repo, "add", ".gatekeeper.yml")
        _run(repo, "commit", "-q", "-m", "chore: change gate config")
        _run(repo, "update-ref", "ORIG_HEAD", base)
        assert post_merge.main() == 0
        assert post_merge._relevant([".gatekeeper.yml"]) == [".gatekeeper.yml"]

    def test_ignores_unrelated_changes(self, repo):
        assert post_merge._relevant(["docs/HUSKY.md", "app.jsx"]) == []

    def test_stays_quiet_without_history(self, repo):
        assert post_merge.main() == 0


class TestPostCheckout:
    def test_reports_on_branch_checkout(self, repo):
        assert post_checkout.main(["a", "b", "1"]) == 0

    def test_silent_on_file_checkout(self, repo):
        assert post_checkout.main(["a", "b", "0"]) == 0

    @pytest.mark.parametrize("argv", [[], ["a"], ["a", "b"]])
    def test_tolerates_missing_arguments(self, repo, argv):
        assert post_checkout.main(argv) == 0


class TestPostRewrite:
    def test_reports_rewritten_pairs(self, repo):
        head = _run(repo, "rev-parse", "HEAD").stdout.strip()
        assert post_rewrite.report(f"{head} {head}\n") is None

    def test_tolerant_of_empty_input(self, repo):
        assert post_rewrite.report("") is None


class TestApplyPatchMsg:
    """`git am` never runs commit-msg, so patches need the same rules."""

    def test_rejects_patch_with_unconventional_subject(self, repo, tmp_path):
        msg = tmp_path / "PATCH_MSG"
        msg.write_text("bogus patch subject\n", encoding="utf-8")
        assert applypatch_msg.main([str(msg)]) == 1

    def test_accepts_patch_with_conventional_subject(self, repo, tmp_path):
        msg = tmp_path / "PATCH_MSG"
        msg.write_text("feat(cli): add a flag\n\nbody\n", encoding="utf-8")
        assert applypatch_msg.main([str(msg)]) == 0

    def test_ignores_the_body_when_checking(self, repo, tmp_path):
        msg = tmp_path / "PATCH_MSG"
        msg.write_text("fix: real subject\n\nnot conventional at all\n", encoding="utf-8")
        assert applypatch_msg.main([str(msg)]) == 0

    def test_tolerates_a_missing_file(self, repo, tmp_path):
        assert applypatch_msg.main([str(tmp_path / "absent")]) == 0

    def test_tolerates_no_argument(self, repo):
        assert applypatch_msg.main([]) == 0


class TestBootstrapShell:
    """The sh layer git actually runs, exercised through a real shell.

    These caught a defect reading the file could not: a failed `.` aborts a
    non-interactive shell, so the defensively-written `source ... || true`
    never ran and a wrong $0 silently killed every hook.
    """

    @staticmethod
    def _sh(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
        merged = {**os.environ, **(env or {})}
        return subprocess.run(
            ["sh", "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=merged,
            check=False,
        )

    def test_sourcing_without_a_hook_path_does_not_abort(self):
        result = self._sh('. .husky/lib/bootstrap.sh; echo survived')
        assert result.returncode == 0
        assert "survived" in result.stdout

    def test_sourcing_from_another_directory_does_not_abort(self, tmp_path):
        bootstrap = (rt.REPO_ROOT / ".husky" / "lib" / "bootstrap.sh").as_posix()
        result = self._sh(f'cd "{tmp_path.as_posix()}" && . "{bootstrap}"; echo survived')
        assert result.returncode == 0
        assert "survived" in result.stdout

    def test_run_hook_resolves_and_executes_a_module(self):
        # post-checkout with a file-checkout flag exits 0 and prints nothing.
        result = self._sh('. .husky/lib/bootstrap.sh; gatekeeper_run_hook post-checkout 0 0 0')
        assert result.returncode == 0, result.stderr

    def test_bad_gatekeeper_python_falls_through(self):
        result = self._sh(
            '. .husky/lib/bootstrap.sh; gatekeeper_run_hook post-checkout 0 0 0',
            env={"GATEKEEPER_PYTHON": "/nonexistent/python"},
        )
        assert result.returncode == 0, result.stderr

    def test_reports_clearly_when_no_interpreter_works(self):
        # An empty PATH with no override: the hook must fail loudly, not
        # silently succeed and let a bad commit through.
        result = self._sh(
            '. .husky/lib/bootstrap.sh; gatekeeper_run_hook post-checkout 0 0 0',
            env={"PATH": "", "GATEKEEPER_PYTHON": ""},
        )
        assert result.returncode != 0
        assert "no working Python 3" in result.stderr


class TestHookInventory:
    def test_every_hook_has_a_shim_and_a_module(self):
        for hook in rt.HOOKS:
            assert (rt.REPO_ROOT / ".husky" / hook).is_file(), f"missing shim: {hook}"
            module = rt.REPO_ROOT / (rt.module_for(hook).replace(".", "/") + ".py")
            assert module.is_file(), f"missing module for {hook}"

    def test_every_hook_name_is_a_real_git_hook(self):
        for hook in rt.HOOKS:
            assert hook in rt.KNOWN_GIT_HOOKS, f"{hook} is not a git hook name"

    def test_module_for_translates_dashes(self):
        assert rt.module_for("pre-merge-commit") == "scripts.hooks.pre_merge_commit"

    def _doctor_on(self, monkeypatch, tmp_path, *names: str) -> list[str]:
        husky_dir = tmp_path / ".husky"
        husky_dir.mkdir(exist_ok=True)
        for name in names:
            (husky_dir / name).write_text("echo\n", encoding="utf-8")
        monkeypatch.setattr(rt, "REPO_ROOT", tmp_path)
        return verify._check_unknown_hooks()

    def test_flags_misspelled_hook_name(self, tmp_path, monkeypatch):
        problems = self._doctor_on(monkeypatch, tmp_path, "precommit")
        assert len(problems) == 1
        assert "rename it to 'pre-commit'" in problems[0]

    def test_flags_underscored_hook_name(self, tmp_path, monkeypatch):
        problems = self._doctor_on(monkeypatch, tmp_path, "pre_commit")
        assert len(problems) == 1
        assert "pre-commit" in problems[0]

    def test_flags_extensioned_hook_without_the_real_one(self, tmp_path, monkeypatch):
        problems = self._doctor_on(monkeypatch, tmp_path, "pre-commit.sh")
        assert len(problems) == 1
        assert "git looks for 'pre-commit'" in problems[0]

    def test_allows_helper_beside_the_real_hook(self, tmp_path, monkeypatch):
        problems = self._doctor_on(monkeypatch, tmp_path, "pre-commit", "pre-commit.js")
        assert problems == []

    @pytest.mark.parametrize("name", ["install.mjs", "uninstall.mjs", "common.sh"])
    def test_allows_husky_support_files(self, monkeypatch, tmp_path, name):
        assert self._doctor_on(monkeypatch, tmp_path, name) == []

    def test_inventory_lists_every_hook(self):
        assert verify.main(["--list"]) == 0
        assert len(rt.HOOKS) == 11

    def test_every_shipped_hook_is_one_husky_dispatches(self):
        # husky only creates _/ shims for the hooks it knows; a hook outside
        # that set would be a file that never runs.
        husky_hooks = set(rt.HOOKS)
        assert husky_hooks.issubset(set(rt.KNOWN_GIT_HOOKS))


class TestShimChain:
    """The real .husky/_ -> .husky/<hook> -> module chain, in this repo."""

    @pytest.mark.parametrize("hook", ["pre-commit", "commit-msg"])
    def test_husky_shim_is_executable_and_delegates(self, hook):
        shim = rt.REPO_ROOT / ".husky" / hook
        bootstrap = rt.REPO_ROOT / ".husky" / "lib" / "bootstrap.sh"
        assert shim.is_file() and bootstrap.is_file()
        text = shim.read_text(encoding="utf-8")
        assert "lib/bootstrap.sh" in text
        assert f"gatekeeper_run_hook {hook}" in text

    def test_bootstrap_sources_common_sh(self):
        text = (rt.REPO_ROOT / ".husky" / "lib" / "bootstrap.sh").read_text(encoding="utf-8")
        assert "lib/common.sh" in text

    def test_common_sh_does_not_hijack_stdin(self):
        """The tty workaround must stay opt-in.

        pre-push and post-rewrite read git's payload from stdin; an
        unconditional `exec < /dev/tty` would silently discard it.
        """
        text = (rt.REPO_ROOT / ".husky" / "lib" / "common.sh").read_text(encoding="utf-8")
        assert "gatekeeper_restore_tty()" in text
        assert "exec < /dev/tty" in text
        for hook in ("pre-push", "post-rewrite"):
            body = (rt.REPO_ROOT / ".husky" / hook).read_text(encoding="utf-8")
            assert "restore_tty" not in body
