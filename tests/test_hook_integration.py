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
import time

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.hooks import (
    _checks,
    _events,
    applypatch_msg,
    post_applypatch,
    post_checkout,
    post_commit,
    post_merge,
    post_rewrite,
    pre_applypatch,
    pre_auto_gc,
    pre_merge_commit,
    pre_rebase,
    prepare_commit_msg,
    verify,
)
from scripts.hooks import _policy as policy
from scripts.hooks import _runtime as rt

# The generated dispatcher, which only exists after an install. Tests that
# exercise husky's own `h` layer need it and skip without it.
HUSKY_DISPATCHER = rt.REPO_ROOT / ".husky" / "_" / "h"
needs_husky = pytest.mark.skipif(
    not HUSKY_DISPATCHER.is_file(),
    reason="hooks are not generated (.husky/_/h missing) - run `npm install`",
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
    monkeypatch.delenv(rt.STATE_DIR_ENV, raising=False)
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

    @pytest.fixture(autouse=True)
    def isolated_state_dir(self, tmp_path, monkeypatch):
        """Keep hook state out of *this* repository.

        These tests run the real shims against the real checkout, so a hook
        that answers by reading repository state is reading the state of
        whatever is happening right now. `pre-auto-gc` declines while a gate
        is running, and the pre-push hook -- which runs this very suite --
        holds exactly that marker. Both tests below passed standalone and
        failed inside the push, reporting the gate's state rather than the
        hook's.
        """
        monkeypatch.setenv(rt.STATE_DIR_ENV, str(tmp_path / "hook-state"))

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

    def test_survives_an_errexit_set_u_shell(self):
        """Upstream's `8_set_u.sh`, one layer stricter.

        husky runs the shim with `sh -e`; `-u` is added on top by anyone who
        sets it in a startup file. An unset variable read at the top level
        would abort the shell before the hook ever starts.
        """
        result = self._sh("set -u; . .husky/lib/bootstrap.sh; echo survived")
        assert result.returncode == 0, result.stderr
        assert "survived" in result.stdout

    def test_survives_set_u_all_the_way_into_a_hook(self):
        result = self._sh(
            "set -u; . .husky/lib/bootstrap.sh; gatekeeper_run_hook pre-auto-gc"
        )
        assert result.returncode == 0, result.stderr

    def test_pre_auto_gc_shim_allows_collection_under_isolated_state(self):
        """The state redirect is what makes this the hook's answer, not the gate's."""
        result = self._sh(". .husky/lib/bootstrap.sh; gatekeeper_run_hook pre-auto-gc")
        assert result.returncode == 0, result.stderr

    def test_pre_auto_gc_shim_declines_when_its_state_holds_a_marker(self, tmp_path):
        """The same hook and the same repository, the opposite answer.

        Proves the redirect above decides the outcome, rather than luck about
        whether a gate happened to be running when the suite ran.
        """
        state = tmp_path / "gate-state"
        state.mkdir()
        (state / rt.GATE_RUN_FILE).write_text(
            json.dumps({"label": "pre-push", "pid": 1, "started_at": time.time()}),
            encoding="utf-8",
        )
        result = self._sh(
            ". .husky/lib/bootstrap.sh; gatekeeper_run_hook pre-auto-gc",
            env={rt.STATE_DIR_ENV: state.as_posix()},
        )
        assert result.returncode == 1
        assert "skipping background gc" in result.stderr

    @needs_husky
    def test_xdg_startup_file_is_sourced_by_husky(self, tmp_path):
        """The documented `init.sh` hook, through the real dispatcher.

        Verified by side effect rather than by reading the shim: the file
        writes a marker, so the assertion is that husky actually sourced it.
        """
        config = tmp_path / "config" / "husky"
        config.mkdir(parents=True)
        marker = tmp_path / "marker"
        (config / "init.sh").write_text(
            f'printf "sourced\\n" > "{marker.as_posix()}"\n', encoding="utf-8"
        )
        result = self._sh(
            "sh .husky/_/pre-auto-gc",
            env={"XDG_CONFIG_HOME": (tmp_path / "config").as_posix()},
        )
        assert result.returncode == 0, result.stderr
        assert marker.read_text(encoding="utf-8").strip() == "sourced"

    @needs_husky
    def test_husky_zero_from_a_startup_file_disables_every_hook(self, tmp_path):
        """Upstream's `9_husky_0.sh`: an init.sh can turn the hooks off.

        The control run comes first, so a hook that exits 0 because it is
        broken cannot be mistaken for one that was correctly disabled.
        """
        message = tmp_path / "msg"
        message.write_text("not a conventional subject\n", encoding="utf-8")
        call = f'sh .husky/_/commit-msg "{message.as_posix()}"'

        assert self._sh(call).returncode != 0

        config = tmp_path / "config" / "husky"
        config.mkdir(parents=True)
        (config / "init.sh").write_text("export HUSKY=0\n", encoding="utf-8")
        disabled = self._sh(
            call, env={"XDG_CONFIG_HOME": (tmp_path / "config").as_posix()}
        )
        assert disabled.returncode == 0, disabled.stderr


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
        assert len(rt.HOOKS) == 14

    def test_every_shipped_hook_is_one_husky_dispatches(self):
        # husky only creates _/ shims for the hooks it knows; a hook outside
        # that set would be a file that never runs, and one it dispatches
        # with nothing behind it is a silently ignored no-op.
        assert set(rt.HOOKS) == set(rt.HUSKY_HOOKS)
        assert set(rt.HOOKS).issubset(set(rt.KNOWN_GIT_HOOKS))

    def test_generated_dispatchers_cover_every_hook(self):
        """Every hook we ship has a real dispatcher in the generated dir."""
        generated = rt.REPO_ROOT / ".husky" / "_"
        if not generated.is_dir():
            pytest.skip("hooks are not generated - run `npm install`")
        for hook in rt.HOOKS:
            assert (generated / hook).is_file(), f"no generated dispatcher for {hook}"


class TestPreApplyPatch:
    """`git am` runs no commit hooks, so it needs its own content gate."""

    def test_blocks_on_conflict_markers(self, repo):
        marked = repo / "notes.txt"
        marked.write_text(
            "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> other\n", encoding="utf-8"
        )
        _run(repo, "add", "notes.txt")
        assert _checks.conflict_markers() == ["notes.txt"]
        assert pre_applypatch.main() == 1

    def test_accepts_a_clean_staged_change(self, repo):
        (repo / "clean.py").write_text("x = 1\n", encoding="utf-8")
        _run(repo, "add", "clean.py")
        assert pre_applypatch.main() == 0

    def test_finds_the_same_files_pre_commit_would(self, repo):
        (repo / "clean.py").write_text("x = 1\n", encoding="utf-8")
        (repo / "notes.txt").write_text("plain\n", encoding="utf-8")
        _run(repo, "add", "clean.py", "notes.txt")
        assert _checks.staged_python_files() == ["clean.py"]

    def test_skip_env_disables_the_hook(self, repo, monkeypatch):
        monkeypatch.setenv(rt.SKIP_ENV, "pre-applypatch")
        assert pre_applypatch.main() == 0


class TestPostApplyPatch:
    """`git am` never runs post-commit, so the patch path records its own."""

    def test_records_the_patch_commit(self, repo):
        sha = _commit(repo, "a.txt")
        assert post_applypatch.main() == 0
        events = _events.read_events()
        assert events[-1]["event"] == "applypatch"
        assert events[-1]["sha"] == sha
        assert events[-1]["branch"] == "main"

    def test_writes_into_the_commit_event_log(self, repo):
        _commit(repo, "a.txt")
        post_commit.main()
        post_applypatch.main()
        events = _events.read_events()
        assert [event["event"] for event in events] == ["commit", "applypatch"]

    def test_never_fails_without_a_commit(self, repo, monkeypatch):
        monkeypatch.setattr(rt, "REPO_ROOT", repo / "does-not-exist")
        assert post_applypatch.main() == 0


class TestGateRunMarker:
    """The marker that keeps background housekeeping away from a live gate."""

    def test_lifecycle(self, repo):
        assert rt.gate_run_active() is False
        assert rt.begin_gate_run("pre-push") is True
        assert rt.gate_run_active() is True
        rt.end_gate_run()
        assert rt.gate_run_active() is False

    def test_marker_lives_inside_the_git_directory(self, repo):
        rt.begin_gate_run("pre-push")
        assert (repo / ".git" / rt.STATE_DIR / rt.GATE_RUN_FILE).is_file()
        rt.end_gate_run()
        assert not (repo / ".git" / rt.STATE_DIR / rt.GATE_RUN_FILE).exists()

    def test_abandoned_marker_is_ignored(self, repo):
        directory = repo / ".git" / rt.STATE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        (directory / rt.GATE_RUN_FILE).write_text(
            json.dumps({"label": "pre-push", "pid": 1, "started_at": 0}), encoding="utf-8"
        )
        assert rt.gate_run_active() is False

    def test_unreadable_marker_is_ignored(self, repo):
        directory = repo / ".git" / rt.STATE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        (directory / rt.GATE_RUN_FILE).write_text("not json", encoding="utf-8")
        assert rt.gate_run_active() is False

    def test_is_a_no_op_outside_a_repository(self, repo, monkeypatch):
        monkeypatch.setattr(rt, "REPO_ROOT", repo / "does-not-exist")
        assert rt.begin_gate_run("pre-push") is False
        assert rt.gate_run_active() is False
        assert rt.end_gate_run() is None

    def test_state_directory_can_be_relocated(self, repo, tmp_path, monkeypatch):
        """The escape hatch a read-only git directory -- and these tests -- need."""
        elsewhere = tmp_path / "elsewhere"
        monkeypatch.setenv(rt.STATE_DIR_ENV, str(elsewhere))
        assert rt.begin_gate_run("pre-push") is True
        assert (elsewhere / rt.GATE_RUN_FILE).is_file()
        assert not (repo / ".git" / rt.STATE_DIR).exists()
        assert rt.gate_run_active() is True

    def test_relocated_state_ignores_a_marker_in_the_repository(
        self, repo, tmp_path, monkeypatch
    ):
        """A live marker in the repo must be invisible once state is moved."""
        directory = repo / ".git" / rt.STATE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        (directory / rt.GATE_RUN_FILE).write_text(
            json.dumps({"label": "pre-push", "pid": 1, "started_at": time.time()}),
            encoding="utf-8",
        )
        assert rt.gate_run_active() is True
        monkeypatch.setenv(rt.STATE_DIR_ENV, str(tmp_path / "elsewhere"))
        assert rt.gate_run_active() is False

    def test_resolving_state_does_not_create_it(self, tmp_path, monkeypatch):
        """A hook that only reads state must not leave a directory behind."""
        target = tmp_path / "never-made"
        monkeypatch.setenv(rt.STATE_DIR_ENV, str(target))
        assert rt.state_dir(create=False) == target
        assert rt.gate_run_active() is False
        assert not target.exists()


class TestPreAutoGc:
    """A non-zero exit here is not an error: it postpones the collection."""

    def test_allows_collection_when_idle(self, repo):
        assert pre_auto_gc.main() == 0

    def test_declines_while_a_gate_is_running(self, repo):
        rt.begin_gate_run("pre-push")
        assert pre_auto_gc.main() == 1

    def test_allows_collection_after_the_gate_finishes(self, repo):
        rt.begin_gate_run("pre-push")
        rt.end_gate_run()
        assert pre_auto_gc.main() == 0

    def test_ignores_an_abandoned_marker(self, repo):
        directory = repo / ".git" / rt.STATE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        (directory / rt.GATE_RUN_FILE).write_text(
            json.dumps({"label": "pre-push", "pid": 1, "started_at": 0}), encoding="utf-8"
        )
        assert pre_auto_gc.main() == 0

    def test_skip_env_leaves_collection_alone(self, repo, monkeypatch):
        rt.begin_gate_run("pre-push")
        monkeypatch.setenv(rt.SKIP_ENV, "pre-auto-gc")
        assert pre_auto_gc.main() == 0

    def test_pre_push_clears_the_marker_it_sets(self, repo, monkeypatch):
        """A blocked push must not leave the marker behind."""
        from scripts.hooks import pre_push

        monkeypatch.setattr(pre_push, "_gate", lambda _stdin: 1)
        monkeypatch.setattr(sys, "stdin", type("S", (), {"isatty": lambda self: True})())
        assert pre_push.main() == 1
        assert rt.gate_run_active() is False


class TestEventLog:
    """One writer, one file, one format -- shared by every reporting hook."""

    def test_post_merge_records_a_merge_event(self, repo):
        base = _run(repo, "rev-parse", "HEAD").stdout.strip()
        (repo / ".gatekeeper.yml").write_text("rules: {}\n", encoding="utf-8")
        _run(repo, "add", ".gatekeeper.yml")
        _run(repo, "commit", "-q", "-m", "chore: change gate config")
        _run(repo, "update-ref", "ORIG_HEAD", base)

        assert post_merge.main() == 0
        event = _events.read_events()[-1]
        assert event["event"] == "merge"
        assert event["gate_files_changed"] == 1
        assert "gate" not in event  # a merge is not a Gate 1 commit

    def test_post_rewrite_stores_the_sha_mapping(self, repo):
        head = _run(repo, "rev-parse", "HEAD").stdout.strip()
        post_rewrite.report(f"{head} {head}\n", "amend")

        event = _events.read_events()[-1]
        assert event["event"] == "rewrite"
        assert event["command"] == "amend"
        assert event["rewrites"] == [{"old": head, "new": head}]
        # A rewrite is branch-independent: it can be recorded mid-detach.
        assert "branch" not in event

    def test_parse_pairs_ignores_malformed_lines(self):
        assert post_rewrite.parse_pairs("garbage\n\na b\n") == [("a", "b")]

    def test_read_events_respects_the_limit(self, repo):
        for name in ("a.txt", "b.txt", "c.txt"):
            _commit(repo, name)
            post_commit.main()
        assert len(_events.read_events(2)) == 2
        assert len(_events.read_events()) == 3

    def test_a_torn_last_line_does_not_hide_the_records_before_it(self, repo):
        _commit(repo, "a.txt")
        post_commit.main()
        with open(_events.events_path(), "a", encoding="utf-8") as handle:
            handle.write('{"event": "commit"')  # a hook killed mid-write
        assert len(_events.read_events()) == 1

    def test_empty_log_is_not_an_error(self, repo):
        assert _events.read_events() == []

    def test_write_event_reports_failure_outside_a_repository(self, repo, monkeypatch):
        monkeypatch.setattr(rt, "REPO_ROOT", repo / "does-not-exist")
        assert _events.write_event("commit") is False
        assert _events.record_commit("commit") is None

    def test_the_log_follows_the_relocated_state_directory(
        self, repo, tmp_path, monkeypatch
    ):
        elsewhere = tmp_path / "elsewhere"
        monkeypatch.setenv(rt.STATE_DIR_ENV, str(elsewhere))
        _commit(repo, "a.txt")
        assert post_commit.main() == 0
        assert (elsewhere / _events.EVENTS_FILE).is_file()
        assert not (repo / ".git" / rt.STATE_DIR).exists()
        assert _events.read_events()[-1]["event"] == "commit"


class TestHookLatency:
    """Upstream's `11_time.sh`: a slow hook gets disabled by its users."""

    def test_reporting_hooks_stay_fast(self, repo, monkeypatch):
        class _TtyStdin:
            """post-rewrite reads stdin; under pytest it is not a tty."""

            def isatty(self) -> bool:
                return True

            def read(self) -> str:
                return ""

        monkeypatch.setattr(sys, "stdin", _TtyStdin())
        calls = {
            "post-checkout": lambda: post_checkout.main(["a", "b", "1"]),
            "post-commit": post_commit.main,
            "post-merge": post_merge.main,
            "post-rewrite": lambda: post_rewrite.main(["amend"]),
            "pre-auto-gc": pre_auto_gc.main,
        }
        for name, call in calls.items():
            started = time.monotonic()
            assert call() == 0
            elapsed = time.monotonic() - started
            assert elapsed < 2.0, f"{name} took {elapsed:.2f}s"


class TestDoctorEnvironment:
    """The doctor's environment checks, which need a breakable repo."""

    def test_flags_a_flag_shaped_hooks_path(self, repo):
        """The exact damage `npx husky --version` does."""
        _run(repo, "config", "core.hooksPath", "--version/_")
        name, failures, _warnings = verify._check_hooks_path()
        assert name == "core.hooksPath"
        assert any("looks like a command-line flag" in problem for problem in failures)
        assert any("npm run hooks:install" in problem for problem in failures)

    def test_reports_a_missing_hooks_path(self, repo):
        _run(repo, "config", "--unset", "core.hooksPath")
        _name, failures, _warnings = verify._check_hooks_path()
        assert any("not set" in problem for problem in failures)

    def test_shadowed_hooks_reports_every_hook_not_just_pre_push(self, repo):
        hooks = repo / ".git" / "hooks"
        (hooks / "pre-commit").write_text("#!/bin/sh\n", encoding="utf-8")
        (hooks / "pre-push").write_text("#!/bin/sh\n", encoding="utf-8")
        _name, failures, _warnings = verify._check_git_dir_shadowing()
        assert any("pre-commit" in problem for problem in failures)
        assert any("pre-push" in problem for problem in failures)
        # The stock `.sample` files git writes are not shadowing hooks.
        assert not any(".sample" in problem for problem in failures)

    def test_quiet_when_nothing_shadows_the_hooks(self, repo):
        _name, failures, _warnings = verify._check_git_dir_shadowing()
        assert failures == []

    def test_generated_dir_warns_rather_than_fails_when_absent(self, repo):
        """A fresh Python-only clone must not be told the repo is broken."""
        _name, failures, warnings = verify._check_generated_dir()
        assert failures == []
        assert any("npm install" in warning for warning in warnings)

    def test_generated_dir_flags_a_missing_dispatcher(self, repo):
        generated = repo / ".husky" / "_"
        generated.mkdir(parents=True)
        (generated / "h").write_text("#!/usr/bin/env sh\n", encoding="utf-8")
        _name, failures, _warnings = verify._check_generated_dir()
        assert any("generated dispatcher" in problem for problem in failures)

    def test_generated_dir_wants_the_ignore_file(self, repo):
        generated = repo / ".husky" / "_"
        generated.mkdir(parents=True)
        (generated / "h").write_text("#!/usr/bin/env sh\n", encoding="utf-8")
        for hook in rt.HUSKY_HOOKS:
            (generated / hook).write_text("#!/usr/bin/env sh\n", encoding="utf-8")
        _name, failures, warnings = verify._check_generated_dir()
        assert failures == []
        assert any("gitignore" in warning for warning in warnings)

    def test_shim_shape_accepts_the_delegators_we_ship(self, repo):
        husky = repo / ".husky"
        husky.mkdir()
        for hook in rt.HOOKS:
            (husky / hook).write_text(
                f'. "${{0%/*}}/lib/bootstrap.sh"\ngatekeeper_run_hook {hook}\n',
                encoding="utf-8",
            )
        _name, failures, _warnings = verify._check_shim_shape()
        assert failures == []

    def test_shim_shape_rejects_logic_in_shell(self, repo):
        """Logic in a shim is invisible to the hook test suite."""
        husky = repo / ".husky"
        husky.mkdir()
        for hook in rt.HOOKS:
            (husky / hook).write_text(
                f'. "${{0%/*}}/lib/bootstrap.sh"\ngatekeeper_run_hook {hook}\n',
                encoding="utf-8",
            )
        (husky / "pre-commit").write_text(
            "#!/usr/bin/env sh\nruff check .\npytest -q\npython scripts/hooks/pre_commit.py\n",
            encoding="utf-8",
        )
        _name, failures, _warnings = verify._check_shim_shape()
        assert any("does not reference lib/bootstrap.sh" in problem for problem in failures)
        assert any("is 4 lines of shell" in problem for problem in failures)

    def test_hooks_present_reports_every_missing_shim(self, repo):
        _name, failures, _warnings = verify._check_hooks_present()
        assert len([p for p in failures if "missing hook shim" in p]) == len(rt.HOOKS)

    def test_toolchain_rejects_a_git_too_old_for_hooks_path(self, monkeypatch):
        monkeypatch.setattr(verify, "_git_version", lambda: (2, 8))
        _name, failures, _warnings = verify._check_toolchain()
        assert any("too old" in problem for problem in failures)

    def test_toolchain_accepts_modern_git(self):
        _name, failures, _warnings = verify._check_toolchain()
        assert not any("git" in problem for problem in failures)

    def test_missing_husky_is_a_warning_not_a_failure(self, repo):
        """git is present in the temp repo; only the Node side is absent."""
        _name, failures, warnings = verify._check_toolchain()
        assert failures == []
        assert any("husky" in warning for warning in warnings)

    def test_strict_promotes_warnings_to_failures(self, monkeypatch):
        code, value = rt.git("config", "--get", "core.hooksPath")
        if code != 0 or value != verify.EXPECTED_HOOKS_PATH:
            pytest.skip("hooks are not installed in this checkout")
        monkeypatch.setattr(
            verify,
            "_check_toolchain",
            lambda: ("toolchain", [], ["pretend node is missing"]),
        )
        assert verify.main([]) == 0
        assert verify.main(["--strict"]) == 1

    def test_environment_notes_name_a_deprecated_huskyrc(self, monkeypatch, tmp_path):
        (tmp_path / ".huskyrc").write_text("export PATH=$PATH\n", encoding="utf-8")
        monkeypatch.setenv("HOME", tmp_path.as_posix())
        monkeypatch.setenv("USERPROFILE", tmp_path.as_posix())
        monkeypatch.setenv("XDG_CONFIG_HOME", (tmp_path / "config").as_posix())
        assert any("DEPRECATED" in note for note in verify._environment_notes())

    def test_environment_notes_name_the_startup_file_in_use(self, monkeypatch, tmp_path):
        config = tmp_path / "config" / "husky"
        config.mkdir(parents=True)
        (config / "init.sh").write_text("export A=1\n", encoding="utf-8")
        monkeypatch.setenv("HOME", tmp_path.as_posix())
        monkeypatch.setenv("USERPROFILE", tmp_path.as_posix())
        monkeypatch.setenv("XDG_CONFIG_HOME", (tmp_path / "config").as_posix())
        assert any("startup file in use" in note for note in verify._environment_notes())


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
