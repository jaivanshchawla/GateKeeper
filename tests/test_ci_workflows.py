"""The CI/CD workflows, asserted from the suite.

Why this file exists
--------------------
Two workflows were red for weeks, and every reason was invisible from a
developer machine:

* the `hooks` job installed pytest and nothing else, so two tests that read a
  *configured* protected-branches list failed on all four matrix legs while
  passing on any machine that had PyYAML;
* the `suite` job ran the project suite from a fresh clone, which has neither
  data/commit_features.csv nor repos/, so the tests that read them errored
  during collection instead of skipping;
* Gate 3's `docker compose build` COPYs dashboard/dist/, a gitignored
  directory, so it failed on all 95 runs from a clean checkout.

Each of those is an invariant about a file rather than a behaviour of the
library, which is why the ordinary suite could not see them. They are asserted
here, in the suite, so the next version of the same mistake is caught by a red
test instead of a red push.
"""

from __future__ import annotations

import ast
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
TESTS = REPO_ROOT / "tests"

HOOK_TEST_FILES = (
    "tests/test_hooks.py",
    "tests/test_hook_integration.py",
    "tests/test_husky_fidelity.py",
)


# --------------------------------------------------------------------------
# Reading the workflows
# --------------------------------------------------------------------------


def _workflow_paths() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _workflow(name: str) -> dict:
    path = WORKFLOWS / name
    assert path.is_file(), f"{name} is missing from .github/workflows/"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _steps(workflow: dict, job: str) -> list:
    jobs = workflow.get("jobs", {})
    assert job in jobs, f"no `{job}` job in this workflow (has: {sorted(jobs)})"
    return jobs[job].get("steps", [])


def _step(steps: list, name: str) -> dict:
    for step in steps:
        if step.get("name") == name:
            return step
    raise AssertionError(f"no step named {name!r} (has: {[s.get('name') for s in steps]})")


def _script(step: dict) -> str:
    """The shell a step runs, whether it is a `run` or an action's inputs."""
    return step.get("run", "") or ""


def _step_names(steps: list) -> list[str]:
    return [step.get("name", "") for step in steps]


# --------------------------------------------------------------------------
# Every workflow still parses and still triggers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda path: path.name)
def test_every_workflow_is_valid_yaml(path: Path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path.name} is not a mapping"
    assert document.get("jobs"), f"{path.name} declares no jobs"


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda path: path.name)
def test_every_workflow_declares_a_trigger(path: Path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML resolves the bare key `on` to True under YAML 1.1.
    trigger = document.get("on", document.get(True))
    assert trigger, f"{path.name} has no `on:` trigger, so it never runs"


def test_the_husky_workflow_still_covers_three_operating_systems():
    """The matrix is the point of that job: husky's bootstrap is per-platform."""
    matrix = (
        _workflow("hooks.yml")["jobs"]["hooks"]["strategy"]["matrix"]
    )
    assert {"ubuntu-latest", "macos-latest", "windows-latest"} <= set(matrix["os"])
    assert {"3.11", "3.12"} <= set(matrix["python"])


# --------------------------------------------------------------------------
# The PyYAML failure: a job must install what its tests import
# --------------------------------------------------------------------------


def test_the_hooks_job_installs_pyyaml():
    """The exact bug: the job ran the hook tests without the library they read."""
    steps = _steps(_workflow("hooks.yml"), "hooks")
    install = _script(_step(steps, "Install test dependencies"))
    assert "pyyaml" in install.lower(), (
        "the `hooks` job installs its test dependencies here; "
        "tests/test_hook_integration.py reads .gatekeeper.yml through PyYAML"
    )


def test_the_hooks_job_asserts_that_import_works():
    """Otherwise dropping `pyyaml` silently turns the config tests into skips."""
    steps = _steps(_workflow("hooks.yml"), "hooks")
    scripts = "\n".join(_script(step) for step in steps)
    assert re.search(r"import yaml", scripts), (
        "the `hooks` job needs a step that fails when PyYAML is not importable, "
        "so its absence cannot look like a skip"
    )


def test_the_config_tests_declare_that_they_need_yaml():
    from tests import test_hook_integration as hook_tests

    # The module writes the marker from what it detected, so the condition must
    # be the negation of availability rather than a constant.
    assert hook_tests.needs_yaml.mark.args == (not hook_tests.YAML_AVAILABLE,)
    assert hook_tests.needs_yaml.mark.kwargs.get("reason")

    for name in ("test_reads_configured_list", "test_rule_severity_reads_config"):
        function = getattr(hook_tests.TestProtectedBranchPolicy, name)
        marked = {mark.name for mark in getattr(function, "pytestmark", [])}
        assert "requires_yaml" in marked, f"{name} asserts a configured list"
        assert "skipif" in marked, f"{name} would fail, not skip, without PyYAML"


def test_the_policy_still_works_without_pyyaml():
    """The other half of the contract, on the side the suite can always reach."""
    from scripts.hooks import _policy as policy

    assert policy.DEFAULT_PROTECTED_BRANCHES, "there must be defaults to fall back to"
    assert "main" in policy.DEFAULT_PROTECTED_BRANCHES


# --------------------------------------------------------------------------
# The ruff failure: a hook the tests *run* is a dependency too
# --------------------------------------------------------------------------
#
# ruff is imported by nothing, so no import graph mentions it. The hook tests
# nonetheless call pre-commit and pre-applypatch, which lint staged files by
# running `python -m ruff`. Without it the hook blocked a clean staged change,
# and every job that ran those tests went red for a missing binary.


@pytest.mark.parametrize("job", ["hooks", "suite", "verification"])
def test_every_job_that_runs_the_hook_tests_installs_ruff(job: str):
    steps = _steps(_workflow("hooks.yml"), job)
    install = _script(_step(steps, "Install test dependencies"))
    assert "ruff" in install.lower(), (
        f"the `{job}` job runs the hook tests, which lint through ruff; "
        "installing only the libraries they import misses it"
    )


def test_the_hooks_job_proves_the_linter_runs():
    """Otherwise deleting it from the install line is silent until a test fails."""
    steps = _steps(_workflow("hooks.yml"), "hooks")
    scripts = "\n".join(_script(step) for step in steps)
    assert re.search(r"python -m ruff\b", scripts), (
        "the `hooks` job needs a step that fails when ruff is not runnable"
    )


# --------------------------------------------------------------------------
# The suite job: a red run, and a green-by-absence one, are both failures
# --------------------------------------------------------------------------


def test_the_suite_job_installs_pyyaml_and_reports_skips():
    steps = _steps(_workflow("hooks.yml"), "suite")
    assert "pyyaml" in _script(_step(steps, "Install test dependencies")).lower()
    run = _script(_step(steps, "Run the test suite"))
    assert "-rs" in run, (
        "-rs explains every skip: without it, 'the data is not here' and "
        "'the test never ran' are indistinguishable in the log"
    )


def test_the_suite_job_still_runs_the_whole_suite():
    steps = _steps(_workflow("hooks.yml"), "suite")
    assert "tests/" in _script(_step(steps, "Run the test suite"))


class _FakeItem:
    """The smallest thing conftest's hook needs: markers in, decisions out."""

    def __init__(self, markers: dict):
        self._markers = dict(markers)
        self.added: list = []

    def get_closest_marker(self, name: str):
        return self._markers.get(name)

    def add_marker(self, marker) -> None:
        self.added.append(marker)


def test_an_absent_training_csv_becomes_a_skip(tmp_path, monkeypatch):
    """A missing input must produce a skip with a reason, never an error.

    This is the collection-time decision that turned the `suite` job red, so it
    is exercised directly rather than through a full pytest run.
    """
    from tests import conftest

    missing = tmp_path / "commit_features.csv"
    monkeypatch.setattr(conftest, "DATA_CSV", missing)

    item = _FakeItem({"needs_data": object()})
    conftest.pytest_collection_modifyitems(None, [item])

    assert len(item.added) == 1, "a data-dependent test must be skipped, not collected"
    reason = item.added[0].kwargs["reason"]
    assert "commit_features.csv" in reason and "not present" in reason


def test_present_inputs_are_never_skipped(tmp_path, monkeypatch):
    from tests import conftest

    csv = tmp_path / "commit_features.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    repos = tmp_path / "repos"
    (repos / "rust" / ".git").mkdir(parents=True)
    monkeypatch.setattr(conftest, "DATA_CSV", csv)
    monkeypatch.setattr(conftest, "REPOS_DIR", repos)

    data_item = _FakeItem({"needs_data": object()})
    repo_item = _FakeItem({"needs_repos": pytest.mark.needs_repos("rust")})
    conftest.pytest_collection_modifyitems(None, [data_item, repo_item])

    assert data_item.added == []
    assert repo_item.added == []


def test_missing_repositories_are_named_in_the_reason(tmp_path, monkeypatch):
    from tests import conftest

    monkeypatch.setattr(conftest, "REPOS_DIR", tmp_path / "nope")
    item = _FakeItem({"needs_repos": pytest.mark.needs_repos("rust", "django")})
    conftest.pytest_collection_modifyitems(None, [item])

    reason = item.added[0].kwargs["reason"]
    assert "rust" in reason and "django" in reason


# --------------------------------------------------------------------------
# Every test that reads an input a fresh clone does not have must say so
# --------------------------------------------------------------------------


def _declared_markers(path: Path) -> set[str]:
    """Marker names declared at module level or on any test in the file."""
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def collect(node: ast.AST) -> None:
        candidates = [node]
        if isinstance(node, ast.Call):
            candidates.append(node.func)
        for candidate in candidates:
            if isinstance(candidate, ast.Attribute):
                names.add(candidate.attr)
            elif isinstance(candidate, ast.Name):
                names.add(candidate.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            for element in (
                node.value.elts
                if isinstance(node.value, (ast.List, ast.Tuple))
                else [node.value]
            ):
                collect(element)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                collect(decorator)
    return names


def test_data_reading_tests_declare_what_they_need():
    """CI has no training data and no clones; reading either must be declared.

    A test may either carry the marker (which conftest turns into a skip with a
    reason) or guard itself explicitly. What it may not do is read an input that
    is not there and fail, which is what made the `suite` job red on every run.
    """
    offenders: list[str] = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue  # this file names the paths; it does not read them
        text = path.read_text(encoding="utf-8")
        markers = _declared_markers(path)
        guarded = "pytest.skip(" in text or "importorskip(" in text
        if "commit_features.csv" in text and "needs_data" not in markers and not guarded:
            offenders.append(f"{path.name}: reads the training CSV")
        if "repos/" in text and "needs_repos" not in markers and not guarded:
            offenders.append(f"{path.name}: reads a clone under repos/")
    assert not offenders, (
        "these tests read inputs a fresh clone does not have and neither declare "
        "a marker nor guard themselves:\n  " + "\n  ".join(offenders)
    )


def test_the_tests_that_made_the_suite_job_red_stay_marked():
    """A named pin, because these four are the ones that actually failed."""
    determinism = _declared_markers(TESTS / "test_determinism.py")
    parity = _declared_markers(TESTS / "test_feature_parity.py")
    fuzz = _declared_markers(TESTS / "test_fuzz_thresholds.py")
    performance = _declared_markers(TESTS / "test_perf_regression.py")

    assert "needs_data" in determinism
    assert {"needs_data", "needs_repos"} <= parity
    assert "needs_data" in fuzz
    assert "needs_repos" in performance


def test_markers_are_registered_with_pytest(request):
    """An unregistered marker is a warning, and warnings get ignored."""
    registered = request.config.getini("markers")
    for name in ("needs_data", "needs_repos", "requires_yaml"):
        assert any(entry.startswith(name) for entry in registered), name


# --------------------------------------------------------------------------
# Gate 3: the docker build needs a directory that is not in the repository
# --------------------------------------------------------------------------


def test_gate3_builds_the_dashboard_before_building_the_images():
    steps = _steps(_workflow("gate3_post_deploy.yml"), "smoke-tests")
    names = _step_names(steps)
    build = _script(_step(steps, "Build the dashboard"))
    assert "npm" in build and "build" in build
    assert names.index("Build the dashboard") < names.index("Build Docker containers"), (
        "api/Dockerfile COPYs dashboard/dist/, so the dashboard has to exist first"
    )


def test_the_dashboard_dist_directory_exists_in_a_fresh_clone():
    """Belt and braces: the build step above, and a tracked place-holder."""
    placeholder = REPO_ROOT / "dashboard" / "dist" / ".gitkeep"
    assert placeholder.is_file(), "a fresh clone has no dashboard/dist/ without it"

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "dashboard/dist/.gitkeep"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode == 0, "the place-holder must be committed, not just present"

    dockerfile = (REPO_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY dashboard/dist/" in dockerfile, (
        "if this COPY goes away the place-holder is no longer needed either"
    )


def test_the_docker_build_context_keeps_the_dashboard_dist():
    """A .dockerignore entry for `dist/` would recreate the Gate 3 failure.

    The Dockerfile's COPY only fails if the path is missing *from the build
    context*, so excluding it there is the other half of the same bug.
    """
    patterns = [
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    def excluded(path: str) -> bool:
        verdict = False
        for pattern in patterns:
            negated = pattern.startswith("!")
            rule = pattern.lstrip("!").rstrip("/")
            matches = fnmatch(path, rule) or any(
                fnmatch(part, rule) for part in Path(path).parts
            )
            if matches:
                verdict = not negated
        return verdict

    assert not excluded("dashboard/dist/.gitkeep"), (
        ".dockerignore must not exclude the directory api/Dockerfile COPYs"
    )


def test_the_dashboard_build_output_is_still_ignored():
    """The place-holder must not turn into a committed build artifact."""
    for artifact in ("dashboard/dist/index.html", "dashboard/dist/assets/app.js"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", artifact],
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.returncode == 0, f"{artifact} would be committed with the dist"


# --------------------------------------------------------------------------
# A failed run has to be readable without a signed-in browser
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["hooks.yml", "gate3_post_deploy.yml"])
def test_a_failed_run_uploads_its_output(name: str):
    workflow = _workflow(name)
    for job, definition in workflow.get("jobs", {}).items():
        uploads = [
            step
            for step in definition.get("steps", [])
            if "upload-artifact" in str(step.get("uses", ""))
            and "failure()" in str(step.get("if", ""))
        ]
        assert uploads, f"{name}:{job} fails without leaving its output behind"


def test_the_hooks_job_writes_its_test_output_to_a_file():
    """Uploading an artifact only helps if a step wrote one."""
    steps = _steps(_workflow("hooks.yml"), "hooks")
    run = _script(_step(steps, "Run the hook tests"))
    assert "tee" in run and "pipefail" in run, (
        "the test output has to be captured, and pipefail keeps pytest's exit status"
    )


def test_diagnostic_logs_are_not_committed():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for name in ("verify_all.log", "hook-tests.log", "suite.log"):
        assert name in ignored, f"{name} would show up as untracked noise"


# --------------------------------------------------------------------------
# The local matrix and the CI matrix must stay the same list
# --------------------------------------------------------------------------


def test_the_verification_job_runs_the_shared_matrix():
    steps = _steps(_workflow("hooks.yml"), "verification")
    run = "\n".join(_script(step) for step in steps)
    assert "scripts.ci.verify_all" in run
    assert "--profile ci" in run


def test_every_verify_all_step_names_a_step_that_exists_in_a_workflow():
    from scripts.ci import verify_all

    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in _workflow_paths()
    )
    missing = [
        f"{step.name} -> {step.ci_step!r}"
        for step in verify_all.STEPS
        if step.ci_step not in corpus
    ]
    assert not missing, (
        "these verification steps claim a CI equivalent that no longer exists, so "
        "the local matrix and CI have drifted apart:\n  " + "\n  ".join(missing)
    )


def test_verify_all_wires_the_hooks_before_it_checks_them():
    """The verification job failed on every OS because of this order.

    A fresh clone has neither core.hooksPath nor a generated .husky/_/, and
    `doctor --strict` reports both as failures. That is a real property of the
    checkout, not of the wiring, so the matrix has to install first -- which
    it did not, and every developer machine already had husky installed.
    """
    from scripts.ci import verify_all

    order = [step.name for step in verify_all.STEPS]
    assert order.index("install") < order.index("doctor"), (
        "`doctor --strict` fails on a fresh clone until the hooks are installed"
    )
    assert order.index("install") < order.index("hook-tests")
    assert order.index("install") < order.index("hostile-environment")


def test_verify_all_still_covers_the_hook_layer():
    from scripts.ci import verify_all

    names = {step.name for step in verify_all.STEPS}
    assert {
        "doctor",
        "inventory",
        "install",
        "hook-tests",
        "hostile-environment",
        "trace",
        "real-shims",
        "bootstrap",
        "doctor-rejects-flag",
        "conformance",
        "shellcheck",
        "project-suite",
    } <= names
    for path in verify_all.HOOK_TEST_FILES:
        assert (REPO_ROOT / path).is_file(), f"{path} is gone"
    assert tuple(verify_all.HOOK_TEST_FILES) == HOOK_TEST_FILES
