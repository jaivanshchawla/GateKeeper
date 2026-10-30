"""The onboarding pipeline: accept a repository, or refuse it with a reason.

Two things are being tested. First, that the stages compose -- a URL becomes a
checkout becomes scores becomes a gate decision becomes a registry entry.
Second, and more important, that the refusal path is real: a repository whose
history cannot support the label is not registered, and the gate that refused
it says which measurement failed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

import registry
from pipelines import onboard as pipeline
from tools import make_demo_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Stage 1: resolve ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/django/django.git", "django"),
        ("https://github.com/facebook/react", "react"),
        ("https://github.com/Owner/Some.Repo.git/", "some.repo"),
        ("https://gitlab.com/group/sub/project.git", "project"),
    ],
)
def test_derive_name_reads_the_project_from_the_url(url, expected):
    assert pipeline.derive_name(url) == expected


def test_derive_name_prefers_an_explicit_override():
    assert pipeline.derive_name("https://github.com/facebook/react.git", "react-18") == "react-18"


def test_derive_name_refuses_a_name_the_registry_would_reject():
    # `---` and `@scope/pkg` are both real things in URLs and both are invalid
    # registry names; failing here is far better than failing mid-registration.
    with pytest.raises(pipeline.OnboardError):
        pipeline.derive_name("https://github.com/owner/@scope.git")
    with pytest.raises(pipeline.OnboardError):
        pipeline.derive_name("https://github.com/owner/.git")


def test_check_remote_refuses_a_non_https_url():
    with pytest.raises(pipeline.OnboardError, match="https"):
        pipeline.check_remote("git@github.com:django/django.git")
    with pytest.raises(pipeline.OnboardError, match="https"):
        pipeline.check_remote("http://github.com/django/django.git")


# ── Stage 3: calibration ────────────────────────────────────────────────────


def test_calibration_puts_high_at_the_ninetieth_percentile():
    scores = [index / 100 for index in range(101)]
    thresholds = pipeline.calibrate(scores)
    assert thresholds == {"high": 0.9, "medium": 0.75}


def test_calibration_keeps_the_bands_ordered_on_a_flat_distribution():
    thresholds = pipeline.calibrate([0.42] * 40)
    assert thresholds["high"] > thresholds["medium"]
    assert thresholds["high"] <= 1.0


def test_calibration_of_no_scores_is_empty():
    assert pipeline.calibrate([]) == {}


def test_rebucket_reproduces_the_documented_ten_fifteen_seventy_five():
    scores = [index / 1000 for index in range(1000)]
    thresholds = pipeline.calibrate(scores)
    counts = pipeline.rebucket(scores, thresholds)
    assert counts["high"] == 100
    assert counts["medium"] == 150
    assert counts["low"] == 750


def test_rebucket_without_thresholds_is_all_low():
    assert pipeline.rebucket([0.1, 0.9], {}) == {"low": 2}


# ── Stage 4: gates ──────────────────────────────────────────────────────────


def _scored(**overrides):
    base = {
        "commits_scored": 40,
        "distinct_scores": 30,
        "score_min": 0.10,
        "score_max": 0.70,
        "observable_outcomes": 40,
        "label_rate": 0.12,
    }
    return base | overrides


def test_a_usable_history_passes_every_gate():
    passed, gates = pipeline.evaluate_gates(_scored())
    assert passed
    assert {gate["name"] for gate in gates} == {
        "enough_commits_scored",
        "model_discriminates",
        "outcomes_are_observable",
        "label_rate_is_usable",
    }


@pytest.mark.parametrize(
    ("overrides", "expected_gate"),
    [
        ({"commits_scored": 4}, "enough_commits_scored"),
        ({"distinct_scores": 1, "score_min": 0.5, "score_max": 0.5}, "model_discriminates"),
        ({"observable_outcomes": 3}, "outcomes_are_observable"),
        ({"label_rate": 0.0}, "label_rate_is_usable"),
        ({"label_rate": 1.0}, "label_rate_is_usable"),
        ({"label_rate": None}, "label_rate_is_usable"),
    ],
)
def test_a_history_that_cannot_support_the_label_is_refused(overrides, expected_gate):
    passed, gates = pipeline.evaluate_gates(_scored(**overrides))
    assert not passed
    failing = [gate["name"] for gate in gates if not gate["passed"]]
    assert failing == [expected_gate]


# ── Stage 5: register ───────────────────────────────────────────────────────

SAMPLE_REGISTRY = """\
# The registry.
schema_version: 1

repos:
  - name: django
    url: https://github.com/django/django.git
    clone_dir: repos/django
    role: train
    description: Keep this comment.
"""


@pytest.fixture
def registry_file(tmp_path, monkeypatch):
    path = tmp_path / "repos.yaml"
    path.write_text(SAMPLE_REGISTRY, encoding="utf-8")
    monkeypatch.setenv(registry.ENV_VAR, str(path))
    registry.clear_cache()
    yield path
    registry.clear_cache()


def test_register_appends_without_losing_the_files_comments(registry_file):
    result = pipeline.register(
        "flask",
        "https://github.com/pallets/flask.git",
        "repos/flask",
        "validate",
        "A candidate.",
        path=registry_file,
    )
    assert result["added"] is True
    assert result["declared_repositories"] == 2

    text = registry_file.read_text(encoding="utf-8")
    assert "# The registry." in text
    assert "Keep this comment." in text

    parsed = yaml.safe_load(text)
    assert [entry["name"] for entry in parsed["repos"]] == ["django", "flask"]
    assert parsed["repos"][1]["role"] == "validate"


def test_registering_the_same_url_again_is_a_no_op(registry_file):
    result = pipeline.register(
        "django", "https://github.com/django/django.git", "repos/django", "train", path=registry_file
    )
    assert result["added"] is False
    assert result["reason"] == "already declared"
    assert registry_file.read_text(encoding="utf-8") == SAMPLE_REGISTRY


def test_registering_a_known_name_with_a_different_url_is_refused(registry_file):
    with pytest.raises(pipeline.OnboardError, match="already declared"):
        pipeline.register(
            "django",
            "https://github.com/someone/else.git",
            "repos/else",
            "train",
            path=registry_file,
        )
    assert registry_file.read_text(encoding="utf-8") == SAMPLE_REGISTRY


def test_a_rejected_entry_is_rolled_back(registry_file, monkeypatch):
    """A bug in the append must not leave an unloadable registry behind."""
    real_load = registry.load
    calls = {"count": 0}

    def boom(*args, **kwargs):
        # The pre-write lookup succeeds; only the reload after the append fails,
        # which is the case that has to restore the file.
        calls["count"] += 1
        if calls["count"] > 1:
            raise registry.RegistryError("nope")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(registry, "load", boom)
    with pytest.raises(pipeline.OnboardError, match="rolled back"):
        pipeline.register(
            "flask",
            "https://github.com/pallets/flask.git",
            "repos/flask",
            "train",
            path=registry_file,
        )
    assert registry_file.read_text(encoding="utf-8") == SAMPLE_REGISTRY


# ── Thresholds ──────────────────────────────────────────────────────────────

SAMPLE_CONFIG = """\
feature_columns:
  - lines_added

thresholds:
  django:
    high: 0.8677
    medium: 0.6817
  _global:
    high: 0.8677
    medium: 0.6817
"""


def test_thresholds_are_appended_and_the_file_still_loads(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(SAMPLE_CONFIG, encoding="utf-8")

    result = pipeline.write_thresholds("flask", {"high": 0.5, "medium": 0.4}, path=config)
    assert result["written"] is True

    parsed = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert parsed["thresholds"]["flask"] == {"high": 0.5, "medium": 0.4}
    # The existing entries survive, so calibration of one repo cannot silently
    # retune another.
    assert parsed["thresholds"]["django"]["high"] == 0.8677


def test_a_thresholds_write_that_does_not_reload_is_rolled_back(tmp_path):
    config = tmp_path / "config.yaml"
    broken = "thresholds: [this is not a mapping\n"
    config.write_text(broken, encoding="utf-8")

    result = pipeline.write_thresholds("flask", {"high": 0.5, "medium": 0.4}, path=config)
    assert result["written"] is False
    assert config.read_text(encoding="utf-8") == broken


# ── End to end, offline ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def demo_checkout(tmp_path_factory):
    """The generated fixture repository, built once per session."""
    destination = tmp_path_factory.mktemp("demo") / "dummy-service"
    make_demo_repo.build(destination, force=True)
    return destination


def test_the_fixture_is_deterministic_in_shape(demo_checkout):
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=demo_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert int(count) == len(make_demo_repo.PLAN)
    # No uncommitted state: the pipeline scores commits, so a dirty tree would
    # change the result of the next run rather than the history.
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=demo_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status.strip() == ""


def test_onboarding_the_fixture_scores_gates_and_reports(demo_checkout, tmp_path):
    report = pipeline.onboard(
        "https://github.com/example/dummy-service.git",
        local=demo_checkout,
        name="dummy-service",
        role="validate",
        sample=12,
        min_commits=8,
        registry_path=tmp_path / "repos.yaml",
        report_dir=tmp_path,
        dry_run=True,
    )

    assert report["passed"] is True, [g for g in report["gates"] if not g["passed"]]
    assert report["repo"]["name"] == "dummy-service"
    assert report["dry_run"] is True
    assert report["registry"]["added"] is False

    scored = report["scored"]
    assert scored["commits_scored"] == 12
    assert sum(scored["band_counts"].values()) == scored["commits_scored"]
    assert len(scored["commits"]) == scored["commits_scored"]
    assert all(0.0 <= commit["score"] <= 1.0 for commit in scored["commits"])

    thresholds = report["thresholds"]
    assert thresholds["high"] > thresholds["medium"]
    # Calibration is what makes the bands reachable: on a fresh repository the
    # global cut-offs put everything in low.
    assert scored["band_counts"].get("high", 0) > 0

    written = json.loads((tmp_path / "dummy-service.json").read_text(encoding="utf-8"))
    assert written["passed"] is True
    assert written["scored"]["commits_scored"] == 12


def test_a_dry_run_never_touches_the_registry(demo_checkout, tmp_path):
    path = tmp_path / "repos.yaml"
    path.write_text(SAMPLE_REGISTRY, encoding="utf-8")

    pipeline.onboard(
        "https://github.com/example/dummy-service.git",
        local=demo_checkout,
        name="dummy-service",
        sample=6,
        min_commits=4,
        registry_path=path,
        report_dir=tmp_path,
        dry_run=True,
    )
    assert path.read_text(encoding="utf-8") == SAMPLE_REGISTRY


def test_a_checkout_that_is_not_git_is_refused_before_scoring(tmp_path):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    with pytest.raises(pipeline.OnboardError, match="not a git checkout"):
        pipeline.onboard(
            "https://github.com/example/nope.git",
            local=not_a_repo,
            name="nope",
            report_dir=tmp_path,
            dry_run=True,
        )


def test_an_unknown_role_is_refused(demo_checkout, tmp_path):
    with pytest.raises(pipeline.OnboardError, match="role"):
        pipeline.onboard(
            "https://github.com/example/dummy-service.git",
            local=demo_checkout,
            name="dummy-service",
            role="production",
            report_dir=tmp_path,
            dry_run=True,
        )
