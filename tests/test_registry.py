#!/usr/bin/env python3
"""The registry is the only place a repository is declared.

These tests exist because the five repositories used to be written out by
hand in eight files. Adding a sixth meant finding all eight, and forgetting
one produced no error -- the API simply served a list the pipeline did not
mine. So the tests do three things:

1. assert the shipped registry is well formed;
2. assert every consumer derives its list from it, by comparing each
   module's list against ``registry.repo_names()``;
3. assert no consumer goes back to writing the list out, by scanning the
   runtime sources for a line that names four or more of the repositories.

The invalid-registry cases are here for the same reason: a typo in the YAML
should fail loudly, at load, with the file and entry named -- not silently
drop a repository from the corpus.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path, PurePosixPath

import pytest
import yaml

import registry
from registry import RegistryError

REPO_ROOT = Path(__file__).resolve().parents[1]

# The consumers that used to carry a hand-written list. A new consumer
# belongs here too, so the scan below covers it.
CONSUMERS = (
    "api/main.py",
    "ml/history_features.py",
    "ml/m1_compute_features.py",
    "ml/build_multi_repo_dataset.py",
    "ml/rebuild_b2.py",
    "pipelines/run_retrain.py",
    "tests/test_band_calibration.py",
    "tests/test_scoring_consistency.py",
    "tests/test_perf_regression.py",
)

# A line naming this many repositories is a list of the corpus, not a
# selection from it. Selections (e.g. which three repos a perf test needs)
# stay allowed.
UNIVERSE_THRESHOLD = 4


def _write(tmp_path: Path, repos: list[dict]) -> Path:
    path = tmp_path / "repos.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "repos": repos}, sort_keys=False),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Every test starts from an empty parse cache."""
    registry.clear_cache()
    yield
    registry.clear_cache()


# ── The shipped registry ────────────────────────────────────────────────


class TestShippedRegistry:
    def test_loads_and_declares_the_training_corpus(self):
        declared = registry.load()
        assert len(declared) >= 5
        train = registry.repo_names(role="train")
        assert len(train) == len(declared), "the shipped corpus is all training repos"
        assert len(set(registry.repo_names())) == len(declared)

    def test_every_entry_is_a_clonable_https_url(self):
        for repo in registry.load():
            assert repo.url.startswith("https://"), repo.name
            assert repo.url.endswith(".git"), repo.name
            assert repo.role in registry.ROLES, repo.name

    def test_clone_dirs_are_inside_the_project_and_match_the_name(self):
        for repo in registry.load():
            assert Path(repo.clone_dir) == Path("repos") / repo.name, repo.name
            assert repo.path.parent == REPO_ROOT / "repos"
            assert REPO_ROOT in repo.path.parents

    def test_every_training_repo_has_a_threshold_band(self):
        """A repo the model scores needs thresholds, or the global fallback."""
        config = yaml.safe_load((REPO_ROOT / "ml" / "config.yaml").read_text(encoding="utf-8"))
        thresholds = config.get("thresholds", {})
        assert "_global" in thresholds
        missing = [name for name in registry.repo_names(role="train") if name not in thresholds]
        # Not a failure: `_global` is the documented fallback. It is a failure
        # to have neither, which the assertion above rules out.
        if missing:
            assert thresholds["_global"]["high"] > thresholds["_global"]["medium"]


# ── Every consumer derives from the registry ────────────────────────────


class TestConsumersAgree:
    def test_api_repo_list_is_the_registry(self):
        from api.main import REPO_MAP, REPO_NAMES

        assert list(REPO_NAMES) == list(registry.repo_names())
        assert set(REPO_MAP) == set(registry.repo_names())
        for name, path in REPO_MAP.items():
            assert path == registry.get(name).path

    def test_history_features_repo_list_is_the_registry(self):
        from ml.history_features import REPO_NAMES

        assert REPO_NAMES == list(registry.repo_names())

    def test_m1_feature_repo_list_is_the_registry(self):
        from ml.m1_compute_features import REPO_NAMES

        assert REPO_NAMES == list(registry.repo_names())

    def test_dataset_builder_mines_the_registry_corpus(self):
        from ml.build_multi_repo_dataset import REPOS

        assert [entry["name"] for entry in REPOS] == list(registry.repo_names(role="train"))
        assert [entry["url"] for entry in REPOS] == list(registry.urls(role="train"))

    def test_rebuild_b2_mines_the_registry_corpus(self):
        from ml.rebuild_b2 import REPOS

        assert [entry["name"] for entry in REPOS] == list(registry.repo_names(role="train"))
        for entry in REPOS:
            assert Path(entry["path"]) == registry.get(entry["name"]).path

    def test_retrain_dag_defaults_come_from_the_registry(self):
        """Read as source: importing the DAG pulls in Docker and KFP."""
        source = (REPO_ROOT / "pipelines" / "run_retrain.py").read_text(encoding="utf-8")
        assert "registered_urls" in source
        assert "github.com/" not in source, "the retrain DAG still spells out clone URLs"

    def test_band_calibration_scores_the_registry_corpus(self):
        from tests.test_band_calibration import REPOS

        assert REPOS == list(registry.repo_names(role="train"))

    def test_perf_regression_clone_paths_come_from_the_registry(self):
        from tests.test_perf_regression import REPO_PATH, SMALL_REPOS

        assert Path(REPO_PATH) == registry.clone_dir("rust")
        for path in SMALL_REPOS:
            assert Path(path).parent == REPO_ROOT / "repos"


# ── Nobody writes the corpus out again ──────────────────────────────────


class TestNoHardcodedCorpus:
    def test_no_consumer_lists_the_corpus_by_hand(self):
        names = set(registry.repo_names())
        offenders = []
        for relative in CONSUMERS:
            path = REPO_ROOT / relative
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                hits = sum(1 for name in names if f'"{name}"' in line or f"'{name}'" in line)
                if hits >= UNIVERSE_THRESHOLD:
                    offenders.append(f"{relative}:{lineno}: {line.strip()}")
        assert not offenders, "these lines hardcode the repository corpus:\n" + "\n".join(offenders)

    def test_scan_has_something_to_scan(self):
        """A refactor that renames these files should fail here, not silently pass."""
        missing = [relative for relative in CONSUMERS if not (REPO_ROOT / relative).is_file()]
        assert not missing, f"consumers listed but absent: {missing}"


# ── Adding a repository is one YAML entry ───────────────────────────────


class TestAddingARepository:
    def test_a_new_entry_is_visible_without_touching_code(self, tmp_path):
        entries = [
            {"name": name, "url": registry.get(name).url}
            for name in registry.repo_names()
        ]
        entries.append(
            {
                "name": "flask",
                "url": "https://github.com/pallets/flask.git",
                "role": "validate",
                "description": "a sixth repository, declared only in YAML",
            }
        )
        path = _write(tmp_path, entries)

        assert "flask" in registry.repo_names(path=path)
        assert registry.get("flask", path=path).clone_dir == PurePosixPath("repos", "flask").as_posix()
        assert registry.repo_names(role="validate", path=path) == ("flask",)
        assert registry.repo_names(role="train", path=path) == registry.repo_names()

    def test_the_env_var_relocates_the_registry(self, tmp_path, monkeypatch):
        path = _write(
            tmp_path,
            [{"name": "onlyrepo", "url": "https://example.com/onlyrepo.git"}],
        )
        monkeypatch.setenv(registry.ENV_VAR, str(path))

        assert registry.repo_names() == ("onlyrepo",)
        assert registry.get("onlyrepo").path == REPO_ROOT / "repos" / "onlyrepo"

    def test_an_edit_is_picked_up_without_a_restart(self, tmp_path):
        path = _write(tmp_path, [{"name": "one", "url": "https://example.com/one.git"}])
        assert registry.repo_names(path=path) == ("one",)

        path.write_text(
            yaml.safe_dump(
                {
                    "repos": [
                        {"name": "one", "url": "https://example.com/one.git"},
                        {"name": "two", "url": "https://example.com/two.git"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        # Filesystems with coarse timestamps can land in the same tick; a new
        # mtime is what the cache keys on, so make the edit visible the way a
        # real edit seconds later would be.
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        assert registry.repo_names(path=path) == ("one", "two")


# ── A bad registry fails loudly ─────────────────────────────────────────


class TestInvalidRegistries:
    @pytest.mark.parametrize(
        ("entries", "message"),
        [
            pytest.param(
                [
                    {"name": "one", "url": "https://example.com/one.git"},
                    {"name": "one", "url": "https://example.com/two.git"},
                ],
                "duplicate name",
                id="duplicate-name",
            ),
            pytest.param(
                [
                    {"name": "one", "url": "https://example.com/one.git"},
                    {
                        "name": "two",
                        "url": "https://example.com/two.git",
                        "clone_dir": PurePosixPath("repos", "one").as_posix(),
                    },
                ],
                "share clone_dir",
                id="duplicate-clone-dir",
            ),
            pytest.param(
                [{"name": "one", "url": "http://example.com/one.git"}],
                "https",
                id="plain-http",
            ),
            pytest.param(
                [{"name": "one", "url": "https://example.com/one.git", "clone_dir": "/tmp/one"}],
                "inside the project",
                id="absolute-clone-dir",
            ),
            pytest.param(
                [{"name": "one", "url": "https://example.com/one.git", "clone_dir": "../one"}],
                "inside the project",
                id="escaping-clone-dir",
            ),
            pytest.param(
                [{"name": "one", "url": "https://example.com/one.git", "role": "traning"}],
                "role must be one of",
                id="role-typo",
            ),
            pytest.param(
                [
                    {
                        "name": "one",
                        "url": "https://example.com/one.git",
                        "clone_dr": PurePosixPath("repos", "one").as_posix(),
                    }
                ],
                "unknown key",
                id="unknown-key",
            ),
            pytest.param(
                [{"name": "One Repo", "url": "https://example.com/one.git"}],
                "name must match",
                id="invalid-name",
            ),
        ],
    )
    def test_a_malformed_entry_names_itself(self, tmp_path, entries, message):
        path = _write(tmp_path, entries)
        with pytest.raises(RegistryError) as excinfo:
            registry.load(path=path)
        assert message in str(excinfo.value), str(excinfo.value)

    def test_an_empty_repos_list_is_an_error(self, tmp_path):
        path = tmp_path / "repos.yaml"
        path.write_text("repos: []\n", encoding="utf-8")
        with pytest.raises(RegistryError, match="non-empty list"):
            registry.load(path=path)

    def test_a_missing_file_names_the_env_var(self, tmp_path):
        with pytest.raises(RegistryError) as excinfo:
            registry.load(path=tmp_path / "absent.yaml")
        assert registry.ENV_VAR in str(excinfo.value)

    def test_invalid_yaml_is_reported_as_such(self, tmp_path):
        path = tmp_path / "repos.yaml"
        path.write_text(textwrap.dedent("""\
            repos:
              - name: one
                url: https://example.com/one.git
                 role: train
        """), encoding="utf-8")
        with pytest.raises(RegistryError, match="not valid YAML"):
            registry.load(path=path)

    def test_unknown_role_filter_is_rejected(self):
        with pytest.raises(ValueError, match="role must be one of"):
            registry.repos(role="validation")
