"""Shared test configuration.

The hook layer reads its configuration from the environment, which makes every
test *of* it sensitive to whatever the shell that started the suite happens to
export. Both halves of that have already bitten this repository:

* two shell tests assumed no gate was running, and failed inside the pre-push
  gate that was running them;
* the tracing test failed for anyone with `export GATEKEEPER_TRACE=1` in their
  profile -- and because pre-push runs this suite, a debugging habit would
  have blocked their own push.

Clearing the hook's own variables for the session makes the suite answer about
the code rather than about the machine. Tests that need a variable set it
themselves with `monkeypatch`, which is undone after each test, so a test can
still be explicit about the environment it means to exercise.

The same reasoning applies to a test's *inputs*, which is what the markers
below are for. `data/commit_features.csv` and `repos/*` are gitignored, so the
fresh clone CI checks out has neither: a test that reads one of them errors out
instead of testing anything, which is exactly how the `suite` job went red on
every run while passing on any machine that happened to have the data. The
markers move that decision into one place, so a test declares what it needs and
the skip reason stays truthful.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# tests/ is a package, so the project root is already on sys.path under
# pytest's default import mode; add it anyway so this module also works when
# collected directly.
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.hooks import _runtime as rt

# Inputs that are gitignored, i.e. absent from any fresh clone.
DATA_CSV = Path(REPO_ROOT) / "data" / "commit_features.csv"
REPOS_DIR = Path(REPO_ROOT) / "repos"

# Registered here rather than in a pytest.ini so the project keeps its single
# test configuration and the markers cannot be added somewhere pytest will not
# read them.
MARKERS = (
    "needs_data: needs data/commit_features.csv, which is not in a fresh clone",
    (
        "needs_repos: needs a cloned upstream repository under repos/, which is "
        "not in a fresh clone; pass repository names as arguments to name them"
    ),
    "requires_yaml: needs PyYAML, which the hook layer itself treats as optional",
)


def pytest_configure(config):
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)


def _missing_repos(names: tuple) -> list:
    """Which of the requested clones are absent, or ['repos/*'] when unnamed."""
    if names:
        return [name for name in names if not (REPOS_DIR / name / ".git").exists()]
    return [] if any(REPOS_DIR.glob("*/.git")) else ["repos/*"]


def pytest_collection_modifyitems(config, items):
    """Degrade an absent input to a skip with a reason, never to an error."""
    data_missing = not DATA_CSV.is_file()
    for item in items:
        if data_missing and item.get_closest_marker("needs_data"):
            item.add_marker(
                pytest.mark.skip(
                    reason=f"{DATA_CSV.name} is not present (gitignored: a fresh "
                    "clone has no training data)"
                )
            )
        marker = item.get_closest_marker("needs_repos")
        if marker:
            missing = _missing_repos(tuple(str(arg) for arg in marker.args))
            if missing:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"missing cloned repository: {', '.join(missing)} "
                        "(repos/ is gitignored)"
                    )
                )


def _hook_env_vars() -> tuple[str, ...]:
    """Every switch the hook layer reads, from its single owner where it has one."""
    return (
        "HUSKY",
        rt.SKIP_ENV,
        rt.TRACE_ENV,
        rt.STATE_DIR_ENV,
        "GATEKEEPER_PYTHON",
        "GATEKEEPER_HOOK_TESTS",
        "GATEKEEPER_ALLOW_PROTECTED",
        "GATE1_BLOCK",
    )


@pytest.fixture(autouse=True, scope="session")
def isolated_hook_environment():
    """Hide the ambient hook configuration for the whole session."""
    names = _hook_env_vars()
    saved = {name: os.environ.pop(name, None) for name in names}
    yield
    for name, value in saved.items():
        if value is not None:
            os.environ[name] = value
