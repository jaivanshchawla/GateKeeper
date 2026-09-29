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
