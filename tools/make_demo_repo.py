#!/usr/bin/env python3
"""Build the disposable repository the onboarding pipeline is demonstrated on.

The demo needs history that looks like a real project -- several authors,
tests, config churn, a hotfix that gets re-fixed four days later, a revert --
without cloning anything. This writes it deterministically: same commits, same
dates, same authors on every run, so a reported score can be compared across
runs.

    python -m tools.make_demo_repo            # -> demo/dummy-service
    python -m tools.make_demo_repo --force    # rebuild from scratch

Then::

    python -m pipelines.onboard \\
        https://github.com/example/dummy-service.git \\
        --local demo/dummy-service --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "demo" / "dummy-service"

# Three authors with different habits, so author_prior_commits and the
# author-file features have something to separate.
AUTHORS = {
    "ana": ("Ana Ruiz", "ana@demo.invalid"),
    "ben": ("Ben Osei", "ben@demo.invalid"),
    "cara": ("Cara Lind", "cara@demo.invalid"),
}

START = datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc)

# (day offset, author, kind, path, message). `kind` decides what the commit
# does; the pipeline only sees the resulting history, exactly as it would on a
# real repository.
PLAN: list[tuple[int, str, str, str, str]] = [
    (0, "ana", "add", "service/__init__.py", "Bootstrap the service package"),
    (0, "ana", "add", "service/app.py", "Add the HTTP entry point"),
    (1, "ana", "add", "requirements.txt", "Pin the runtime dependencies"),
    (2, "ben", "add", "tests/test_app.py", "First smoke test for the entry point"),
    (3, "ana", "add", "service/handlers.py", "Split request handling into handlers"),
    (3, "ana", "edit", "service/app.py", "Wire handlers into the app factory"),
    (4, "ben", "add", "service/store.py", "Add the in-memory store"),
    (5, "cara", "add", "README.md", "Document how to run the service"),
    (7, "ana", "edit", "service/store.py", "Store: expire entries after a timeout"),
    (8, "ben", "edit", "tests/test_app.py", "Cover the timeout behaviour"),
    # A hotfix cycle: the file is touched again four days later, which is the
    # positive label the whole model is trained to predict.
    (10, "cara", "edit", "service/handlers.py", "Handle a missing payload field"),
    (14, "cara", "edit", "service/handlers.py", "Re-fix the missing payload field"),
    (15, "ben", "edit", "tests/test_app.py", "Regression test for the payload fix"),
    (17, "ana", "add", ".github/workflows/ci.yml", "Run the tests on every push"),
    (18, "ana", "add", "service/config.py", "Configuration from the environment"),
    (21, "ana", "edit", "service/config.py", "Config: default the port to 8080"),
    (22, "ana", "edit", "service/config.py", "Config: honour the PORT override"),
    (24, "ben", "edit", "service/store.py", "Store: bound the dictionary size"),
    (28, "cara", "edit", "README.md", "Document the configuration variables"),
    (30, "cara", "edit", "README.md", "README: fix a broken link"),
    (33, "ben", "edit", "tests/test_app.py", "Test the store bound"),
    (36, "ana", "edit", "service/app.py", "app: add request logging"),
    (37, "ana", "edit", "service/app.py", "app: quiet the noisy log line"),
    (40, "ben", "add", "tests/test_store.py", "Dedicated store tests"),
    # A revert: this commit is undone two days later, the other positive label.
    (45, "cara", "edit", "service/handlers.py", "Rewrite handler dispatch"),
    (47, "cara", "edit", "service/handlers.py", "Revert \"Rewrite handler dispatch\""),
    (48, "ben", "edit", "tests/test_store.py", "Store tests: cover eviction"),
    (52, "ana", "add", "Makefile", "Add a Makefile for the common tasks"),
    (55, "ana", "edit", "requirements.txt", "Bump the pinned dependency"),
    (58, "cara", "edit", "service/handlers.py", "Handlers: validate the request body"),
    (63, "cara", "edit", "service/handlers.py", "Handlers: reject an empty body"),
    (66, "ben", "edit", "tests/test_app.py", "Test empty request bodies"),
    (70, "ben", "edit", "service/store.py", "Store: drop the unused helper"),
    (72, "ana", "add", "docs/architecture.md", "Describe the request flow"),
    (75, "ana", "edit", "docs/architecture.md", "Docs: correct the module names"),
    (80, "cara", "add", "service/metrics.py", "Count requests and errors"),
    (84, "cara", "edit", "service/metrics.py", "Metrics: label counters by route"),
    (86, "ben", "edit", "tests/test_app.py", "Assert the counters move"),
    (90, "ana", "edit", "Makefile", "Makefile: add a lint target"),
    (95, "ben", "edit", "service/store.py", "Store: cheaper expiry sweep"),
    (99, "ana", "edit", ".github/workflows/ci.yml", "CI: cache the dependency install"),
    (104, "cara", "edit", "service/metrics.py", "Metrics: expose a text endpoint"),
    (109, "ben", "edit", "tests/test_store.py", "Store tests: split fixtures"),
    (114, "ana", "edit", "service/handlers.py", "Handlers: return 400 on bad JSON"),
    (120, "ana", "edit", "service/handlers.py", "Handlers: keep the 400 status"),
    (126, "cara", "edit", "README.md", "README: add the metrics section"),
    (133, "ben", "edit", "tests/test_app.py", "Test the 400 path"),
    (140, "ana", "edit", "service/app.py", "app: close over the config once"),
    (147, "cara", "edit", "docs/architecture.md", "Docs: describe the metrics endpoint"),
    (154, "ben", "add", "tests/test_metrics.py", "Cover the metrics endpoint"),
    (161, "ana", "edit", "requirements.txt", "requirements: relax the pin"),
    (168, "cara", "edit", "service/metrics.py", "Metrics: reset between tests"),
    (175, "ben", "edit", "service/store.py", "Store: preserve insertion order"),
    (182, "ana", "edit", "service/config.py", "Config: read a config file too"),
    (189, "ben", "edit", "tests/test_store.py", "Store tests: ordering"),
    (196, "cara", "edit", "README.md", "README: document the config file"),
    (203, "ana", "edit", "Makefile", "Makefile: run the tests quietly"),
    (210, "ben", "edit", "tests/test_app.py", "Cover the config file"),
    (217, "cara", "edit", "service/handlers.py", "Handlers: quote error messages"),
    (224, "ana", "edit", "service/app.py", "app: log the request id"),
    (231, "ben", "edit", "tests/test_app.py", "Assert the request id is logged"),
    (238, "cara", "edit", "docs/architecture.md", "Docs: note the request id"),
    (245, "ana", "edit", "service/store.py", "Store: cap the value size"),
    (252, "ben", "edit", "tests/test_store.py", "Test the value cap"),
    (259, "ana", "edit", "requirements.txt", "requirements: pin dev tools"),
    (266, "cara", "edit", "service/metrics.py", "Metrics: expose latency"),
    (273, "ben", "edit", "tests/test_metrics.py", "Test the latency bucket"),
    (280, "ana", "edit", "service/app.py", "app: 404 for unknown routes"),
    # A recent hotfix cycle, so evaluating the newest commits -- what a review
    # of a live branch actually sees -- has outcomes to measure.
    (284, "ana", "edit", "service/handlers.py", "Handlers: fix the 400 status"),
    (287, "ana", "edit", "service/handlers.py", "Handlers: re-fix the 400 status"),
    (288, "ben", "edit", "tests/test_app.py", "Test the 400 status again"),
    (290, "cara", "edit", "service/store.py", "Store: fix the eviction race"),
    (292, "cara", "edit", "service/store.py", "Store: re-fix the eviction race"),
    (295, "ana", "edit", "service/metrics.py", "Rewrite the latency bucket"),
    (296, "ana", "edit", "service/metrics.py", 'Revert "Rewrite the latency bucket"'),
    (299, "ben", "edit", "tests/test_metrics.py", "Cover the latency revert"),
]

BODY = {
    "service/__init__.py": '"""A tiny demo service, used as onboarding fixture."""\n\n__version__ = "0.1.0"\n',
    "service/app.py": (
        '"""HTTP entry point."""\n\n'
        "from service.handlers import handle\n\n\n"
        "def create_app():\n"
        "    return handle\n"
    ),
    "service/handlers.py": (
        '"""Request handling."""\n\n\n'
        "def handle(request):\n"
        '    return {"ok": True}\n'
    ),
    "service/store.py": '"""An in-memory store."""\n\nSTORE = {}\n',
    "service/config.py": '"""Configuration."""\n\nPORT = 8080\n',
    "service/metrics.py": '"""Counters."""\n\nREQUESTS = 0\nERRORS = 0\n',
    "requirements.txt": "flask==3.0.0\n",
    "tests/test_app.py": 'def test_import():\n    from service import app\n\n    assert app\n',
    "tests/test_store.py": 'def test_store_import():\n    from service import store\n\n    assert store.STORE == {}\n',
    "tests/test_metrics.py": 'def test_metrics_import():\n    from service import metrics\n\n    assert metrics.REQUESTS == 0\n',
    "README.md": "# dummy-service\n\nA tiny service used to demonstrate Gatekeeper onboarding.\n",
    "docs/architecture.md": "# Architecture\n\nOne process, one module per concern.\n",
    ".github/workflows/ci.yml": "name: ci\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - run: python -m pytest\n",
    "Makefile": "test:\n\tpython -m pytest\n",
}


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _force_remove(path: Path) -> None:
    """Delete a checkout, including git's read-only object files.

    Windows marks objects read-only, and ``shutil.rmtree`` gives up on the
    first one; clearing the bit and retrying is the only portable fix.
    """

    def retry(func, target, _exc):
        os.chmod(target, 0o700)
        func(target)

    shutil.rmtree(path, onerror=retry)


def build(destination: Path, force: bool = False) -> Path:
    """Write the history to ``destination`` and return it."""
    if destination.exists():
        if not force:
            print(f"{destination} already exists; pass --force to rebuild it")
            return destination
        _force_remove(destination)
    destination.mkdir(parents=True)

    _run(["git", "init", "--quiet", "-b", "main"], destination, os.environ.copy())

    base_env = os.environ.copy()
    base_env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@demo.invalid",
        }
    )

    for index, (day, who, kind, path, message) in enumerate(PLAN):
        name, email = AUTHORS[who]
        when = (START + timedelta(days=day, hours=index % 6)).isoformat()
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)

        if not target.exists():
            target.write_text(BODY.get(path, f"# {path}\n"), encoding="utf-8")
        # Every commit must change the tree; the marker is what makes two
        # edits of the same file two commits instead of one and a no-op.
        existing = target.read_text(encoding="utf-8")
        marker = message if kind == "edit" else ""
        target.write_text(
            existing + (f"\n# {marker}\n" if marker else f"\n# v{index}\n"),
            encoding="utf-8",
        )

        env = base_env | {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
            "GIT_AUTHOR_DATE": when,
            "GIT_COMMITTER_DATE": when,
        }
        _run(["git", "add", "--all"], destination, env)
        _run(["git", "commit", "--quiet", "-m", message], destination, env)

    print(f"{destination}: {len(PLAN)} commits")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="delete and rebuild an existing fixture")
    args = parser.parse_args(argv)
    build(args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
