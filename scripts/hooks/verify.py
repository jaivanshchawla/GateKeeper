#!/usr/bin/env python3
"""Doctor for the husky hook wiring.

Run:  npm run hooks:verify     (or: python -m scripts.hooks.verify)

Checks the invariants that are easy to break and hard to notice:

* ``core.hooksPath`` points at ``.husky/_`` so git actually runs the hooks
  (husky's bin treats *any* argument as the hooks directory, so a stray
  ``npx husky --version`` silently rewrites this to ``--version/_``).
* every hook husky dispatches has a shim and a module behind it -- and the
  shim is still the two-line delegator rather than a hand-written script
  that never reaches Python.
* the generated ``.husky/_/`` really contains a dispatcher per hook.
* ``.git/hooks`` does not still hold hooks that ``core.hooksPath`` shadows.
* the toolchain can support the wiring at all: git new enough for
  ``core.hooksPath``, Node and husky present at the declared version.

Two severities, because they fail for different reasons:

* **failures** mean the hooks cannot work -- they decide the exit code.
* **warnings** mean something is missing from *this* machine but the
  repository is still correct (Node not installed, hooks not installed
  yet). A contributor running the test suite on a Python-only checkout
  should not be told the repository is broken. ``--strict`` promotes them,
  which is what CI uses after installing everything.

Also: ``--list`` prints the hook inventory, ``--events`` the event log.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.hooks import _events
from scripts.hooks import _runtime as rt

EXPECTED_HOOKS_PATH = ".husky/_"

# core.hooksPath support landed in git 2.9.
MINIMUM_GIT = (2, 9)

# A hook shim is a delegator, not an implementation. Anything longer is a
# sign someone wrote the logic in shell, where the tests cannot see it.
MAX_SHIM_LINES = 3
SHIM_MARKERS = ("lib/bootstrap.sh", "gatekeeper_run_hook")

# A doctor "check" is (name, failures, warnings).
Check = tuple[str, list[str], list[str]]


# -- the wiring itself -------------------------------------------------


def _check_hooks_path() -> Check:
    code, value = rt.git("config", "--get", "core.hooksPath")
    if code != 0 or not value:
        return (
            "core.hooksPath",
            [
                "core.hooksPath is not set - run `npm install` (which runs `husky`)",
                "husky is not managing this repo's hooks",
            ],
            [],
        )
    if value != EXPECTED_HOOKS_PATH:
        problems = [
            f"core.hooksPath is '{value}', expected '{EXPECTED_HOOKS_PATH}'",
            f"fix with: git config core.hooksPath {EXPECTED_HOOKS_PATH}",
        ]
        if value.startswith("-"):
            # The exact shape `npx husky --version` leaves behind.
            problems.append(
                f"'{value}' looks like a command-line flag: this is what "
                "`npx husky <argument>` writes, and it disables every hook"
            )
            problems.append("re-run: npm run hooks:install")
        else:
            problems.append(
                "cause is usually `npx husky <arg>` - husky treats any argument "
                "as a directory"
            )
        return ("core.hooksPath", problems, [])
    return ("core.hooksPath", [], [])


def _check_hooks_present() -> Check:
    """Every hook we ship has a shim and a module, and husky dispatches it."""
    failures: list[str] = []
    for hook in rt.HOOKS:
        if not (rt.REPO_ROOT / ".husky" / hook).is_file():
            failures.append(f"missing hook shim: .husky/{hook}")
        module = rt.REPO_ROOT / (rt.module_for(hook).replace(".", "/") + ".py")
        if not module.is_file():
            failures.append(f"missing hook module: {rt.module_for(hook).replace('.', '/')}.py")
        if hook not in rt.HUSKY_HOOKS:
            failures.append(
                f"{hook} is not a hook husky dispatches, so its shim can never run"
            )
    missing = [hook for hook in rt.HUSKY_HOOKS if hook not in rt.HOOKS]
    if missing:
        failures.append(
            "husky dispatches these with no implementation behind them: "
            + ", ".join(missing)
        )
    return (f"hook shims and modules ({len(rt.HOOKS)} hooks)", failures, [])


def _check_shim_shape() -> Check:
    """A shim stays a delegator, and delegates to its own hook."""
    failures: list[str] = []
    for hook in rt.HOOKS:
        shim = rt.REPO_ROOT / ".husky" / hook
        if not shim.is_file():
            continue  # already reported by _check_hooks_present
        text = shim.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        for marker in SHIM_MARKERS:
            if marker not in text:
                failures.append(f".husky/{hook} does not reference {marker}")
        if f"gatekeeper_run_hook {hook}" not in text:
            failures.append(f".husky/{hook} does not call gatekeeper_run_hook {hook}")
        if len(lines) > MAX_SHIM_LINES:
            failures.append(
                f".husky/{hook} is {len(lines)} lines of shell; logic belongs in "
                f"scripts/hooks/{hook.replace('-', '_')}.py where it is tested"
            )
    return ("shim shape", failures, [])


def _check_unknown_hooks() -> list[str]:
    """Catch files in .husky/ that look like hooks but will never run.

    Marks that look right and are inert are the first thing husky's own
    troubleshooting guide lists, so three shapes are flagged:

    * `precommit` / `pre_commit` -- a hook name without its dashes.
    * `pre-commit.sh` -- a hook name with an extension, when the real
      extensionless hook is not there to run.

    `.husky/install.mjs`, `.husky/common.sh` and helpers such as
    `.husky/pre-commit.js` next to a real `.husky/pre-commit` are legitimate
    -- husky documents exactly those -- so anything else is left alone
    rather than guessed at.
    """
    husky_dir = rt.REPO_ROOT / ".husky"
    if not husky_dir.is_dir():
        return []

    present = {entry.name for entry in husky_dir.iterdir() if entry.is_file()}
    squashed = {hook.replace("-", ""): hook for hook in rt.KNOWN_GIT_HOOKS}
    problems: list[str] = []

    for name in sorted(present):
        if name in rt.KNOWN_GIT_HOOKS:
            continue

        # precommit / pre_commit instead of pre-commit
        if "-" not in name and name.replace("_", "").lower() in squashed:
            correct = squashed[name.replace("_", "").lower()]
            problems.append(f".husky/{name} will never run -- rename it to '{correct}'")
            continue

        # pre-commit.sh / pre-commit.txt when there is no real pre-commit
        stem = name.rsplit(".", 1)[0]
        if stem != name and stem in rt.KNOWN_GIT_HOOKS and stem not in present:
            problems.append(
                f".husky/{name} will never run -- git looks for '{stem}'"
            )

    return problems


def _check_generated_dir() -> Check:
    """`.husky/_/` holds one dispatcher per hook, and stays out of git."""
    husky_dir = rt.REPO_ROOT / ".husky" / "_"
    if not husky_dir.is_dir():
        # Not an error: a fresh clone has no generated directory until
        # `npm install` runs, and no Python-only contributor needs one.
        return (
            "generated .husky/_/",
            [],
            ["not generated yet - run `npm install` (or `npm run hooks:install`)"],
        )

    failures: list[str] = []
    warnings: list[str] = []

    if not (husky_dir / "h").is_file():
        failures.append("missing dispatcher: .husky/_/h")
    for hook in rt.HUSKY_HOOKS:
        if not (husky_dir / hook).is_file():
            failures.append(f"missing generated dispatcher: .husky/_/{hook}")

    ignore = husky_dir / ".gitignore"
    if not ignore.is_file() or ignore.read_text(encoding="utf-8").strip() != "*":
        warnings.append(
            ".husky/_/.gitignore should contain '*', or the generated shims "
            "show up as untracked files"
        )
    return ("generated .husky/_/", failures, warnings)


def _check_git_dir_shadowing() -> Check:
    """Leftover .git/hooks/* are ignored under husky - say so."""
    hooks_dir = rt.REPO_ROOT / ".git" / "hooks"
    if not hooks_dir.is_dir():
        return ("shadowed .git/hooks", [], [])

    shadowed = sorted(
        entry.name
        for entry in hooks_dir.iterdir()
        if entry.is_file()
        and entry.name in rt.KNOWN_GIT_HOOKS
        and not entry.name.endswith(".sample")
    )
    if not shadowed:
        return ("shadowed .git/hooks", [], [])

    failures = [
        f".git/hooks/{name} is shadowed by core.hooksPath and will never run"
        for name in shadowed
    ]
    failures.append("delete them, or they will mislead the next reader")
    return ("shadowed .git/hooks", failures, [])


def _check_toolchain() -> Check:
    """git new enough for core.hooksPath, Node and husky as declared."""
    failures: list[str] = []
    warnings: list[str] = []

    git_version = _git_version()
    if git_version is None:
        failures.append("could not read `git --version`")
    elif git_version < MINIMUM_GIT:
        failures.append(
            f"git {git_version[0]}.{git_version[1]} is too old for core.hooksPath "
            f"(needs {MINIMUM_GIT[0]}.{MINIMUM_GIT[1]}+)"
        )

    required_node = _declared_node_engine()
    node = shutil.which("node")
    if node is None:
        warnings.append(
            "node not on PATH: husky cannot be installed or upgraded here "
            "(hooks already generated keep working)"
        )
    else:
        node_version = _node_version(node)
        if node_version is None:
            warnings.append("could not read `node --version`")
        elif required_node is not None and node_version < required_node:
            warnings.append(
                f"node {node_version[0]}.{node_version[1]} is older than the "
                f"declared engines.node ({_node_range_text()})"
            )

    husky = rt.REPO_ROOT / "node_modules" / "husky" / "package.json"
    if not husky.is_file():
        warnings.append(
            "husky is not installed in node_modules - run `npm install`"
        )
    return ("toolchain (git, node, husky)", failures, warnings)


def _environment_notes() -> list[str]:
    """Informational lines about machine-local configuration."""
    notes: list[str] = []

    if (Path.home() / ".huskyrc").is_file():
        notes.append(
            "~/.huskyrc exists but is DEPRECATED; move it to "
            "~/.config/husky/init.sh (husky warns about this on every hook)"
        )

    configured = _init_sh_path()
    if configured.is_file():
        notes.append(f"startup file in use: {configured}")
    return notes


# -- small readers -----------------------------------------------------


def _git_version() -> tuple[int, int] | None:
    """(major, minor) of the installed git, or None."""
    result = subprocess.run(
        ["git", "--version"], capture_output=True, text=True, check=False
    )
    text = result.stdout.strip()
    if "version" not in text:
        return None
    parts = text.split("version", 1)[1].strip().split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return None


def _node_version(node: str) -> tuple[int, int] | None:
    result = subprocess.run([node, "--version"], capture_output=True, text=True, check=False)
    parts = result.stdout.strip().lstrip("v").split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return None


def _package_json() -> dict:
    try:
        with open(rt.REPO_ROOT / "package.json", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _node_range_text() -> str:
    return str(_package_json().get("engines", {}).get("node", "unspecified"))


def _declared_node_engine() -> tuple[int, int] | None:
    """Lowest Node version allowed by ``engines.node``, as (major, minor)."""
    parts = _node_range_text().lstrip("^~>=< ").split(".")
    try:
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (IndexError, ValueError):
        return None


def _init_sh_path() -> Path:
    """Where husky reads its startup file, honouring XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "husky" / "init.sh"


# -- reporting ---------------------------------------------------------


def _print_inventory() -> None:
    """List the shipped hooks, the stage they run at and whether they block."""
    rt.log("")
    rt.log(f"{rt.BANNER} shipped hooks ({len(rt.HOOKS)}/{len(rt.HUSKY_HOOKS)} husky hooks)")
    rt.log("  blocks: can abort the git command. reports: advisory only.")
    rt.log("  (within pre-push, Gate 1 is advisory and ruff/pytest block)")
    for hook in rt.HOOKS:
        marker = "blocks" if hook in rt.BLOCKING_HOOKS else "reports"
        rt.log(f"  {hook:<20} {marker:<8} {rt.module_for(hook)}")
    rt.log("")


def _print_events(limit: int) -> None:
    events = _events.read_events(limit)
    rt.log("")
    if not events:
        rt.log(f"{rt.BANNER} no hook events recorded yet")
        rt.log("")
        return
    rt.log(f"{rt.BANNER} last {len(events)} hook event(s)")
    for event in events:
        sha = (event.get("sha") or "")[:8]
        detail = event.get("subject") or event.get("command") or ""
        if event.get("event") == "rewrite" and event.get("rewrites"):
            detail = f"{len(event['rewrites'])} sha pair(s)"
        rt.log(
            f"  {event.get('recorded_at', '?'):<32} {event.get('event', '?'):<11} "
            f"{sha:<8} {detail}"
        )
    rt.log("")


def main(argv: list[str]) -> int:
    if "--list" in argv:
        _print_inventory()
        return 0
    if "--events" in argv:
        _print_events(_events.DEFAULT_TAIL)
        return 0

    checks: list[Check] = [
        _check_hooks_path(),
        _check_hooks_present(),
        _check_shim_shape(),
        ("unknown files in .husky/", _check_unknown_hooks(), []),
        _check_generated_dir(),
        _check_git_dir_shadowing(),
        _check_toolchain(),
    ]

    strict = "--strict" in argv
    failures = 0
    warnings = 0

    rt.log("")
    rt.log(f"{rt.BANNER} husky hook wiring")
    for name, problems, cautions in checks:
        if problems:
            failures += 1
        warnings += len(cautions)
        if problems:
            rt.log(f"  FAIL  {name}")
            for problem in problems:
                rt.log(f"          {problem}")
        elif cautions and strict:
            rt.log(f"  FAIL  {name} (warning promoted by --strict)")
            for caution in cautions:
                rt.log(f"          {caution}")
        elif cautions:
            rt.log(f"  warn  {name}")
            for caution in cautions:
                rt.log(f"          {caution}")
        else:
            rt.log(f"  ok    {name}")

    for note in _environment_notes():
        rt.log(f"  note  {note}")

    rt.log(f"  ok    python: {rt.project_python()}")
    rt.log("")

    if failures or (strict and warnings):
        rt.log(f"{rt.BANNER} {failures} check(s) failed, {warnings} warning(s)")
        return 1
    rt.log(f"{rt.BANNER} all checks passed ({warnings} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
