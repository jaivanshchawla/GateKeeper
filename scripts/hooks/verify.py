#!/usr/bin/env python3
"""Doctor for the husky hook wiring.

Run:  npm run hooks:verify   (or: python scripts/hooks/verify.py)

Checks the invariants that are easy to break and hard to notice:

* ``core.hooksPath`` points at ``.husky/_`` so git actually runs the hooks
  (husky's bin treats *any* argument as the hooks directory, so a stray
  ``npx husky --version`` silently rewrites this to ``--version/_``).
* every hook we ship has a shim and a script behind it.
* a usable Python 3 can be resolved for the hook scripts.
* ``.git/hooks`` does not still hold a stale pre-push that is shadowed.

Exits non-zero when any requirement fails.
"""

from __future__ import annotations

import sys

from scripts.hooks import _runtime as rt

EXPECTED_HOOKS_PATH = ".husky/_"


def _check_hooks_path() -> list[str]:
    code, value = rt.git("config", "--get", "core.hooksPath")
    if code != 0 or not value:
        return [
            "core.hooksPath is not set - run `npm install` (which runs `husky`)",
            "husky is not managing this repo's hooks",
        ]
    if value != EXPECTED_HOOKS_PATH:
        return [
            f"core.hooksPath is '{value}', expected '{EXPECTED_HOOKS_PATH}'",
            "fix with: git config core.hooksPath .husky/_",
            "cause is usually `npx husky <arg>` - husky treats any argument as a directory",
        ]
    return []


def _check_hooks_present() -> list[str]:
    problems: list[str] = []
    for hook in rt.HOOKS:
        shim = rt.REPO_ROOT / ".husky" / hook
        module = rt.REPO_ROOT / (rt.module_for(hook).replace(".", "/") + ".py")
        if not shim.exists():
            problems.append(f"missing hook shim: .husky/{hook}")
        if not module.exists():
            problems.append(f"missing hook module: {rt.module_for(hook).replace('.', '/')}.py")
    return problems


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


def _check_git_dir_shadowing() -> list[str]:
    """A leftover .git/hooks/pre-push is ignored under husky - say so."""
    stale = rt.REPO_ROOT / ".git" / "hooks" / "pre-push"
    if stale.exists():
        return [
            ".git/hooks/pre-push exists but is shadowed by core.hooksPath",
            "it will never run; delete it or it will mislead the next reader",
        ]
    return []


def _print_inventory() -> None:
    """List the shipped hooks, the stage they run at and whether they block."""
    blocking = {
        "pre-commit",
        "commit-msg",
        "pre-merge-commit",
        "pre-rebase",
        "pre-push",
    }
    rt.log("")
    rt.log(f"{rt.BANNER} shipped hooks")
    rt.log("  blocks: can abort the git command. reports: advisory only.")
    rt.log("  (within pre-push, Gate 1 is advisory and ruff/pytest block)")
    for hook in rt.HOOKS:
        marker = "blocks" if hook in blocking else "reports"
        rt.log(f"  {hook:<20} {marker:<8} {rt.module_for(hook)}")
    rt.log("")


def main(argv: list[str]) -> int:
    if "--list" in argv:
        _print_inventory()
        return 0

    checks = (
        ("core.hooksPath", _check_hooks_path()),
        (f"hook shims and modules ({len(rt.HOOKS)} hooks)", _check_hooks_present()),
        ("unknown files in .husky/", _check_unknown_hooks()),
        ("shadowed .git/hooks", _check_git_dir_shadowing()),
    )

    failures = 0
    rt.log("")
    rt.log(f"{rt.BANNER} husky hook wiring")
    for name, problems in checks:
        if problems:
            failures += 1
            rt.log(f"  FAIL  {name}")
            for problem in problems:
                rt.log(f"          {problem}")
        else:
            rt.log(f"  ok    {name}")

    interpreter = rt.project_python()
    rt.log(f"  ok    python: {interpreter}")
    rt.log("")
    if failures:
        rt.log(f"{rt.BANNER} {failures} check(s) failed")
        return 1
    rt.log(f"{rt.BANNER} all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
