# Husky integration log

Living record of the `husky` branch: what was reviewed, what was designed,
what was built, what was verified, and what is still open.

**Branch:** `husky` — a copy of `main` at `c3a8264`. `main` has not been touched.
**Scope:** integrate [typicode/husky](https://github.com/typicode/husky) as the
repo's git-hook manager, and close the gaps that integration exposed.
**Reference docs:** [`docs/HUSKY.md`](docs/HUSKY.md) (how it works) and
[`docs/HUSKY-UPSTREAM.md`](docs/HUSKY-UPSTREAM.md) (husky mapped file by file).

Two phases. **Phase 1** replaced the two competing hook installers and wired
three hooks. **Phase 2** went through husky upstream feature by feature, adopted
the documented behaviours that were missing, put the layer under CI on three
operating systems, and finished the hook set: **all 14 hooks husky dispatches**
now carry policy rather than three of them being deliberate no-ops.

Phase 2 took three passes. The third was not planned: the first attempt to push
the branch was **blocked by the repository's own pre-push gate**, which found a
real defect in the tests (§4, B17). That finding is the best evidence available
that the gate works — it caught its own author, on the change that introduced
it, and it found something no amount of reading the diff would have shown.

---

## 1. Review of the project as it stood

The repo is an MLOps quality gate: a Python core (26.5k lines of tracked
Python) with a FastAPI service, a Flask webhook, a React dashboard and a Node
CLI, plus 19 test files. Checks run at review time: `pytest tests/test_rules.py
tests/test_determinism.py` → **38 passed**, `node cli/src/index.js` → prints
usage.

Findings that mattered for this work:

| # | Finding | Evidence |
|---|---------|----------|
| R1 | **Two hook installers fighting over one file.** `.pre-commit-config.yaml` and `gatekeeper init` both write `.git/hooks/pre-push`. Git runs only one. | `.pre-commit-config.yaml`, `cli/src/index.js` `cmdInit()` |
| R2 | **The installed hook was untracked and machine-specific.** `.git/hooks/pre-push` baked in `C:/Users/Jaivansh Chawla/.../cli/src/index.js`. A fresh clone got no hook at all. | `cat .git/hooks/pre-push` |
| R3 | **Band thresholds exist in five places with five values.** `ml/config.yaml` `_global` 0.8677/0.6817; `ml/scoring.py` fallback 0.8936/0.8230; `api/main.py` fallback 0.8619/0.7536; `scripts/pre_push_score.py` 0.8619/0.7536; `cli/src/index.js` 0.90/0.75. | grep for `0.86`/`0.75`/`0.89` across those files |
| R4 | **`GET /repos/{id}` parses a YAML file with `json.loads`.** `api/main.py` does `_json.loads(Path("ml/config.yaml").read_text())`; it always raises and the bare `except` hides it, so per-repo thresholds silently fall back to `DEFAULT_THRESHOLDS`. | `api/main.py` `get_repo()` |
| R5 | **The CLI reimplements rule policy.** `cmdCheck()` scores commits with four JS rule checks, while `rules/engine.py` implements 18. Two sources of truth for the same decision — the exact failure `ml/scoring.py`'s docstring says cost the project "B2.4, O.1, and Q-T". | `cli/src/index.js` vs `rules/engine.py` |
| R6 | **`cli/package.json` declares `node --test src/**/*.test.js` but no test file exists.** The script is dead. | `ls cli/src/*.test.js` |
| R7 | **The repo does not pass its own linter.** `ruff check .` → **732 errors**. Any whole-tree lint in a hook would block every commit. | `ruff check .` |
| R8 | Dirty tree at branch time: `models/gatekeeper_risk_model.skops` and `smoke_tests/drift_report.html` were already modified. Left untouched throughout. | `git status` |

R1–R3 and R5 are out of scope for a hook integration and are **reported, not
fixed**, except where a hook must depend on them (see §5).

---

## 2. Design

Husky becomes the single hook *installer*; the Python `scripts/hooks/` layer
owns all hook *policy*. Three rules:

1. `.husky/<hook>` is a two-line shim — source a bootstrap, delegate. No logic.
2. Bootstrap (`bootstrap.sh`) decides only which interpreter can *start* a hook.
   `_runtime.py` decides which interpreter runs the project's tooling.
   Separate problems, separate files, no duplicated resolution policy.
3. Every hook reuses the project's own tools (`ruff`, `pytest`,
   `scripts/pre_push_score.py`) instead of reimplementing anything.

Data flow: `git event → .husky/_/<hook> (husky shim) → .husky/<hook> (2 lines)
→ scripts/hooks/<hook>.py → existing project tooling → exit code`.

Coexistence with `.pre-commit-config.yaml`: it stays as the declarative record
of hook policy and for environments where the `pre-commit` tool is the
installer. It is not installed here because installing it would recreate R1.
`gatekeeper init` now detects husky and stands down.

---

## 2b. Phase 2 — what was taken from upstream

Every husky source file, behaviour and documented feature was mapped with an
adopt/skip decision; the full table is in
[`docs/HUSKY-UPSTREAM.md`](docs/HUSKY-UPSTREAM.md). Summarised:

**Adopted from upstream**

| Upstream | What we did with it |
|----------|---------------------|
| `HUSKY=0` / `HUSKY=2` | Skip and trace-tracing, honoured in the sh layer too |
| `~/.config/husky/init.sh` startup file | Relied on, documented, checked by the doctor |
| `.husky/install.mjs` prepare pattern | Adopted and extended so `HUSKY=1` can force an install in CI |
| `husky init` | `npm run hooks:init` — install *and* verify |
| `husky uninstall` (removed in v9) | Reimplemented: unset `core.hooksPath` **and** remove `.husky/_` |
| Windows `winpty` tty workaround | Adopted, made opt-in (upstream's version discards stdin — see below) |
| The 14-hook dispatch set | **All 14 implemented.** The three that looked like no-ops were holes, not redundancies — see §4 B16 |
| `HUSKY=2` debug tracing | Adopted, and **continued past the shell boundary** into `rt.trace`, where the decisions are actually made |
| Non-shell hooks, POSIX-only shell, `--no-verify` | Documented, and the doctor permits the helper-file pattern |
| Upstream's own test structure | Mirrored: a throwaway real git repo per test |

**Where we corrected upstream rather than copying it**

1. **The tty workaround.** Upstream's `common.sh` runs `exec < /dev/tty`
   unconditionally, which *replaces stdin*. `pre-push` reads git's ref lines
   from stdin and `post-rewrite` reads sha pairs, so copying it verbatim would
   silently discard both payloads. Ours is opt-in, with a test asserting the
   two stdin-reading hooks never call it.
2. **Exit 127.** Upstream reports "command not found" after the hook has
already failed. We probe every candidate interpreter first and diagnose it at
   the source, which is what the Windows Store `python3` alias required.
3. **The hook-name footgun.** Upstream's `bin.js` treats any argument as the
   hooks directory, so `npx husky --version` breaks the install silently. We
   cannot fix upstream from here, so the doctor detects it and names the cause.

---

## 3. Commit log

Phase 1 (1-19) and Phase 2 (20-53). Each commit is one thing.

```
 1 0cbf199 chore(husky): add root package manifest for git-hook tooling
 2 95ee57e chore(husky): lock husky 9.1.7 at the repo root
 3 0da0f08 feat(hooks): add shared runtime for husky hook scripts
 4 61a0c1c feat(hooks): add commit-msg conventional commit linter
 5 c102fbb feat(hooks): add staged-file linter for pre-commit
 6 c23af85 feat(hooks): scope linting to the outgoing diff in the hook runtime
 7 91ea08d feat(hooks): add pre-push gate running Gate 1, diff-scoped ruff and pytest
 8 848f3dc feat(husky): add sh bootstrap that probes for a working Python 3
 9 bf5b3ab feat(husky): wire the pre-commit shim
10 e07df44 feat(husky): wire the commit-msg shim
11 5758694 feat(husky): wire the pre-push shim
12 e67748c fix(cli): make `gatekeeper init` husky-aware and its hook portable
13 07ebac0 feat(hooks): add a wiring doctor exposed as `npm run hooks:verify`
14 08d8983 test(hooks): cover commit rules, ref parsing, skip semantics and wiring
15 7a08490 docs(husky): document the hook architecture and log the integration
16 85932e4 chore(lint): apply the existing per-file ignores to pre_push_score.py
17 9002dfc fix(gate1): make the pre-push scorer work when run as a script
18 070cce8 fix(hooks): stop non-ASCII output from crashing hook scripts
19 791e3ab docs(husky): log the Gate 1 silent no-op, the lint exemption gap and the ASCII fix
   -- phase 2 --
20 d10015c refactor(hooks): invoke hook modules with -m and drop the sys.path shim
21 613a6df feat(hooks): add a branch and configuration policy module
22 84ae1b9 feat(hooks): add an advisory runner for non-blocking hooks
23 f249f5e chore(lint): exempt hook scripts from the broad-except rule
24 132cce5 chore(config): declare protected branches
25 1f20c25 feat(hooks): refuse to merge or rebase protected branches
26 8fc4556 feat(hooks): warn on direct_to_main and seed the commit template
27 dc56ed9 feat(hooks): record commits locally for outcome tracking
28 0ab8e65 feat(hooks): flag merges that changed the gate itself
29 75cae8b feat(hooks): report branch gate context on checkout
30 16806ed feat(hooks): report rewritten history
31 9e2843a feat(husky): add shared shell helpers with an opt-in tty guard
32 19339bd refactor(hooks): give the hook inventory a single owner
33 66cc952 test(hooks): exercise every hook against real git repositories
34 5cb5660 feat(husky): CI-safe install and the uninstall husky v9 removed
35 3af991d fix(hooks): include hotfix/* in the default protected branches
36 19a6ce8 feat(hooks): list the hook inventory and sharpen the name check
37 cb0ed32 test(hooks): cover the doctor's hook-name checks and inventory
38 c5ebf03 fix(husky): stop a wrong $0 from silently killing every hook
39 230d377 ci: validate the hook layer on three operating systems
40 6913355 feat(hooks): apply the message rules to git am patches
   -- phase 2, second pass: husky upstream re-read file by file --
41 43968ba docs(husky): map husky upstream file by file, feature by feature
42 2216963 docs: document the eleven hooks, the new commands and the CI
43 4b9c93f ci: install the runtime-only deps the test suite actually needs
44 945d8d3 feat(hooks): implement every hook husky dispatches
45 67a16de feat(hooks): check the toolchain and the generated shims, not just the wiring
46 fd32d1d test(hooks): cover the new hooks, the doctor and husky's shell layer
47 7ebc809 ci(hooks): cover the new hooks, the strict doctor and the install lifecycle
48 c5e5b46 docs(husky): document all fourteen hooks and what has no upstream peer
   -- phase 2, third pass: the push that exposed B17 --
49 3706beb fix(hooks): make the hook state directory relocatable
50 3831772 ci(hooks): re-run the hook tests under a live gate marker
51 b788992 docs(husky): record the hermeticity bug the gate caught
   -- phase 2, fourth pass: hardening what the third pass revealed --
52 a41b4df test(hooks): make the suite independent of the ambient hook environment
53         docs(husky): record B18, the ambient-environment dependency
           (53 is the tip of this branch; its own hash is not recorded because
           it contains this line)
```

Nothing is merged into `main`.

---

## 4. Bugs found and fixed while integrating

These are the cases where building the integration surfaced a defect. Each was
reproduced before it was fixed.

**B1 — `npx husky --version` silently disabled every hook.**
Husky's `bin.js` reads `process.argv[2]` and passes it to the installer as the
hooks directory. Any unrecognised argument is therefore treated as a path:
`npx husky --version` set `core.hooksPath` to `--version/_` and created a stray
`--version/` directory. Every hook stopped running and git reported nothing.
*Fixed:* the wiring doctor now asserts `core.hooksPath` and names the cause;
documented in `docs/HUSKY.md`.

**B2 — `command -v python3` matched the Windows Store stub.**
The first version of the shims resolved the interpreter with
`command -v python3 || command -v python`. On Windows that matches the "App
execution alias" stub, which exists, exits non-zero, and prints an install
advert — so every commit failed with `code 49`. *Fixed:* `bootstrap.sh` probes
each candidate with a real `python -c` before trusting it.

**B3 — hook name to script name mismatch.**
Hooks are `pre-commit`; the scripts are `pre_commit.py`. The first delegation
attempt looked for `pre-commit.py` and died with `[Errno 2]`. *Fixed:* the
bootstrap translates `-` to `_`.

**B4 — a whole-tree `ruff check .` in pre-push would block every push.**
Found only by running `ruff check .` and getting 732 errors (R7). *Fixed:*
`_runtime.changed_files()` resolves git's pre-push refs into a diff spec
(falling back to `origin/main` for a new branch) and the hook lints only the
outgoing diff, matching pre-commit's changed-files semantics.

**B5 — a stale `.git/hooks/pre-push` was left shadowed.**
The doctor's first run flagged it: with `core.hooksPath` set, git ignores
`.git/hooks/`, so the old hand-written hook was dead code that would mislead
the next reader. *Fixed:* removed, and the doctor now checks for it.

**B6 — the old `gatekeeper init` wrote a hook that could never run** (R1/R2).
*Fixed:* `init` detects a hook manager and stands down with a clear message; the
fallback hook resolves the repo root at runtime instead of hardcoding a path.

**B7 — Gate 1 scored nothing at all, and printed nothing at all.**
By far the most serious finding. `python scripts/pre_push_score.py` sets
`sys.path[0]` to `scripts/`, so `from ml.extract_features import ...` raised
`ImportError` inside `score_commit()`, whose `except Exception: return None`
swallowed it. Every commit was skipped, `scored` stayed 0, and the script exited
0 with no output — a gate that looks installed and does nothing.

This affected the CLI hook, the `.pre-commit-config.yaml` entry and the new
husky pre-push hook alike. It was found only by running the hook end to end and
noticing that five outgoing commits produced zero output. Fixing the import is
three lines; the important part is that the failure is now *reported*, because
an unscored commit is indistinguishable from a safe one.

**B8 — `pre_push_score.py` was missing from the project's own lint exemptions.**
Fixing B7 meant touching a long-untouched file, and the new pre-commit hook
(which lints changed files) then reported that file's entire pre-existing debt —
seven errors — and blocked the commit. `ruff.toml` already exempts
`scripts/score_pr.py`, the same shape of code, from exactly those rules; its
sibling had simply been missed. Added to the existing list rather than loosening
the hook.

**B9 — non-ASCII output crashed the hook scripts on Windows.**
The em dashes in the hook log lines rendered as garbage in this repo's own
shell, and actively raised `UnicodeEncodeError` when a stream is not a console.
A hook that dies for a cosmetic reason blocks a commit. *Fixed:* scripts are now
ASCII, and the runtime reconfigures stdout/stderr with
`errors="backslashreplace"` so stray non-ASCII degrades to an escape.

### Found in phase 2

**B10 — a wrong `$0` silently killed every hook.**
The bootstrap sourced its helpers with
`. "$(dirname "$0")/lib/common.sh" 2>/dev/null || true`. That guard cannot
work: POSIX requires a non-interactive shell to **abort** when `.` cannot find
its file, so the shell is already gone by the time `|| true` would run. Any
invocation where `$0` is not the hook path — `sh -c`, some GUI clients, a
wrapper — took every hook down with no error from git at all.

This is the most dangerous class of defect in the whole integration: hooks that
vanish silently look exactly like hooks that pass. *Fixed:* source only after a
`[ -f ]` test, with a fallback to the hooks directory in the repository root.
Found by a CI step written to source the bootstrap directly.

**B11 — `GATEKEEPER_PYTHON` was documented but never honoured.**
The bootstrap's own failure message told people to set it, while the candidate
list never read it — so the one escape hatch for a machine without Python on
`PATH` did nothing. *Fixed:* it is now tried first.

**B12 — `hotfix/*` was missing from the default protected branches.**
The configured list in `.gatekeeper.yml` had it; the built-in fallback did not,
so a repo without a config would happily let a hotfix branch be rebased. Caught
by integration tests that assert the defaults rather than the config.

**B13 — hook modules broke when invoked by file path.**
Moving to `python -m scripts.hooks.<hook>` removed the four-line `sys.path` shim
duplicated in five files, but silently invalidated `npm run hooks:verify`, which
still called the doctor by path. Caught while wiring the new npm scripts.

**B14 — `project_python()` trusted a PATH lookup over the interpreter that was
already proven to work.**
The resolution order put `python3` on `PATH` above `sys.executable`. On Windows
`python3` is routinely the Microsoft Store alias stub, which is on `PATH`,
exits non-zero, and prints an install advert — so every tool invocation failed
and `ruff` was reported as *lint errors* rather than as a broken interpreter,
which is precisely the misleading failure B2 made the bootstrap avoid. The
interpreter already running the hook now outranks the lookup, because
`bootstrap.sh` has already proved it starts. Caught by the integration tests
written in the same pass (commit `fd32d1d`), which is the point of writing them
against a real temporary repo rather than a mock.

**B15 — `rt.git()` raised instead of degrading when the working directory had
gone.**
`subprocess.run(..., cwd=REPO_ROOT)` raises `FileNotFoundError` if that
directory no longer exists — a hook firing after a `git clean` of the checkout,
or on a CI runner tearing its workspace down. A hook is the wrong place for a
traceback, and the failure was indistinguishable from a hook crash. It now
reports `127` like any other git failure and lets the caller decide.

**B16 — the three "no-op" hooks were holes, not redundancies.**
`pre-applypatch`, `post-applypatch` and `pre-auto-gc` were left unimplemented on
the reasoning that they duplicated `applypatch-msg` and had nothing to decide.
Reading git's actual dispatch order showed otherwise: `git am` runs
`applypatch-msg`, `pre-applypatch` and `post-applypatch` and **none** of the
commit hooks. So a patch series landed content that never went through
`pre-commit`, and its commits were never recorded for outcome tracking either.
`pre-auto-gc` was meanwhile free to repack the object store underneath a
multi-minute `pre-push` that was reading history and scoring commits.
*Fixed:* all three carry policy now, and the two shared concerns moved into one
module each (`_checks.py`, `_events.py`) so the commit path and the patch path
cannot drift apart again.

**Test bug, not a code bug.** The first doctor test asserted that
`python -m scripts.hooks.verify` exits 0 — which asserts that *this machine has
run `npm install`*, and is therefore never true on a fresh CI checkout. It now
checks the structural invariants everywhere and the full doctor only where the
hooks are actually installed.

**B17 — the hook tests were not hermetic, and the gate that runs them is what
exposed it.**
Two shell-level tests drove `pre-auto-gc` and asserted it exits 0. Both passed
locally and both failed on the very first attempt to push this branch. The
cause is not in the hook: `pre-auto-gc` answers by reading
`.git/gatekeeper/gate_run.json`, and the **pre-push hook writes exactly that
marker while it runs the test suite**. So the tests were asserting "no gate is
running" from *inside* a gate run, and reporting the surrounding push's state
instead of the hook's.

This is the more interesting failure of the two, because the hook was right and
the test was wrong — and the wrongness was invisible until the tests ran in the
one context they were written for. A suite that only ever passes standalone is
not evidence about a hook that runs under `git push`.

*Fixed:* local state is now relocatable with `GATEKEEPER_STATE_DIR`. The shell
tests point it at a temporary directory, so their answer no longer depends on
what the repository is doing, and two new tests assert both sides of the
decision under that redirect — same hook, same repository, opposite answers.
The escape hatch earns its keep twice over: it is also the right answer for a
read-only git directory. Verified by re-running the whole hook suite with a
live marker deliberately planted in `.git/gatekeeper/` — the condition that
failed — which now passes, while the real shim still declines (exit 1) under
that same marker.

**B18 — the same mistake one layer over: a test reading the ambient
environment.**
Re-running the gate by hand turned up a second instance of B17's shape.
`TestTracing::test_husky_zero_does_not_trace` asserts that `HUSKY=0` does not
enable tracing — but it set only `HUSKY`, leaving `GATEKEEPER_TRACE` to the
environment. Anyone with `export GATEKEEPER_TRACE=1` in their profile failed
that test, and since pre-push runs this suite, **a debugging habit would have
blocked their own push**. It surfaced only because the gate was run with
tracing on, to get per-stage timings.

*Fixed* in two layers rather than one:

1. The test now clears both switches, because it asserts something about one.
2. `tests/conftest.py` clears **every** environment variable the hook layer
   reads, for the whole session. A test that needs one sets it with
   `monkeypatch`, which restores it afterwards — so the suite now answers about
   the code rather than about the machine that started it.

This is the general form of B17, and worth stating plainly: a hook layer
configured by the environment cannot be tested by a suite that inherits the
environment. The verification below runs the whole suite with all eight
switches set to hostile values *and* a live gate marker, which is the check
that was missing.

**B19 — `verify --fix` repaired the wiring and then reported the damage it had
just repaired.**
`--fix` re-runs husky's installer, the one documented repair for a rewritten
`core.hooksPath` or a `.husky/_/` that was never generated; it is deliberately
never automatic. It ran at the top of `main()`, but the check list beneath it is
built eagerly — each `_check_*()` runs and captures its result before anything
is printed. So `--fix` printed the installer's output and then the *pre-repair*
state: a repair that had worked, reported as two failures and exit 1.

The way it was found is the useful part. Four independent breakages were planted
at once (a truncated `_/h`, a flag-shaped `core.hooksPath`, and a deleted
`_/husky.sh` and `_/pre-push`), so a repair that silently handled only the
obvious one would have looked like success. It was the step after it — re-running
the doctor and reading the report — that showed the report was lying.

*Fixed:* repair runs before the list is built. A test stubs the other checks out
and asserts that a repair correcting `core.hooksPath` exits 0, so the ordering
cannot regress silently.

**The generated directory stopped being a formality.** `_/` used to be checked
for existence only. It is now checked for *shape*: `_/h` must look like husky's
dispatcher, every generated hook must actually delegate to it, and a missing
`_/husky.sh` — husky's v8 deprecation guard — is a warning. The failure this
closes is the nastiest shape in the layer: a `_/` that exists, is executable,
and does nothing, which git reports as success.

---

## 5. Verification

| Check | Command | Result |
|-------|---------|--------|
| Hook unit tests | `python -m pytest tests/test_hooks.py -q` | **58 passed** |
| Hook integration tests | `python -m pytest tests/test_hook_integration.py -q` | **114 passed** |
| Generated-husky fidelity | `python -m pytest tests/test_husky_fidelity.py -q` | **14 passed** — `.husky/_/` byte-identical to a fresh install, dispatcher contract pinned |
| Hook layer tests | `python -m pytest tests/test_hooks.py tests/test_hook_integration.py tests/test_husky_fidelity.py -q` | **186 passed** |
| Upstream conformance suite | `sh tools/husky-conformance/run.sh --require` | **12/12 passed** — husky 9.1.7, MINGW64, node v24.12.0, ~59 s |
| Whole suite | `python -m pytest tests/ -q` | **282 passed** (2 pre-existing pydantic deprecation warnings) |
| Lint on new code | `ruff check scripts/hooks/ tests/test_hooks.py tests/test_hook_integration.py tests/test_husky_fidelity.py` | clean |
| Bad message rejected | `git commit -m "totally bogus message"` | exit 1, hook blocked |
| Good message accepted | `git commit -m "feat(hooks): ..."` | exit 0 |
| Lint error blocked | committing a file with a ruff violation | exit 1, hook blocked (happened twice, both times on real findings) |
| Skip hatch | `GATEKEEPER_SKIP_HOOKS=pre-commit git commit ...` | hook skipped |
| `init` without husky | scratch repo, no `.husky/` | wrote portable runtime-resolved hook |
| `init` with husky | scratch repo with `.husky/` | stood down, wrote no hook |
| Gate 1 end-to-end | pre-push with simulated refs | scored 5 commits with bands and SHAP reasons |
| Full pre-push | Gate 1 + ruff + pytest | passed; ruff clean on 7 changed files |
| Doctor | `python -m scripts.hooks.verify` | all 7 checks pass, 0 warnings, 14 hooks |
| Doctor repair (B19) | four planted breakages, then `npm run hooks:repair` | repaired; `verify` exits 0 afterwards |
| Doctor, strict | `python -m scripts.hooks.verify --strict` | passes on an installed machine; the CI runner uses this |
| Doctor inventory | `python -m scripts.hooks.verify --list` | `14/14 husky hooks`, with the block/report kind per hook |
| Event log | `python -m scripts.hooks.verify --events` | the last 10 records, one per commit and per `git am` patch |
| Flag-shaped `core.hooksPath` | `git config core.hooksPath --version/_` then the doctor | exit 1 (B1 reproduced on purpose, then restored) |
| `git am` content gate | a patch that would fail `pre-commit` | refused by `pre-applypatch` (B16) |
| Gate-run marker | marker written, then read by `pre-auto-gc` | collection declined while fresh; allowed once stale |
| Relocatable state | `GATEKEEPER_STATE_DIR=<tmp>` | the marker and the event log both follow it (B17) |
| Hook suites under a live gate | suites re-run with a marker planted in `.git/gatekeeper/` | **168 passed** — the run that failed before the fix (B17) |
| Hook layer, hostile environment | `GATEKEEPER_TRACE=1 HUSKY=2 GATEKEEPER_SKIP_HOOKS=all GATEKEEPER_HOOK_TESTS=0 GATE1_BLOCK=1 GATEKEEPER_ALLOW_PROTECTED=1`, **and** a live gate marker | **186 passed** — was 1 failed before B18 |
| Real shim under a live gate | `sh .husky/_/pre-auto-gc` with the marker present | exit 1, "skipping background gc"; exit 0 without it |
| Tracing | `HUSKY=2 sh .husky/_/pre-auto-gc` | sh `set -x` **and** `pre-auto-gc: interpreter=...` from Python |
| Install / uninstall round trip | `node .husky/uninstall.mjs` then `install.mjs` | `core.hooksPath` unset and restored; `.husky/_` removed and regenerated |
| CI-safe install | `CI=true node .husky/install.mjs` | skips; `HUSKY=1` forces a real install |
| Real shims | `sh .husky/_/pre-commit`, `.husky/_/commit-msg` | 0 on a clean tree; 1 for a bad subject |
| Bootstrap robustness | `sh -c` with a wrong `$0`, and from another directory | survives; `gatekeeper_run_hook` still resolves a module |
| Bootstrap with no Python | empty `PATH`, no override | exits non-zero and says so |
| Merge into a protected branch | `pre_merge_commit.main()` on `main` | exit 1; 0 on a feature branch; 0 with the override |
| Protected-branch globs | `release/2.0`, `hotfix/urgent` | both refused |
| Editor template seeding | `prepare_commit_msg` with `template` vs `message` | seeded for the editor, untouched for `-m` |
| Outcome log | `post_commit` | one JSON line per commit with sha, branch, author |

The pre-commit hook blocked its own author's commit twice, both times on real
findings — three ruff nits in `tests/test_hooks.py` (commit `08d8983`) and the
seven pre-existing errors in `pre_push_score.py` that led to B8.

The unit test for skip semantics also failed on first run: the assertion was
wrong (`commit-msg,pre-push` was expected to disable `pre-commit`). The code was
right, the test was fixed.

Gate 1's silent no-op (B7) was the only defect that reached a green end-to-end
run before being caught: the hook exited 0 and printed nothing, which is why the
verification table above checks the *content* of Gate 1's output, not just its
exit code.

---

## 6. How to use it

See [`docs/HUSKY.md`](docs/HUSKY.md) for the full reference. Short version:

```bash
npm install              # installs husky and activates the hooks
npm run hooks:init       # install and verify the wiring
npm run hooks:verify     # the doctor
npm run hooks:repair     # re-run husky's installer when the doctor fails
npm run hooks:list       # print the hook inventory
npm run hooks:uninstall  # unset core.hooksPath and remove .husky/_
```

Bypass hatches: `GATEKEEPER_SKIP_HOOKS=<hook|all>`,
`GATEKEEPER_HOOK_TESTS=0`, `GATEKEEPER_ALLOW_PROTECTED=1`, `GATE1_BLOCK=1`,
`HUSKY=0`, `GATEKEEPER_PYTHON`, `GATEKEEPER_STATE_DIR`, and git's own
`git commit -n`.

Debug hatches: `HUSKY=2` (or `GATEKEEPER_TRACE=1` when husky is bypassed) traces
the whole chain, including the interpreter each hook resolved to.

CI (`.github/workflows/hooks.yml`) is now three jobs. `hooks` installs husky on
ubuntu, macOS and Windows across Python 3.11 and 3.12, runs the doctor with
`--strict`, asserts the `14/14` inventory, runs the hook layer's 186 tests,
**re-runs them against a live gate marker and every hook switch set to a
hostile value (B17, B18)**, **runs upstream husky's own twelve-test conformance
suite on all three operating systems** (12/12 locally on Git Bash), exercises
the real `.husky/_` shims through the shell git uses, traces one hook end to
end, reproduces the flag-shaped `core.hooksPath`, and shellchecks *every* shim —
the list comes from `rt.HOOKS`, so a new hook cannot skip the check.
`installer` walks install → uninstall → reinstall on Node 18, 20 and 22, plus a
`NODE_ENV=production` install. `suite` runs the project's own test suite.

---

## 7. Open items (deliberately not done)

Carried forward, **not** fixed on this branch:

- **R3 — resolve the band thresholds to one source.** Five values in five files
  is a correctness hazard for the model's own output contract. Not a hook
  concern; fixing it changes scoring behaviour and needs its own validation.
- **R4 — `get_repo()` parses YAML as JSON**, so per-repo thresholds never load.
- **R5 — the CLI's JS rule checks duplicate `rules/engine.py`.** The hook layer
  deliberately routes through the Python engine for Gate 1 scoring and does not
  add a third implementation, but `cmdCheck()` still carries its own rules.
- **R6 — `cli/package.json` test script matches no files.**
- **R7 — 732 pre-existing ruff errors** are now tolerated by scoping (B4) rather
  than fixed. A repo-wide lint gate is not possible until this is paid down.
  Note the new side effect found in B8: because the hook lints *changed* files,
  any edit to a file with old debt now surfaces that whole file's errors. That is
  arguably correct, but it means paying down debt is now on the path of any
  change to those files.
- **Upstream's conformance suite runs in CI and on demand, not inside `pytest`.**
  It needs node and npm, and each of its twelve tests does a real `npm install`
  in a throwaway repository: about a minute, which would roughly double the
  pre-push gate. The part that does not need node — the generated-`_/` fidelity
  module — *is* in the suite, and skips cleanly where husky is not installed.
- **`.pre-commit-config.yaml` is still not installed anywhere.** It is kept as
  the declarative policy record, but nothing verifies it still matches what the
  husky hooks actually enforce. Two descriptions of one policy can drift.
- **A custom hooks directory is not supported.** The hook modules import
  `scripts/hooks/*`, which only resolve at the repository root, so a nested
  hooks directory (upstream's `husky sub/.husky`) cannot work here. The doctor
  reads `core.hooksPath` rather than assuming it, so a moved directory is at
  least reported rather than silently ignored.
- **The `suite` CI job installs a dependency subset, not `requirements.txt`.**
  `tests/` does not need tensorflow, feast or kfp, and pulling them turns a
  one-minute job into a ten-minute one. The list was derived by walking the
  import graph from `tests/` rather than guessed, which is how `httpx` was kept
  — nothing imports it, but starlette's `TestClient` needs it at runtime. If a
  test ever needs a new package, the job fails loudly and the list needs it.
- **`pre-push` runs the whole suite, which is ~65s warm but ran ~525s during a
  push under load.** The variance is contention (Gate 1's feature extraction
  plus the push itself), not the tests. `GATEKEEPER_HOOK_TESTS=0` skips the
  stage. Scoping the test stage to the test files affected by the outgoing diff
  — the way ruff is already scoped — would be the next improvement, though it
  needs a dependency map from source files to tests to be trustworthy.
- **`gate1` and the other npm scripts still call Python scripts by path.**
  That is correct for them (they set their own `sys.path`), but it is a second
  invocation convention next to the hooks' `-m` one.
- **12 of git's 28 hooks are still not implemented**, and that is the right
  call: `pre-receive`, `update`, `post-receive`, `post-update`,
  `reference-transaction`, `push-to-checkout` and `sendemail-validate` are
  server-side, `fsmonitor-watchman` is a filesystem-daemon integration, the four
  `p4-*` hooks are Perforce bridges, and `post-index-change` fires on every
  index touch. husky's 14 are the ones that apply to a *client* repository, and
  now all 14 are ours. `KNOWN_GIT_HOOKS` lists the full 28 so the doctor can
  still tell a real hook name from a typo.
- **Not merged into `main`**, as requested.
