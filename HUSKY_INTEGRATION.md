# Husky integration log

Living record of the `husky` branch: what was reviewed, what was designed,
what was built, what was verified, and what is still open.

**Branch:** `husky` — a copy of `main` at `c3a8264`. `main` has not been touched.
**Scope:** integrate [typicode/husky](https://github.com/typicode/husky) as the
repo's git-hook manager, and close the gaps that integration exposed.
**Reference doc:** [`docs/HUSKY.md`](docs/HUSKY.md).

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

## 3. Commit log

```
0cbf199 chore(husky): add root package manifest for git-hook tooling
95ee57e chore(husky): lock husky 9.1.7 at the repo root
0da0f08 feat(hooks): add shared runtime for husky hook scripts
61a0c1c feat(hooks): add commit-msg conventional commit linter
c102fbb feat(hooks): add staged-file linter for pre-commit
c23af85 feat(hooks): scope linting to the outgoing diff in the hook runtime
91ea08d feat(hooks): add pre-push gate running Gate 1, diff-scoped ruff and pytest
848f3dc feat(husky): add sh bootstrap that probes for a working Python 3
bf5b3ab feat(husky): wire the pre-commit shim
e07df44 feat(husky): wire the commit-msg shim
5758694 feat(husky): wire the pre-push shim
e67748c fix(cli): make `gatekeeper init` husky-aware and its hook portable
07ebac0 feat(hooks): add a wiring doctor exposed as `npm run hooks:verify`
08d8983 test(hooks): cover commit rules, ref parsing, skip semantics and wiring
```

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

---

## 5. Verification

| Check | Command | Result |
|-------|---------|--------|
| Hook unit tests | `python -m pytest tests/test_hooks.py -q` | **35 passed** |
| Whole suite | `python -m pytest tests/ -q` | **131 passed** (2 pre-existing deprecation warnings) |
| Lint on new code | `ruff check scripts/hooks/ tests/test_hooks.py` | clean |
| Wiring doctor | `python scripts/hooks/verify.py` | all checks passed |
| Bad message rejected | `git commit -m "totally bogus message"` | exit 1, hook blocked |
| Good message accepted | `git commit -m "feat(hooks): ..."` | exit 0 |
| Lint error blocked | committing a file with a ruff violation | exit 1, hook blocked |
| Skip hatch | `GATEKEEPER_SKIP_HOOKS=pre-commit git commit ...` | hook skipped |
| `init` without husky | scratch repo, no `.husky/` | wrote portable runtime-resolved hook |
| `init` with husky | scratch repo with `.husky/` | stood down, wrote no hook |

The pre-commit hook blocked its own author's commit during this work with three
genuine ruff findings in `tests/test_hooks.py` — recorded in commit `08d8983`.

The unit test for skip semantics also failed on first run: the assertion was
wrong (`commit-msg,pre-push` was expected to disable `pre-commit`). The code was
right, the test was fixed.

---

## 6. How to use it

See [`docs/HUSKY.md`](docs/HUSKY.md) for the full reference. Short version:

```bash
npm install            # installs husky and activates the hooks
npm run hooks:verify   # confirm the wiring
```

Bypass hatches: `GATEKEEPER_SKIP_HOOKS=<hook|all>`,
`GATEKEEPER_HOOK_TESTS=0`, `GATE1_BLOCK=1`, `HUSKY=0`, `GATEKEEPER_PYTHON`.

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
- **No CI job runs `npm run hooks:verify`.** The hooks are verified locally and
  by `tests/test_hooks.py`; a workflow step would catch a broken install sooner.
- **Not merged into `main`**, as requested.
