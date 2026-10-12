# Merging `husky` into `main`

This is the record of the merge: what was backed up, why it is a fast-forward,
what was broken before it could land, and how to undo it. The CI work that came
with it is documented separately in [docs/CI.md](CI.md).

---

## 1. Why a backup branch existed before anything else

`main` had 57 commits' worth of behaviour that this branch does not touch — the
API, the dashboard, the model, the smoke tests — and the request was explicit:
keep the old `main` available in case a fallback is wanted, and make sure the 57
husky commits appear on the same commit graph.

Those two wants are both satisfied by a fast-forward:

* the 57 commits *are* `main`'s history afterwards, not squashed or replayed;
* `copy-main-before-husky` points at the old `main`, so the previous state is one
  `git checkout` away.

| Ref | Commit | Meaning |
|---|---|---|
| `copy-main-before-husky` | `c3a8264` | `main` exactly as it was before the merge |
| `main` (before) | `c3a8264` | `W7.3 fix: _get_db never worked — invalid SQL and missing psycopg2` |
| `husky` | `5e287f4` | the branch's 57 commits, plus the CI fixes described below |

`git rev-list --left-right --count main...husky` returned `0 57`, so `main` was a
strict ancestor of `husky`: no merge commit, no conflict, nothing to resolve.

## 2. The order of operations

1. `git branch copy-main-before-husky c3a8264493999f0368fee0db8745f73b1b70b848`
   — the backup, from a working tree that also carries the developer's own
   untracked files (`data/`, `repos/`, `mlflow.db.bak`, …). None of those were
   touched: nothing was staged with `git add -A`, and neither `git stash` nor
   `git clean` was run.
2. Before switching branches, both locally-modified tracked files
   (`models/gatekeeper_risk_model.skops`, `smoke_tests/drift_report.html`) were
   confirmed **identical** in `main` and `husky`
   (`git diff --stat main husky -- <paths>` was empty), so a checkout carries them
   over untouched rather than refusing or clobbering.
3. The CI failures were fixed on `husky` and verified locally, so `main` never
   received a red state. This is deliberate: a fast-forward onto `main` cannot be
   followed by a "fix CI" commit without putting a red run on the protected
   branch's history.
4. `husky` was pushed and its workflows watched to green.
5. `main` was fast-forwarded and pushed; `Gate 3` runs on that push.

## 3. What was fixed before the merge

Both failures are described in full in [docs/CI.md](CI.md) §2. In one line each:

* **`hooks` job** — the job installed `pytest` without PyYAML, and two tests read
  a configured `.gatekeeper.yml`, so they failed on all four matrix legs and on
  every run. Now: PyYAML is installed, its import is asserted, and the tests skip
  with a reason where the library is genuinely absent.
* **`suite` job** — four tests read `data/commit_features.csv` and `repos/`, both
  gitignored and therefore absent from the fresh clone CI checks out. Now:
  `needs_data` / `needs_repos` markers turn a missing input into a skip with a
  truthful reason, and `-rs` prints them.
* **Gate 3** — `api/Dockerfile` COPYs `dashboard/dist/`, which no clean checkout
  has; the workflow had failed 95 runs in a row. Now: the workflow builds the
  dashboard first, and `dashboard/dist/.gitkeep` is tracked so the directory
  exists regardless.

## 4. Verification before the merge

Run on the merged content, on this machine (Windows, Git Bash, node v24.12.0,
Python 3.11.15 in `venv/`).

| Check | Command | Result |
|---|---|---|
| The whole local matrix | `python -m scripts.ci.verify_all` | **14/14 steps** (shellcheck skipped: not installed here; four POSIX-only steps skipped on Windows) |
| Project suite | `python -m scripts.ci.verify_all --only project-suite` | **316 passed**, 2 pre-existing warnings (~258 s) |
| Workflow assertions | `pytest tests/test_ci_workflows.py -q` | **33 passed** |
| Hook layer | `pytest tests/test_hooks.py tests/test_hook_integration.py tests/test_husky_fidelity.py -q` | **186 passed** |
| Hook layer, hostile environment | matrix step `hostile-environment` | passed |
| Install lifecycle | matrix step `install` | install → uninstall → reinstall clean |
| Upstream conformance | `sh tools/husky-conformance/run.sh --require` | 12/12 (Git Bash) |
| The `hooks` CI condition | PyYAML hidden from the project's modules, `-k TestProtectedBranchPolicy` | **4 passed, 3 skipped, exit 0** — previously 2 failures |
| The `suite` CI condition | fresh `git clone` (no CSV, no `repos/`) + `pytest tests/ -q -rs` | **276 passed, 40 skipped, 0 errors** — previously errors at ~12 s |
| Gate 3 rehearsal | `python -m scripts.ci.verify_all --docker` | not run locally: the Docker daemon was not running on this machine. Verified by the workflow's own run on `main`, whose failure artifact is now readable either way |

The two "CI condition" lines are the ones that matter: they reproduce the exact
environment the failing jobs were in, rather than re-running the suite where the
inputs happen to be present.

## 5. Undoing it

The backup branch is a complete, independent copy of the pre-merge `main`:

```bash
git fetch origin
git checkout copy-main-before-husky      # inspect the old state
git log --oneline -3                     # expect c3a8264 at the tip
```

To put the old state back on `main`:

```bash
git checkout main
git reset --hard copy-main-before-husky  # local main only
git push --force-with-lease origin main  # rewrites published history: deliberate act
```

Force-pushing a protected branch is not something to do casually, which is why
the backup exists as a *branch* rather than as a note: nothing about the old
state depends on remembering a SHA.

## 6. Loose ends

* `repos/`, `data/commit_features.csv`, `mlflow.db.bak`, `api_server.log`,
  `scripts/rescore_prs.py` and `start_api.py` are still untracked in the working
  tree, exactly as they were before this work. They were never staged, stashed
  or deleted.
* `dashboard/package-lock.json` is still untracked, which is why Gate 3 and the
  rehearsal use `npm install` rather than `npm ci` for the dashboard. Committing
  that lockfile would let both use `npm ci` and pin the build.
* `shellcheck` is not installed on this machine, so that step is proven only by
  CI (Ubuntu's runner image ships it).
