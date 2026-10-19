# CI/CD in this repository

Four workflows, one shared verification matrix, and a set of tests that assert
the workflows themselves. This document is the map: what runs, when it runs,
what it proves, and how to reproduce any of it without pushing.

The reason it exists is in the next section. Two of these workflows had **never
passed**, and both were failing for reasons that were invisible from a
developer machine.

---

## 1. The inventory

| Workflow | Trigger | Jobs | What it proves |
|---|---|---|---|
| `.github/workflows/hooks.yml` | every push, every PR | `hooks` (ubuntu + macOS + Windows × py3.11 + 3.12), `installer` (node 18/20/22), `suite`, `verification` (3 OSes) | The husky-managed git hooks are wired correctly on every platform and nothing they gate has broken |
| `.github/workflows/gate3_post_deploy.yml` | push to `main` | `smoke-tests` | The stack builds in Docker, comes up healthy, and behaves under `smoke_tests/` |
| `.github/workflows/gate2_pr_risk.yml` | pull requests | `score-pr` | A PR's commits are scored and the comment is posted |
| `.github/workflows/keep_warm.yml` | cron `*/10 * * * *`, manual | `ping` | The deployed API stays warm |
| `.github/workflows/retrain_trigger.yml` | manual | `retrain` | The retraining pipeline can be kicked off on demand |

`hooks.yml` is the one that gates every branch, so it is also the one that was
paid the most attention.

## 2. The failures this document was written about

### 2.1 `hooks` job — a test needed a library the job did not install

`tests/test_hook_integration.py` asserts that a *configured* protected-branches
list is honoured, which it does by writing a temporary `.gatekeeper.yml` and
reading it back through `scripts/hooks/_policy.py`. PyYAML is deliberately
**optional** to the hook layer — `gatekeeper_config()` falls back to
`DEFAULT_PROTECTED_BRANCHES` when `import yaml` raises — but it is not optional
to those tests.

The job installed `pytest` and nothing else. So the tests failed on all four
matrix legs while passing on any machine that happened to have PyYAML, which is
the classic green-laptop/red-CI shape. Three of the five previous runs of this
workflow fail at exactly this step.

Fixed by:

* installing `pyyaml` in the job;
* a step that does `python -c "import yaml"`, so removing it fails loudly
  instead of silently turning those tests into skips;
* `requires_yaml` + `skipif` on the tests that need a configured list, so the
  *same tests* on a machine without PyYAML skip with a reason instead of failing;
* `tests/test_hook_integration.py::test_defaults_survive_a_missing_pyyaml`,
  which pins the degradation contract from the side that never needs YAML.

### 2.2 `suite` job — the tests read inputs a fresh clone does not have

`data/commit_features.csv` and `repos/` are both gitignored, so the checkout CI
gets has neither. Four test files read them unguarded
(`test_determinism.py`, `test_feature_parity.py`, `test_fuzz_thresholds.py`,
`test_perf_regression.py`), and pytest reported them as **errors**, which fails
the job within seconds.

Fixed by a marker policy, implemented once in `tests/conftest.py`:

| Marker | Means | Behaviour when the input is absent |
|---|---|---|
| `needs_data` | reads `data/commit_features.csv` | skip, naming the file |
| `needs_repos("rust", ...)` | needs cloned repositories under `repos/` | skip, naming the missing clones |
| `requires_yaml` | needs PyYAML to read a configured list | skip, naming the library |

The `suite` job now runs `pytest tests/ -q -rs`, and `-rs` is load-bearing: it
prints every skip with its reason, so "this input is not here" can never be
mistaken for "this passed".

Measured in a fresh clone (no CSV, no `repos/`): **276 passed, 40 skipped, 0
errors**. Before the markers, the same run errored during collection.

### 2.3 Gate 3 — the Docker build COPYs a gitignored directory

`api/Dockerfile` contains `COPY dashboard/dist/ ./dashboard/dist/`, and
`dashboard/dist/` is gitignored (`.gitignore` has a bare `dist/`). The directory
therefore does not exist in a clean checkout, and `docker compose build` failed
on **95 consecutive runs** — the last one on `main` at `c3a8264`, after which
every later step was skipped.

Fixed twice over, because either fix alone leaves a hole:

1. the workflow now runs `npm --prefix dashboard install` and
   `npm --prefix dashboard run build` before `docker compose build`, so the
   image carries a real dashboard;
2. `dashboard/dist/.gitkeep` is tracked (via `!dashboard/dist/` negations in
   `.gitignore`), so the directory exists even for a `docker compose build` that
   skips step 1. Build output stays ignored.

### 2.4 `verification` job — the matrix checked the wiring before installing it

`verify_all` ran `doctor --strict` first. On a developer machine that passes,
because husky was installed months ago. A fresh clone has no `core.hooksPath`
and no generated `.husky/_/`, and `--strict` promotes the resulting warning to a
failure — so the job failed on Ubuntu, macOS and Windows before it reached a
single real problem.

Fixed by ordering: `install` (install → uninstall → reinstall, which ends
installed) now runs before `doctor`. The matrix wires the hooks it is about to
check, so it no longer depends on the machine having done it already.

### 2.5 The next run — three tests that reported on the runner

With the above fixed, the next run still failed, for three reasons that have the
same shape: a test answered a question about the machine it ran on rather than
about the code.

| Symptom | Cause | Fix |
|---|---|---|
| `git rev-parse HEAD~1` raised | `actions/checkout` clones with depth 1; there is no parent commit | the test builds a two-commit throwaway repository and asserts the exact changed-file list, so it also stops accepting an empty result |
| `FileNotFoundError: 'sh'` | the test empties `PATH` on purpose, and exec search uses the child's `PATH`, so `sh` could not launch | the helper resolves an absolute shell path before the test's env is applied |
| `No module named ruff`, then a hook blocked a clean file | nothing imports ruff, so no import graph mentioned it, but `pre-commit` and `pre-applypatch` *run* it | `ruff` is installed in `hooks`, `suite` and `verification`, and the `hooks` job asserts `python -m ruff --version` |

The first one is the general lesson: a test that depends on how deep the
checkout is will be green forever on a developer's full clone.

## 3. Running the whole matrix locally

```bash
python -m scripts.ci.verify_all            # the local matrix
python -m scripts.ci.verify_all --list     # what would run, and why
python -m scripts.ci.verify_all --docker   # + the Gate 3 rehearsal (needs Docker)
python -m scripts.ci.verify_all --only trace --only conformance
python -m scripts.ci.verify_all --fail-fast
```

`scripts/ci/verify_all.py` is the single definition of the matrix. The
`verification` job in `hooks.yml` runs the same script with `--profile ci` on
Ubuntu, macOS and Windows, so "it passed locally" and "it passed in CI" are the
same sentence rather than two lists that drift.

What it covers, in order:

| Step | What it proves |
|---|---|
| `install` | install → uninstall → reinstall leaves exactly the expected `core.hooksPath` and `.husky/_` (first, so the checks below have something to check) |
| `doctor` | `scripts.hooks.verify --strict`: the wiring is intact |
| `inventory` | `--list` reports `14/14 husky hooks` |
| `hook-tests` | the hook layer's tests pass with only pytest, PyYAML and ruff present |
| `hostile-environment` | the same tests pass with every hook switch exported against them and a gate-run marker planted |
| `trace` | `HUSKY=2` reaches the Python layer and names the interpreter |
| `real-shims` | the real `.husky/_` → `.husky/<hook>` chain exits 0 through `sh` |
| `commit-msg-shim` | an invalid subject is rejected by the real shim |
| `bootstrap` | `.husky/lib/bootstrap.sh` resolves an interpreter (the Windows-critical path) |
| `doctor-rejects-flag` | a flag-shaped `core.hooksPath` is caught, then restored |
| `conformance` | husky's own vendored 12-test suite passes |
| `shellcheck` | no shell-level syntax error can take every hook down silently |
| `project-suite` | `pytest tests/` (local profile only — CI's `suite` job runs it separately) |
| `workflow-invariants` | the workflow assertions in `tests/test_ci_workflows.py` |
| `docker-rehearsal` | Gate 3's steps, locally, before a push to `main` runs them for real |

Steps are skipped — with the reason printed — when the platform or a tool rules
them out, which is the same set CI gates with `if: runner.os == ...`.
`shellcheck` is not installed on every developer machine; on Ubuntu CI it is, so
the step runs there.

Everything is appended to `verify_all.log` (gitignored) as well as the console.

## 4. A red run has to be readable

The three failures above stayed unexplained for weeks partly because a GitHub
job log needs a signed-in browser. Every job that can fail now leaves its output
behind:

* the hook, suite, installer and smoke-test steps run through `tee` with
  `set -o pipefail` (without `pipefail` the pipeline would report `tee`'s
  success, not pytest's failure);
* each job uploads those logs with `actions/upload-artifact` under
  `if: failure()`;
* Gate 3 also dumps `docker compose logs` *before* its unconditional teardown.

So a failed run can be diagnosed from the artifact list alone.

## 5. The workflows are tested too

`tests/test_ci_workflows.py` asserts the invariants that were silently broken,
because none of them is a behaviour of the library and the ordinary suite could
not see them:

* every workflow parses and declares a trigger;
* the `hooks` job installs `pyyaml` **and** asserts that importing it works;
* every job that runs the hook tests installs `ruff`, and the `hooks` job
  asserts that `python -m ruff` runs;
* the YAML-dependent tests carry a skip derived from availability, not a constant;
* the `suite` job installs `pyyaml` and `ruff`, and runs with `-rs`;
* `verify_all` installs the hooks before `doctor --strict` checks them;
* every test that reads `commit_features.csv` or `repos/` declares a marker or
  guards itself — checked by parsing the test files with `ast`;
* Gate 3 builds the dashboard *before* building the images;
* `dashboard/dist/.gitkeep` is tracked, and `.dockerignore` does not exclude the
  directory the Dockerfile COPYs;
* the built dashboard output is still gitignored (the place-holder must not
  become a committed artifact);
* every job that can fail uploads its output;
* every step in `scripts/ci/verify_all.py` names a CI step that still exists in a
  workflow — the drift guard for the two matrices.

## 6. Deliberate non-goals

* **The `suite` job still has no training data.** It installs a dependency
  subset rather than `requirements.txt` (tensorflow, feast and kfp would turn a
  one-minute job into a ten-minute one), and generating a synthetic CSV would
  test the generator, not the model. The data-dependent tests therefore skip
  there, and `-rs` is how you see that they did.
* **Job logs still require authentication.** Only the uploaded artifacts do not;
  that is a GitHub property, not something this repository can change.
* **`shellcheck` is not vendored.** It comes from the runner image on Ubuntu; a
  local machine without it skips that step rather than installing anything.
* **The pre-push gate is not part of this matrix.** `verify_all` is what a
  developer runs deliberately; the pre-push hook is what runs whether they meant
  it or not.
* **One perf test is machine-sensitive.**
  `test_perf_regression.py::test_single_commit_warm` compares a warm run against
  a 2.0 s budget and measures 2.1–2.4 s on a loaded laptop, on unchanged code.
  It carries `needs_repos`, so it skips in CI and anywhere without `repos/`;
  `GATEKEEPER_HOOK_TESTS=0 git push` skips the gate's test phase while Gate 1
  scoring and the ruff check still run.
