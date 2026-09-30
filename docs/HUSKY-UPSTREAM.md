# husky upstream, mapped

A pass over [typicode/husky](https://github.com/typicode/husky) — every source
file, every behaviour, and what this project does with each one.

Upstream is deliberately tiny: **7 published files, 120 lines total**. Almost
all of its behaviour lives in three of them, plus a test suite that is the real
specification. Version mapped: **husky 9.1.7**.

Legend: **adopted** (we use it or reproduce it), **relied on** (husky does it
and we depend on that), **replaced** (we do it differently on purpose),
**skipped** (not applicable here).

---

## 1. Published source files

| File | Lines | What it does | Decision |
|------|-------|--------------|----------|
| `package.json` | 25 | `bin: husky -> bin.js`, `exports: ./index.js`, `engines: node >=18` | **Adopted** — same dependency and version |
| `index.js` | 25 | The installer. Sets `core.hooksPath`, generates `.husky/_/` | **Adopted** — this is the mechanism we build on |
| `bin.js` | 26 | The CLI: `init`, deprecated `add`/`set`/`uninstall`, custom dir argument | **Partly** — see §4 |
| `husky` | 22 | The `h` shim template, copied to `.husky/_/h` | **Relied on** — our shims are called by it |
| `index.d.ts` | 1 | `export default function (dir?: string): string` | **Relied on** — the API our `install.mjs` calls |
| `README.md` | 1 | Link to the docs site | n/a |
| `LICENSE` | 21 | MIT | n/a |

### 1.1 In the repository but not in the package

| Path | What it is | Decision |
|------|------------|----------|
| `test.sh`, `test/{functions.sh,1_default.sh … 12_deprecated.sh}` | The suite that verified this version: `test.sh` packs the module, then each script builds a throwaway repo and git config via `functions.sh` (`setup()`, `install()`, `expect*()`) | **Run** — vendored byte-for-byte under `tools/husky-conformance/upstream/` and executed by `tools/husky-conformance/run.sh` (§9) |
| `.github/workflows/node.js.yml` | Upstream CI: node 18/20/22 × ubuntu/macOS/Windows, `npm ci --ignore-scripts`, `./test.sh` | **Mirrored** — our `hooks.yml` runs the same twelve tests on the same three operating systems |
| `.github/workflows/npm_publish.yml` | Publishes on tag | **Skipped** — this package is private |
| `.github/workflows/deploy.yml` | Builds and deploys the docs site | **Skipped** — the docs are read, not redeployed |
| `docs/index.md`, `get-started.md`, `how-to.md`, `migrate-from-v4.md`, `troubleshoot.md` (+ `es/`, `ru/`, `zh/`) | The manual, in four languages | **Read** — every documented feature is mapped in §7; English only, untranslated here |
| `.shellcheckrc` | shellcheck settings for their sh code | **Adopted in spirit** — our `sh` layer is shellchecked in CI with explicit flags rather than a repointed config |
| `.editorconfig`, `.gitattributes`, `.gitignore`, `.npmignore` | Repo housekeeping | **Skipped** — this repo has its own conventions |
| `.husky/pre-commit` (9 B) | husky's own hook in its own repo (`npm test`) | **Skipped** — `.husky/pre-commit` here is a real hook, not a wrapper around this repo's tests |
| `.github/ISSUE_TEMPLATE/issue.md`, `.github/README.md`, `.github/FUNDING.yml` | GitHub metadata | **Skipped** — not applicable |
| `LICENSE` | MIT, © typicode | **Vendored** with the suite as `tools/husky-conformance/upstream/LICENSE` |

The upstream files are unmodified and are the only third-party source in this
repository; their provenance, sizes and hashes are recorded in
[`tools/husky-conformance/README.md`](../tools/husky-conformance/README.md).

---

## 2. Installer behaviour (`index.js`)

| Behaviour | Decision | Notes |
|-----------|----------|-------|
| `HUSKY=0` skips installation | **Adopted** | `install.mjs` skips on `HUSKY=0`, `NODE_ENV=production` or `CI` |
| Rejects `..` in the target directory | **Skipped** | We never pass a directory; the guard exists for their CLI |
| Requires a `.git` directory | **Skipped** | Implied — there are no hooks without a repo |
| `git config core.hooksPath <dir>/_` | **Adopted** | The whole mechanism; our doctor asserts the value |
| Handles `git` being absent | **Skipped** | Their Node-API edge case; our hooks run *from* git |
| Writes `.husky/_/.gitignore` containing `*` | **Relied on** | Keeps generated shims out of every `git status` |
| Copies `husky` to `.husky/_/h` with mode `0755` | **Relied on** | The dispatcher in §3 |
| Writes 14 hook shims into `.husky/_/` | **Relied on** | Husky dispatches all 14; we implement all 14 (§5) |

**A useful consequence:** husky's `_/` shims exist for all 14 names it knows,
so adding a hook is only ever "create `.husky/<name>` and a module". We ship all
14, and `rt.HOOKS` and `rt.HUSKY_HOOKS` are now the same set — the doctor
asserts that, because a shim husky never dispatches is a file that can never
run.

**Checked, not assumed.** `tests/test_husky_fidelity.py` generates a fresh
`.husky/_/` with the installed husky in a temporary repository and asserts this
checkout's directory is byte-identical, file by file. A `_/` from a different
husky version, a shim someone edited, or a dispatcher that was never copied in
would all pass every other test in this repository and fail that one. The
doctor checks the same directory structurally (dispatcher looks like the real
one, every hook delegates to it, `_/husky.sh` carries the deprecation guard),
and `npm run hooks:repair` re-runs husky's installer for when a check fails.

---

## 3. The `h` shim

Every `.husky/_/<hook>` is `. "$(dirname "$0")/h"`. `h` then:

| Behaviour | Decision | Notes |
|-----------|----------|-------|
| `[ "$HUSKY" = "2" ] && set -x` — debug tracing | **Adopted and extended** | Reproduced in `bootstrap.sh`, then continued into Python via `rt.trace`/`rt.enter`, so one variable traces the whole chain (§6.1) |
| Resolves `$s` to `.husky/<hook>` | **Relied on** | How our shims get invoked |
| Missing `.husky/<hook>` exits 0 | **Relied on** | An unimplemented hook is a no-op, not an error |
| Warns when `~/.huskyrc` exists | **Relied on** | Upstream's deprecation notice |
| Sources `${XDG_CONFIG_HOME:-$HOME/.config}/husky/init.sh` | **Adopted** | Documented and checked by the doctor (§7) |
| `HUSKY=0` exits 0 | **Adopted** | Ours additionally honours it in Python and `.mjs` |
| Prepends `node_modules/.bin` to `PATH` | **Relied on** | Not something we need, but it shapes the environment |
| `sh -e "$s" "$@"` (errexit) | **Relied on** | Why our shims must not run loose commands |
| `husky - <hook> script failed (code N)` | **Relied on** | Our modules also report their own cause |
| Special-cases exit 127 as "command not found in PATH" | **Replaced** | We fail earlier with a specific message (§3.1) |

### 3.1 Where we diverge, and why

127 is a *helpful but late* signal: by then the hook has already run and git
reports a generic failure. Our `bootstrap.sh` probes each candidate interpreter
before use, so a missing Python is diagnosed at the source with an actionable
message — and it had to, because on Windows `command -v python3` succeeds
against the Microsoft Store alias stub, which exits non-zero.

---

## 4. CLI behaviour (`bin.js`)

| Command | Upstream behaviour | Decision |
|---------|--------------------|----------|
| `husky` (no args) | Install | **Adopted** as `npm run hooks:install` |
| `husky init` | Writes `prepare`, installs, creates `.husky/pre-commit` = `<pm> test` | **Replaced** as `npm run hooks:init` — install *and* verify, and it does not overwrite our tracked hooks |
| `husky install` | Deprecated warning, still installs | **Replaced** — `hooks:install` is the modern name |
| `husky add` / `set` | Deprecated error, exit 1 | **Skipped** — adding a hook is creating a file |
| `husky uninstall` | Deprecated error, exit 1 | **Replaced** — we implement it properly; see below |
| `husky <dir>` | Custom hooks directory, e.g. `husky sub/.husky` | **Skipped** — see §8 |

`uninstall` is worth calling out: v9 removed the command and documents only the
manual `git config --unset core.hooksPath`. Doing just that leaves the generated
`.husky/_/` shims on disk for the next install to inherit, so
`.husky/uninstall.mjs` unsets the key **and** removes the directory.

**The footgun we inherited.** `bin.js` reads `process.argv[2]` and treats
anything unrecognised as the hooks directory. `npx husky --version` — an
entirely reasonable thing to type — therefore sets `core.hooksPath` to
`--version/_`, creates a stray `--version/` directory, and every hook stops
running with no error from git. This is not hypothetical; it happened while
building this integration. The doctor now asserts the value and names the cause.

---

## 5. Hook inventory

husky supports 14 and git supports 28. We ship **all 14 of husky's**: every
shim husky writes a dispatcher for has a real script behind it.

| Hook | Upstream shim | Ours | Kind |
|------|---------------|------|------|
| `applypatch-msg` | yes | `applypatch_msg.py` | blocks |
| `pre-applypatch` | yes | `pre_applypatch.py` | blocks |
| `post-applypatch` | yes | `post_applypatch.py` | reports |
| `pre-commit` | yes | `pre_commit.py` | blocks |
| `pre-merge-commit` | yes | `pre_merge_commit.py` | blocks |
| `prepare-commit-msg` | yes | `prepare_commit_msg.py` | reports |
| `commit-msg` | yes | `commit_msg.py` | blocks |
| `post-commit` | yes | `post_commit.py` | reports |
| `pre-rebase` | yes | `pre_rebase.py` | blocks |
| `post-checkout` | yes | `post_checkout.py` | reports |
| `post-rewrite` | yes | `post_rewrite.py` | reports |
| `post-merge` | yes | `post_merge.py` | reports |
| `pre-push` | yes | `pre_push.py` | blocks (Gate 1 within it is advisory) |
| `pre-auto-gc` | yes | `pre_auto_gc.py` | declines housekeeping |

Two of those exist to close gaps in git's own hook *ordering* rather than to add
policy:

* `applypatch-msg` exists because `git am` never runs `commit-msg`, so without
  it a patch series could land with subjects the rest of the gate would reject.
* `pre-applypatch` exists because `git am` never runs `pre-commit` either. A
  hand-made commit and an applied patch would otherwise go through *different*
  content checks; both hooks call the same `_checks.run()`.

The remaining two report rather than judge. `post-applypatch` records the commit
`git am` created into the same event log as `post-commit`, so a patch series is
visible to outcome tracking. `pre-auto-gc` is the only hook where a non-zero
exit is not a failure at all — it tells git to skip its background `gc --auto`
while a gate hook holds the run marker (§6.1).

---

## 6. Where we go beyond upstream

Upstream says nothing about *what* a hook should do. Each of ours carries real
Gatekeeper policy, and every one reuses project tooling rather than restating
it:

| Hook | Policy | Reuses |
|------|--------|--------|
| `pre-commit` | ruff on staged files; conflict markers | `ruff`, git's own `diff --cached --check` |
| `pre-applypatch` | the same staged-content checks, for `git am` | `_checks.run()` |
| `commit-msg` | conventional subjects | allow-list read from this repo's `git log` |
| `applypatch-msg` | the same subject rules for `git am` | `commit_msg.lint_subject` |
| `prepare-commit-msg` | `direct_to_main`; seeds the type list | `.gatekeeper.yml` |
| `pre-merge-commit` | refuses merges into protected branches | `_policy.protected_branches()` |
| `pre-rebase` | refuses rebasing protected branches | same |
| `pre-push` | Gate 1 scoring, diff-scoped ruff, pytest | `scripts/pre_push_score.py` |
| `post-commit` | writes an outcome-tracking record | `_events.record_commit()` |
| `post-applypatch` | the same record, for a `git am` patch | `_events.record_commit()` |
| `post-merge` | records the merge and flags gate changes | `_events`, `.gatekeeper.yml` |
| `post-checkout` | reports branch gate context | same |
| `post-rewrite` | records amended/rebased-away shas | `_events`, `.gatekeeper.yml` |
| `pre-auto-gc` | declines background `gc --auto` during a gate run | `_runtime.gate_run_active()` |

`protected_branches` is new configuration this work added to `.gatekeeper.yml`,
kept separate from the existing `branch_rules` — the latter tunes *which rules*
block on a branch, the former says whether the branch is directly writable at
all.

### 6.1 What has no upstream equivalent at all

Upstream's surface is an installer, a dispatcher and a naming convention. Four
things here are ours, and each exists because a hook is a poor place for
untestable logic:

| Addition | Why | Where |
|----------|-----|-------|
| **A continued trace.** `HUSKY=2` makes husky's sh dispatcher run `set -x`, but the trace stops the instant the hook hands over to Python — which is where every decision is actually made. `rt.enter()`/`rt.trace()` continue it, printing the resolved repo, the interpreter the hook really got, and the argv. | "Which Python did the hook get?" is the first question when a hook behaves differently in two terminals | `_runtime.py` |
| **A gate-run marker.** Long gate hooks drop `.git/gatekeeper/gate_run.json` while they run; `pre-auto-gc` declines to repack the object store underneath them. Relocatable with `GATEKEEPER_STATE_DIR`, and ignored once older than 30 minutes. | `gc --auto` repacking during a multi-minute pre-push is contention, and a lock conflict at worst | `begin_gate_run` / `gate_run_active` |
| **One event log.** `post-commit`, `post-applypatch`, `post-merge` and `post-rewrite` all append to one newline-delimited JSON file, so a reader never has to know which hook wrote a line. | Outcome tracking needs the commit recorded at the one moment only a hook can observe it | `_events.py` |
| **Shared content checks.** `pre-commit` and `pre-applypatch` call the same function. | Duplicating "lint the staged files, look for conflict markers" in two files is how the two drift | `_checks.py` |

All four are inside the git directory, never the work tree: a hook that litters
the checkout with an untracked file would show up in every `git status`, which
is itself something these hooks read.

---

## 7. Documented features (`docs/**`)

| Documented feature | Decision |
|--------------------|----------|
| Startup files: `$XDG_CONFIG_HOME/husky/init.sh`, `~/.config/husky/init.sh`, deprecated `~/.huskyrc` | **Adopted** — relied on for env, documented, and checked by the doctor |
| `HUSKY=0` for one command / a session / a GUI / globally | **Adopted** — plus `GATEKEEPER_SKIP_HOOKS` for per-hook control |
| CI and Docker: `env: HUSKY: 0` | **Adopted** — `install.mjs` skips on `CI=true`, with `HUSKY=1` to override |
| `.husky/install.mjs` so `prepare` never fails without husky | **Adopted** — the same file, extended |
| Testing hooks without committing (`exit 1` in the hook) | **Adopted** — documented |
| `git commit -n/--no-verify` | **Adopted** — documented |
| Non-shell hooks (`.husky/pre-commit` calling `.husky/pre-commit.js`) | **Adopted** — the doctor permits the pattern |
| POSIX-only shell, no bash dependency | **Adopted** — our sh layer is POSIX and linted by shellcheck in CI |
| Bash heredoc pattern for bash-only teams | **Adopted** — documented |
| Node version managers / GUIs and `PATH` | **Adopted** — `init.sh` is the documented answer |
| Yarn on Windows "stdin is not a tty" → `.husky/common.sh` with `winpty` | **Adopted, with a correction** — see §7.1 |
| Sub-directory projects (`prepare: "cd .. && husky frontend/.husky"`) | **Skipped** — see §8 |
| Troubleshoot: hook file *names*, `core.hooksPath`, git > 2.9, `--unset` after uninstall | **Adopted** — all four are doctor checks |

### 7.1 The tty workaround needs a caveat upstream does not state

Upstream's `common.sh` does `exec < /dev/tty` unconditionally when `winpty`
exists. That **replaces stdin**, which would silently discard the payload git
hands a hook. `pre-push` reads ref lines from stdin and `post-rewrite` reads
sha pairs, so applying it globally would break both.

Ours is therefore opt-in via `gatekeeper_restore_tty()`, with a test asserting
neither stdin-reading hook ever calls it.

---

## 8. Deliberately not adopted

| Upstream feature | Why not |
|------------------|---------|
| Custom hooks directory (`husky sub/.husky`) | The hooks here are repo-specific: they import `scripts/hooks/*`, which only exist at the repository root. Supporting a nested directory would mean the modules could not resolve. The doctor reads `core.hooksPath` rather than assuming it, so a moved directory is at least reported |
| Sub-directory project scaffolding | Same reason |
| `husky add` / `set` | Deprecated upstream; creating a file is clearer than a command that templates one |
| Server-side hooks (`pre-receive`, `update`, `post-receive`, `fsmonitor-watchman`, the `p4-*` family) | They do not run in a client repository. Of git's 28 hook names, husky dispatches the 14 that do, and all 14 are implemented here. `KNOWN_GIT_HOOKS` still carries the full 28 so the doctor can tell a real name from a typo |
| `pinst` (Yarn publish-time disable) | This package is `private: true` and is never published |
| Translations of the docs | Not our docs to translate |

---

## 9. Upstream's test suite — executed, not just mapped

The twelve tests are vendored byte-for-byte under
`tools/husky-conformance/upstream/` and run by
`tools/husky-conformance/run.sh`, which packs the husky installed in
`node_modules/` — the bytes a consumer would get — instead of this repository,
because upstream's `test.sh` only works inside upstream's own checkout. See
[`tools/husky-conformance/README.md`](../tools/husky-conformance/README.md) for
provenance and hashes.

```sh
sh tools/husky-conformance/run.sh            # all twelve, about a minute
sh tools/husky-conformance/run.sh --only 9   # one test, for debugging
```

Result: **12/12 pass** locally (`MINGW64_NT-10.0-26200`, node v24.12.0,
husky 9.1.7), and CI runs the same twelve on ubuntu, macOS and Windows
(`.github/workflows/hooks.yml`, job `hooks`). Nothing is listed as allowed to
fail; the driver shouts if a test in that list starts passing, so it cannot
quietly become a bin for real failures.

| Upstream test | What it proves | Status here |
|---------------|----------------|-------------|
| `1_default.sh` | install sets `.husky/_`; a failing hook blocks the commit | **Run.** Ours too: `TestPreMergeCommit`, `TestShimChain`, CI's real-shim step |
| `2_in-sub-dir.sh` | a custom hooks directory is honoured | **Run** (upstream's feature); this repo deliberately does not use one (§8) |
| `3_from-sub-dir.sh` | install from a subdirectory via `prepare` | **Run** (upstream's feature); skipped here (§8) |
| `4_not-git-dir.sh` | install does not fail outside a repo | **Run.** Ours too: `install.mjs` never throws |
| `5_git_command_not_found.sh` | the Node API degrades without git | **Run.** n/a to our hooks — they run *inside* git |
| `6_command_not_found.sh` | exit 127 is reported clearly | **Run.** Ours is earlier and more specific: `TestBootstrapShell` |
| `7_node_modules_path.sh` | `node_modules/.bin` is on `PATH` inside a hook | **Run**, and asserted directly in `tests/test_husky_fidelity.py` |
| `8_set_u.sh` | a `set -u` `init.sh` does not break dispatch | **Run.** CI also sources the bootstrap in a real shell |
| `9_husky_0.sh` | `HUSKY=0` skips install, and `init.sh` can set it | **Run.** Ours too: `TestRuntimeResolution`, the `install.mjs` skip test |
| `10_init.sh` | `husky init` completes | **Run.** Ours: `npm run hooks:init` |
| `11_time.sh` | a hook adds no perceptible commit latency | **Run.** Our advisory hooks are single-digit milliseconds |
| `12_deprecated.sh` | a v8-era shim header warns instead of failing | **Run.** Our own hooks are v9 format only |

The hook layer's own tests are `tests/test_hooks.py` (58, unit),
`tests/test_hook_integration.py` (114, real git repositories and the real
shell) and `tests/test_husky_fidelity.py` (14, the generated `_/` and the
dispatcher's contract) — 186, inside a project suite of 282. Together with
upstream's twelve they cover both halves of one story: ours says what the hooks
decide, upstream's says the chain that reaches them is the chain we assume.

---

## 10. Summary

Upstream gives us a dispatcher, a naming convention, and a set of documented
escape hatches. We take all three, keep the sh layer two lines deep, put every
decision in Python where it can be tested, and implement all 14 hooks it
dispatches rather than stopping at the ones with obvious policy. The three
places we knowingly diverge — diagnosing the interpreter before the hook runs,
making the tty workaround opt-in, and implementing `uninstall` properly — are
all cases where the upstream behaviour was right for a general-purpose tool and
wrong for a hook that has to read git's stdin and fail visibly.

We also do not take the integration on faith: the generated `_/` is compared
byte-for-byte against a fresh install, and upstream's own twelve tests are run
against the husky in `node_modules/` on three operating systems. "We depend on
husky" is checked in both directions — that our copy is husky's, and that
husky still behaves the way this integration reads it.
