# gatekeeper-cli

Score the risk of every commit you push, before you push it.

The CLI installs a pre-push hook that runs two things on the commits you are
about to send:

1. **Deterministic rules** — change size, files touched, tests included,
   config-and-code in one commit, weekend deploys, direct pushes to `main`.
   These always run, offline, and they decide whether a push is blocked.
2. **The risk model** — when a Gatekeeper API is configured, each commit's
   features are scored by the same LightGBM model the dashboard serves, and
   the returned band and top SHAP factors are printed.

If the API is unreachable the hook fails **open**: it warns and blocks only on
rules. A network blip must never be able to stop a push.

## Install

```bash
npx gatekeeper-cli init           # inside a git repository
# or
npm install --save-dev gatekeeper-cli && npx gatekeeper init
# or
npm install -g gatekeeper-cli
```

`init` writes `.gatekeeper.yml`, installs `.git/hooks/pre-push` (unless husky
owns `core.hooksPath`, in which case it says so and defers), and registers the
repository name in `~/.gatekeeper/config.json`.

## Commands

| Command | What it does |
| --- | --- |
| `gatekeeper init` | Write config, install the hook, register the repo |
| `gatekeeper check [--json]` | Score unpushed commits; `--json` for CI |
| `gatekeeper status` | Open findings for this repository |
| `gatekeeper explain <sha>` | Why one commit scored the way it did |
| `gatekeeper config [--api-url <url>]` | Show or set configuration |
| `gatekeeper auth --token <jwt>` | Save a Clerk session token (`--clear` to remove) |
| `gatekeeper whoami` | Who the API thinks you are |
| `gatekeeper --version` | CLI version |

`check` exits `1` only when a rule whose severity is `block` fires **and**
`fail_on` includes `block`. Model bands are advisory: a high band is printed,
not enforced, unless you add `high` to `fail_on`.

## Pointing it at an API

```bash
gatekeeper config --api-url https://gatekeeper.example.com
```

or set `GATEKEEPER_API_URL`. The CLI computes the model's feature vector from
the local diff using the same formulas as `ml/history_features.py`
(`churn_ratio`, `change_entropy`, `max_file_churn`, `is_test_only`,
`test_to_code_ratio`, `config_touch`), so scores are comparable with the
dashboard's. Features that need repository history the CLI does not have are
left at the model's defaults rather than guessed.

## Authentication

A deployment with Clerk enabled rejects anonymous requests. Either:

```bash
gatekeeper auth --token "$(your session jwt)"   # stored in ~/.gatekeeper/config.json
```

or, in CI, export `GATEKEEPER_TOKEN`. `gatekeeper whoami` confirms the token.
A rejected token stops `check` with a clear message instead of silently
degrading.

## CI usage

```yaml
- run: npx gatekeeper-cli check --json > gatekeeper.json
  env:
    GATEKEEPER_API_URL: ${{ vars.GATEKEEPER_API_URL }}
    GATEKEEPER_TOKEN: ${{ secrets.GATEKEEPER_TOKEN }}
```

The JSON shape is stable:

```json
{
  "version": "1.1.0",
  "repo": "owner/name",
  "commits": [
    { "sha": "...", "band": "high", "score": 0.93, "rule_results": [], "explanations": ["..."] }
  ],
  "summary": { "total": 1, "passed": 0, "warnings": 1, "blocked": 0 }
}
```

## Development

```bash
npm test        # node:test, no network, no git checkout required
npm pack        # the tarball a published release would ship
```
