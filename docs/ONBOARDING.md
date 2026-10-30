# Onboarding a repository

Adding a repository used to be five manual steps spread over four files. The
registry made the *declaration* one line of YAML; this pipeline makes the whole
decision one command:

```bash
python -m pipelines.onboard https://github.com/owner/repo.git
```

It either leaves you with a registry entry, per-repository band thresholds and a
report in `artifacts/onboarding/<name>.json`, or  it refuses the repository and
tells you which measurement failed. There is no third outcome.

## The five stages

| Stage | What happens | Failure |
| --- | --- | --- |
| 1. resolve | derive a registry name from the URL; `git ls-remote` proves the repository is public and reachable | `OnboardError`, nothing cloned |
| 2. clone | full clone (not shallow: the label needs history after the tip) into `repos/<name>` | `OnboardError` |
| 3. score | score a sample of real commits with the production model, and measure what actually happened next | one unscoreable commit is recorded, not fatal |
| 4. gate | four gates decide whether the history can be corpus | `passed: false`, registry untouched |
| 5. register | append the entry to `registry/repos.yaml` and the thresholds to `ml/config.yaml`, then reload both | rolled back if either fails to reload |

## What "score" measures

For each sampled commit the pipeline computes the same 35-feature vector the
API uses and asks the production model for `P(risky)`. Then it looks *forward*
from the commit, for the same seven-day window the training label uses:

- was any file this commit touched changed again inside the window?
- was this commit named in a revert inside the window?

Commits newer than the window have no observable outcome yet, so they are
excluded from the rate rather than counted as safe.

`precision_at_high` is the fraction of high-band commits that really did have a
bad outcome. It is the one number that says whether the band means anything on
this repository, so it is reported even when it is uncomfortable.

## The gates

| Gate | Requirement | Why the repository is refused |
| --- | --- | --- |
| `enough_commits_scored` | ≥ 25 scored | a handful of commits cannot show a distribution |
| `model_discriminates` | ≥ 5 distinct scores and spread ≥ 0.02 | identical scores mean the model cannot rank this history |
| `outcomes_are_observable` | ≥ 10 commits with a full window | a history younger than the label window cannot be labelled |
| `label_rate_is_usable` | 2% ≤ risky rate ≤ 95% | 0% or 100% risky is a broken label, not an easy corpus |

Thresholds are deliberately low: a gate only large projects pass would refuse
the small repositories this pipeline exists to onboard.

## Calibration, and why bands were meaningless before

`ml/config.yaml` gives each corpus repository `high` at the 90th percentile of
its scores and `medium` at the 75th. A repository without an entry falls back to
`_global`, and on a new history that fallback labels **everything low**.

Onboarding therefore calibrates from the sample and writes the entry:

```
dummy-service, 40 commits scored
  global fallback     {'low': 40}
  calibrated          {'high': 3, 'medium': 7, 'low': 30}
  thresholds          {'high': 0.5147, 'medium': 0.3775}
```

The write is append-only so the file's comments survive, and it is verified by
reloading: if the entry does not come back, the file is restored.

## Flags

| Flag | Use |
| --- | --- |
| `--role train\|validate\|holdout` | how the corpus uses it. Default `validate`: score it, do not train on it until it has earned that |
| `--dry-run` | run everything but the registry write — the way to evaluate a candidate |
| `--local PATH` | score an existing checkout instead of cloning, and skip the remote check. This is how the pipeline runs offline |
| `--since`, `--sample` | the window and the number of commits to score |
| `--name` | override the derived registry name |
| `--registry PATH` | append to a different registry file (tests use this) |

## Demonstrated on a dummy repository

The fixture is generated, not downloaded, so the demonstration is reproducible:

```bash
python -m tools.make_demo_repo --force          # demo/dummy-service, 76 commits
python -m pipelines.onboard \
    https://github.com/example/dummy-service.git \
    --local demo/dummy-service --dry-run
```

Result (`artifacts/onboarding/dummy-service.json`):

```
[3/5] score    up to 40 commits since 2023-01-01
[4/5] gate
      pass  enough_commits_scored: 40
      pass  model_discriminates: {'distinct': 31, 'spread': 0.39}
      pass  outcomes_are_observable: 40
      pass  label_rate_is_usable: 0.1
      thresholds {'high': 0.5147, 'medium': 0.3775} from 40 scores
dummy-service: ACCEPTED
  band distribution  {'high': 3, 'medium': 7, 'low': 30}
  precision @ high   0.33
```

The same command against a public repository with three commits is refused, and
says why:

```bash
python -m pipelines.onboard https://github.com/octocat/Hello-World.git \
    --name hello-world-demo --since 2010-01-01 --dry-run
```

```
      FAIL  enough_commits_scored: 2 (>= 25 commits scored)
      FAIL  model_discriminates: {'distinct': 2, 'spread': 0.2522}
      FAIL  outcomes_are_observable: 2 (>= 10 commits with a full outcome window)
      FAIL  label_rate_is_usable: 0.0 (0.02 <= risky rate <= 0.95)
hello-world-demo: REJECTED          # exit status 1
```

Both reports are committed under `artifacts/onboarding/`.

## Adding a repository without the pipeline

Still supported, and still one file:

```yaml
  - name: flask
    url: https://github.com/pallets/flask.git
    clone_dir: repos/flask
    role: validate
```

Then run `python -m scripts.ci.verify_all`. The pipeline's advantage is not the
YAML line — it is the gates and the calibration, which are the parts a human
forgets.
