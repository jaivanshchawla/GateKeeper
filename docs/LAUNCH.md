# Launch checklist

Everything that is built is below the line "Needs your values". Everything
above it is done, committed on `finisher`, and verified locally — the exact
commands that verified it are in each section.

## Done and verified

| Area | State | Verified by |
| --- | --- | --- |
| Artifact-truth layer | every metric `/model/health`, `/drift`, `/repos` returns comes from `artifacts/*.json`; a number without a source fails `tests/test_artifact_truth.py` | `pytest tests/test_artifact_truth.py` |
| Repo registry | `registry/repos.yaml` is the only place a repository is declared; eight hardcoded lists are gone | `pytest tests/test_registry.py` |
| Clerk auth (server) | `auth/` verifies a Clerk session JWT, exposes `/auth/*`, `401`s anonymous callers when enabled | `pytest tests/test_auth.py` |
| Clerk auth (dashboard) | sign-in screen and token attach in `dashboard/src` | `npm --prefix dashboard run build` |
| Repo onboarding | `python -m pipelines.onboard <url>` clones, scores, gates, registers, calibrates | `pytest tests/test_onboard.py` |
| CLI package | `gatekeeper-cli@1.1.0`, 8 files, 12.6 kB tarball | `npm pack`, install of the tarball, `gatekeeper --version` |
| CLI → API end to end | three unpushed commits in the dummy repo scored with rules + model + SHAP | see below |
| Docker image | one image serves API and dashboard; multi-arch | `docker build`, `docker run` healthcheck |
| Terraform (Oracle) | `terraform validate` clean | `terraform init && terraform validate` |
| Deploy workflow | pushes to GHCR, restarts the VM only when secrets exist | file assertion in `tests/test_ci_workflows.py` |
| npm publish workflow | tag `cli-v*` publishes; green without a token | `pytest tests/test_ci_workflows.py` |

### The demo that was actually run

```bash
python -m tools.make_demo_repo --force                    # 76-commit fixture
python -m pipelines.onboard https://github.com/example/dummy-service.git \
    --local demo/dummy-service --dry-run                  # ACCEPTED
python -m pipelines.onboard https://github.com/octocat/Hello-World.git \
    --name hello-world-demo --since 2010-01-01 --dry-run  # REJECTED
```

With the API on `127.0.0.1:8010` and the packed CLI installed:

```
🛡️  Gatekeeper: scoring 3 unpushed commit(s) (rules, plus the model when the API answers)
  ✅ f3171ca3 Handlers: fix the payload shape (score 0.346)
     [WARN] no_tests: No test file touched
     [WARN] direct_to_main: Commit lands directly on main
     ↳ Author has 35 prior commits (elevates risk)
     ...
```

## Needs your values

Each item is a credential or a setting only the account owner can provide. The
code is ready; nothing else is required from this side.

### 1. Clerk — sign-in

1. Create an application at <https://dashboard.clerk.com>.
2. Copy the **Publishable key** (`pk_test_…` / `pk_live_…`) and the **Issuer**
   URL (Frontend API, `https://<slug>.clerk.accounts.dev`).
3. Set these wherever the app runs:

   | Variable | Value |
   | --- | --- |
   | `CLERK_ISSUER` | the issuer URL |
   | `CLERK_PUBLISHABLE_KEY` | the publishable key |
   | `CLERK_AUTHORIZED_PARTIES` | the deployed origin, e.g. `https://ip-1-2-3-4.sslip.io` |
   | `GATEKEEPER_AUTH_MODE` | `on` |

   Locally they go in `.env` (already git-ignored). On the VM they are Terraform
   variables (`clerk_issuer`, `clerk_publishable_key`). On GitHub they are
   repository secrets used by the deploy workflow.
4. The secret key is **not** needed: this stack only verifies sessions, it never
   calls Clerk's backend API.

### 2. npm — publish the CLI

1. Create an automation token at <https://www.npmjs.com/settings/~/tokens>.
2. Add it as the repository secret `NPM_TOKEN`.
3. Release:

   ```bash
   # bump cli/package.json version first
   git tag cli-v1.1.0 && git push origin cli-v1.1.0
   ```

   The workflow tests, packs, installs the tarball to prove the `bin` works,
   checks the tag matches `package.json`, then publishes. Without `NPM_TOKEN`
   it packs only and stays green.

### 3. Oracle Cloud — the long-term free deployment

1. Create an API key in the OCI console and download the PEM.
2. ```bash
   cd deploy/oracle
   cp terraform.tfvars.example terraform.tfvars
   # fill in tenancy/user/fingerprint/region/compartment, ssh_cidr, image
   terraform init && terraform apply
   ```
3. Add the SSH deploy secrets so CI can restart it: `DEPLOY_SSH_KEY`,
   `DEPLOY_HOST`, `DEPLOY_USER` (`ubuntu`).
4. Capacity for `VM.Standard.A1.Flex` is often exhausted in a region; reducing
   `ocpus` to 1 or trying another availability domain is the documented fix.

### 4. The default branch

`finisher` is the branch everything is built on. Making it the default and
renaming `main` to `starter` is a GitHub-side setting:

```bash
gh api -X POST repos/jaivanshchawla/GateKeeper/branches/finisher/rename -f new_name=finisher   # no-op, names the branch
gh api -X PATCH repos/jaivanshchawla/GateKeeper -f default_branch=finisher
gh api -X POST repos/jaivanshchawla/GateKeeper/branches/main/rename -f new_name=starter
```

Order matters: set the default to `finisher` first, or GitHub refuses to rename
the branch that is still default.

## Deliberately not built

Named here so they are decisions rather than omissions:

- **Clerk organisations / per-user dashboard data.** Auth identifies the caller
  and protects the API; rows are not yet scoped per user. That is the next real
  feature, not a loose end — it changes the schema.
- **A hosted onboarding service.** Onboarding runs where the corpus lives. A web
  form that accepts a URL would need a job queue, which is a different
  deployment shape than one VM.
- **Kubeflow / Airflow.** The retraining DAG is KFP-local and the CI is GitHub
  Actions; a cluster scheduler buys nothing at this corpus size and costs the one
  thing the free tier optimises for.
- **Billing.** Nothing here is metered.
