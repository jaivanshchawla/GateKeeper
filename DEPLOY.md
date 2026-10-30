# Deploying Gatekeeper

Three paths, in increasing order of how long you want the thing to live:

| Path | Best for | Persistence | Cost |
| --- | --- | --- | --- |
| [Local Docker Compose](#1-local-docker-compose) | development, screenshots, the college demo | named volume | free |
| [Render Blueprint](#2-render-blueprint) | a URL to share today | **none** (ephemeral disk) | free, sleeps when idle |
| [Oracle Cloud Always Free](#3-oracle-cloud-always-free-recommended) | the real deployment | boot volume, always on | free forever |

Every path runs the same image: one multi-stage `Dockerfile` that builds the
dashboard with Node, then serves the API and the dashboard from one FastAPI
process on one port. No proxy, no CORS, no second service to keep alive.

---

## 1. Local Docker Compose

```bash
docker compose up -d --build      # http://localhost:8000
curl localhost:8000/health
```

The model, the mined corpus, the artifact files and the dashboard are all baked
into the image, so this works from a clean clone with no `pip install` and no
`npm install`.

Optional extras:

```bash
docker compose --profile postgres up -d        # Postgres instead of SQLite
docker compose --profile observability up -d   # Prometheus :9090, Grafana :3100
```

**Sign-in is off** unless you set Clerk variables:

```bash
CLERK_PUBLISHABLE_KEY=pk_test_... CLERK_ISSUER=https://your-app.clerk.accounts.dev docker compose up -d
```

With either variable set, the API rejects anonymous requests, `/auth/config`
tells the dashboard to render the sign-in card, and `/auth/me` is the endpoint
that proves a session works.

---

## 2. Render Blueprint

`render.yaml` is committed and points at the root `Dockerfile`.

1. Render dashboard → **New +** → **Blueprint** → pick this repository.
2. Apply. Render asks for `CLERK_ISSUER` and `CLERK_PUBLISHABLE_KEY` because
   they are `sync: false` — leave them blank to deploy without sign-in.
3. First build is 8-15 minutes; the image installs LightGBM, SHAP and MLflow.

Set the repository secret `DEPLOYED_API_URL` (your Render URL, no trailing
slash) so `.github/workflows/keep_warm.yml` keeps the free instance awake.

**The limitation that matters:** Render's free plan has an ephemeral
filesystem. `data/gatekeeper.db` — issues, risk budget — resets on every
deploy. The model and every artifact are in the image and are unaffected, so
the dashboard is fully functional. If you need those rows to survive, use
Oracle (below) or point `DATABASE_URL` at a free managed Postgres.

---

## 3. Oracle Cloud Always Free (recommended)

Oracle's Always Free tier is the only free option that gives a real always-on
machine: **4 ARM OCPUs / 24 GB RAM / 200 GB storage / 10 TB egress, forever**.
That is enough to run the API, the dashboard, Prometheus and Grafana on one
box with no cold starts and no sleeping services.

```
deploy/oracle/
├── main.tf                    # VCN, subnet, security list, ARM instance
├── variables.tf               # shape, size, keys, Clerk values
├── cloud-init.yaml.tftpl      # installs Docker, writes the stack, starts it
├── terraform.tfvars.example   # copy to terraform.tfvars and fill in
└── outputs.tf                 # public IP, URL, SSH command
```

### One-time setup

1. **Create an API key** in the Oracle console (Profile → API keys → Add).
   Download the private key to `~/.oci/oci_api_key.pem`.
2. **Install Terraform**, or use the container:
   `docker run --rm -it -v "$PWD:/tf" -w /tf hashicorp/terraform:1.9 ...`
3. Copy the variables file and fill it in:

   ```bash
   cd deploy/oracle
   cp terraform.tfvars.example terraform.tfvars
   ```

   `ssh_cidr` must be *your* address (`curl ifconfig.me`) with a `/32` — the
   security list never opens port 22 to the world.
4. Apply:

   ```bash
   terraform init && terraform apply
   ```

   `terraform validate` passes on this configuration as committed.

### What happens at boot

`cloud-init` installs Docker from Docker's repository, enables `ufw` for 22/80/443,
writes `/opt/gatekeeper/docker-compose.yml` and a `Caddyfile`, pulls the image
published by `.github/workflows/deploy.yml`, and starts the stack. It also
enables a `gatekeeper.service` systemd unit so the stack comes back after a
reboot.

Caddy terminates TLS. If you set `domain`, Caddy gets a certificate for it. If
you leave it empty, cloud-init resolves the instance's public IP at boot and
uses `<ip-with-dashes>.sslip.io`, which resolves to that IP — so you get a
**real HTTPS certificate without owning a domain**.

### Deploys after that

Push to `finisher` (or `main` once you promote it). The `release` workflow
builds `linux/amd64` **and** `linux/arm64` images to GHCR, then SSHes to the VM,
pulls, restarts, and waits for the container's health check to report healthy.
ARM is not optional: Oracle's free compute is Ampere, and an amd64-only image
fails there with `exec format error` *after* a successful pull.

Add these repository secrets to enable the deploy job (the job skips cleanly
when they are absent):

| Secret | Value |
| --- | --- |
| `DEPLOY_HOST` | the instance's public IP |
| `DEPLOY_SSH_KEY` | the private key matching `ssh_public_key_path` |
| `DEPLOY_USER` | `ubuntu` (default) |

### Cost and limits, honestly

- The instance is free permanently, but **capacity is the catch**: if A1
  capacity is unavailable in your region, `terraform apply` fails with
  `OutOfHostCapacity`. Retry, or try another availability domain.
- Ephemeral public IPs change if you stop and start the instance. Reserve the
  IP in the console (Networking → IP management) before pointing a domain at it.
- 10 TB/month egress and 200 GB block storage are far beyond what this uses.

### Migrating away

Nothing here is Oracle-specific except the VM. The image is a plain container,
the database is SQLite on a volume, and Caddy is configuration. To move:
`docker save`/`compose up` the same image on any host with 2 GB of RAM and a
public IP — Fly.io, Hetzner, a Raspberry Pi, or another free tier. There is no
lock-in to unwind.

---

## The npm CLI (optional, for the repositories you gate)

```bash
npx gatekeeper-cli init
npx gatekeeper-cli check --json
gatekeeper config --api-url https://your-deployment
gatekeeper auth --token "$SESSION_JWT"     # when Clerk is on
```

`npm pack` in `cli/` produces the tarball; publishing needs an npm account and
`npm publish` from that directory (the package is `gatekeeper-cli`, and the
name is currently unclaimed on the registry).

---

## Verification

The deployment work is checked in CI, not by hand:

- `tests/test_ci_workflows.py` asserts there is exactly one Dockerfile, that it
  builds the dashboard in a Node stage, that compose and Render both use it,
  and that `.dockerignore` keeps the sources and the mined corpus in the build
  context.
- `tests/test_auth.py` signs real RS256 tokens against a locally generated
  keypair and proves the API rejects anything else.
- `scripts/ci/verify_all.py` runs the whole matrix locally, which is the same
  thing CI runs.
