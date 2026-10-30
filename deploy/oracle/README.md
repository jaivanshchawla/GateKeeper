# Oracle Cloud Always Free deployment

One ARM VM, one container, one volume. This is the long-term deployment: it is
free permanently, always on, and has enough RAM (12 GB of the 24 GB allowance)
to run the API plus Prometheus and Grafana without swapping.

## Why this and not a PaaS free tier

| | Oracle A1 (this) | Render free | Fly.io free |
| --- | --- | --- | --- |
| Machine sleeps | never | after 15 min | scales to zero |
| RAM | 12 GB | 512 MB | 256 MB |
| Persistent disk | 50 GB boot volume | none | 3 GB volume |
| ARM | yes (must build arm64) | no | yes |
| Egress | 10 TB/month | 100 GB | 100 GB |

The only real cost is that you own the operating system. `cloud-init.yaml.tftpl`
and `docker-compose.yml` written at boot keep that surface small.

## Apply

```bash
cp terraform.tfvars.example terraform.tfvars
# fill in the tenancy/user/fingerprint/region, your ssh_cidr, and the image
terraform init
terraform validate
terraform apply
```

`terraform validate` passes as committed (checked with Terraform 1.9).

The first boot takes 2-4 minutes: apt, Docker, then the image pull. Follow it
with:

```bash
ssh ubuntu@<public_ip> tail -f /var/log/gatekeeper-bootstrap.log
```

The URL from `terraform output url` answers once Caddy has its certificate.

## What the VM runs

- `api` — the Gatekeeper image, SQLite on a named volume, not published on the
  host (Caddy is the only thing listening on 80/443).
- `caddy` — TLS termination and HTTP→HTTPS redirect, with HSTS/nosniff headers.

## Operations

```bash
cd /opt/gatekeeper
docker compose ps
docker compose logs -f api
docker compose pull api && docker compose up -d   # what CI does
```

CI (`.github/workflows/deploy.yml`) does the pull/restart on every push to the
default branch and fails the job if the container does not report healthy.

## Common failures

| Symptom | Cause | Fix |
| --- | --- | --- |
| `OutOfHostCapacity` on apply | A1 capacity in that AD | retry, try another AD, or reduce ocpus to 1 |
| `exec format error` | image built only for amd64 | CI builds `linux/arm64` too; check the tag |
| Caddy cannot get a certificate | ports 80/443 closed | OCI security list **and** `ufw` both allow them; cloud-init sets ufw |
| Dashboard loads, API returns 401 | Clerk enabled, no session | sign in, or clear `CLERK_ISSUER` and redeploy |
| Data disappears after recreate | `terraform apply` replaced the instance | the volume lives on the boot volume; keep `ignore_changes = [metadata]` and do not taint the instance |

## Teardown

```bash
terraform destroy
```

That removes the VCN, subnet, security list and instance. GHCR keeps the image,
and `docker compose` files on the VM go with it — nothing else is left behind.
