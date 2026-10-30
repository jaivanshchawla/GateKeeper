# syntax=docker/dockerfile:1
#
# One image that serves the API and the dashboard from the same origin.
#
# Two stages, because only one of them has to be large: the dashboard is built
# with Node and the runtime never sees Node or node_modules. The previous
# Dockerfile copied a *prebuilt* dashboard/dist, which is git-ignored and
# therefore absent in a clean clone -- `docker build` failed for anyone who had
# not run `npm run build` first.

# ── Dashboard ───────────────────────────────────────────────────────────────
FROM node:22-alpine AS dashboard
WORKDIR /ui

# Lockfile first: this layer is cached until the dependencies actually change.
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY dashboard/index.html dashboard/vite.config.js ./
COPY dashboard/src ./src

# VITE_API_URL is deliberately unset: the SPA calls its own origin, and the API
# serves it, so no build-time knowledge of the deployment URL is needed.
RUN npm run build

# ── Runtime ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# libgomp1 is LightGBM's OpenMP runtime; git and curl are used by the file
# history endpoints and the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements-slim.txt ./requirements-slim.txt
RUN pip install -r requirements-slim.txt

# Exactly the packages the API imports. Listing them one by one keeps the image
# honest: a new top-level package that is not copied is a build failure, not a
# 500 in production.
COPY api/ ./api/
COPY auth/ ./auth/
COPY artifacts/ ./artifacts/
COPY registry/ ./registry/
COPY ml/ ./ml/
COPY models/ ./models/
COPY policy/ ./policy/
COPY rules/ ./rules/
COPY webhook/ ./webhook/

# The data the read-only endpoints serve. The rest of data/ is 258 MB of mined
# archives and graphs that nothing at runtime opens.
#
# `data/commit_features.csv` is deliberately absent: it is git-ignored (DVC
# tracks it), so a build from a clean clone -- which is what CI does -- failed
# on this COPY with "/data/commit_features.csv: not found". Every endpoint that
# reads it is guarded by `if csv_path.exists()`, so the image serves without it
# and the training-corpus numbers are simply not claimed in that deployment.
COPY data/scored_prs.json ./data/
COPY data/u69_*_oow.json ./data/

COPY --from=dashboard /ui/dist ./dashboard/dist

# Non-root. /app is owned by the runtime user so SQLite (DATABASE_URL=
# sqlite:////app/data/gatekeeper.db) and the outcome log stay writable.
RUN useradd --create-home --uid 10001 gatekeeper \
    && chown -R gatekeeper:gatekeeper /app
USER gatekeeper

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

EXPOSE 8000
ENTRYPOINT ["/app/api/entrypoint.sh"]
