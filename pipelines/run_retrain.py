#!/usr/bin/env python3
"""
Run the Gatekeeper retraining pipeline through kfp.local.

This intentionally executes the real @dsl.pipeline graph with
kfp.local.DockerRunner instead of calling component functions directly.

DockerRunner is the KFP-recommended runner for local execution. It runs
each component in an isolated Docker container, providing faithful
environment isolation while remaining local (no Kubernetes needed).

Windows workaround: KFP's DockerRunner has Windows-specific issues:
1. Volume mount strings use colons (source:bind:mode), which conflict
   with Windows drive letter colons (C:).
2. os.path.join() produces backslash paths that break in Linux containers.

This script:
- Uses Docker Mount API (key/value, not colon-delimited) for mounts
- Patches os.path.join within KFP to use forward slashes
- Mounts pipeline_root at /C:<rest-of-path> so executor URIs resolve
  inside Linux containers (C: becomes a valid Linux directory name).
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Windows Docker fixes ---
if sys.platform == "win32":
    # Fix 1: os.path.join → forward slashes (KFP uses it for executor URIs)
    _orig_join = os.path.join

    def _fwd_join(*args):
        return _orig_join(*args).replace("\\", "/")

    os.path.join = _fwd_join

    # Fix 2: Use Docker Mount API instead of volumes dict
    import docker as _docker
    import kfp.local.docker_task_handler as _dth

    _original_run = _dth.run_docker_container

    def _fixed_run(client, image, command, volumes, **kwargs):
        """Convert volumes dict to Docker Mount objects."""
        mounts = []
        for src, cfg in volumes.items():
            src_f = src.replace("\\", "/")
            bind_f = cfg["bind"].replace("\\", "/")
            # The container target must be a Linux-style path.
            # KFP sets both source=bind=pipeline_root (Windows path like C:/Temp/x).
            # On Linux, C: is just a directory name, so mount to /C:/Temp/x.
            if not bind_f.startswith("/"):
                bind_f = "/" + bind_f
            mounts.append(
                _docker.types.Mount(
                    target=bind_f,
                    source=src_f,
                    type="bind",
                    read_only=False,
                )
            )

        kwargs.pop("volumes", None)
        # Ensure container uses UTF-8 encoding (fixes Unicode in PyCaret/KFP logs)
        env = kwargs.pop("environment", None) or {}
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return _original_run(client, image, command, volumes={}, mounts=mounts, environment=env, **kwargs)

    _dth.run_docker_container = _fixed_run

from kfp import local  # noqa: E402

from pipelines.retrain_pipeline import retrain_pipeline  # noqa: E402


def _default_since_date() -> str:
    params_path = PROJECT_ROOT / "params.yaml"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = yaml.safe_load(f) or {}
        if params.get("since"):
            return str(params["since"])
    return (datetime.now(timezone.utc) - timedelta(days=3 * 365)).strftime("%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gatekeeper retraining DAG using kfp.local."
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/django/django.git",
        help="Git repository URL or local path to mine (host-side path).",
    )
    parser.add_argument(
        "--since-date",
        default=_default_since_date(),
        help="Date to start mining commits from (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--label-window-days",
        type=int,
        default=7,
        help="Window for risky commit labeling.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Minimum number of extracted rows required.",
    )
    parser.add_argument(
        "--min-positive-pct",
        type=float,
        default=0.05,
        help="Minimum positive class fraction before warning.",
    )
    parser.add_argument(
        "--pipeline-root",
        default=str(PROJECT_ROOT / "local_outputs" / "retrain"),
        help="Directory for kfp.local task outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Normalize to forward slashes
    pipeline_root = str(Path(args.pipeline_root).resolve()).replace("\\", "/")

    # On Windows, compute the container-side path for repo_url
    # pipeline_root on host: C:/Temp/gk_retrain
    # pipeline_root in container: /C:/Temp/gk_retrain (C: becomes dir name)
    container_root = "/" + pipeline_root if not pipeline_root.startswith("/") else pipeline_root

    print("=" * 60)
    print("Running Gatekeeper retraining DAG with kfp.local.DockerRunner")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Pipeline root (host): {pipeline_root}")
    print(f"Pipeline root (container): {container_root}")
    print()

    local.init(
        runner=local.DockerRunner(),
        pipeline_root=pipeline_root,
        raise_on_error=True,
        enable_caching=False,
    )

    # Convert repo_url to container-accessible path if it's a local path
    repo_url = args.repo_url
    if Path(repo_url).exists():
        # Local path — translate to container-side path
        abs_repo = str(Path(repo_url).resolve()).replace("\\", "/")
        repo_url = "/" + abs_repo
        print(f"Translated repo path to container: {repo_url}")

    retrain_pipeline(
        repo_url=repo_url,
        since_date=args.since_date,
        label_window_days=args.label_window_days,
        min_rows=args.min_rows,
        min_positive_pct=args.min_positive_pct,
    )


if __name__ == "__main__":
    main()
