#!/usr/bin/env python3
"""
Run the Gatekeeper retraining pipeline through kfp.local.

Windows Docker compatibility for KFP's DockerRunner.

Key problem: KFP uses os.path.join() to construct executor URIs.
On Windows, this produces C:\\Temp\\... (backslashes) which break in Linux
containers. We patch os.path.join to produce /C:/Temp/... (forward slashes,
absolute Linux paths). But KFP on the host also uses these paths to read
output files — and /C:/... is invalid on Windows.

Solution: patch both os.path.join AND builtins.open:
- os.path.join: C:\\Temp\\... → /C:/Temp/... (for container)
- builtins.open: /C:/Temp/... → C:/Temp/... (for host-side reading)

IMPORTANT: All patches are applied AFTER local.init() because it uses
tempfile which would break with the patched os.path.join.
"""

import argparse
import builtins
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Docker Mount fix is safe to apply at import time (no os.path interaction)
if sys.platform == "win32":
    import docker as _docker
    import kfp.local.docker_task_handler as _dth

    _original_run = _dth.run_docker_container

    def _fixed_run(client, image, command, volumes, **kwargs):
        """Convert volumes dict to Docker Mount objects."""
        mounts = []
        for src, cfg in volumes.items():
            src_f = src.replace("\\", "/")
            bind_f = cfg["bind"].replace("\\", "/")
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


def _backup_mlflow_db() -> None:
    """Back up the MLflow tracking database before running the pipeline.

    SAFETY: This prevents accidental data loss if someone runs
    'rm -f mlflow.db' or the pipeline overwrites the DB.
    The backup is mlflow.db.backup-<YYYYMMDD-HHMMSS>.
    Never delete or overwrite the tracking database directly.
    """
    import shutil

    db_path = PROJECT_ROOT / "mlflow.db"
    if not db_path.exists():
        print("No mlflow.db found — nothing to back up.")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_path = PROJECT_ROOT / f"mlflow.db.backup-{ts}"
    shutil.copy2(db_path, backup_path)
    print(f"Backed up mlflow.db -> {backup_path.name} ({backup_path.stat().st_size:,} bytes)")

    # Clean up old backups: keep only the 5 most recent
    backups = sorted(PROJECT_ROOT.glob("mlflow.db.backup-*"), key=lambda p: p.stat().st_mtime)
    while len(backups) > 5:
        oldest = backups.pop(0)
        oldest.unlink()
        print(f"  Removed old backup: {oldest.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gatekeeper retraining DAG using kfp.local."
    )
    parser.add_argument("--repo-url", default="https://github.com/django/django.git")
    parser.add_argument("--since-date", default=_default_since_date())
    parser.add_argument("--label-window-days", type=int, default=7)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-positive-pct", type=float, default=0.05)
    parser.add_argument("--pipeline-root", default=str(PROJECT_ROOT / "local_outputs" / "retrain"))
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset MLflow tracking DB before running (creates backup first, then deletes).",
    )
    return parser.parse_args()


def _normalize_path(path_str: str) -> str:
    """Normalize a Windows path to forward slashes, stripping extended-length prefix."""
    resolved = str(Path(path_str).resolve())
    resolved = resolved.removeprefix("\\\\?\\")
    return resolved.replace("\\", "/")


def _translate_c_drive(path):
    """Translate /C:/... to C:/... on Windows host."""
    if isinstance(path, str) and len(path) >= 4 and path.startswith("/"):
        translated = path[1:]
        if len(translated) >= 2 and translated[1] == ":":
            return translated
    return path


def _apply_windows_patches():
    """Apply patches for Docker container compatibility.

    os.path.join: produces /C:/... paths (valid inside Linux containers)
    os.path.exists/isfile/open: translates /C:/... back to C:/... (valid on Windows host)
    """
    _orig_join = os.path.join

    def _fwd_join(*args):
        result = _orig_join(*args).replace("\\", "/")
        if len(result) >= 2 and result[1] == ":" and not result.startswith("/"):
            result = "/" + result
        return result

    os.path.join = _fwd_join

    _orig_exists = os.path.exists
    _orig_isfile = os.path.isfile
    _orig_open = builtins.open
    _orig_getsize = os.path.getsize

    def _host_exists(path):
        return _orig_exists(_translate_c_drive(path))

    def _host_isfile(path):
        return _orig_isfile(_translate_c_drive(path))

    def _host_open(file, *args, **kwargs):
        return _orig_open(_translate_c_drive(file), *args, **kwargs)

    def _host_getsize(path):
        return _orig_getsize(_translate_c_drive(path))

    os.path.exists = _host_exists
    os.path.isfile = _host_isfile
    builtins.open = _host_open
    os.path.getsize = _host_getsize


def main() -> None:
    args = parse_args()
    pipeline_root = _normalize_path(args.pipeline_root)

    print("=" * 60)
    print("Running Gatekeeper retraining DAG with kfp.local.DockerRunner")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Pipeline root: {pipeline_root}")

    # SAFETY: Always back up mlflow.db before running the pipeline.
    # This is the #1 guard against accidental data loss.
    # Never delete mlflow.db without an explicit --reset flag.
    _backup_mlflow_db()

    if args.reset:
        db_path = PROJECT_ROOT / "mlflow.db"
        if db_path.exists():
            db_path.unlink()
            print("--reset: Deleted mlflow.db (backup preserved above).")
        else:
            print("--reset: No mlflow.db to reset.")

    print()

    # Step 1: Initialize KFP (calls tempfile.mkdtemp — must NOT be patched yet)
    local.init(
        runner=local.DockerRunner(),
        pipeline_root=pipeline_root,
        raise_on_error=True,
        enable_caching=False,
    )

    # Step 2: Apply Docker compatibility patches AFTER init
    if sys.platform == "win32":
        _apply_windows_patches()

    # Step 2b: Set up cached CSV if data/commit_features.csv is fresh
    cached_csv_container_path = ""
    host_csv = PROJECT_ROOT / "data" / "commit_features.csv"
    if host_csv.exists():
        import shutil

        # Copy to pipeline root so Docker containers can access it
        pipeline_root_dir = Path(pipeline_root)
        pipeline_root_dir.mkdir(parents=True, exist_ok=True)
        dest = pipeline_root_dir / "cached_features.csv"
        shutil.copy2(host_csv, dest)
        # Inside the container, pipeline_root is at /<normalized_pipeline_root>
        cached_csv_container_path = "/" + str(dest).replace("\\", "/")
        print(f"Cached CSV copied to container path: {cached_csv_container_path}")

    repo_url = args.repo_url
    if Path(repo_url).exists():
        abs_repo = _normalize_path(repo_url)
        repo_url = "/" + abs_repo
        print(f"Translated repo path to container: {repo_url}")

    retrain_pipeline(
        repo_url=repo_url,
        since_date=args.since_date,
        label_window_days=args.label_window_days,
        min_rows=args.min_rows,
        min_positive_pct=args.min_positive_pct,
        cached_csv_path=cached_csv_container_path,
    )


if __name__ == "__main__":
    main()
