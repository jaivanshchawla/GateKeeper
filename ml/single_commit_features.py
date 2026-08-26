#!/usr/bin/env python3
"""
Q.2: Compute M.1 features for a single commit at serving time.

Uses ml/m1_shared.py — the SAME functions as bulk extraction.
Uses graph file paths and author names (not PyDriller's) to ensure parity.
Caches the full repo graph per-repo.
"""

from collections import defaultdict
from datetime import datetime, timezone

import json as _json
import os as _os
import subprocess as _subprocess
from pathlib import Path as _Path

# Module-level cache: avoids rebuilding the full graph per commit
_graph_cache: dict[str, dict] = {}       # repo_path -> full graph
_risky_cache: dict[str, set] = {}        # repo_path -> risky hashes
_sorted_cache: dict[str, list] = {}      # repo_path -> sorted graph list
_snapshot_cache: dict[str, list] = {}    # repo_path -> list of (hash, state_dict)

# Must match dataset rebuild parameters exactly
WINDOW_START = "2024-07-01"
FORWARD_LOOK_END = "2026-07-07"
LABEL_WINDOW_DAYS = 7
SNAPSHOT_INTERVAL = 1000  # persist state every N commits
SNAPSHOT_DIR = _Path(__file__).parent.parent / "data" / "graph_snapshots"


def _get_repo_head(repo_path: str) -> str:
    """Get current HEAD sha for cache keying."""
    r = _subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path, capture_output=True, text=True, timeout=10,
    )
    return r.stdout.strip()[:12]


def _load_snapshots(repo_path: str, head_sha: str) -> list | None:
    """Load persisted snapshots for this repo+HEAD. Returns None if not found."""
    snapshot_file = SNAPSHOT_DIR / f"{head_sha}.json"
    if snapshot_file.exists():
        try:
            data = _json.loads(snapshot_file.read_text())
            # Reconstruct datetime objects from ISO strings
            for entry in data:
                if "state" in entry and "author_counts" in entry["state"]:
                    entry["state"]["author_counts"] = {
                        k: v for k, v in entry["state"]["author_counts"].items()
                    }
                if "state" in entry and "file_history" in entry["state"]:
                    entry["state"]["file_history"] = {
                        k: [(h, datetime.fromisoformat(ts), a, m) for h, ts, a, m in v]
                        for k, v in entry["state"]["file_history"].items()
                    }
            return data
        except Exception:
            pass
    return None


def _save_snapshots(repo_path: str, head_sha: str, snapshots: list) -> None:
    """Persist snapshots to disk."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_file = SNAPSHOT_DIR / f"{head_sha}.json"
    # Serialize datetime objects to ISO strings
    serializable = []
    for entry in snapshots:
        s_entry = {"hash": entry["hash"], "idx": entry["idx"]}
        state = entry["state"]
        s_state = {
            "author_counts": dict(state.get("author_counts", {})),
            "file_history": {
                k: [(h, ts.isoformat() if hasattr(ts, 'isoformat') else str(ts), a, m)
                    for h, ts, a, m in v]
                for k, v in state.get("file_history", {}).items()
            },
            "file_risky_counts": dict(state.get("file_risky_counts", {})),
            "file_revert_counts": dict(state.get("file_revert_counts", {})),
            "file_first_seen": {
                k: (ts.isoformat() if hasattr(ts, 'isoformat') else str(ts))
                for k, ts in state.get("file_first_seen", {}).items()
            },
            "author_file_counts": {
                k: dict(v) for k, v in state.get("author_file_counts", {}).items()
            },
            "author_dir_counts": {
                k: dict(v) for k, v in state.get("author_dir_counts", {}).items()
            },
            "author_last_commit": dict(state.get("author_last_commit", {})),
        }
        s_entry["state"] = s_state
        serializable.append(s_entry)
    snapshot_file.write_text(_json.dumps(serializable))


def _get_full_graph(repo_path: str) -> tuple[dict, set, list]:
    """Get the full repo graph, risky hashes, AND sorted list, all cached.

    Returns (graph, risky_hashes, sorted_graph_list).
    The sorted list avoids re-sorting on every walk_graph_to_state call.
    """
    if repo_path not in _graph_cache:
        from ml.m1_shared import build_graph, compute_risky_hashes
        graph = build_graph(repo_path, WINDOW_START, FORWARD_LOOK_END)
        risky = compute_risky_hashes(graph, label_window_days=LABEL_WINDOW_DAYS)
        _graph_cache[repo_path] = graph
        _risky_cache[repo_path] = risky
        _sorted_cache[repo_path] = sorted(graph.items(), key=lambda x: x[1]["date"])
    return _graph_cache[repo_path], _risky_cache[repo_path], _sorted_cache[repo_path]


def clear_cache():
    """Clear all cached graphs. Call after repo updates."""
    _graph_cache.clear()
    _risky_cache.clear()
    _sorted_cache.clear()
    _snapshot_cache.clear()


def compute_single_commit_m1_features(
    repo_path: str,
    commit_hash: str,
    commit_date: datetime,
    author_name: str,
    touched_files: set[str],
    lines_added: int = 0,
    lines_deleted: int = 0,
    since_date: str = None,
) -> dict:
    """Compute all M.1 features for a single commit.

    Calls ml/m1_shared.py — the SAME code path as bulk extraction.
    Uses graph file paths and author names to ensure parity.
    """
    from ml.m1_shared import compute_change_shape, compute_m1_features, walk_graph_to_state

    # Normalize commit_date
    cd = commit_date
    if cd.tzinfo is not None:
        cd = cd.astimezone(timezone.utc).replace(tzinfo=None)

    # Get full graph, risky hashes, AND sorted list (all cached per repo)
    graph, risky_hashes, sorted_graph = _get_full_graph(repo_path)

    # Try to find a snapshot to resume from
    start_index = 0
    start_state = None
    head_sha = _get_repo_head(repo_path)
    cache_key = f"{repo_path}:{head_sha}"

    if cache_key in _snapshot_cache:
        # Use in-memory snapshot cache
        snapshots = _snapshot_cache[cache_key]
        # Find nearest snapshot before target
        for snap in reversed(snapshots):
            if snap["idx"] <= len(sorted_graph) * 0.9:  # Don't use snapshots too close to end
                start_index = snap["idx"] + 1
                start_state = snap["state"]
                break
    else:
        # Try disk cache
        snapshots = _load_snapshots(repo_path, head_sha)
        if snapshots:
            _snapshot_cache[cache_key] = snapshots
            for snap in reversed(snapshots):
                if snap["idx"] <= len(sorted_graph) * 0.9:
                    start_index = snap["idx"] + 1
                    start_state = snap["state"]
                    break

    # Walk graph to build state up to (not including) the target commit.
    state, target_info = walk_graph_to_state(
        graph, risky_hashes,
        stop_hash=commit_hash,
        sorted_graph=sorted_graph,
        start_index=start_index,
        start_state=start_state,
    )

    # Persist snapshot periodically
    if commit_hash in graph:
        target_idx = next((i for i, (h, _) in enumerate(sorted_graph) if h == commit_hash), None)
        if target_idx is not None and target_idx % SNAPSHOT_INTERVAL < 10:
            snapshots_to_save = _snapshot_cache.get(cache_key, [])
            # Check if we already have a snapshot near this index
            existing = [s for s in snapshots_to_save if abs(s["idx"] - target_idx) < SNAPSHOT_INTERVAL // 2]
            if not existing:
                snapshots_to_save.append({"hash": commit_hash, "idx": target_idx, "state": state})
                _snapshot_cache[cache_key] = snapshots_to_save
                _save_snapshots(repo_path, head_sha, snapshots_to_save)

    # CRITICAL: Use graph file paths and author, NOT PyDriller's.
    # PyDriller and git log can produce different paths/names.
    # Bulk extraction uses graph paths, so SC must too.
    graph_author_used = author_name
    if target_info:
        graph_files = target_info.get("files", set())
        graph_author = target_info.get("author", author_name)
        if graph_files:
            touched_files = graph_files
        author_name = graph_author
        graph_author_used = graph_author

    # Compute M.1 features using the shared function
    m1 = compute_m1_features(
        state=state,
        graph=graph,
        target_hash=commit_hash,
        target_date=cd,
        author=author_name,
        files_touched=touched_files,
        is_merge=0,
        risky_hashes=risky_hashes,
    )

    # Compute change-shape features using the shared function
    files_touched_count = len(touched_files)
    shape = compute_change_shape(
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        files_touched_count=files_touched_count,
        touched_files=touched_files,
    )

    # Merge M.1 and change-shape
    result = {**m1, **shape}

    # Remove co-change features (not in current config)
    result.pop("co_change_strength_max", None)
    result.pop("co_change_strength_mean", None)

    # Return graph author so extract_single_commit can override PyDriller's email
    result["_graph_author"] = graph_author_used

    return result
