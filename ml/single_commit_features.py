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
_author_prior_cache: dict[str, dict] = {}  # repo_path -> {hash: count}

# Hot walk cache: persists state between calls for sequential commits.
# For batch/simulator/backfill, consecutive commits walk from here.
_hot_state: dict[str, dict] = {}  # repo_path -> {"idx": int, "state": dict}

# Must match dataset rebuild parameters exactly
WINDOW_START = "2024-07-01"
FORWARD_LOOK_END = "2026-07-07"
LABEL_WINDOW_DAYS = 7
SNAPSHOT_INTERVAL = 5000  # persist state every N commits
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

    Tries pickle cache first (0.1s on Rust), falls back to git log (11s).
    Returns (graph, risky_hashes, sorted_graph_list).
    """
    if repo_path not in _graph_cache:
        # Try loading from pickle cache first
        loaded = _try_load_pickle(repo_path)
        if loaded is not None:
            _graph_cache[repo_path] = loaded["graph"]
            _risky_cache[repo_path] = loaded["risky"]
            _sorted_cache[repo_path] = loaded["sorted"]
            return _graph_cache[repo_path], _risky_cache[repo_path], _sorted_cache[repo_path]

        # Fall back to git log
        from ml.m1_shared import build_graph, compute_risky_hashes
        graph = build_graph(repo_path, WINDOW_START, FORWARD_LOOK_END)
        risky = compute_risky_hashes(graph, label_window_days=LABEL_WINDOW_DAYS)
        _graph_cache[repo_path] = graph
        _risky_cache[repo_path] = risky
        _sorted_cache[repo_path] = sorted(graph.items(), key=lambda x: x[1]["date"])
        # Persist for next time
        _try_save_pickle(repo_path)
    return _graph_cache[repo_path], _risky_cache[repo_path], _sorted_cache[repo_path]


def _try_load_pickle(repo_path: str) -> dict | None:
    """Try to load graph from pickle cache. Returns None if not found or stale."""
    import pickle as _pickle
    head_sha = _get_repo_head(repo_path)
    cache_file = SNAPSHOT_DIR / f"graph_{head_sha[:12]}.pkl"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            data = _pickle.load(f)
        # Validate window matches
        if data.get("window_start") != WINDOW_START or data.get("forward_look_end") != FORWARD_LOOK_END:
            return None
        return data
    except Exception:
        return None


def _try_save_pickle(repo_path: str) -> None:
    """Persist graph to pickle for next cold start."""
    import pickle as _pickle
    head_sha = _get_repo_head(repo_path)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = SNAPSHOT_DIR / f"graph_{head_sha[:12]}.pkl"
    try:
        with open(cache_file, "wb") as f:
            _pickle.dump({
                "graph": _graph_cache[repo_path],
                "risky": _risky_cache[repo_path],
                "sorted": _sorted_cache[repo_path],
                "head": head_sha,
                "window_start": WINDOW_START,
                "forward_look_end": FORWARD_LOOK_END,
            }, f, protocol=_pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _precompute_author_prior(repo_path: str) -> dict[str, int]:
    """Precompute author_prior_commits for ALL commits in the graph.

    Walks the sorted graph once, tracking per-author commit counts.
    Returns {commit_hash: author_prior_count} for every commit.
    This replaces count_authors_before() subprocess calls at scoring time.
    """
    if repo_path in _author_prior_cache:
        return _author_prior_cache[repo_path]

    graph, risky_hashes, sorted_graph = _get_full_graph(repo_path)

    result: dict[str, int] = {}
    author_total: dict[str, int] = defaultdict(int)

    for h, v in sorted_graph:
        author = v.get("author", "")
        result[h] = author_total[author]
        author_total[author] += 1

    _author_prior_cache[repo_path] = result
    return result


def clear_cache():
    """Clear all cached graphs. Call after repo updates."""
    _graph_cache.clear()
    _risky_cache.clear()
    _sorted_cache.clear()
    _snapshot_cache.clear()
    _author_prior_cache.clear()
    _hot_state.clear()


def _ensure_walk_snapshots(repo_path: str) -> list[dict]:
    """Ensure walk-state snapshots exist for this repo, building if needed.

    Returns list of {"idx": int, "state": dict} sorted by idx.
    Each snapshot covers SNAPSHOT_INTERVAL commits in the sorted graph.
    """
    import pickle as _pickle
    head_sha = _get_repo_head(repo_path)
    snap_file = SNAPSHOT_DIR / f"walk_{head_sha[:12]}.pkl"

    if snap_file.exists():
        try:
            with open(snap_file, "rb") as f:
                snaps = _pickle.load(f)
            if snaps and len(snaps) > 0 and "state" in snaps[0]:
                return snaps
        except Exception:
            pass

    from ml.m1_shared import walk_graph_to_state
    graph, risky_hashes, sorted_graph = _get_full_graph(repo_path)
    snapshots = []
    state = None
    prev_idx = 0

    for snap_point in range(SNAPSHOT_INTERVAL, len(sorted_graph), SNAPSHOT_INTERVAL):
        target_hash = sorted_graph[snap_point][0]
        state, _ = walk_graph_to_state(
            graph, risky_hashes, stop_hash=target_hash,
            sorted_graph=sorted_graph, start_index=prev_idx, start_state=state,
        )
        # Convert to plain dicts — fast, pickle-friendly, no deepcopy needed
        plain = {
            "file_change_count": dict(state["file_change_count"]),
            "file_risky_count": dict(state["file_risky_count"]),
            "file_revert_count": dict(state["file_revert_count"]),
            "file_first_seen": dict(state["file_first_seen"]),
            "file_last_touch_hash": dict(state["file_last_touch_hash"]),
            "file_authors": {k: list(v) for k, v in state["file_authors"].items()},
            "author_state": {
                a: {"files": dict(s["files"]), "dirs": dict(s["dirs"]),
                    "last_date": s["last_date"], "total": s.get("total", 0)}
                for a, s in state["author_state"].items()
            },
            "co_change": {k: v for k, v in state["co_change"].items()},
        }
        snapshots.append({"idx": snap_point, "state": plain})
        prev_idx = snap_point + 1

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(snap_file, "wb") as f:
            _pickle.dump(snapshots, f, protocol=_pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

    return snapshots


def _find_nearest_snapshot(snapshots: list[dict], target_idx: int) -> tuple[int, dict | None]:
    """Find the nearest snapshot before target_idx. Returns (start_index, state)."""
    best = None
    for snap in snapshots:
        if snap["idx"] < target_idx:
            best = snap
    if best:
        return best["idx"] + 1, best["state"]
    return 0, None


def _state_needs_conversion(state: dict) -> bool:
    """Check if state dicts are defaultdicts (needs conversion for walk resume)."""
    fcc = state.get("file_change_count", {})
    return hasattr(fcc, "default_factory") if fcc else False


def batch_score_commits(
    repo_path: str,
    commit_hashes: list[str],
) -> list[dict]:
    """Score multiple commits in a single graph walk (fast for PRs).

    Uses persistent walk snapshots to skip to the nearest checkpoint.
    For Rust (66K commits), this reduces a 10s walk to ~2s.
    """
    from ml.m1_shared import compute_change_shape, compute_m1_features, walk_graph_to_state
    import subprocess as _sp

    graph, risky_hashes, sorted_graph = _get_full_graph(repo_path)
    _precompute_author_prior(repo_path)

    # Build index for sorted_graph
    sorted_idx = {h: i for i, (h, _) in enumerate(sorted_graph)}

    # Sort commits by position in sorted_graph
    indexed = [(i, h) for h in commit_hashes if (i := sorted_idx.get(h)) is not None]
    indexed.sort(key=lambda x: x[0])

    if not indexed:
        return [{}] * len(commit_hashes)

    # Find nearest snapshot for the FIRST commit
    snapshots = _ensure_walk_snapshots(repo_path)
    first_idx = indexed[0][0]
    start_index, start_state = _find_nearest_snapshot(snapshots, first_idx)

    state = start_state
    prev_idx = start_index
    result_map = {}

    for target_idx, target_hash in indexed:
        if target_idx < prev_idx:
            continue  # shouldn't happen after sorting, but safety
        state, target_info = walk_graph_to_state(
            graph, risky_hashes, stop_hash=target_hash,
            sorted_graph=sorted_graph, start_index=prev_idx, start_state=state,
        )
        if target_info:
            touched_files = target_info.get("files", set())
            author = target_info.get("author", "")
            m1 = compute_m1_features(
                state=state, graph=graph, target_hash=target_hash,
                target_date=target_info["date"], author=author,
                files_touched=touched_files, is_merge=1 if target_info.get("is_merge") else 0,
                risky_hashes=risky_hashes,
            )
            shape = compute_change_shape(0, 0, len(touched_files), touched_files)
            result = {**m1, **shape}
            result.pop("co_change_strength_max", None)
            result.pop("co_change_strength_mean", None)
            result["_graph_author"] = author
            apc = _author_prior_cache.get(repo_path, {})
            result["_author_prior_commits"] = apc.get(target_hash, 0)
            result_map[target_hash] = result
        prev_idx = target_idx + 1

    # Update hot cache
    if indexed:
        last_idx, _ = indexed[-1]
        _hot_state[repo_path] = {"idx": last_idx, "state": state}

    return [result_map.get(h, {}) for h in commit_hashes]


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
    # Precompute author_prior_commits if not already done
    _precompute_author_prior(repo_path)
    from ml.m1_shared import compute_change_shape, compute_m1_features, walk_graph_to_state

    # Normalize commit_date
    cd = commit_date
    if cd.tzinfo is not None:
        cd = cd.astimezone(timezone.utc).replace(tzinfo=None)

    # Get full graph, risky hashes, AND sorted list (all cached per repo)
    graph, risky_hashes, sorted_graph = _get_full_graph(repo_path)

    # Try to find a state to resume from: hot cache > snapshot cache > from scratch
    start_index = 0
    start_state = None
    head_sha = _get_repo_head(repo_path)
    cache_key = f"{repo_path}:{head_sha}"

    # 1. Hot cache: in-memory state from last call (fastest path)
    if repo_path in _hot_state:
        hs = _hot_state[repo_path]
        if hs["idx"] < len(sorted_graph):
            # Check if the target is after the hot position
            target_idx = next((i for i, (h, _) in enumerate(sorted_graph) if h == commit_hash), None)
            if target_idx is not None and target_idx > hs["idx"]:
                start_index = hs["idx"] + 1
                start_state = hs["state"]

    # 2. Walk snapshots (persistent pickle, skips large gaps)
    if start_state is None:
        walk_snaps = _ensure_walk_snapshots(repo_path)
        target_idx = next((i for i, (h, _) in enumerate(sorted_graph) if h == commit_hash), None)
        if target_idx is not None:
            start_index, start_state = _find_nearest_snapshot(walk_snaps, target_idx)

    # Walk graph to build state up to (not including) the target commit.
    state, target_info = walk_graph_to_state(
        graph, risky_hashes,
        stop_hash=commit_hash,
        sorted_graph=sorted_graph,
        start_index=start_index,
        start_state=start_state,
    )

    # Update hot cache for next call
    if commit_hash in graph:
        target_idx = next((i for i, (h, _) in enumerate(sorted_graph) if h == commit_hash), None)
        if target_idx is not None:
            _hot_state[repo_path] = {"idx": target_idx, "state": state}

    # Persist snapshot periodically
    if commit_hash in graph:
        target_idx = target_idx if target_idx is not None else next((i for i, (h, _) in enumerate(sorted_graph) if h == commit_hash), None)
        if target_idx is not None and target_idx % SNAPSHOT_INTERVAL < 10:
            snapshots_to_save = _snapshot_cache.get(cache_key, [])
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

    # Precomputed author_prior_commits from the graph walk.
    # This replaces the subprocess-based count_authors_before call.
    apc_cache = _author_prior_cache.get(repo_path, {})
    result["_author_prior_commits"] = apc_cache.get(commit_hash, 0)

    return result
