#!/usr/bin/env python3
"""
Q.2: Compute M.1 features for a single commit at serving time.

Uses ml/m1_shared.py — the SAME functions as bulk extraction.
Uses graph file paths and author names (not PyDriller's) to ensure parity.
Caches the full repo graph per-repo.
"""

from collections import defaultdict
from datetime import datetime, timezone

# Module-level cache: avoids rebuilding the full graph per commit
_graph_cache: dict[str, dict] = {}       # repo_path -> full graph
_risky_cache: dict[str, set] = {}        # repo_path -> risky hashes
_sorted_cache: dict[str, list] = {}      # repo_path -> sorted graph list

# Must match dataset rebuild parameters exactly
WINDOW_START = "2024-07-01"
FORWARD_LOOK_END = "2026-07-07"
LABEL_WINDOW_DAYS = 7


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

    # Walk graph to build state up to (not including) the target commit.
    # CRITICAL: use stop_hash, NOT stop_date. Multiple commits can share
    # the same timestamp (e.g. 3 commits at 2025-11-05T12:20:57), so
    # stop_date stops at the WRONG commit, including the target's own
    # changes in the state. stop_hash gives exact matching.
    # Pass pre-sorted graph to avoid O(n log n) sort on every call.
    state, target_info = walk_graph_to_state(
        graph, risky_hashes,
        stop_hash=commit_hash,
        sorted_graph=sorted_graph,
    )

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
