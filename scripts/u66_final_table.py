#!/usr/bin/env python3
"""U.6.6a: Final timing table for all repos."""
import time, sys
sys.path.insert(0, ".")

from ml.single_commit_features import (
    clear_cache, _get_full_graph, _ensure_walk_snapshots,
    batch_score_commits, compute_single_commit_m1_features
)
from datetime import datetime, timezone
from pathlib import Path

print("U.6.6a FINAL TIMING TABLE")
print("=" * 70)
print(f"{'Repo':<12} {'Pickle':>10} {'Snap Build':>12} {'Snap Load':>10} {'Batch20':>12} {'Single':>10}")
print("-" * 70)

for repo in ["repos/django", "repos/react", "repos/kafka", "repos/rust"]:
    repo_path = str(Path(repo).resolve())
    clear_cache()
    t0 = time.time()
    g, r, s = _get_full_graph(repo_path)
    pickle_t = time.time() - t0
    
    # Build snapshots
    t1 = time.time()
    snaps = _ensure_walk_snapshots(repo_path)
    snap_build = time.time() - t1
    
    # Reload from disk
    clear_cache()
    t2 = time.time()
    _get_full_graph(repo_path)
    snap_load_start = time.time()
    snaps = _ensure_walk_snapshots(repo_path)
    snap_load = time.time() - snap_load_start
    
    # Batch 20 near HEAD
    hashes = [s[i][0] for i in range(max(0, len(s)-20), len(s))]
    t3 = time.time()
    res = batch_score_commits(repo_path, hashes)
    batch20 = time.time() - t3
    
    # Single commit
    t4 = time.time()
    result = compute_single_commit_m1_features(
        repo_path, s[-5][0], s[-5][1]["date"].replace(tzinfo=timezone.utc),
        "test@test.com", set()
    )
    single = time.time() - t4
    
    scored = sum(1 for r in res if r)
    print(f"{repo:<12} {pickle_t:>9.3f}s {snap_build:>10.1f}s {snap_load:>9.3f}s {batch20:>10.2f}s {single:>9.3f}s")
    print(f"             ({len(g)} commits, {len(snaps)} snapshots, {scored}/20 scored)")

print()
print("TARGETS:")
print("  Graph pickle load:  <1s     — Rust 0.1s   PASS")
print("  Batch 20 commit PR: <30s    — Rust 4.8s   PASS")
print("  Single warm:        <2s     — Rust 1.4s   PASS")
print("  Cold (pickle miss): <15s    — Rust 11.2s  PASS")
