#!/usr/bin/env python3
"""U.6.6a: Test batch scoring with walk snapshots."""
import time
import sys
sys.path.insert(0, ".")

from ml.single_commit_features import (
    clear_cache, _get_full_graph, _ensure_walk_snapshots,
    batch_score_commits
)

print("=== Building walk snapshots (one-time) ===")
t0 = time.time()
snaps = _ensure_walk_snapshots("repos/rust")
print(f"  Built {len(snaps)} snapshots in {time.time()-t0:.1f}s")

# Reload from disk
clear_cache()
t0 = time.time()
_get_full_graph("repos/rust")
print(f"  Graph load: {time.time()-t0:.3f}s")
t1 = time.time()
snaps = _ensure_walk_snapshots("repos/rust")
print(f"  Snapshots load: {time.time()-t1:.3f}s")

# Test batch scoring near HEAD (20 commits)
print("\n=== Batch 20 commits near HEAD (Rust) ===")
_, _, sg = _get_full_graph("repos/rust")
hashes = [sg[i][0] for i in range(len(sg)-20, len(sg))]

t0 = time.time()
results = batch_score_commits("repos/rust", hashes)
elapsed = time.time() - t0
print(f"  Batch: {elapsed:.2f}s ({elapsed/20*1000:.0f}ms/commit)")
print(f"  Results: {sum(1 for r in results if r)} scored")

# Test 1-commit scoring (hot path)
print("\n=== Single commit (warm, Rust) ===")
t0 = time.time()
from ml.single_commit_features import compute_single_commit_m1_features
from datetime import datetime, timezone
result = compute_single_commit_m1_features(
    "repos/rust", sg[-5][0], sg[-5][1]["date"].replace(tzinfo=timezone.utc),
    "test@test.com", set()
)
print(f"  Single: {time.time()-t0:.3f}s")

# Test 5-commit batch
print("\n=== Batch 5 commits (Rust) ===")
hashes5 = [sg[i][0] for i in range(len(sg)-5, len(sg))]
t0 = time.time()
results5 = batch_score_commits("repos/rust", hashes5)
elapsed5 = time.time() - t0
print(f"  Batch: {elapsed5:.2f}s ({elapsed5/5*1000:.0f}ms/commit)")

# Summary table
print("\n=== U.6.6a FINAL TABLE ===")
print(f"{'Repo':<12} {'Pickle':<10} {'Cold':<10} {'Warm':<10} {'Batch20':<12}")
print("-" * 54)

for repo in ["django", "react", "kafka", "rust"]:
    clear_cache()
    t0 = time.time()
    g, r, s = _get_full_graph(repo)
    pickle_time = time.time() - t0
    n_commits = len(g)
    hashes = [s[i][0] for i in range(max(0, len(s)-20), len(s))]
    t1 = time.time()
    res = batch_score_commits(repo, hashes)
    batch_time = time.time() - t1
    print(f"{repo:<12} {pickle_time:.3f}s{'':<5} {'':<10} {'':<10} {batch_time:.2f}s ({batch_time/20*1000:.0f}ms/commit)")
