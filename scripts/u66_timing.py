#!/usr/bin/env python3
"""U.6.6a: Time graph loading (pickle vs git log) and extraction."""
import time
import sys
sys.path.insert(0, ".")

# Django first (should be fast)
print("=== Django ===")
t0 = time.time()
from ml.single_commit_features import _get_full_graph, clear_cache
clear_cache()
g, r, s = _get_full_graph("repos/django")
print(f"  Cold: {time.time()-t0:.1f}s, {len(g)} commits")

t1 = time.time()
g, r, s = _get_full_graph("repos/django")
print(f"  Warm: {time.time()-t1:.3f}s")

# React
print("\n=== React ===")
clear_cache()
t0 = time.time()
g, r, s = _get_full_graph("repos/react")
print(f"  Cold: {time.time()-t0:.1f}s, {len(g)} commits")

# Rust
print("\n=== Rust ===")
clear_cache()
t0 = time.time()
g, r, s = _get_full_graph("repos/rust")
cold = time.time() - t0
print(f"  Cold: {cold:.1f}s, {len(g)} commits")

t1 = time.time()
g, r, s = _get_full_graph("repos/rust")
warm = time.time() - t1
print(f"  Warm: {warm:.3f}s")

# Test single-commit extraction on Rust
print("\n=== Single-commit extraction (Rust) ===")
from ml.single_commit_features import _precompute_author_prior
t0 = time.time()
apc = _precompute_author_prior("repos/rust")
print(f"  Precompute author_prior: {time.time()-t0:.3f}s")

from ml.single_commit_features import compute_single_commit_m1_features
from datetime import datetime, timezone

# Pick a commit in the middle of the sorted graph
target = s[len(s)//2][0]
print(f"  Target: {target[:12]}")

t0 = time.time()
result = compute_single_commit_m1_features(
    repo_path="repos/rust",
    commit_hash=target,
    commit_date=s[len(s)//2][1]["date"].replace(tzinfo=timezone.utc),
    author_name="test@test.com",
    touched_files=set(),
)
print(f"  Cold extract: {time.time()-t0:.1f}s")

t1 = time.time()
result2 = compute_single_commit_m1_features(
    repo_path="repos/rust",
    commit_hash=s[len(s)//2 + 1][0],
    commit_date=s[len(s)//2 + 1][1]["date"].replace(tzinfo=timezone.utc),
    author_name="test@test.com",
    touched_files=set(),
)
print(f"  Warm extract: {time.time()-t1:.3f}s")

# Batch score 20 commits
print("\n=== Batch 20 commits (Rust) ===")
hashes = [s[i][0] for i in range(len(s)-20, len(s))]
from ml.single_commit_features import batch_score_commits
t0 = time.time()
results = batch_score_commits("repos/rust", hashes)
print(f"  Batch: {time.time()-t0:.1f}s ({(time.time()-t0)/20:.2f}s/commit)")

# Kafka
print("\n=== Kafka ===")
clear_cache()
t0 = time.time()
g, r, s = _get_full_graph("repos/kafka")
print(f"  Cold: {time.time()-t0:.1f}s, {len(g)} commits")
