#!/usr/bin/env python3
"""
U.6.8a: Permanent performance regression tests for Rust extraction.

Rust's extraction has regressed THREE times after correctness fixes:
1. U.6.0: PyDriller removed, encoding fix
2. U.6.7a: author_prior_commits changed to full-history index
3. U.6.8a: walk snapshot loading (65MB pickle -> individual files)

These tests run on every commit to prevent a fourth regression.
"""
import subprocess
import time
from datetime import datetime

import pytest


REPO_PATH = "repos/rust"
SMALL_REPOS = ["repos/django", "repos/react", "repos/kafka"]

# Performance targets (seconds)
RUST_SINGLE_WARM_MAX = 2.0   # single commit, snapshot loaded
RUST_BATCH20_MAX = 30.0      # 20-commit batch
SMALL_REPO_COLD_MAX = 5.0    # cold start for small repos


def _get_non_merge_commit(repo_path: str, since: str, until: str) -> str:
    """Get a non-merge commit hash within the date range."""
    result = subprocess.check_output(
        ["git", "log", "--no-merges", "--format=%H|%ct",
         f"--since={since}", f"--until={until}"],
        cwd=repo_path, text=True, timeout=30,
    ).strip().split("\n")
    return result[0].split("|")[0]


class TestRustPerformance:
    """Performance regression tests for Rust (66K commits)."""

    def test_single_commit_warm(self):
        """Rust single commit with snapshot must complete in <2s."""
        from ml.single_commit_features import (
            clear_cache, compute_single_commit_m1_features,
            _get_full_graph, _precompute_author_prior, _ensure_walk_snapshots,
            _hot_state,
        )
        clear_cache()

        # Build everything (uses pickles if available)
        _get_full_graph(REPO_PATH)
        _precompute_author_prior(REPO_PATH)
        _ensure_walk_snapshots(REPO_PATH)

        # Pick a non-merge commit near end of window
        h = _get_non_merge_commit(REPO_PATH, "2026-05-01", "2026-06-15")
        ts = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", h],
            cwd=REPO_PATH, text=True, timeout=10,
        ).strip()
        cd = datetime.fromtimestamp(int(ts))

        # Clear hot state but keep all caches
        _hot_state.clear()

        t0 = time.time()
        feats = compute_single_commit_m1_features(
            REPO_PATH, h, cd, "test@test.com", {"src/lib.rs"}, 10, 5
        )
        elapsed = time.time() - t0

        assert len(feats) > 0, "Features should not be empty"
        assert elapsed < RUST_SINGLE_WARM_MAX, (
            f"Rust single commit took {elapsed:.2f}s, budget {RUST_SINGLE_WARM_MAX}s"
        )

    def test_batch20_commits(self):
        """Rust batch 20 commits must complete in <30s."""
        from ml.single_commit_features import (
            clear_cache, batch_score_commits,
            _get_full_graph, _precompute_author_prior, _ensure_walk_snapshots,
            _hot_state,
        )
        clear_cache()
        _get_full_graph(REPO_PATH)
        _precompute_author_prior(REPO_PATH)
        _ensure_walk_snapshots(REPO_PATH)

        # Get 20 non-merge commits near end of window
        hashes = subprocess.check_output(
            ["git", "log", "--no-merges", "--format=%H",
             "--since=2026-04-01", "--until=2026-06-15"],
            cwd=REPO_PATH, text=True, timeout=30,
        ).strip().split("\n")[:20]

        _hot_state.clear()
        t0 = time.time()
        results = batch_score_commits(REPO_PATH, hashes)
        elapsed = time.time() - t0

        scored = sum(1 for r in results if r)
        assert scored == len(hashes), f"Only scored {scored}/{len(hashes)} commits"
        assert elapsed < RUST_BATCH20_MAX, (
            f"Rust batch 20 took {elapsed:.2f}s, budget {RUST_BATCH20_MAX}s"
        )


class TestSmallRepoPerformance:
    """Performance tests for small repos (django, react, kafka)."""

    @pytest.mark.parametrize("repo_path", SMALL_REPOS)
    def test_cold_start(self, repo_path):
        """Small repo cold start must complete in <5s."""
        from ml.single_commit_features import clear_cache, _get_full_graph
        clear_cache()

        t0 = time.time()
        graph, _, _ = _get_full_graph(repo_path)
        elapsed = time.time() - t0

        assert len(graph) > 0, f"Graph should not be empty for {repo_path}"
        assert elapsed < SMALL_REPO_COLD_MAX, (
            f"{repo_path} cold start took {elapsed:.2f}s, budget {SMALL_REPO_COLD_MAX}s"
        )
