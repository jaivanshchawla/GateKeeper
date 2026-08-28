#!/usr/bin/env python3
"""U.6.6a: Profile graph build, persist to disk, batch walk."""
import cProfile
import pstats
import io
import time
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

REPO = str(Path("repos/rust").resolve())
WINDOW_START = "2024-07-01"
FORWARD_LOOK_END = "2026-07-07"

def profile_graph_build():
    """Profile build_graph to find where time is spent."""
    print("=" * 60)
    print("U.6.6a: Profile graph build on Rust")
    print("=" * 60)

    from ml.m1_shared import build_graph

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    graph = build_graph(REPO, WINDOW_START, FORWARD_LOOK_END)
    elapsed = time.time() - t0
    pr.disable()

    print(f"\nGraph build: {elapsed:.1f}s, {len(graph)} commits")

    # Print top 20 by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

    # Also profile per-stage
    fmt = "%H|%ct|%aE|%s"
    print("\n--- Stage timing ---")

    t1 = time.time()
    r1 = subprocess.run(
        ["git", "log", f"--since={WINDOW_START}", f"--until={FORWARD_LOOK_END}",
         f"--pretty=format:{fmt}", "--name-only", "--no-merges", "HEAD"],
        cwd=REPO, capture_output=True, timeout=600,
        encoding="utf-8", errors="replace",
    )
    t_log = time.time() - t1
    lines = r1.stdout.split("\n")
    print(f"  git log --name-only: {t_log:.1f}s, {len(lines)} lines")

    t2 = time.time()
    for line in lines:
        parts = line.split("|", 3)
        if len(parts) == 4 and len(parts[0]) == 40:
            _ = int(parts[1])
    t_parse = time.time() - t2
    print(f"  Parse loop: {t_parse:.3f}s")

    # Merge pass
    t3 = time.time()
    r2 = subprocess.run(
        ["git", "log", f"--since={WINDOW_START}", f"--until={FORWARD_LOOK_END}",
         f"--pretty=format:{fmt}", "--merges", "HEAD"],
        cwd=REPO, capture_output=True, timeout=600,
        encoding="utf-8", errors="replace",
    )
    t_merge_log = time.time() - t3
    print(f"  git log --merges: {t_merge_log:.1f}s")

    # resolve_author calls
    t4 = time.time()
    from ml.m1_shared import resolve_author, normalize_author_id
    for line in lines[:100]:
        parts = line.split("|", 3)
        if len(parts) == 4 and len(parts[0]) == 40:
            raw_email = parts[2]
            _ = resolve_author(REPO, normalize_author_id(raw_email))
    t_resolve_100 = time.time() - t4
    print(f"  resolve_author x100: {t_resolve_100:.3f}s ({t_resolve_100*len(graph)/100:.1f}s total)")

    # risky_hashes computation
    t5 = time.time()
    from ml.m1_shared import compute_risky_hashes
    risky = compute_risky_hashes(graph)
    t_risky = time.time() - t5
    print(f"  compute_risky_hashes: {t_risky:.3f}s, {len(risky)} risky")

    # Sort
    t6 = time.time()
    sorted_graph = sorted(graph.items(), key=lambda x: x[1]["date"])
    t_sort = time.time() - t6
    print(f"  Sort graph: {t_sort:.3f}s")

    return graph, risky, sorted_graph


def persist_graph(graph, risky, sorted_graph, repo_path):
    """Persist graph to disk as JSON, keyed on HEAD sha."""
    head_r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path,
                            capture_output=True, timeout=10, encoding="utf-8", errors="replace")
    head_sha = head_r.stdout.strip()
    print(f"\nHEAD: {head_sha}")

    snap_dir = Path(repo_path).parent / "data" / "graph_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    cache_file = snap_dir / f"graph_{head_sha[:12]}.json"

    # Serialize
    t0 = time.time()
    serializable = {}
    for h, info in graph.items():
        serializable[h] = {
            "date": info["date"].isoformat(),
            "files": info["files"],
            "subject": info["subject"],
            "author": info["author"],
            "is_merge": info["is_merge"],
        }

    payload = {
        "head_sha": head_sha,
        "window_start": WINDOW_START,
        "forward_look_end": FORWARD_LOOK_END,
        "graph": serializable,
        "risky": list(risky),
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False))
    t_write = time.time() - t0
    size_mb = cache_file.stat().st_size / (1024 * 1024)
    print(f"Persisted graph: {size_mb:.1f}MB, {t_write:.1f}s -> {cache_file}")

    # Now time loading it back
    t1 = time.time()
    loaded = json.loads(cache_file.read_text())
    t_load = time.time() - t1
    print(f"Load from disk: {t_load:.1f}s, {len(loaded['graph'])} commits")

    # Reconstruct graph from loaded
    t2 = time.time()
    restored_graph = {}
    for h, info in loaded["graph"].items():
        restored_graph[h] = {
            "date": datetime.fromisoformat(info["date"]),
            "files": info["files"],
            "subject": info["subject"],
            "author": info["author"],
            "is_merge": info["is_merge"],
        }
    restored_risky = set(loaded["risky"])
    t_restore = time.time() - t2
    print(f"Restore graph objects: {t_restore:.3f}s")
    print(f"Total cold start from disk: {t_load + t_restore:.1f}s (vs {size_mb:.1f}MB git log)")

    return head_sha, cache_file


def batch_walk_test(graph, risky, sorted_graph, repo_path, n_commits=20):
    """Walk ONCE for N commits, emit features at each boundary."""
    print(f"\n--- Batch walk: {n_commits} commits ---")
    from ml.m1_shared import walk_graph_to_state

    # Pick the last N commits
    targets = [(h, v) for h, v in sorted_graph[-n_commits:]]

    t0 = time.time()
    # Walk once from beginning to end
    state, _ = walk_graph_to_state(
        graph, risky, stop_hash=None, stop_date=None,
        sorted_graph=sorted_graph,
    )
    t_full_walk = time.time() - t0
    print(f"  Full walk (all {len(sorted_graph)} commits): {t_full_walk:.1f}s")

    # Now walk to each target
    t1 = time.time()
    prev_idx = 0
    for h, v in targets:
        idx = next(i for i, (hh, _) in enumerate(sorted_graph) if hh == h)
        # Walk from prev to this target
        walk_graph_to_state(
            graph, risky, stop_hash=h,
            sorted_graph=sorted_graph,
            start_index=prev_idx,
            start_state=state if prev_idx > 0 else None,
        )
        prev_idx = idx
    t_batch = time.time() - t1
    print(f"  Batch walk to {n_commits} targets: {t_batch:.1f}s ({t_batch/n_commits:.2f}s/commit)")

    # Sequential walk (what current code does)
    t2 = time.time()
    for h, v in targets:
        walk_graph_to_state(
            graph, risky, stop_hash=h,
            sorted_graph=sorted_graph,
            start_index=0,
            start_state=None,
        )
    t_sequential = time.time() - t2
    print(f"  Sequential walk (no cache): {t_sequential:.1f}s ({t_sequential/n_commits:.2f}s/commit)")

    return t_batch, t_sequential


if __name__ == "__main__":
    graph, risky, sorted_graph = profile_graph_build()
    head_sha, cache_file = persist_graph(graph, risky, sorted_graph, REPO)
    t_batch, t_sequential = batch_walk_test(graph, risky, sorted_graph, REPO)

    print("\n" + "=" * 60)
    print("U.6.6a SUMMARY")
    print("=" * 60)
    print(f"  Graph build (git log): see above")
    print(f"  Persist to disk: see above")
    print(f"  Batch walk 20 commits: {t_batch:.1f}s")
    print(f"  Sequential walk 20 commits: {t_sequential:.1f}s")
