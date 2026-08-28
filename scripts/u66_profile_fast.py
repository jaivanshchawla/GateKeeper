#!/usr/bin/env python3
"""U.6.6a: Fast profiling of just the git log stages."""
import time
import subprocess
from pathlib import Path

REPO = str(Path("repos/rust").resolve())
WINDOW_START = "2024-07-01"
FORWARD_LOOK_END = "2026-07-07"
fmt = "%H|%ct|%aE|%s"

print("=" * 60)
print("U.6.6a FAST: Stage timing on Rust")
print("=" * 60)

# Stage 1: git log --name-only --no-merges
t1 = time.time()
r1 = subprocess.run(
    ["git", "log", f"--since={WINDOW_START}", f"--until={FORWARD_LOOK_END}",
     f"--pretty=format:{fmt}", "--name-only", "--no-merges", "HEAD"],
    cwd=REPO, capture_output=True, timeout=600,
    encoding="utf-8", errors="replace",
)
t_log = time.time() - t1
stdout = r1.stdout
lines = stdout.split("\n")
print(f"Stage 1 - git log --name-only --no-merges: {t_log:.1f}s, {len(lines)} lines, {len(stdout)} bytes")

# Stage 2: parse loop
t2 = time.time()
ch = None
cf = []
ct = 0
ca = ""
cs = ""
graph_count = 0
for line in lines:
    line = line.rstrip()
    if not line:
        continue
    parts = line.split("|", 3)
    if len(parts) == 4 and len(parts[0]) == 40:
        if ch is not None:
            graph_count += 1
        ch = parts[0]
        ct = int(parts[1])
        ca = parts[2]
        cs = parts[3]
        cf = []
    else:
        cf.append(line)
if ch is not None:
    graph_count += 1
t_parse = time.time() - t2
print(f"Stage 2 - Parse loop: {t_parse:.3f}s, {graph_count} commits")

# Stage 3: git log --merges
t3 = time.time()
r2 = subprocess.run(
    ["git", "log", f"--since={WINDOW_START}", f"--until={FORWARD_LOOK_END}",
     f"--pretty=format:{fmt}", "--merges", "HEAD"],
    cwd=REPO, capture_output=True, timeout=600,
    encoding="utf-8", errors="replace",
)
t_merge = time.time() - t3
merge_lines = r2.stdout.strip().split("\n")
merge_count = sum(1 for l in merge_lines if len(l.split("|", 3)) == 4 and len(l.split("|", 3)[0]) == 40)
print(f"Stage 3 - git log --merges: {t_merge:.1f}s, {merge_count} merges")

# Stage 4: resolve_author (loading .mailmap)
t4 = time.time()
from ml.m1_shared import resolve_author, normalize_author_id
for line in lines[:200]:
    parts = line.split("|", 3)
    if len(parts) == 4 and len(parts[0]) == 40:
        _ = resolve_author(REPO, normalize_author_id(parts[2]))
t_resolve = time.time() - t4
print(f"Stage 4 - resolve_author x200: {t_resolve:.3f}s (est total: {t_resolve * graph_count / 200:.1f}s)")

# Stage 5: risky hashes
t5 = time.time()
# Build minimal graph for risky computation
from collections import defaultdict
from datetime import datetime, timezone
graph = {}
ch2 = None
cf2 = []
ct2 = 0
ca2 = ""
cs2 = ""
for line in lines:
    line = line.rstrip()
    if not line:
        continue
    parts = line.split("|", 3)
    if len(parts) == 4 and len(parts[0]) == 40:
        if ch2 is not None:
            resolved = resolve_author(REPO, normalize_author_id(ca2))
            graph[ch2] = {
                "date": datetime.fromtimestamp(ct2, tz=timezone.utc).replace(tzinfo=None),
                "files": cf2, "subject": cs2, "author": resolved, "is_merge": False,
            }
        ch2 = parts[0]
        ct2 = int(parts[1])
        ca2 = parts[2]
        cs2 = parts[3]
        cf2 = []
    else:
        cf2.append(line)
if ch2 is not None:
    resolved = resolve_author(REPO, normalize_author_id(ca2))
    graph[ch2] = {
        "date": datetime.fromtimestamp(ct2, tz=timezone.utc).replace(tzinfo=None),
        "files": cf2, "subject": cs2, "author": resolved, "is_merge": False,
    }

from ml.m1_shared import compute_risky_hashes
risky = compute_risky_hashes(graph, 7)
t_risky = time.time() - t5
print(f"Stage 5 - compute_risky_hashes: {t_risky:.3f}s, {len(risky)} risky")

# Stage 6: Sort
t6 = time.time()
sorted_graph = sorted(graph.items(), key=lambda x: x[1]["date"])
t_sort = time.time() - t6
print(f"Stage 6 - Sort: {t_sort:.3f}s")

# Stage 7: Write to disk as pickle (not JSON)
import pickle
t7 = time.time()
snap_dir = Path(REPO).parent / "data" / "graph_snapshots"
snap_dir.mkdir(parents=True, exist_ok=True)
head_r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                        capture_output=True, timeout=10, encoding="utf-8", errors="replace")
head_sha = head_r.stdout.strip()
cache_file = snap_dir / f"graph_{head_sha[:12]}.pkl"
with open(cache_file, "wb") as f:
    pickle.dump({"graph": graph, "risky": risky, "sorted": sorted_graph, "head": head_sha}, f, protocol=pickle.HIGHEST_PROTOCOL)
t_write_pkl = time.time() - t7
size_mb = cache_file.stat().st_size / (1024 * 1024)
print(f"Stage 7 - Persist pickle: {t_write_pkl:.3f}s, {size_mb:.1f}MB -> {cache_file.name}")

# Stage 8: Load from disk
t8 = time.time()
with open(cache_file, "rb") as f:
    loaded = pickle.load(f)
t_load_pkl = time.time() - t8
print(f"Stage 8 - Load pickle: {t_load_pkl:.3f}s")

print("\n" + "=" * 60)
print("TOTAL COLD START (git log + parse + merges + resolve + risky + sort):")
total = t_log + t_parse + t_merge + t_resolve + t_risky + t_sort
print(f"  {total:.1f}s")
print(f"\nPERSISTED COLD START (load pickle only):")
print(f"  {t_load_pkl:.3f}s")
print(f"\nBEST CASE WARM (pickle load):")
print(f"  {t_load_pkl:.3f}s")
