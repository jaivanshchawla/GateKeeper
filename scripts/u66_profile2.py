#!/usr/bin/env python3
"""Profile resolve_author and build_graph speed on Rust."""
import time
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

REPO = str(Path("repos/rust").resolve())

# Step 1: Time identity map load
from ml.m1_shared import _load_identity_map, _identity_map, resolve_author, normalize_author_id

t0 = time.time()
_load_identity_map(REPO)
t1 = time.time()
print(f"Identity map load: {t1-t0:.3f}s, {len(_identity_map)} entries")

# Step 2: Test resolve speed
for e in ["test@test.com", "foo@bar.com", "rust@example.com", "esteban@kuber.com"]:
    t = time.time()
    resolve_author(REPO, e)
    print(f"  resolve {e}: {(time.time()-t)*1000:.2f}ms")

# Step 3: Time git log
t2 = time.time()
r1 = subprocess.run(
    ["git", "log", "--since=2024-07-01", "--until=2026-07-07",
     "--pretty=format:%H|%ct|%aE|%s", "--name-only", "--no-merges", "HEAD"],
    cwd=REPO, capture_output=True, timeout=600,
    encoding="utf-8", errors="replace",
)
t_log = time.time() - t2
print(f"\ngit log --name-only: {t_log:.1f}s, {len(r1.stdout)} bytes")

# Step 4: Parse loop with resolve_author
t3 = time.time()
lines = r1.stdout.split("\n")
graph = {}
ch = None; cf = []; ct = 0; ca = ""; cs = ""
resolve_count = 0
for line in lines:
    line = line.rstrip()
    if not line:
        continue
    parts = line.split("|", 3)
    if len(parts) == 4 and len(parts[0]) == 40:
        if ch is not None:
            resolved = resolve_author(REPO, normalize_author_id(ca))
            graph[ch] = {"date": datetime.fromtimestamp(ct, tz=timezone.utc).replace(tzinfo=None),
                         "files": cf, "subject": cs, "author": resolved, "is_merge": False}
            resolve_count += 1
        ch, ct, ca, cs = parts[0], int(parts[1]), parts[2], parts[3]
        cf = []
    else:
        cf.append(line)
if ch is not None:
    resolved = resolve_author(REPO, normalize_author_id(ca))
    graph[ch] = {"date": datetime.fromtimestamp(ct, tz=timezone.utc).replace(tzinfo=None),
                 "files": cf, "subject": cs, "author": resolved, "is_merge": False}
    resolve_count += 1
t_parse = time.time() - t3
print(f"Parse + resolve {resolve_count} authors: {t_parse:.1f}s ({t_parse/resolve_count*1000:.1f}ms/author)")

# Step 5: risky hashes
from ml.m1_shared import compute_risky_hashes
t4 = time.time()
risky = compute_risky_hashes(graph, 7)
t_risky = time.time() - t4
print(f"risky hashes: {t_risky:.3f}s, {len(risky)} risky")

# Step 6: Sort
t5 = time.time()
sorted_graph = sorted(graph.items(), key=lambda x: x[1]["date"])
t_sort = time.time() - t5
print(f"Sort: {t_sort:.3f}s")

# Step 7: Persist
import pickle
t6 = time.time()
head_r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, timeout=10, encoding="utf-8", errors="replace")
head_sha = head_r.stdout.strip()
snap_dir = Path(REPO).parent / "data" / "graph_snapshots"
snap_dir.mkdir(parents=True, exist_ok=True)
cache_file = snap_dir / f"graph_{head_sha[:12]}.pkl"
with open(cache_file, "wb") as f:
    pickle.dump({"graph": graph, "risky": risky, "sorted": sorted_graph, "head": head_sha}, f, protocol=pickle.HIGHEST_PROTOCOL)
t_write = time.time() - t6
size_mb = cache_file.stat().st_size / (1024 * 1024)
print(f"Persist pickle: {t_write:.3f}s, {size_mb:.1f}MB")

# Step 8: Load pickle
t7 = time.time()
with open(cache_file, "rb") as f:
    loaded = pickle.load(f)
t_load = time.time() - t7
print(f"Load pickle: {t_load:.3f}s")

total_cold = t_log + t_parse + t_risky + t_sort
print(f"\n=== TOTALS ===")
print(f"Cold start: {total_cold:.1f}s")
print(f"  git log: {t_log:.1f}s ({t_log/total_cold*100:.0f}%)")
print(f"  parse+resolve: {t_parse:.1f}s ({t_parse/total_cold*100:.0f}%)")
print(f"  risky: {t_risky:.3f}s")
print(f"  sort: {t_sort:.3f}s")
print(f"Persisted cold (pickle load): {t_load:.3f}s")
