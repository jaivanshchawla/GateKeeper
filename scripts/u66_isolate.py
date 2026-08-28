#!/usr/bin/env python3
"""Isolate which stage takes >60s on Rust."""
import time
import subprocess
from pathlib import Path

REPO = str(Path("repos/rust").resolve())

# Stage 1: just git log
t1 = time.time()
r1 = subprocess.run(
    ["git", "log", "--since=2024-07-01", "--until=2026-07-07",
     "--pretty=format:%H|%ct|%aE|%s", "--name-only", "--no-merges", "HEAD"],
    cwd=REPO, capture_output=True, timeout=600,
    encoding="utf-8", errors="replace",
)
t_log = time.time() - t1
print(f"git log: {t_log:.1f}s")

lines = r1.stdout.split("\n")
print(f"Lines: {len(lines)}")

# Stage 2: just parsing (no resolve)
t2 = time.time()
ch = None; cf = []; ct = 0; ca = ""; cs = ""
commits = []
for line in lines:
    line = line.rstrip()
    if not line:
        continue
    parts = line.split("|", 3)
    if len(parts) == 4 and len(parts[0]) == 40:
        if ch is not None:
            commits.append((ch, ct, ca, cs, cf[:]))
        ch, ct, ca, cs = parts[0], int(parts[1]), parts[2], parts[3]
        cf = []
    else:
        cf.append(line)
if ch is not None:
    commits.append((ch, ct, ca, cs, cf[:]))
t_parse_raw = time.time() - t2
print(f"Parse raw (no resolve): {t_parse_raw:.3f}s, {len(commits)} commits")

# Stage 3: just resolve_author on all commits
from ml.m1_shared import _load_identity_map, resolve_author, normalize_author_id
t_load = time.time()
_load_identity_map(REPO)
t_load_done = time.time()
print(f"Identity map load: {t_load_done - t_load:.3f}s")

t3 = time.time()
for ch, ct, ca, cs, cf in commits:
    resolve_author(REPO, normalize_author_id(ca))
t_resolve = time.time() - t3
print(f"Resolve all {len(commits)} authors: {t_resolve:.1f}s ({t_resolve/len(commits)*1000:.2f}ms each)")

# Stage 4: normalize_author_id only
t4 = time.time()
for ch, ct, ca, cs, cf in commits:
    normalize_author_id(ca)
t_norm = time.time() - t4
print(f"normalize_author_id x{len(commits)}: {t_norm:.3f}s")

print(f"\nTotal: git log {t_log:.1f}s + parse {t_parse_raw:.3f}s + resolve {t_resolve:.1f}s = {t_log+t_parse_raw+t_resolve:.1f}s")
