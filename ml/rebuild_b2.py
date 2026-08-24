#!/usr/bin/env python3
"""
Gatekeeper Dataset Rebuild B2 — Unified extractor, complete labeling graph.

Two separate concerns:
  1. LABELING GRAPH: built via `git log` (fast, complete coverage, no cap)
  2. FEATURE EXTRACTION: built via CommitFeatureExtractor (PyDriller bulk)
     — the SAME code path used by Gate 2's extract_single_commit

Window: [2024-07-01, 2026-06-30), identical for all five repos.
Forward-look buffer: +7 days (until 2026-07-07) for labeling graph only.
Sampling: cap 2000 per repo, every Nth in committer_date order.
"""

import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = PROJECT_ROOT / "repos"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "commit_features.csv"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

# ── Window (FIXED, not selected) ──────────────────────────────────────
WINDOW_START = "2024-07-01"
WINDOW_END = "2026-06-30"          # exclusive: rows with committer_date <= this
FORWARD_LOOK_END = "2026-07-07"    # labeling graph extends 7 days past window
LABEL_WINDOW_DAYS = 7
MAX_ROWS_PER_REPO = 2000

# ── Repos ──────────────────────────────────────────────────────────────
REPOS = [
    {"name": "django",      "url": "https://github.com/django/django.git"},
    {"name": "react",       "url": "https://github.com/facebook/react.git"},
    {"name": "rust",        "url": "https://github.com/rust-lang/rust.git"},
    {"name": "kubernetes",  "url": "https://github.com/kubernetes/kubernetes.git"},
    {"name": "kafka",       "url": "https://github.com/apache/kafka.git"},
]

# ── Feature columns (single source of truth from config.yaml) ─────────
FEATURE_COLS = [
    "lines_added", "lines_deleted", "files_touched", "dirs_touched",
    "author_prior_commits", "hour_of_day", "day_of_week",
    "commit_msg_length", "is_fix_bug_revert",
]

META_COLS = ["hash", "source_repo", "author", "author_date",
             "committer_date", "touched_files"]


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1 — LABELING GRAPH via git log  (B2.1)
# ═══════════════════════════════════════════════════════════════════════

def build_labeling_graph(repo_path: str, since: str, until: str) -> dict:
    """Build the labeling corpus using git log (fast, no PyDriller).

    Returns: {hash: {"committer_date": datetime(UTC), "files": [str, ...], "subject": str}}
    Uses --no-merges to match PyDriller's default (merge commits have empty
    modified_files).  %ct is committer timestamp in UTC epoch.
    """
    fmt = "%H|%ct|%s"
    result = subprocess.run(
        ["git", "log",
         f"--since={since}", f"--until={until}",
         f"--pretty=format:{fmt}",
         "--name-only",
         "--no-merges",
         "HEAD"],
        cwd=repo_path, capture_output=True, text=True, timeout=600,
    )

    graph: dict[str, dict] = {}
    current_hash = None
    current_files: list[str] = []
    current_ct = 0
    current_subject = ""

    for line in result.stdout.split("\n"):
        line = line.rstrip()
        if not line:
            continue

        parts = line.split("|", 2)
        if len(parts) == 3 and len(parts[0]) == 40:
            # Save previous commit
            if current_hash is not None:
                graph[current_hash] = {
                    "committer_date": datetime.fromtimestamp(
                        current_ct, tz=timezone.utc
                    ),
                    "files": current_files,
                    "subject": current_subject,
                }
            current_hash = parts[0]
            current_ct = int(parts[1])
            current_subject = parts[2]
            current_files = []
        else:
            # File path line
            current_files.append(line)

    # Save last commit
    if current_hash is not None:
        graph[current_hash] = {
            "committer_date": datetime.fromtimestamp(
                current_ct, tz=timezone.utc
            ),
            "files": current_files,
            "subject": current_subject,
        }

    return graph


def label_from_graph(graph: dict, label_window_days: int) -> set[str]:
    """Compute risky hashes from the complete labeling graph.

    Two criteria:
      1. commit message contains 'revert'
      2. any touched file is touched again by another commit within label_window_days
    """
    risky: set[str] = set()

    # Criterion 1: revert in subject
    for h, info in graph.items():
        if "revert" in info["subject"].lower():
            risky.add(h)

    # Criterion 2: file re-touch within window
    file_touches: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    for h, info in graph.items():
        cd = info["committer_date"]
        for fp in info["files"]:
            file_touches[fp].append((h, cd))

    for fp, touches in file_touches.items():
        if len(touches) < 2:
            continue
        touches.sort(key=lambda x: x[1])
        for i, (h_i, d_i) in enumerate(touches):
            if h_i in risky:
                continue
            for j in range(i + 1, len(touches)):
                h_j, d_j = touches[j]
                if (d_j - d_i).days <= label_window_days:
                    risky.add(h_i)
                    break
                else:
                    break

    return risky


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2 — AUTHOR PRIOR COUNTS from full history  (B2.5)
# ═══════════════════════════════════════════════════════════════════════

def count_authors_before(repo_path: str, since: str) -> dict[str, int]:
    """Count commits per author BEFORE the window (full history)."""
    result = subprocess.run(
        ["git", "log", f"--until={since}", "--format=%aN",
         "--no-merges", "HEAD"],
        cwd=repo_path, capture_output=True, text=True, timeout=300,
    )
    counts: dict[str, int] = defaultdict(int)
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            counts[line] += 1
    return dict(counts)


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3 — FEATURE EXTRACTION via CommitFeatureExtractor  (B2.4)
# ═══════════════════════════════════════════════════════════════════════

def extract_features_bulk(
    repo_path: str,
    repo_name: str,
    sampled_hashes: set[str],
    author_prior: dict[str, int],
    graph: dict,
) -> pd.DataFrame:
    """Extract features for sampled commits using CommitFeatureExtractor.

    Uses PyDriller bulk traversal (fast, ~3-8k commits/sec) but only
    extracts features for commits in the sampled set.  author_prior_commits
    is seeded from full history and incremented in traversal order.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from pydriller import Repository as PRepository

    from ml.extract_features import CommitFeatureExtractor

    # Parse since date for PyDriller (requires naive datetime)
    since_dt = datetime.strptime(WINDOW_START, "%Y-%m-%d")

    extractor = CommitFeatureExtractor(
        repo_path=repo_path,
        since=WINDOW_START,
        label_window_days=LABEL_WINDOW_DAYS,
    )
    extractor.seed_author_counts(author_prior)

    rows = []
    processed = 0
    skipped = 0

    t0 = time.time()
    repo = PRepository(repo_path, since=since_dt)

    for commit in repo.traverse_commits():
        # Skip if not in sampled set
        if commit.hash not in sampled_hashes:
            # Still count author for correct sequence, but don't extract
            # Actually, we must increment for sampled commits to match
            # extract_single_commit behavior.  For non-sampled commits,
            # we also need to count them so the counter is correct when
            # we hit a sampled commit.
            extractor.author_prior_counts[commit.author.name] += 1
            skipped += 1
            continue

        # Extract features using the single shared code path
        feat = extractor._extract_features_from_commit(commit)
        feat["source_repo"] = repo_name

        # Get metadata from graph
        info = graph.get(commit.hash, {})
        feat["committer_date"] = info.get(
            "committer_date", commit.committer_date
        )
        if hasattr(feat["committer_date"], "isoformat"):
            feat["committer_date"] = feat["committer_date"].isoformat()
        # author_date: use PyDriller's author_date
        ad = commit.author_date
        feat["author_date"] = ad.isoformat() if hasattr(ad, "isoformat") else str(ad)
        # touched_files as pipe-separated string for leakage checks
        feat["touched_files"] = "|".join(sorted(
            info.get("files", [])
        ))

        rows.append(feat)
        processed += 1

        if processed % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {repo_name}: {processed} features in {elapsed:.1f}s "
                  f"({processed/elapsed:.0f} f/s)", flush=True)

    elapsed = time.time() - t0
    print(f"    {repo_name}: done — {processed} features, "
          f"{skipped} skipped, {elapsed:.1f}s total", flush=True)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 4 — SAMPLING  (B2.3)
# ═══════════════════════════════════════════════════════════════════════

def sample_commits(graph: dict, max_rows: int, window_end: str) -> set[str]:
    """Select every-Nth commits in committer_date order, capped at max_rows.

    Only includes commits with committer_date <= window_end (training window).
    Buffer commits (beyond window_end) are in the graph for labeling only.
    Returns set of hashes for O(1) lookup during extraction.
    """
    until_dt = datetime.strptime(window_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # Filter to only training-window commits
    window_items = [
        (h, info) for h, info in graph.items()
        if info["committer_date"] <= until_dt
    ]
    n = len(window_items)

    if n <= max_rows:
        return {h for h, _ in window_items}

    step = n / max_rows
    indices = [int(i * step) for i in range(max_rows)]
    return {window_items[i][0] for i in indices}


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 5 — GATE CHECKS  (B2.6)
# ═══════════════════════════════════════════════════════════════════════

def gate_checks(
    df: pd.DataFrame,
    repo_heads: dict[str, dict],
    graph_sizes: dict[str, int],
    true_counts: dict[str, int],
) -> bool:
    """B2.6 gate: abort on any failure."""
    print(f"\n{'='*60}")
    print("B2.6 GATE CHECKS")
    print(f"{'='*60}")
    all_ok = True

    # 1. All repos share identical window
    print("\n1. Window uniformity:")
    for repo in df["source_repo"].unique():
        sub = df[df["source_repo"] == repo]
        print(f"   {repo}: {sub['committer_date'].min()} to {sub['committer_date'].max()}")

    # 2. No shallow clones, HEAD within 7 days
    print("\n2. Clone freshness:")
    for r in REPOS:
        name = r["name"]
        head_info = repo_heads.get(name, {})
        head_date = head_info.get("date", "N/A")
        age = head_info.get("age_days", 999)
        shallow = head_info.get("shallow", True)
        status = "OK" if (age <= 7 and not shallow) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"   {name}: HEAD {head_date[:10]}  age={age}d  shallow={shallow}  [{status}]")

    # 3. Labeling graph completeness
    print("\n3. Graph completeness (graph commits == git rev-list count):")
    for name in [r["name"] for r in REPOS]:
        gs = graph_sizes.get(name, 0)
        tc = true_counts.get(name, 0)
        match = "OK" if gs == tc else "MISMATCH"
        if gs != tc:
            all_ok = False
        print(f"   {name}: graph={gs}  git_count={tc}  [{match}]")

    # 4. Zero rows with committer_date > WINDOW_END
    until_dt = datetime.strptime(WINDOW_END, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dates = pd.to_datetime(df["committer_date"])
    if dates.dt.tz is None:
        dates = dates.dt.tz_localize("UTC")
    over = df[dates > until_dt]
    print(f"\n4. Rows after {WINDOW_END}: {len(over)}")
    if len(over) > 0:
        all_ok = False

    if all_ok:
        print("\n✓ ALL GATE CHECKS PASSED")
    else:
        print("\n✗ GATE CHECK FAILED")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 60, flush=True)
    print("GATEKEEPER DATASET REBUILD B2", flush=True)
    print(f"Window: [{WINDOW_START}, {WINDOW_END})", flush=True)
    print(f"Forward-look: +7d → {FORWARD_LOOK_END}", flush=True)
    print(f"Max rows/repo: {MAX_ROWS_PER_REPO}", flush=True)
    print(f"Label window: {LABEL_WINDOW_DAYS}d", flush=True)
    print("=" * 60, flush=True)

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── B2.1: Build labeling graphs + verify clones ───────────────────
    print(f"\n{'='*60}", flush=True)
    print("PHASE 1: LABELING GRAPH (git log)", flush=True)
    print(f"{'='*60}", flush=True)

    graphs: dict[str, dict] = {}
    repo_heads: dict[str, dict] = {}
    graph_sizes: dict[str, int] = {}
    true_counts: dict[str, int] = {}

    for r in REPOS:
        name = r["name"]
        rp = str(REPOS_DIR / name)

        # Verify clone
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=rp, text=True
        ).strip()
        head_date_str = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci", "HEAD"], cwd=rp, text=True
        ).strip()
        head_dt = datetime.strptime(head_date_str[:19], "%Y-%m-%d %H:%M:%S")
        age_days = (datetime.now() - head_dt).days
        is_shallow = subprocess.check_output(
            ["git", "rev-parse", "--is-shallow-repository"], cwd=rp, text=True
        ).strip() == "true"

        repo_heads[name] = {
            "sha": head_sha,
            "date": head_date_str,
            "age_days": age_days,
            "shallow": is_shallow,
        }
        print(f"\n  {name}: HEAD={head_sha[:12]}  age={age_days}d  "
              f"shallow={is_shallow}", flush=True)

        # True commit count in graph window (same dates + --no-merges as graph)
        true_count = int(subprocess.check_output(
            ["git", "rev-list", "--count",
             "--no-merges",
             f"--since={WINDOW_START}", f"--until={FORWARD_LOOK_END}", "HEAD"],
            cwd=rp, text=True, timeout=60,
        ).strip())
        true_counts[name] = true_count

        # Build labeling graph (window + 7 days forward)
        t0 = time.time()
        graph = build_labeling_graph(rp, WINDOW_START, FORWARD_LOOK_END)
        elapsed = time.time() - t0
        graphs[name] = graph
        graph_sizes[name] = len(graph)
        print(f"  Labeling graph: {len(graph)} commits in {elapsed:.1f}s "
              f"({len(graph)/max(elapsed,0.01):.0f} commits/sec)", flush=True)

    # ── B2.3: Sampling ────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("PHASE 2: SAMPLING", flush=True)
    print(f"{'='*60}", flush=True)

    sampled: dict[str, set[str]] = {}
    for r in REPOS:
        name = r["name"]
        hashes = sample_commits(graphs[name], MAX_ROWS_PER_REPO, WINDOW_END)
        sampled[name] = hashes
        print(f"  {name}: {len(graphs[name])} total → {len(hashes)} sampled "
              f"(every {len(graphs[name])/max(len(hashes),1):.1f}th)", flush=True)

    # ── B2.4: Feature extraction via CommitFeatureExtractor ───────────
    print(f"\n{'='*60}", flush=True)
    print("PHASE 3: FEATURE EXTRACTION (CommitFeatureExtractor)", flush=True)
    print(f"{'='*60}", flush=True)

    all_dfs = []
    for r in REPOS:
        name = r["name"]
        rp = str(REPOS_DIR / name)

        # B2.5: author_prior from full history
        t0 = time.time()
        author_prior = count_authors_before(rp, WINDOW_START)
        print(f"\n  {name}: {len(author_prior)} authors before window "
              f"({time.time()-t0:.1f}s)", flush=True)

        # Extract features for sampled commits via bulk traversal
        t0 = time.time()
        df = extract_features_bulk(
            rp, name, sampled[name], author_prior, graphs[name]
        )
        all_dfs.append(df)

    # ── Combine ───────────────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)

    # Apply labels from the graph
    print(f"\n{'='*60}", flush=True)
    print("PHASE 4: APPLYING LABELS FROM GRAPH", flush=True)
    print(f"{'='*60}", flush=True)

    for name in [r["name"] for r in REPOS]:
        risky_hashes = label_from_graph(graphs[name], LABEL_WINDOW_DAYS)
        mask = combined["source_repo"] == name
        n_risky = combined.loc[mask, "hash"].apply(lambda h: h in risky_hashes).sum()
        combined.loc[mask, "risky"] = combined.loc[mask, "hash"].apply(
            lambda h: 1 if h in risky_hashes else 0
        )
        n_total = mask.sum()
        print(f"  {name}: {n_risky}/{n_total} risky "
              f"({n_risky/n_total*100:.1f}%)", flush=True)

    # Order columns
    label_col = ["risky"]
    other = [c for c in combined.columns
             if c not in META_COLS + FEATURE_COLS + label_col]
    combined = combined[META_COLS + FEATURE_COLS + label_col + other]

    # Drop commit_msg if present
    if "commit_msg" in combined.columns:
        combined = combined.drop(columns=["commit_msg"])

    # Save
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_CSV}", flush=True)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("FINAL PER-REPO TABLE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Repo':<15} {'Graph':>6} {'True':>6} {'Rows':>6} "
          f"{'Span(d)':>8} {'R/wk':>7} {'Risky%':>7}", flush=True)
    print("-" * 60, flush=True)
    for r in REPOS:
        name = r["name"]
        sub = combined[combined["source_repo"] == name]
        dates = pd.to_datetime(sub["committer_date"], utc=True)
        span = (dates.max() - dates.min()).days
        rpw = len(sub) / (span / 7) if span > 0 else 0
        risky = sub["risky"].mean() * 100
        gs = graph_sizes.get(name, 0)
        tc = true_counts.get(name, 0)
        print(f"{name:<15} {gs:>6} {tc:>6} {len(sub):>6} "
              f"{span:>8} {rpw:>7.1f} {risky:>6.2f}%", flush=True)
    print(f"{'TOTAL':<15} {'':>6} {'':>6} {len(combined):>6}", flush=True)

    # author_prior stats
    print("\nauthor_prior_commits:", flush=True)
    for r in REPOS:
        name = r["name"]
        sub = combined[combined["source_repo"] == name]
        print(f"  {name}: min={sub['author_prior_commits'].min()} "
              f"max={sub['author_prior_commits'].max()} "
              f"mean={sub['author_prior_commits'].mean():.1f} "
              f"zeros={(sub['author_prior_commits']==0).sum()}/{len(sub)}",
              flush=True)

    # ── Manifest ──────────────────────────────────────────────────────
    import pydriller
    script_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "forward_look_end": FORWARD_LOOK_END,
        "label_window_days": LABEL_WINDOW_DAYS,
        "max_rows_per_repo": MAX_ROWS_PER_REPO,
        "script_git_sha": script_sha,
        "pydriller_version": pydriller.__version__,
        "total_runtime_seconds": round(time.time() - t_start, 1),
        "repos": {},
    }
    for r in REPOS:
        name = r["name"]
        sub = combined[combined["source_repo"] == name]
        manifest["repos"][name] = {
            "url": r["url"],
            "head_sha": repo_heads[name]["sha"],
            "head_date": repo_heads[name]["date"],
            "graph_commits": graph_sizes[name],
            "true_window_commits": true_counts[name],
            "sampled_rows": len(sub),
            "risky_rate": round(float(sub["risky"].mean()), 4),
        }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {MANIFEST_PATH}", flush=True)

    # ── Gate ──────────────────────────────────────────────────────────
    passed = gate_checks(combined, repo_heads, graph_sizes, true_counts)
    if not passed:
        sys.exit(1)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s", flush=True)
    print("REBUILD COMPLETE", flush=True)


if __name__ == "__main__":
    main()
