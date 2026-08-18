#!/usr/bin/env python3
"""
Multi-repo dataset builder for Gatekeeper.

Clones 5 repos (if not already present), mines commits from each with
max-commits cap, concatenates results into a single CSV with a
source_repo column for leave-one-repo-out validation.

Repos:
  - django/django      (Python/web)
  - facebook/react     (JS/UI)
  - rust-lang/rust     (Rust/systems)
  - kubernetes/kubernetes (Go/infra)
  - apache/kafka       (Java/distributed)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.extract_features import CommitFeatureExtractor

# ── Configuration ──────────────────────────────────────────────────────
REPOS = [
    {
        "name": "django",
        "url": "https://github.com/django/django.git",
        "clone_dir": "repos/django",
    },
    {
        "name": "react",
        "url": "https://github.com/facebook/react.git",
        "clone_dir": "repos/react",
    },
    {
        "name": "rust",
        "url": "https://github.com/rust-lang/rust.git",
        "clone_dir": "repos/rust",
    },
    {
        "name": "kubernetes",
        "url": "https://github.com/kubernetes/kubernetes.git",
        "clone_dir": "repos/kubernetes",
    },
    {
        "name": "kafka",
        "url": "https://github.com/apache/kafka.git",
        "clone_dir": "repos/kafka",
    },
]

SINCE = "2024-08-15"  # ~2 years ago
MAX_COMMITS = 3000
OUTPUT_DIR = Path("data")
CONFIG_PATH = Path("ml/config.yaml")
COMBINED_OUTPUT = OUTPUT_DIR / "commit_features.csv"


def clone_repo(repo: dict) -> Path:
    """Clone repo if not already present. Returns local path.
    
    Does NOT use --depth 1 because PyDriller needs full history
    for author_prior_commits (author's total prior commit count).
    """
    local_path = Path(repo["clone_dir"])
    if local_path.exists() and (local_path / ".git").exists():
        print(f"  ✓ {repo['name']} already cloned at {local_path}")
        return local_path

    print(f"  Cloning {repo['url']} into {local_path} ...")
    local_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", repo["url"], str(local_path)],
        check=True,
        timeout=600,
    )
    print(f"  ✓ Cloned {repo['name']}")
    return local_path


def mine_repo(repo: dict, config: dict) -> pd.DataFrame | None:
    """Run feature extraction on a single repo. Returns DataFrame or None."""
    try:
        clone_path = clone_repo(repo)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"  ✗ Failed to clone {repo['name']}: {e}")
        return None

    output_csv = OUTPUT_DIR / f"{repo['name']}_features.csv"
    label_window = config.get("label_window_days", 7)

    print(f"\n{'='*60}")
    print(f"Mining {repo['name']} (since={SINCE}, max={MAX_COMMITS})")
    print(f"{'='*60}")

    t0 = time.time()

    try:
        extractor = CommitFeatureExtractor(
            repo_path=str(clone_path),
            since=SINCE,
            label_window_days=label_window,
            max_commits=MAX_COMMITS,
        )
        df = extractor.extract_and_save(str(output_csv))
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"  ✗ Failed to mine {repo['name']}: {e}")
        return None

    elapsed = time.time() - t0
    n_risky = int(df["risky"].sum())
    n_safe = len(df) - n_risky

    print(f"\n  {repo['name']}: {len(df)} commits mined in {elapsed:.1f}s")
    print(f"  Class balance: {n_risky} risky ({n_risky/len(df)*100:.1f}%), "
          f"{n_safe} safe ({n_safe/len(df)*100:.1f}%)")

    return df


def main():
    print("=" * 60)
    print("Gatekeeper Multi-Repo Dataset Builder")
    print(f"Repos: {', '.join(r['name'] for r in REPOS)}")
    print(f"Since: {SINCE}, Max commits per repo: {MAX_COMMITS}")
    print("=" * 60)

    # Load config
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mine each repo
    all_dfs = []
    t_start = time.time()

    for repo in REPOS:
        df = mine_repo(repo, config)
        if df is not None and len(df) > 0:
            df["source_repo"] = repo["name"]
            all_dfs.append(df)

    # Combine
    if not all_dfs:
        print("\nERROR: No data mined from any repo!")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns — source_repo after hash
    cols = list(combined.columns)
    if "source_repo" in cols:
        cols.remove("source_repo")
        hash_idx = cols.index("hash") + 1
        cols.insert(hash_idx, "source_repo")
    combined = combined[cols]

    # Save combined dataset
    combined.to_csv(COMBINED_OUTPUT, index=False)

    t_total = time.time() - t_start

    # ── Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMBINED DATASET SUMMARY")
    print("=" * 60)

    total = len(combined)
    total_risky = int(combined["risky"].sum())
    total_safe = total - total_risky
    print(f"Total rows: {total}")
    print(f"Total risky: {total_risky} ({total_risky/total*100:.2f}%)")
    print(f"Total safe:  {total_safe} ({total_safe/total*100:.2f}%)")
    print(f"Total time:  {t_total:.1f}s")
    print(f"Output:      {COMBINED_OUTPUT}")

    # Per-repo breakdown
    print("\nPer-repo breakdown:")
    print(f"{'Repo':<15} {'Commits':>8} {'Risky':>8} {'Safe':>8} {'Risky%':>8}")
    print("-" * 50)
    for name in [r["name"] for r in REPOS]:
        sub = combined[combined["source_repo"] == name]
        n = len(sub)
        r = int(sub["risky"].sum())
        s = n - r
        pct = r / n * 100 if n > 0 else 0
        print(f"{name:<15} {n:>8} {r:>8} {s:>8} {pct:>7.1f}%")

    print(f"\nColumns: {list(combined.columns)}")
    print("Done!")


if __name__ == "__main__":
    main()
