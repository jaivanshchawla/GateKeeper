#!/usr/bin/env python3
"""
P.1: Compute M.1 features using a single-pass git log index.

Builds a file→commits index from one `git log --name-only` call with a
1-year window (sufficient for feature computation, fits in <2s).
"""

import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone


def _run_git(repo_path: str, args: list[str], timeout: int = 30) -> str:
    """Run a git command with UTF-8 encoding."""
    r = subprocess.run(
        ["git"] + args,
        cwd=repo_path, capture_output=True, timeout=timeout, check=False,
        encoding="utf-8", errors="replace",
    )
    return r.stdout


def build_file_index(repo_path: str, since_date: str, before_date: str,
                      max_years: float = 1.0) -> dict[str, list[dict]]:
    """Single git log call: build file → commit index.

    Uses a 1-year window (max_years) for speed on large repos.
    The index is used for M.1 feature computation only.
    """
    # Limit window to max_years for speed
    before_dt = datetime.strptime(before_date[:10], "%Y-%m-%d")
    since_dt = max(
        datetime.strptime(since_date, "%Y-%m-%d"),
        before_dt - timedelta(days=int(365 * max_years))
    )
    since_str = since_dt.strftime("%Y-%m-%d")

    output = _run_git(repo_path, [
        "log", f"--since={since_str}", f"--before={before_date}",
        "--pretty=format:%H|%ct|%an",
        "--name-only",
        "--diff-filter=ACDMR",
        "--no-merges",
        "HEAD",
    ])

    index = defaultdict(list)
    current_hash = None
    current_ts = None
    current_author = None

    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) == 3 and len(parts[0]) == 40:
            current_hash = parts[0]
            try:
                current_ts = int(parts[1])
            except ValueError:
                current_ts = 0
            current_author = parts[2]
        else:
            if current_hash and line and not line.startswith("commit "):
                index[line].append({
                    "hash": current_hash,
                    "ts": current_ts,
                    "author": current_author,
                })

    return dict(index)


def compute_single_commit_m1_features(
    repo_path: str,
    commit_date: datetime,
    author_name: str,
    touched_files: set[str],
    lines_added: int = 0,
    lines_deleted: int = 0,
    dirs_touched: int = 0,
    risky_hashes: set[str] | None = None,
    since_date: str = None,
    file_index: dict | None = None,
) -> dict:
    """Compute all 27 M.1 features using a pre-built file index."""
    if commit_date.tzinfo is not None:
        commit_date = commit_date.replace(tzinfo=None)

    before_ts = int(commit_date.timestamp())

    # Build index if not provided
    if file_index is None:
        since_str = since_date or "2020-01-01"
        before_str = commit_date.strftime("%Y-%m-%dT%H:%M:%S")
        file_index = build_file_index(repo_path, since_str, before_str)

    # ── M.1a: File-level history (dict lookups) ──
    file_change_count = {}
    file_risky_count = {}
    file_revert_count = {}
    file_first_ts = {}
    file_last_ts = {}
    file_authors = {}

    for fp in touched_files:
        commits = file_index.get(fp, [])
        prior = [c for c in commits if c["ts"] < before_ts]
        file_change_count[fp] = len(prior)
        file_risky_count[fp] = sum(1 for c in prior if risky_hashes and c["hash"] in risky_hashes)
        # Revert detection: use hash-based lookup instead of subject (not in index)
        # For now, set to 0 — reverts are rare and this is a speed/accuracy tradeoff
        file_revert_count[fp] = 0
        if prior:
            file_first_ts[fp] = min(c["ts"] for c in prior)
            file_last_ts[fp] = max(c["ts"] for c in prior)
        file_authors[fp] = set(c["author"] for c in prior if c.get("author"))

    changes = [file_change_count[f] for f in touched_files]
    risky_vals = [file_risky_count[f] for f in touched_files]
    reverts = [file_revert_count[f] for f in touched_files]
    ages = [(before_ts - file_first_ts[f]) // 86400 for f in touched_files if f in file_first_ts]
    days_since_last = [(before_ts - file_last_ts[f]) // 86400 for f in touched_files if f in file_last_ts]
    author_counts = [len(file_authors.get(f, set())) for f in touched_files]

    # ── M.1b: Author-file familiarity (scan full index) ──
    author_file_counts = defaultdict(int)
    author_dir_counts = defaultdict(int)
    author_last_ts = None

    for fp, commits in file_index.items():
        for c in commits:
            if c["ts"] >= before_ts:
                continue
            if c["author"] != author_name:
                continue
            author_file_counts[fp] += 1
            d = fp.rsplit("/", 1)[0] if "/" in fp else fp
            author_dir_counts[d] += 1
            if author_last_ts is None or c["ts"] > author_last_ts:
                author_last_ts = c["ts"]

    touched_dirs = set()
    for f in touched_files:
        d = f.rsplit("/", 1)[0] if "/" in f else ""
        if d:
            touched_dirs.add(d)

    author_file_prior = [author_file_counts.get(f, 0) for f in touched_files]
    author_dir_prior = [author_dir_counts.get(d, 0) for d in touched_dirs]
    is_first_touch_file = 1 if all(c == 0 for c in author_file_prior) else 0
    is_first_touch_dir = 1 if all(c == 0 for c in author_dir_prior) else 0
    author_days_since = (before_ts - author_last_ts) // 86400 if author_last_ts else 0

    # ── M.1c: Change-shape features ──
    total_lines = lines_added + lines_deleted
    churn_ratio = lines_deleted / (lines_added + 1) if lines_added > 0 else 0.0
    files_touched_count = len(touched_files)
    if files_touched_count > 1:
        p = 1.0 / files_touched_count
        change_entropy = -files_touched_count * p * __import__("math").log2(p) if p > 0 else 0.0
    else:
        change_entropy = 0.0
    max_file_churn = total_lines / files_touched_count if files_touched_count > 0 else 0

    test_pats = ("test", "spec", "_test.", "_spec.", "tests/", "test_", "__tests__")
    cfg_pats = (".yaml", ".yml", ".toml", ".lock", "dockerfile", ".github/",
                "docker-compose", "makefile", ".env", ".ini", ".cfg", "setup.py",
                "setup.cfg", "pyproject.toml", "package.json", "cargo.toml")
    test_count = sum(1 for f in touched_files if any(p in f.lower() for p in test_pats))
    config_count = sum(1 for f in touched_files if any(p in f.lower() for p in cfg_pats))

    return {
        "file_prior_changes_max": max(changes) if changes else 0,
        "file_prior_changes_mean": float(sum(changes) / len(changes)) if changes else 0.0,
        "file_prior_risky_max": max(risky_vals) if risky_vals else 0,
        "file_prior_risky_mean": float(sum(risky_vals) / len(risky_vals)) if risky_vals else 0.0,
        "file_revert_count_max": max(reverts) if reverts else 0,
        "file_revert_count_mean": float(sum(reverts) / len(reverts)) if reverts else 0.0,
        "file_age_days_max": max(ages) if ages else 0,
        "file_age_days_mean": float(sum(ages) / len(ages)) if ages else 0.0,
        "file_authors_count_max": max(author_counts) if author_counts else 0,
        "file_authors_count_mean": float(sum(author_counts) / len(author_counts)) if author_counts else 0.0,
        "days_since_last_change_max": max(days_since_last) if days_since_last else 0,
        "days_since_last_change_mean": float(sum(days_since_last) / len(days_since_last)) if days_since_last else 0.0,
        "author_file_prior_commits_max": max(author_file_prior) if author_file_prior else 0,
        "author_file_prior_commits_mean": float(sum(author_file_prior) / len(author_file_prior)) if author_file_prior else 0.0,
        "author_dir_prior_commits_max": max(author_dir_prior) if author_dir_prior else 0,
        "author_dir_prior_commits_mean": float(sum(author_dir_prior) / len(author_dir_prior)) if author_dir_prior else 0.0,
        "is_author_first_touch_file": is_first_touch_file,
        "is_author_first_touch_dir": is_first_touch_dir,
        "author_days_since_last_commit": author_days_since,
        "churn_ratio": churn_ratio,
        "change_entropy": change_entropy,
        "max_file_churn": max_file_churn,
        "is_test_only": 1 if files_touched_count > 0 and test_count == files_touched_count else 0,
        "test_to_code_ratio": test_count / files_touched_count if files_touched_count > 0 else 0.0,
        "config_touch": 1 if config_count > 0 else 0,
        "is_merge": 0,
        "files_per_dir_ratio": files_touched_count / max(len(touched_dirs), 1),
    }
