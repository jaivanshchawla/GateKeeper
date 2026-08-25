#!/usr/bin/env python3
"""
Compute M.1 features for a single commit.

Uses per-file git log -- <file> (only returns commits touching that file)
which is fast regardless of total repo size. This matches bulk extraction
because it sees the FULL commit history per file.
"""

import subprocess
from collections import defaultdict
from datetime import datetime, timezone


def _run_git(repo_path: str, args: list[str], timeout: int = 15) -> str:
    """Run a git command with UTF-8 encoding."""
    r = subprocess.run(
        ["git"] + args,
        cwd=repo_path, capture_output=True, timeout=timeout, check=False,
        encoding="utf-8", errors="replace",
    )
    return r.stdout


def _get_file_commits_before(repo_path: str, filepath: str, before_ts: int,
                              since_ts: int = 0) -> list[dict]:
    """Get commits in [since, before) window that touched this file."""
    before_date = datetime.fromtimestamp(before_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    args = ["log", f"--before={before_date}"]
    if since_ts > 0:
        since_date = datetime.fromtimestamp(since_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        args.append(f"--since={since_date}")
    args += ["--pretty=format:%H|%ct|%s", "--no-merges", "--", filepath]
    output = _run_git(repo_path, args)
    commits = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) == 3 and len(parts[0]) == 40:
            try:
                dt = datetime.fromtimestamp(int(parts[1]), tz=timezone.utc).replace(tzinfo=None)
            except (ValueError, OSError):
                dt = None
            commits.append({"hash": parts[0], "date": dt, "subject": parts[2]})
    return commits


def _get_author_file_count_before(repo_path: str, author: str, filepath: str,
                                  before_ts: int, since_ts: int = 0) -> int:
    """Count author's commits to this file in [since, before) window."""
    before_date = datetime.fromtimestamp(before_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    args = ["log", f"--before={before_date}"]
    if since_ts > 0:
        since_date = datetime.fromtimestamp(since_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        args.append(f"--since={since_date}")
    args += [f"--author={author}", "--oneline", "--no-merges", "--", filepath]
    output = _run_git(repo_path, args)
    if not output.strip():
        return 0
    return len([l for l in output.strip().split("\n") if l.strip()])


def _get_author_dir_count_before(repo_path: str, author: str, directory: str,
                                 before_ts: int, since_ts: int = 0) -> int:
    """Count author's commits to this directory in [since, before) window."""
    before_date = datetime.fromtimestamp(before_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    args = ["log", f"--before={before_date}"]
    if since_ts > 0:
        since_date = datetime.fromtimestamp(since_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        args.append(f"--since={since_date}")
    args += [f"--author={author}", "--oneline", "--no-merges", "--", f"{directory}/"]
    output = _run_git(repo_path, args)
    if not output.strip():
        return 0
    return len([l for l in output.strip().split("\n") if l.strip()])


def _get_author_last_commit_ts(repo_path: str, author: str, before_ts: int,
                               since_ts: int = 0) -> int | None:
    """Get author's most recent commit timestamp in [since, before) window."""
    before_date = datetime.fromtimestamp(before_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    args = ["log", f"--before={before_date}"]
    if since_ts > 0:
        since_date = datetime.fromtimestamp(since_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        args.append(f"--since={since_date}")
    args += [f"--author={author}", "--pretty=format:%ct", "-1", "--no-merges", "HEAD"]
    output = _run_git(repo_path, args)
    try:
        return int(output.strip())
    except (ValueError, OSError):
        return None


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
) -> dict:
    """Compute all 27 M.1 features for a single commit.

    Uses per-file git log -- <file> which only returns commits touching
    that file, so it's fast even for large repos. Sees FULL history.
    """
    if commit_date.tzinfo is not None:
        commit_date = commit_date.replace(tzinfo=None)

    before_ts = int(commit_date.timestamp())
    since_ts = int(datetime.strptime(since_date, "%Y-%m-%d").timestamp()) if since_date else 0

    # ── Per-file history (M.1a): windowed to match bulk extraction ──
    file_change_count = defaultdict(int)
    file_risky_count = defaultdict(int)
    file_revert_count = defaultdict(int)
    file_first_seen = {}
    file_last_seen = {}

    for fp in touched_files:
        history = _get_file_commits_before(repo_path, fp, before_ts, since_ts)
        file_change_count[fp] = len(history)
        file_risky_count[fp] = sum(1 for c in history if risky_hashes and c["hash"] in risky_hashes)
        file_revert_count[fp] = sum(1 for c in history if "revert" in (c.get("subject") or "").lower())
        if history:
            dates = [c["date"] for c in history if c["date"]]
            if dates:
                file_first_seen[fp] = min(dates)
                file_last_seen[fp] = max(dates)

    # Aggregate across touched files
    changes = [file_change_count[f] for f in touched_files]
    risky_vals = [file_risky_count[f] for f in touched_files]
    reverts = [file_revert_count[f] for f in touched_files]
    ages = [(commit_date - file_first_seen[f]).days for f in touched_files if f in file_first_seen]
    days_since_last = [(commit_date - file_last_seen[f]).days for f in touched_files if f in file_last_seen]

    # ── Author-file familiarity (M.1b) ──
    author_file_prior = []
    for f in touched_files:
        author_file_prior.append(_get_author_file_count_before(repo_path, author_name, f, before_ts, since_ts))

    touched_dirs = set()
    for f in touched_files:
        d = f.rsplit("/", 1)[0] if "/" in f else ""
        if d:
            touched_dirs.add(d)

    author_dir_prior = []
    for d in touched_dirs:
        author_dir_prior.append(_get_author_dir_count_before(repo_path, author_name, d, before_ts, since_ts))

    is_first_touch_file = 1 if all(c == 0 for c in author_file_prior) else 0
    is_first_touch_dir = 1 if all(c == 0 for c in author_dir_prior) else 0

    author_days_since = 0
    last_ts = _get_author_last_commit_ts(repo_path, author_name, before_ts, since_ts)
    if last_ts:
        author_days_since = (before_ts - last_ts) // 86400

    # ── Change-shape features (M.1c) ──
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
        # M.1a
        "file_prior_changes_max": max(changes) if changes else 0,
        "file_prior_changes_mean": float(sum(changes) / len(changes)) if changes else 0.0,
        "file_prior_risky_max": max(risky_vals) if risky_vals else 0,
        "file_prior_risky_mean": float(sum(risky_vals) / len(risky_vals)) if risky_vals else 0.0,
        "file_revert_count_max": max(reverts) if reverts else 0,
        "file_revert_count_mean": float(sum(reverts) / len(reverts)) if reverts else 0.0,
        "file_age_days_max": max(ages) if ages else 0,
        "file_age_days_mean": float(sum(ages) / len(ages)) if ages else 0.0,
        "file_authors_count_max": 0,  # Not computed per-file without author info in git log
        "file_authors_count_mean": 0.0,
        "days_since_last_change_max": max(days_since_last) if days_since_last else 0,
        "days_since_last_change_mean": float(sum(days_since_last) / len(days_since_last)) if days_since_last else 0.0,
        # M.1b
        "author_file_prior_commits_max": max(author_file_prior) if author_file_prior else 0,
        "author_file_prior_commits_mean": float(sum(author_file_prior) / len(author_file_prior)) if author_file_prior else 0.0,
        "author_dir_prior_commits_max": max(author_dir_prior) if author_dir_prior else 0,
        "author_dir_prior_commits_mean": float(sum(author_dir_prior) / len(author_dir_prior)) if author_dir_prior else 0.0,
        "is_author_first_touch_file": is_first_touch_file,
        "is_author_first_touch_dir": is_first_touch_dir,
        "author_days_since_last_commit": author_days_since,
        # M.1c
        "churn_ratio": churn_ratio,
        "change_entropy": change_entropy,
        "max_file_churn": max_file_churn,
        "is_test_only": 1 if files_touched_count > 0 and test_count == files_touched_count else 0,
        "test_to_code_ratio": test_count / files_touched_count if files_touched_count > 0 else 0.0,
        "config_touch": 1 if config_count > 0 else 0,
        "is_merge": 0,
        "files_per_dir_ratio": files_touched_count / max(len(touched_dirs), 1),
    }
