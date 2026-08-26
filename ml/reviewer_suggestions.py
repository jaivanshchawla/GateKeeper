#!/usr/bin/env python3
"""
U.2: Reviewer suggestion engine.

Uses author-file familiarity data to suggest the best reviewers for a PR.
Answers a question no diff-only reviewer can: who actually knows this code.

Features:
- Recency-weighted scoring (half-life ~180 days)
- Bus-factor risk detection
- CODEOWNERS parsing
- Bot exclusion
- Configurable via .gatekeeper.yml
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Default bot patterns
DEFAULT_BOT_PATTERNS = [
    "dependabot",
    "renovate",
    "[bot]",
    "github-actions",
    "codecov",
    "sonarcloud",
    "ci-svc",
]


@dataclass
class ReviewerCandidate:
    """A suggested reviewer with evidence."""
    author: str
    email: str
    score: float  # recency-weighted familiarity score
    commits_to_files: int  # total commits to the PR's files
    most_recent_commit_days: int  # days since most recent commit to these files
    is_codeowner: bool = False
    bus_factor_files: list[str] = field(default_factory=list)  # files where this person is the sole contributor
    evidence: str = ""


@dataclass
class ReviewerSuggestionResult:
    """Result of reviewer suggestion for a PR."""
    suggestions: list[ReviewerCandidate]
    bus_factor_risks: list[dict]  # files with only one contributor
    codeowners: dict[str, list[str]]  # file -> list of codeowners
    pr_author: str
    files: list[str]


def _parse_codeowners(repo_path: str) -> dict[str, list[str]]:
    """Parse CODEOWNERS file if present.

    Returns dict mapping file patterns to list of owners.
    Supports GitHub/GitLab CODEOWNERS format.
    """
    codeowners = {}

    # Check standard locations
    for path in [".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS"]:
        full_path = os.path.join(repo_path, path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        pattern = parts[0]
                        owners = [o.lstrip("@") for o in parts[1:]]
                        codeowners[pattern] = owners
            break

    return codeowners


def _match_codeowners(file_path: str, codeowners: dict[str, list[str]]) -> list[str]:
    """Check if a file matches any CODEOWNERS pattern."""
    import fnmatch
    owners = []
    for pattern, pattern_owners in codeowners.items():
        if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path, f"**/{pattern}"):
            owners.extend(pattern_owners)
    return list(set(owners))


def _is_bot(author: str, email: str, bot_patterns: list[str]) -> bool:
    """Check if an author is a bot."""
    check_str = f"{author} {email}".lower()
    return any(pat.lower() in check_str for pat in bot_patterns)


def _compute_recency_weight(days_ago: float, half_life: float = 180.0) -> float:
    """Compute exponential decay weight based on recency.

    half_life: number of days after which weight drops to 0.5.
    """
    return math.exp(-0.693 * days_ago / half_life)


def suggest_reviewers(
    repo_path: str,
    pr_files: list[str],
    pr_author: str,
    config: dict[str, Any] | None = None,
    git_log_entries: list[dict] | None = None,
) -> ReviewerSuggestionResult:
    """Suggest reviewers for a PR based on author-file familiarity.

    Args:
        repo_path: path to the git repository
        pr_files: list of files touched by the PR
        pr_author: email/name of the PR author (excluded from suggestions)
        config: reviewer_suggestions config from .gatekeeper.yml
        git_log_entries: optional pre-fetched git log entries for efficiency

    Returns:
        ReviewerSuggestionResult with ranked suggestions and evidence
    """
    config = config or {}
    max_suggestions = config.get("max_suggestions", 3)
    half_life = config.get("half_life_days", 180)
    exclude_bots = config.get("exclude_bots", True)
    bot_patterns = config.get("bot_patterns", DEFAULT_BOT_PATTERNS)

    # Parse CODEOWNERS
    codeowners = _parse_codeowners(repo_path)

    # Build file->author history from git log
    now = datetime.now(timezone.utc)

    if git_log_entries is None:
        # Fetch git log for all touched files
        git_log_entries = _fetch_git_log_for_files(repo_path, pr_files)

    # Aggregate per-author, per-file scores
    author_file_commits: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    author_latest: dict[str, datetime] = {}

    for entry in git_log_entries:
        author = entry.get("author", "")
        email = entry.get("email", "")
        file_path = entry.get("file", "")
        timestamp = entry.get("timestamp")

        if not author or not file_path:
            continue

        if file_path not in pr_files:
            continue

        # Skip bots
        if exclude_bots and _is_bot(author, email, bot_patterns):
            continue

        # Skip PR author
        author_key = email or author
        if _is_bot(pr_author, pr_author, bot_patterns):
            pass  # don't skip if PR author is somehow a bot
        elif author_key == pr_author or author == pr_author:
            continue

        author_file_commits[author_key][file_path].append(entry)

        if timestamp:
            ts_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            if author_key not in author_latest or ts_dt > author_latest[author_key]:
                author_latest[author_key] = ts_dt

    # Compute scores
    candidates: list[ReviewerCandidate] = []

    for author_key, file_map in author_file_commits.items():
        total_commits = sum(len(entries) for entries in file_map.values())
        files_covered = list(file_map.keys())

        # Recency-weighted score
        score = 0.0
        most_recent_days = 999999

        for file_path, entries in file_map.items():
            for entry in entries:
                ts = entry.get("timestamp")
                if ts:
                    days_ago = (now - datetime.fromtimestamp(ts, tz=timezone.utc)).days
                    weight = _compute_recency_weight(days_ago, half_life)
                    score += weight
                    if days_ago < most_recent_days:
                        most_recent_days = days_ago

        if most_recent_days == 999999:
            most_recent_days = 0

        # Get display name from entry
        display_name = ""
        email = ""
        for entries in file_map.values():
            if entries:
                display_name = entries[0].get("author", author_key)
                email = entries[0].get("email", "")
                break

        # Check CODEOWNERS
        is_codeowner = False
        for fp in files_covered:
            owners = _match_codeowners(fp, codeowners)
            if author_key in owners or display_name in owners:
                is_codeowner = True
                break

        # Get author name from git
        if not display_name:
            display_name = author_key

        candidates.append(ReviewerCandidate(
            author=display_name,
            email=email,
            score=score,
            commits_to_files=total_commits,
            most_recent_commit_days=most_recent_days,
            is_codeowner=is_codeowner,
            evidence=f"{total_commits} commits to {len(files_covered)} PR file(s), most recent {most_recent_days}d ago",
        ))

    # Sort by score descending, prioritize codeowners
    candidates.sort(key=lambda c: (c.is_codeowner, c.score), reverse=True)

    # Bus-factor risk: files with exactly one contributor
    bus_factor_risks = []
    for file_path in pr_files:
        contributors = set()
        for author_key, file_map in author_file_commits.items():
            if file_path in file_map:
                contributors.add(author_key)
        if len(contributors) == 1:
            sole_contributor = list(contributors)[0]
            bus_factor_risks.append({
                "file": file_path,
                "sole_contributor": sole_contributor,
                "message": f"`{file_path}` has only one contributor ({sole_contributor})",
            })

    return ReviewerSuggestionResult(
        suggestions=candidates[:max_suggestions],
        bus_factor_risks=bus_factor_risks,
        codeowners=codeowners,
        pr_author=pr_author,
        files=pr_files,
    )


def _fetch_git_log_for_files(repo_path: str, files: list[str]) -> list[dict]:
    """Fetch git log entries for specific files.

    Returns list of dicts with author, email, file, timestamp, hash.
    """
    entries = []
    if not files:
        return entries

    # Use git log with --follow for each file (limited to 500 entries per file)
    for file_path in files[:20]:  # Cap at 20 files to avoid slow queries
        try:
            result = subprocess.run(
                ["git", "log", "--pretty=format:%H|%ct|%aN|%aE", "--name-only",
                 "--since=2022-01-01", "--", file_path],
                cwd=repo_path, capture_output=True, text=True, timeout=30,
            )
            current_hash = None
            current_ts = None
            current_author = None
            current_email = None

            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if "|" in line and len(line.split("|")) == 4:
                    parts = line.split("|")
                    current_hash = parts[0]
                    try:
                        current_ts = int(parts[1])
                    except ValueError:
                        current_ts = None
                    current_author = parts[2]
                    current_email = parts[3]
                elif current_hash and line and not line.startswith(" "):
                    entries.append({
                        "hash": current_hash,
                        "timestamp": current_ts,
                        "author": current_author,
                        "email": current_email,
                        "file": line.strip(),
                    })
        except (subprocess.TimeoutExpired, Exception):
            continue

    return entries


def format_reviewer_suggestions(result: ReviewerSuggestionResult) -> str:
    """Format reviewer suggestions as markdown."""
    lines = []

    if result.suggestions:
        lines.append("### 👥 Suggested Reviewers")
        lines.append("")
        for i, s in enumerate(result.suggestions, 1):
            badge = " 👑 CODEOWNER" if s.is_codeowner else ""
            lines.append(f"{i}. **{s.author}**{badge}")
            lines.append(f"   - Score: {s.score:.1f} ({s.commits_to_files} commits, most recent {s.most_recent_commit_days}d ago)")
            lines.append(f"   - {s.evidence}")
        lines.append("")

    if result.bus_factor_risks:
        lines.append("### ⚠️ Bus-Factor Risk")
        lines.append("")
        for risk in result.bus_factor_risks[:5]:
            lines.append(f"- {risk['message']}")
        lines.append("")

    return "\n".join(lines)
