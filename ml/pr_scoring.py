#!/usr/bin/env python3
"""
U.1: PR-level scoring.

Score every commit in a PR, aggregate to a PR-level verdict:
- max band, mean score, count per band, total files/lines
- Aggregate rule results across commits, deduplicated
- Identify the single riskiest commit
- Detect PR-level patterns invisible per-commit:
  - Same file modified in 3+ commits (churn/thrash)
  - A commit that reverts an earlier commit in the same PR
  - Tests added then removed
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rules.base import CommitContext, Severity
from rules.engine import RuleEngine, load_config as load_rules_config


@dataclass
class CommitScore:
    """Score for a single commit within a PR."""
    hash: str
    author: str
    message: str
    risk_score: float
    risk_label: str  # low/medium/high
    files: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0
    files_touched: int = 0
    rule_results: list = field(default_factory=list)
    explanations: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    blocked: bool = False
    warning_count: int = 0


@dataclass
class PRVerdict:
    """Aggregated verdict for an entire PR."""
    verdict: str  # "low", "medium", "high" (max band)
    total_commits: int
    mean_score: float
    max_score: float
    min_score: float
    band_counts: dict[str, int]  # {"low": 3, "medium": 1, "high": 0}
    total_files: int = 0
    total_lines_added: int = 0
    total_lines_deleted: int = 0
    riskiest_commit: CommitScore | None = None
    unique_files: list[str] = field(default_factory=list)
    unique_authors: list[str] = field(default_factory=list)
    blocked_rules: list[dict] = field(default_factory=list)
    warned_rules: list[dict] = field(default_factory=list)
    info_rules: list[dict] = field(default_factory=list)
    should_block: bool = False
    patterns: list[dict] = field(default_factory=list)
    commit_scores: list[CommitScore] = field(default_factory=list)


def detect_pr_patterns(commits: list[CommitScore]) -> list[dict]:
    """Detect PR-level patterns invisible to per-commit scoring.

    Returns list of pattern dicts with type, severity, message, evidence.
    """
    patterns = []

    # 1. File churn: same file modified in 3+ commits
    file_commit_count: dict[str, list[str]] = defaultdict(list)
    for cs in commits:
        for f in cs.files:
            file_commit_count[f].append(cs.hash[:8])

    for file_path, hashes in file_commit_count.items():
        if len(hashes) >= 3:
            patterns.append({
                "type": "file_churn",
                "severity": "warn",
                "message": f"`{file_path}` modified in {len(hashes)} commits within this PR — possible thrash",
                "evidence": {"file": file_path, "commits": hashes, "count": len(hashes)},
            })

    # 2. Revert chain: a commit reverts an earlier commit in the same PR
    pr_messages = {cs.hash[:8]: cs.message.lower() for cs in commits}
    pr_hashes = set(pr_messages.keys())
    for cs in commits:
        msg = cs.message.lower()
        if "revert" in msg:
            # Check if any earlier commit's hash prefix appears in the message
            for earlier_hash in pr_hashes:
                if earlier_hash != cs.hash[:8] and earlier_hash in msg:
                    patterns.append({
                        "type": "revert_chain",
                        "severity": "block",
                        "message": f"Commit `{cs.hash[:8]}` reverts `{earlier_hash}` within the same PR",
                        "evidence": {"reverter": cs.hash[:8], "reverted": earlier_hash},
                    })

    # 3. Tests added then removed
    test_files_added: set[str] = set()
    test_files_removed: set[str] = set()
    test_patterns = ("test", "spec", "_test.", "_spec.", "tests/", "test_", "__tests__")

    for cs in commits:
        # Heuristic: if lines_added > 0 and file is a test file, it was added
        # If lines_deleted > 0 and file is a test file, it was removed
        # (This is approximate — exact diff would need git diff)
        for f in cs.files:
            is_test = any(pat in f.lower() for pat in test_patterns)
            if is_test:
                if cs.lines_added > 0 and cs.lines_deleted == 0:
                    test_files_added.add(f)
                elif cs.lines_deleted > 0 and cs.lines_added == 0:
                    test_files_removed.add(f)

    added_then_removed = test_files_added & test_files_removed
    if added_then_removed:
        patterns.append({
            "type": "test_churn",
            "severity": "warn",
            "message": f"Test files added then removed in same PR: {', '.join(sorted(added_then_removed)[:3])}",
            "evidence": {"files": sorted(added_then_removed)},
        })

    # 4. Large PR (many files or lines)
    total_files = len(set(f for cs in commits for f in cs.files))
    total_lines = sum(cs.lines_added + cs.lines_deleted for cs in commits)
    if total_files > 30:
        patterns.append({
            "type": "large_pr",
            "severity": "warn",
            "message": f"PR touches {total_files} files — consider splitting",
            "evidence": {"total_files": total_files},
        })
    if total_lines > 1000:
        patterns.append({
            "type": "large_pr",
            "severity": "warn",
            "message": f"PR changes {total_lines} lines — consider splitting",
            "evidence": {"total_lines": total_lines},
        })

    return patterns


def aggregate_rule_results(
    all_rule_results: list[tuple[str, list]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Aggregate rule results across commits, deduplicated.

    Returns (blocked, warned, info) lists of rule dicts.
    """
    rule_map: dict[str, dict] = {}  # rule_name -> {rule_name, severity, message, commits}

    for commit_hash, results in all_rule_results:
        for r in results:
            if not r.passed:
                key = r.rule_name
                if key not in rule_map:
                    rule_map[key] = {
                        "rule": r.rule_name,
                        "severity": r.severity.value,
                        "message": r.message,
                        "evidence": r.evidence,
                        "commits": [],
                    }
                rule_map[key]["commits"].append(commit_hash[:8])
                # Keep the most severe message
                if r.severity.value == "block":
                    rule_map[key]["severity"] = "block"
                    rule_map[key]["message"] = r.message

    blocked = [v for v in rule_map.values() if v["severity"] == "block"]
    warned = [v for v in rule_map.values() if v["severity"] == "warn"]
    info = [v for v in rule_map.values() if v["severity"] == "info"]

    return blocked, warned, info


def aggregate_commits_to_pr(
    commit_scores: list[CommitScore],
    patterns: list[dict] | None = None,
) -> PRVerdict:
    """Aggregate per-commit scores into a PR-level verdict."""
    if not commit_scores:
        return PRVerdict(
            verdict="low", total_commits=0,
            mean_score=0, max_score=0, min_score=0,
            band_counts={"low": 0, "medium": 0, "high": 0},
        )

    scores = [cs.risk_score for cs in commit_scores]
    band_counts = Counter(cs.risk_label for cs in commit_scores)

    # Determine PR verdict: highest band present
    if band_counts.get("high", 0) > 0:
        verdict = "high"
    elif band_counts.get("medium", 0) > 0:
        verdict = "medium"
    else:
        verdict = "low"

    # Riskiest commit
    riskiest = max(commit_scores, key=lambda cs: cs.risk_score)

    # Aggregate rule results
    all_rule_results = [(cs.hash, cs.rule_results) for cs in commit_scores]
    blocked, warned, info = aggregate_rule_results(all_rule_results)
    should_block = any(r["severity"] == "block" for r in blocked)

    # Aggregate stats
    unique_files = list(set(f for cs in commit_scores for f in cs.files))
    unique_authors = list(set(cs.author for cs in commit_scores if cs.author))
    total_lines_added = sum(cs.lines_added for cs in commit_scores)
    total_lines_deleted = sum(cs.lines_deleted for cs in commit_scores)

    # Detect PR-level patterns
    if patterns is None:
        patterns = detect_pr_patterns(commit_scores)

    return PRVerdict(
        verdict=verdict,
        total_commits=len(commit_scores),
        mean_score=float(np.mean(scores)),
        max_score=float(np.max(scores)),
        min_score=float(np.min(scores)),
        band_counts=dict(band_counts),
        total_files=len(unique_files),
        total_lines_added=total_lines_added,
        total_lines_deleted=total_lines_deleted,
        riskiest_commit=riskiest,
        unique_files=unique_files,
        unique_authors=unique_authors,
        blocked_rules=blocked,
        warned_rules=warned,
        info_rules=info,
        should_block=should_block,
        patterns=patterns,
        commit_scores=commit_scores,
    )


def format_pr_comment(verdict: PRVerdict, repo_name: str = "") -> str:
    """Format the PR-level verdict as a GitHub comment."""
    # Display labels: internal values stay low/medium/high, display text is honest
    display_labels = {
        "low": "NOT FLAGGED",
        "medium": "ELEVATED",
        "high": "HIGH RISK",
    }
    emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(verdict.verdict, "⚪")
    band_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    display_label = display_labels.get(verdict.verdict, verdict.verdict.upper())

    lines = []
    lines.append(f"## 🛡️ Gatekeeper Risk Assessment — {repo_name}")
    lines.append("")
    lines.append(f"### {emoji} PR Verdict: **{display_label}**")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Commits | {verdict.total_commits} |")
    lines.append(f"| Mean score | {verdict.mean_score:.4f} |")
    lines.append(f"| Max score | {verdict.max_score:.4f} |")
    lines.append(f"| Files changed | {verdict.total_files} |")
    lines.append(f"| Lines changed | +{verdict.total_lines_added}/-{verdict.total_lines_deleted} |")
    lines.append(f"| Authors | {len(verdict.unique_authors)} |")
    lines.append("")

    # Band breakdown
    band_parts = []
    for band in ["high", "medium", "low"]:
        count = verdict.band_counts.get(band, 0)
        if count > 0:
            band_parts.append(f"{band_emoji[band]} {count} {band}")
    if band_parts:
        lines.append(f"**Bands:** {' · '.join(band_parts)}")
        lines.append("")

    # Riskiest commit
    if verdict.riskiest_commit:
        rc = verdict.riskiest_commit
        rc_emoji = band_emoji.get(rc.risk_label, "⚪")
        lines.append(f"### Riskiest Commit")
        lines.append(f"- `{rc.hash[:12]}` by **{rc.author}** — {rc_emoji} **{rc.risk_label.upper()}** ({rc.risk_score:.4f})")
        lines.append(f"  - {rc.lines_added}+/{rc.lines_deleted}- in {rc.files_touched} files")
        if rc.explanations:
            for exp in rc.explanations[:2]:
                lines.append(f"  - {exp.get('human_readable', exp.get('description', ''))}")
        lines.append("")

    # Blocked rules (deduplicated)
    if verdict.blocked_rules:
        lines.append("### 🚫 Blocked")
        for r in verdict.blocked_rules:
            commits_str = ", ".join(r["commits"][:3])
            lines.append(f"- **{r['rule']}**: {r['message']} _({commits_str})_")
        lines.append("")

    # Warned rules
    if verdict.warned_rules:
        lines.append("### ⚠️ Warnings")
        for r in verdict.warned_rules:
            commits_str = ", ".join(r["commits"][:3])
            lines.append(f"- **{r['rule']}**: {r['message']} _({commits_str})_")
        lines.append("")

    # PR-level patterns
    if verdict.patterns:
        lines.append("### 🔍 PR Patterns")
        for p in verdict.patterns:
            icon = "🔴" if p["severity"] == "block" else "🟡" if p["severity"] == "warn" else "ℹ️"
            lines.append(f"- {icon} {p['message']}")
        lines.append("")

    # Info rules (collapsed)
    if verdict.info_rules:
        lines.append("<details>")
        lines.append("<summary>ℹ️ Info rules</summary>")
        lines.append("")
        for r in verdict.info_rules:
            lines.append(f"- **{r['rule']}**: {r['message']}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Per-commit breakdown (collapsed)
    if len(verdict.commit_scores) > 1:
        lines.append("<details>")
        lines.append(f"<summary>📋 Per-commit breakdown ({len(verdict.commit_scores)} commits)</summary>")
        lines.append("")
        lines.append("| Commit | Author | Score | Band | Files | Lines |")
        lines.append("|--------|--------|-------|------|-------|-------|")
        for cs in sorted(verdict.commit_scores, key=lambda x: -x.risk_score):
            ce = band_emoji.get(cs.risk_label, "⚪")
            msg_preview = cs.message[:40] + ("..." if len(cs.message) > 40 else "")
            lines.append(f"| `{cs.hash[:8]}` | {cs.author[:15]} | {cs.risk_score:.4f} | {ce} {cs.risk_label} | {cs.files_touched} | +{cs.lines_added}/-{cs.lines_deleted} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append("*Scored by [Gatekeeper](https://github.com/jaivanshchawla/GateKeeper) — PR-level risk analysis*")

    return "\n".join(lines)
