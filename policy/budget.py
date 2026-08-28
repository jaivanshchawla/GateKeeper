#!/usr/bin/env python3
"""
U.5a: Per-repo rolling risk budget.

Tracks the share of high-band commits over a rolling window.
Warns at 80% of budget, blocks at 100%.
Configured in .gatekeeper.yml under risk_budget.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Any


@dataclass
class BudgetStatus:
    """Current budget status for a repo."""
    repo: str
    window_days: int
    max_high_pct: float  # e.g. 0.20 = 20%
    current_high_pct: float
    total_commits: int
    high_commits: int
    budget_pct_used: float  # current_high_pct / max_high_pct
    is_warning: bool  # >= 80% of budget
    is_over: bool  # >= 100% of budget
    window_start: str  # ISO date

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "window_days": self.window_days,
            "max_high_pct": self.max_high_pct,
            "current_high_pct": round(self.current_high_pct, 4),
            "total_commits": self.total_commits,
            "high_commits": self.high_commits,
            "budget_pct_used": round(self.budget_pct_used, 4),
            "is_warning": self.is_warning,
            "is_over": self.is_over,
            "window_start": self.window_start,
        }


DEFAULT_BUDGET_CONFIG = {
    "enabled": True,
    "window_days": 30,
    "max_high_pct": 0.25,  # 25% high-band commits allowed (per-repo override via .gatekeeper.yml)
    "warn_threshold": 0.80,  # warn at 80% of budget
}


class RiskBudget:
    """Manages per-repo rolling risk budgets using SQLite."""

    def __init__(self, db_path: str = "data/risk_budget.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                band TEXT NOT NULL,
                scored_at TEXT NOT NULL,
                UNIQUE(repo, commit_hash)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_budget_repo_time
            ON budget_scores(repo, scored_at)
        """)
        conn.commit()
        conn.close()

    def record_score(self, repo: str, commit_hash: str, band: str):
        """Record a scored commit's band for budget tracking."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO budget_scores (repo, commit_hash, band, scored_at) VALUES (?, ?, ?, ?)",
                (repo, commit_hash, band, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_status(
        self,
        repo: str,
        config: dict[str, Any] | None = None,
    ) -> BudgetStatus:
        """Get current budget status for a repo."""
        cfg = {**DEFAULT_BUDGET_CONFIG, **(config or {})}
        window_days = cfg["window_days"]
        max_high_pct = cfg["max_high_pct"]
        warn_threshold = cfg["warn_threshold"]

        conn = sqlite3.connect(self.db_path)
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

            total = conn.execute(
                "SELECT COUNT(*) FROM budget_scores WHERE repo = ? AND scored_at >= ?",
                (repo, cutoff),
            ).fetchone()[0]

            high = conn.execute(
                "SELECT COUNT(*) FROM budget_scores WHERE repo = ? AND scored_at >= ? AND band = 'high'",
                (repo, cutoff),
            ).fetchone()[0]
        finally:
            conn.close()

        current_pct = high / total if total > 0 else 0.0
        budget_used = current_pct / max_high_pct if max_high_pct > 0 else 0.0

        window_start = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")

        return BudgetStatus(
            repo=repo,
            window_days=window_days,
            max_high_pct=max_high_pct,
            current_high_pct=current_pct,
            total_commits=total,
            high_commits=high,
            budget_pct_used=budget_used,
            is_warning=budget_used >= warn_threshold,
            is_over=budget_used >= 1.0,
            window_start=window_start,
        )

    def get_all_statuses(
        self,
        repos: list[str],
        config: dict[str, Any] | None = None,
    ) -> list[BudgetStatus]:
        """Get budget status for multiple repos."""
        return [self.get_status(repo, config) for repo in repos]
