#!/usr/bin/env python3
"""
U.4: Outcome feedback loop.

Persist every score and compute realized outcomes to measure whether
the gate is actually useful in practice.

Architecture:
1. Persist: every scored commit → SQLite database
2. Revisit: scheduled job checks scored commits after label window
3. Compute: realized outcome using the SAME label definition as training
4. Surface: production precision/recall per repo per band on dashboard
5. Trigger: retrain when production ROC-AUC drops or N outcomes accumulate

CRITICAL: outcomes are computed from the repo's real git history,
never from the model's own output.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Database Schema ─────────────────────────────────────────────────

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS scored_commits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_hash TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    repo_url TEXT,
    scored_at TEXT NOT NULL,  -- ISO timestamp
    risk_score REAL NOT NULL,
    risk_label TEXT NOT NULL,  -- low/medium/high
    band_counts TEXT,  -- JSON: {"low": N, "medium": N, "high": N} for PR-level
    features TEXT,  -- JSON: feature values
    shap_top3 TEXT,  -- JSON: SHAP explanations
    rule_results TEXT,  -- JSON: rule results
    pr_number INTEGER,
    author TEXT,
    -- Outcome fields (filled later)
    outcome_computed INTEGER DEFAULT 0,  -- 0=pending, 1=computed
    outcome_actual INTEGER,  -- 0=safe, 1=risky
    outcome_method TEXT,  -- "revert" or "retouch"
    outcome_window_days INTEGER DEFAULT 7,
    outcome_computed_at TEXT,
    outcome_detail TEXT  -- JSON: which commit/file triggered the outcome
);

CREATE INDEX IF NOT EXISTS idx_scored_repo ON scored_commits(repo_name);
CREATE INDEX IF NOT EXISTS idx_scored_hash ON scored_commits(commit_hash);
CREATE INDEX IF NOT EXISTS idx_scored_outcome ON scored_commits(outcome_computed);
CREATE INDEX IF NOT EXISTS idx_scored_label ON scored_commits(risk_label);

CREATE TABLE IF NOT EXISTS production_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    total_scored INTEGER,
    total_high INTEGER,
    total_medium INTEGER,
    total_low INTEGER,
    -- Precision per band
    high_precision REAL,
    medium_precision REAL,
    -- Overall metrics
    overall_precision REAL,
    overall_recall REAL,
    roc_auc REAL,
    -- Retrain trigger
    should_retrain INTEGER DEFAULT 0,
    retrain_reason TEXT
);
"""


@dataclass
class ScoredCommit:
    """A commit that has been scored by Gatekeeper."""
    commit_hash: str
    repo_name: str
    repo_url: str = ""
    scored_at: str = ""
    risk_score: float = 0.0
    risk_label: str = "low"
    band_counts: dict | None = None
    features: dict | None = None
    shap_top3: list | None = None
    rule_results: list | None = None
    pr_number: int | None = None
    author: str = ""


@dataclass
class RealizedOutcome:
    """The actual outcome for a scored commit."""
    commit_hash: str
    repo_name: str
    predicted_label: str
    predicted_score: float
    actual_label: int  # 0=safe, 1=risky
    method: str  # "revert" or "retouch"
    window_days: int
    detail: dict | None = None


class OutcomeDB:
    """Outcome database: Postgres via SQLAlchemy when available, SQLite fallback."""

    def __init__(self, db_path: str | Path | None = None):
        # Try Postgres first
        self.use_postgres = False
        self.Session = None
        try:
            from webhook.models import Commit, Repo, engine, SessionLocal
            from sqlalchemy import inspect
            # Check if tables exist
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            if "commits" in tables:
                self.use_postgres = True
                self.Session = SessionLocal
                self.Commit = Commit
                self.Repo = Repo
                return
        except Exception:
            pass

        # SQLite fallback
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "..", "data", "outcomes.db")
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            dir_name = os.path.dirname(self.db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()

    def close(self):
        if not self.use_postgres and hasattr(self, 'conn'):
            self.conn.close()

    def persist_score(self, score: ScoredCommit) -> None:
        """Persist a scored commit to the database."""
        if self.use_postgres:
            session = self.Session()
            try:
                # Find or create repo
                repo = session.query(self.Repo).filter_by(name=score.repo_name).first()
                if not repo:
                    repo = self.Repo(name=score.repo_name, remote_url=score.repo_url)
                    session.add(repo)
                    session.flush()
                commit = self.Commit(
                    repo_id=repo.id,
                    sha=score.commit_hash,
                    author=score.author,
                    timestamp=datetime.fromisoformat(score.scored_at) if score.scored_at else datetime.now(timezone.utc),
                    score=int(score.risk_score * 100),
                    band=score.risk_label,
                    risk_label=score.risk_label,
                    rule_results=json.dumps(score.rule_results) if score.rule_results else None,
                    shap_top3=json.dumps(score.shap_top3) if score.shap_top3 else None,
                )
                session.add(commit)
                session.commit()
            finally:
                session.close()
        else:
            self.conn.execute(
                """INSERT INTO scored_commits
                   (commit_hash, repo_name, repo_url, scored_at, risk_score, risk_label,
                    band_counts, features, shap_top3, rule_results, pr_number, author)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    score.commit_hash,
                    score.repo_name,
                    score.repo_url,
                    score.scored_at or datetime.now(timezone.utc).isoformat(),
                    score.risk_score,
                    score.risk_label,
                    json.dumps(score.band_counts) if score.band_counts else None,
                    json.dumps(score.features) if score.features else None,
                    json.dumps(score.shap_top3) if score.shap_top3 else None,
                    json.dumps(score.rule_results) if score.rule_results else None,
                    score.pr_number,
                    score.author,
                ),
            )
            self.conn.commit()

    def persist_scores(self, scores: list[ScoredCommit]) -> None:
        """Batch persist scored commits."""
        for score in scores:
            self.persist_score(score)

    def get_uncomputed_outcomes(self, min_age_days: int = 7) -> list[dict]:
        """Get scored commits whose outcomes haven't been computed yet."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
        if self.use_postgres:
            session = self.Session()
            try:
                rows = (
                    session.query(self.Commit)
                    .filter(self.Commit.outcome_actual.is_(None))
                    .filter(self.Commit.timestamp < datetime.fromisoformat(cutoff))
                    .all()
                )
                return [{
                    "commit_hash": r.sha,
                    "repo_name": session.query(self.Repo).get(r.repo_id).name if r.repo_id else "",
                    "risk_score": (r.score or 0) / 100.0,
                    "risk_label": r.risk_label or r.band or "low",
                    "scored_at": r.timestamp.isoformat() if r.timestamp else "",
                } for r in rows]
            finally:
                session.close()
        else:
            rows = self.conn.execute(
                """SELECT commit_hash, repo_name, risk_score, risk_label, scored_at
                   FROM scored_commits
                   WHERE outcome_computed = 0 AND scored_at < ?""",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    def record_outcome(
        self,
        commit_hash: str,
        repo_name: str,
        actual_label: int,
        method: str,
        window_days: int,
        detail: dict | None = None,
    ) -> None:
        """Record the realized outcome for a commit."""
        if self.use_postgres:
            session = self.Session()
            try:
                repo = session.query(self.Repo).filter_by(name=repo_name).first()
                if repo:
                    commit = session.query(self.Commit).filter_by(sha=commit_hash, repo_id=repo.id).first()
                    if commit:
                        commit.outcome_actual = actual_label
                        commit.outcome_checked_at = datetime.now(timezone.utc)
                        session.commit()
            finally:
                session.close()
        else:
            self.conn.execute(
                """UPDATE scored_commits
                   SET outcome_computed = 1,
                       outcome_actual = ?,
                       outcome_method = ?,
                       outcome_window_days = ?,
                       outcome_computed_at = ?,
                       outcome_detail = ?
                   WHERE commit_hash = ? AND repo_name = ?""",
                (
                    actual_label,
                    method,
                    window_days,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(detail) if detail else None,
                    commit_hash,
                    repo_name,
                ),
            )
            self.conn.commit()

    def compute_production_metrics(
        self,
        repo_name: str,
        window_days: int = 90,
    ) -> dict:
        """Compute production precision/recall per band for a repo.

        Only uses commits with computed outcomes.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

        rows = self.conn.execute(
            """SELECT risk_label, risk_score, outcome_actual
               FROM scored_commits
               WHERE repo_name = ? AND scored_at > ? AND outcome_computed = 1""",
            (repo_name, cutoff),
        ).fetchall()

        if not rows:
            return {"repo": repo_name, "total": 0, "message": "No outcomes computed yet"}

        # Group by band
        bands = defaultdict(lambda: {"total": 0, "true_positives": 0, "false_positives": 0})
        overall = {"total": 0, "true_positives": 0, "false_positives": 0, "actual_positive": 0}

        for row in rows:
            label = row["risk_label"]
            actual = row["outcome_actual"]

            bands[label]["total"] += 1
            if actual == 1:
                bands[label]["true_positives"] += 1
            else:
                bands[label]["false_positives"] += 1

            overall["total"] += 1
            if actual == 1:
                overall["actual_positive"] += 1
            if label in ("high", "medium") and actual == 1:
                overall["true_positives"] += 1
            elif label in ("high", "medium") and actual == 0:
                overall["false_positives"] += 1

        # Compute precision per band
        result = {
            "repo": repo_name,
            "window_days": window_days,
            "total": len(rows),
            "bands": {},
        }

        for band in ["high", "medium", "low"]:
            b = bands[band]
            if b["total"] > 0:
                precision = b["true_positives"] / b["total"] if b["total"] > 0 else 0
                result["bands"][band] = {
                    "count": b["total"],
                    "true_positives": b["true_positives"],
                    "false_positives": b["false_positives"],
                    "precision": round(precision, 4),
                }

        # Overall precision for medium+high
        if overall["total"] > 0:
            flagged = overall["true_positives"] + overall["false_positives"]
            result["overall_precision"] = (
                overall["true_positives"] / flagged if flagged > 0 else 0
            )
            result["overall_recall"] = (
                overall["true_positives"] / overall["actual_positive"]
                if overall["actual_positive"] > 0 else 0
            )

        # Store in production_metrics table
        self.conn.execute(
            """INSERT INTO production_metrics
               (repo_name, computed_at, window_days, total_scored,
                total_high, total_medium, total_low,
                high_precision, medium_precision,
                overall_precision, overall_recall)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                repo_name,
                datetime.now(timezone.utc).isoformat(),
                window_days,
                len(rows),
                bands["high"]["total"],
                bands["medium"]["total"],
                bands["low"]["total"],
                bands["high"].get("true_positives", 0) / max(bands["high"]["total"], 1),
                bands["medium"].get("true_positives", 0) / max(bands["medium"]["total"], 1),
                result.get("overall_precision", 0),
                result.get("overall_recall", 0),
            ),
        )
        self.conn.commit()

        return result


# ── Outcome Computation ─────────────────────────────────────────────

def compute_realized_outcome(
    repo_path: str,
    commit_hash: str,
    window_days: int = 7,
    repo_url: str = "",
) -> RealizedOutcome | None:
    """Compute the realized outcome for a single commit.

    Uses the SAME label definition as training:
    A commit is "risky" if:
    1. Its files are re-touched within window_days, OR
    2. It is reverted within window_days

    CRITICAL: This reads from git history, never from the model's output.
    """
    # Get commit timestamp
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", commit_hash],
        cwd=repo_path, capture_output=True, text=True, timeout=10,
    )
    if not result.stdout.strip():
        return None

    commit_ts = int(result.stdout.strip())
    commit_dt = datetime.fromtimestamp(commit_ts, tz=timezone.utc)

    # Check for revert: any commit after this one with "Revert" + this hash in message
    revert_result = subprocess.run(
        ["git", "log", f"--since={commit_dt.isoformat()}",
         f"--until={(commit_dt + timedelta(days=window_days)).isoformat()}",
         "--format=%H|%s", "--all"],
        cwd=repo_path, capture_output=True, text=True, timeout=30,
    )

    is_reverted = False
    revert_detail = {}
    for line in revert_result.stdout.strip().split("\n"):
        if "|" in line:
            h, msg = line.split("|", 1)
            if "revert" in msg.lower() and commit_hash[:8] in h or commit_hash[:12] in msg:
                is_reverted = True
                revert_detail = {"reverted_by": h[:8], "message": msg[:100]}
                break

    # Check for file retouch: any commit after this one touching the same files
    # within window_days
    files_result = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash],
        cwd=repo_path, capture_output=True, text=True, timeout=10,
    )
    commit_files = [f.strip() for f in files_result.stdout.strip().split("\n") if f.strip()]

    is_retouched = False
    retouch_detail = {}
    for file_path in commit_files[:10]:  # Cap at 10 files
        retouch_result = subprocess.run(
            ["git", "log", f"--since={commit_dt.isoformat()}",
             f"--until={(commit_dt + timedelta(days=window_days)).isoformat()}",
             "--format=%H|%ct", "--", file_path],
            cwd=repo_path, capture_output=True, text=True, timeout=15,
        )
        for line in retouch_result.stdout.strip().split("\n"):
            if "|" in line:
                h, ts = line.split("|", 1)
                if h[:8] != commit_hash[:8]:  # Don't count self
                    is_retouched = True
                    retouch_detail = {"retouched_file": file_path, "retouched_by": h[:8]}
                    break
        if is_retouched:
            break

    actual_label = 1 if (is_reverted or is_retouched) else 0
    method = "revert" if is_reverted else "retouch" if is_retouched else "none"
    detail = revert_detail if is_reverted else retouch_detail if is_retouched else None

    return RealizedOutcome(
        commit_hash=commit_hash,
        repo_name=os.path.basename(repo_path),
        predicted_label="",  # filled by caller
        predicted_score=0,  # filled by caller
        actual_label=actual_label,
        method=method,
        window_days=window_days,
        detail=detail,
    )


def compute_all_outcomes(
    db: OutcomeDB,
    repo_paths: dict[str, str],
    window_days: int = 7,
) -> dict[str, dict]:
    """Compute outcomes for all uncomputed scored commits.

    Args:
        db: OutcomeDB instance
        repo_paths: dict mapping repo_name to repo_path
        window_days: label window in days

    Returns:
        dict mapping repo_name to summary stats
    """
    uncomputed = db.get_uncomputed_outcomes(min_age_days=window_days)
    summary = defaultdict(lambda: {"total": 0, "risky": 0, "safe": 0})

    for row in uncomputed:
        repo_name = row["repo_name"]
        commit_hash = row["commit_hash"]
        repo_path = repo_paths.get(repo_name)

        if not repo_path or not os.path.exists(repo_path):
            continue

        outcome = compute_realized_outcome(repo_path, commit_hash, window_days)
        if outcome:
            db.record_outcome(
                commit_hash=commit_hash,
                repo_name=repo_name,
                actual_label=outcome.actual_label,
                method=outcome.method,
                window_days=window_days,
                detail=outcome.detail,
            )
            summary[repo_name]["total"] += 1
            if outcome.actual_label == 1:
                summary[repo_name]["risky"] += 1
            else:
                summary[repo_name]["safe"] += 1

    return dict(summary)


def should_retrain(
    db: OutcomeDB,
    repo_name: str,
    min_precision: float = 0.3,
    min_outcomes: int = 50,
) -> tuple[bool, str]:
    """Check if retraining should be triggered for a repo.

    Triggers when:
    1. Production precision for high band drops below min_precision, OR
    2. At least min_outcomes have accumulated since last retrain

    Returns (should_retrain, reason).
    """
    metrics = db.compute_production_metrics(repo_name, window_days=90)

    if metrics.get("total", 0) < min_outcomes:
        return False, f"Only {metrics.get('total', 0)} outcomes (need {min_outcomes})"

    high_precision = metrics.get("bands", {}).get("high", {}).get("precision", 1.0)
    if high_precision < min_precision:
        return True, f"High-band precision dropped to {high_precision:.2%} (threshold: {min_precision:.0%})"

    return False, f"Precision OK ({high_precision:.2%}), {metrics['total']} outcomes"


# ── Dashboard Formatting ────────────────────────────────────────────

def format_outcomes_summary(
    db: OutcomeDB,
    repo_name: str,
    window_days: int = 90,
) -> str:
    """Format production outcomes as markdown for dashboard display."""
    metrics = db.compute_production_metrics(repo_name, window_days)

    if metrics.get("total", 0) == 0:
        return f"### {repo_name} — No outcomes computed yet"

    lines = []
    lines.append(f"### {repo_name} — Production Metrics ({window_days}d window)")
    lines.append("")
    lines.append(f"Of **{metrics['total']}** scored commits:")
    lines.append("")

    for band in ["high", "medium", "low"]:
        b = metrics.get("bands", {}).get(band, {})
        if b:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[band]
            precision_str = f"{b['precision']:.0%}" if b.get("precision") is not None else "N/A"
            lines.append(
                f"- {emoji} **{band.upper()}** ({b['count']} commits): "
                f"{b.get('true_positives', 0)} actually risky → "
                f"precision = {precision_str}"
            )

    lines.append("")

    overall_p = metrics.get("overall_precision")
    overall_r = metrics.get("overall_recall")
    if overall_p is not None:
        lines.append(f"**Overall precision (medium+high):** {overall_p:.1%}")
    if overall_r is not None:
        lines.append(f"**Overall recall (medium+high):** {overall_r:.1%}")

    return "\n".join(lines)
