#!/usr/bin/env python3
"""
U.5d: Policy simulator.

Replays historical commits against a proposed config and reports
what WOULD have been blocked, warned, and passed — with a diff
against the current config's outcome.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rules.base import CommitContext, Severity
from rules.engine import RuleEngine, load_config


@dataclass
class SimResult:
    """Result of simulating one commit against one config."""
    hash: str
    date: str
    files: int
    proposed_band: str  # low/medium/high
    proposed_blocked: bool
    proposed_warnings: int
    proposed_rule_hits: list[str]
    current_band: str
    current_blocked: bool
    current_warnings: int
    current_rule_hits: list[str]
    changed: bool  # different outcome between configs


@dataclass
class SimSummary:
    """Aggregate simulation results."""
    repo: str
    window_days: int
    total_commits: int
    proposed_high: int
    proposed_medium: int
    proposed_low: int
    proposed_blocked_count: int
    current_high: int
    current_medium: int
    current_low: int
    current_blocked_count: int
    changed_count: int
    newly_blocked: int  # blocked by proposed but not current
    newly_freed: int  # freed by proposed but blocked by current
    details: list[SimResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "window_days": self.window_days,
            "total_commits": self.total_commits,
            "proposed": {
                "high": self.proposed_high,
                "medium": self.proposed_medium,
                "low": self.proposed_low,
                "blocked": self.proposed_blocked_count,
            },
            "current": {
                "high": self.current_high,
                "medium": self.current_medium,
                "low": self.current_low,
                "blocked": self.current_blocked_count,
            },
            "diff": {
                "changed": self.changed_count,
                "newly_blocked": self.newly_blocked,
                "newly_freed": self.newly_freed,
            },
        }


class PolicySimulator:
    """Simulates policy configs against historical commits."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model = None
        self._fcols = None

    def _load_model(self):
        if self._model is not None:
            return
        import skops.io as sio
        mp = self.model_path or str(Path(__file__).parent.parent / "models" / "gatekeeper_risk_model.skops")
        trusted = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
                    "numpy.dtype", "numpy.ndarray", "pandas.core.frame.DataFrame", "pandas.core.series.Series"]
        self._model = sio.loads(open(mp, "rb").read(), trusted=trusted)
        config = yaml.safe_load(open(Path(__file__).parent.parent / "ml" / "config.yaml"))
        self._fcols = config["feature_columns"]

    def simulate_repo(
        self,
        repo_path: str,
        repo_name: str,
        proposed_config: dict[str, Any],
        current_config: dict[str, Any],
        window_days: int = 90,
        max_commits: int = 500,
    ) -> SimSummary:
        """Simulate both configs on recent commits from a repo."""
        rp = Path(repo_path)
        if not rp.exists():
            raise FileNotFoundError(f"Repo not found: {repo_path}")

        # Get recent commits
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")
        r = subprocess.run(
            ["git", "log", f"--since={since}", "--no-merges",
             "--format=%H|%ct|%aE|%s", "--max-count", str(max_commits)],
            cwd=str(rp), capture_output=True, text=True, timeout=60,
        )

        commits = []
        for line in r.stdout.strip().split("\n"):
            if "|" not in line:
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            h, ts, author, subject = parts[0], int(parts[1]), parts[2], parts[3]

            # Get files
            fr = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", h[:8]],
                cwd=str(rp), capture_output=True, text=True, timeout=10,
            )
            files = [f.strip() for f in fr.stdout.strip().split("\n") if f.strip()]

            # Get lines added/deleted
            lr = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "-r", "--numstat", h[:8]],
                cwd=str(rp), capture_output=True, text=True, timeout=10,
            )
            la = ld = 0
            for lline in lr.stdout.strip().split("\n"):
                parts2 = lline.split("\t")
                if len(parts2) >= 2:
                    try:
                        la += int(parts2[0]) if parts2[0] != "-" else 0
                        ld += int(parts2[1]) if parts2[1] != "-" else 0
                    except ValueError:
                        pass

            commits.append({
                "hash": h, "ts": ts, "author": author, "subject": subject,
                "files": files, "lines_added": la, "lines_deleted": ld,
                "dirs": len(set(str(Path(f).parent) for f in files if Path(f).parent != Path("."))),
            })

        # Score with model
        self._load_model()
        from ml.extract_features import CommitFeatureExtractor
        from ml.single_commit_features import clear_cache
        clear_cache()
        ext = CommitFeatureExtractor(repo_path=str(rp), since="2024-07-01", label_window_days=7)

        # Load thresholds
        proposed_thresholds = proposed_config.get("ml_scoring", {}).get("band_thresholds", {})
        current_thresholds = current_config.get("ml_scoring", {}).get("band_thresholds", {})

        def get_band(score, thresholds):
            high = thresholds.get("high", 0.86)
            med = thresholds.get("medium", 0.75)
            if score >= high:
                return "high"
            elif score >= med:
                return "medium"
            return "low"

        # Run both engines
        proposed_engine = RuleEngine(proposed_config)
        current_engine = RuleEngine(current_config)

        results = []
        t0 = time.time()
        for i, c in enumerate(commits):
            if i > 0 and i % 50 == 0:
                elapsed = time.time() - t0
                print(f"  ... {i}/{len(commits)} ({elapsed:.0f}s)")

            # Score
            try:
                feat = ext.extract_single_commit(str(rp), c["hash"])
                fv = [feat.get(col, 0) for col in self._fcols]
                score = float(self._model.predict_proba(np.array([fv]))[0][1])
            except Exception:
                score = 0.5

            band = get_band(score, current_thresholds)
            p_band = get_band(score, proposed_thresholds)

            # Build context
            dt = datetime.fromtimestamp(c["ts"], tz=timezone.utc)
            ctx = CommitContext(
                hash=c["hash"],
                author=c["author"],
                message=c["subject"],
                files=c["files"],
                lines_added=c["lines_added"],
                lines_deleted=c["lines_deleted"],
                files_touched=len(c["files"]),
                dirs_touched=c["dirs"],
                hour_of_day=dt.hour,
                day_of_week=dt.weekday(),
                risk_score=score,
                risk_label=band,
            )

            # Evaluate both
            p_results = proposed_engine.evaluate(ctx)
            c_results = current_engine.evaluate(ctx)
            p_blocked = proposed_engine.should_block(p_results)
            c_blocked = current_engine.should_block(c_results)
            p_hits = [r.rule_name for r in p_results if not r.passed]
            c_hits = [r.rule_name for r in c_results if not r.passed]

            changed = (p_band != band) or (p_blocked != c_blocked) or (p_hits != c_hits)

            results.append(SimResult(
                hash=c["hash"],
                date=dt.strftime("%Y-%m-%d"),
                files=len(c["files"]),
                proposed_band=p_band,
                proposed_blocked=p_blocked,
                proposed_warnings=sum(1 for r in p_results if not r.passed and r.severity == Severity.WARN),
                proposed_rule_hits=p_hits,
                current_band=band,
                current_blocked=c_blocked,
                current_warnings=sum(1 for r in c_results if not r.passed and r.severity == Severity.WARN),
                current_rule_hits=c_hits,
                changed=changed,
            ))

        # Aggregate
        proposed_blocked = sum(1 for r in results if r.proposed_blocked)
        current_blocked = sum(1 for r in results if r.current_blocked)
        changed = sum(1 for r in results if r.changed)
        newly_blocked = sum(1 for r in results if r.proposed_blocked and not r.current_blocked)
        newly_freed = sum(1 for r in results if not r.proposed_blocked and r.current_blocked)

        return SimSummary(
            repo=repo_name,
            window_days=window_days,
            total_commits=len(results),
            proposed_high=sum(1 for r in results if r.proposed_band == "high"),
            proposed_medium=sum(1 for r in results if r.proposed_band == "medium"),
            proposed_low=sum(1 for r in results if r.proposed_band == "low"),
            proposed_blocked_count=proposed_blocked,
            current_high=sum(1 for r in results if r.current_band == "high"),
            current_medium=sum(1 for r in results if r.current_band == "medium"),
            current_low=sum(1 for r in results if r.current_band == "low"),
            current_blocked_count=current_blocked,
            changed_count=changed,
            newly_blocked=newly_blocked,
            newly_freed=newly_freed,
            details=results,
        )

    def format_summary(self, summary: SimSummary) -> str:
        """Format simulation results as readable text."""
        lines = [
            f"{'='*60}",
            f"SIMULATION: {summary.repo} (last {summary.window_days}d)",
            f"{'='*60}",
            f"Commits analyzed: {summary.total_commits}",
            "",
            f"{'':>20} {'Current':>10} {'Proposed':>10} {'Delta':>10}",
            f"{'─'*50}",
            f"{'High risk':>20} {summary.current_high:>10} {summary.proposed_high:>10} {summary.proposed_high - summary.current_high:>+10}",
            f"{'Elevated':>20} {summary.current_medium:>10} {summary.proposed_medium:>10} {summary.proposed_medium - summary.current_medium:>+10}",
            f"{'Not flagged':>20} {summary.current_low:>10} {summary.proposed_low:>10} {summary.proposed_low - summary.current_low:>+10}",
            f"{'Blocked':>20} {summary.current_blocked_count:>10} {summary.proposed_blocked_count:>10} {summary.proposed_blocked_count - summary.current_blocked_count:>+10}",
            "",
            f"Changed outcomes: {summary.changed_count}/{summary.total_commits} ({summary.changed_count/summary.total_commits*100:.1f}%)",
            f"  Newly blocked:  {summary.newly_blocked}",
            f"  Newly freed:    {summary.newly_freed}",
        ]
        return "\n".join(lines)
