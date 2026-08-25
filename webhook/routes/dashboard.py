#!/usr/bin/env python3
"""
Dashboard routes for the Gatekeeper webhook.
Provides API endpoints for managing issues from Gates 1-3.
"""

from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request
from sqlalchemy import desc, func

# Handle both local development and Docker context
import json

try:
    from webhook.models import Issue, Repo, Commit, SessionLocal, init_db
except ImportError:
    from models import Issue, Repo, Commit, SessionLocal, init_db

dashboard_bp = Blueprint("dashboard", __name__)

# Initialize database on module load
init_db()


@dashboard_bp.route("/issues", methods=["POST"])
def create_issue():
    """Log a new issue from Gates 1-3."""
    data = request.get_json(force=True)
    
    # Validate required fields
    required_fields = ["gate", "type", "repo"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400
    
    # Validate gate number
    if data["gate"] not in [1, 2, 3]:
        return jsonify({"error": "Gate must be 1, 2, or 3"}), 400
    
    # Create issue
    db = SessionLocal()
    try:
        issue = Issue(
            gate=data["gate"],
            type=data["type"],
            repo=data["repo"],
            status=data.get("status", "open"),
            details=data.get("details"),
            commit_hash=data.get("commit_hash"),
            risk_score=data.get("risk_score"),
        )
        db.add(issue)
        db.commit()
        db.refresh(issue)
        
        return jsonify({
            "message": "Issue created successfully",
            "issue": issue.to_dict()
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@dashboard_bp.route("/issues", methods=["GET"])
def list_issues():
    """List issues with optional filters."""
    status = request.args.get("status")
    repo = request.args.get("repo")
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    
    db = SessionLocal()
    try:
        query = db.query(Issue)
        
        # Apply filters
        if status:
            query = query.filter(Issue.status == status)
        if repo:
            query = query.filter(Issue.repo == repo)
        
        # Order by created_at descending
        query = query.order_by(desc(Issue.created_at))
        
        # Apply pagination
        total = query.count()
        issues = query.offset(offset).limit(limit).all()
        
        return jsonify({
            "issues": [issue.to_dict() for issue in issues],
            "total": total,
            "limit": limit,
            "offset": offset,
        }), 200
    finally:
        db.close()


@dashboard_bp.route("/issues/<int:issue_id>", methods=["PATCH"])
def toggle_issue_status(issue_id):
    """Toggle issue status between open and resolved."""
    db = SessionLocal()
    try:
        issue = db.query(Issue).filter(Issue.id == issue_id).first()
        if not issue:
            return jsonify({"error": "Issue not found"}), 404
        
        # Toggle status
        issue.status = "resolved" if issue.status == "open" else "open"
        issue.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(issue)
        
        return jsonify({
            "message": f"Issue status toggled to {issue.status}",
            "issue": issue.to_dict()
        }), 200
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@dashboard_bp.route("/issues/stats", methods=["GET"])
def issue_stats():
    """Get daily issue counts for the last 30 days, grouped by status."""
    days = request.args.get("days", 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    db = SessionLocal()
    try:
        # Get daily counts grouped by status
        stats = (
            db.query(
                func.date(Issue.created_at).label("date"),
                Issue.status,
                func.count(Issue.id).label("count")
            )
            .filter(Issue.created_at >= start_date)
            .group_by(func.date(Issue.created_at), Issue.status)
            .order_by(func.date(Issue.created_at))
            .all()
        )
        
        # Format for chart
        daily_stats = {}
        for date, status, count in stats:
            date_str = date.isoformat() if hasattr(date, 'isoformat') else str(date)
            if date_str not in daily_stats:
                daily_stats[date_str] = {"date": date_str, "open": 0, "resolved": 0}
            daily_stats[date_str][status] = count
        
        # Convert to list and sort by date
        result = sorted(daily_stats.values(), key=lambda x: x["date"])
        
        # Get total counts
        total_open = db.query(Issue).filter(Issue.status == "open").count()
        total_resolved = db.query(Issue).filter(Issue.status == "resolved").count()
        
        return jsonify({
            "daily": result,
            "totals": {
                "open": total_open,
                "resolved": total_resolved,
                "total": total_open + total_resolved
            },
            "period_days": days
        }), 200
    finally:
        db.close()


@dashboard_bp.route("/issues/stats/by-type", methods=["GET"])
def issue_stats_by_type():
    """Get issue counts grouped by type."""
    db = SessionLocal()
    try:
        stats = (
            db.query(
                Issue.type,
                Issue.status,
                func.count(Issue.id).label("count")
            )
            .group_by(Issue.type, Issue.status)
            .all()
        )
        
        result = {}
        for issue_type, status, count in stats:
            if issue_type not in result:
                result[issue_type] = {"open": 0, "resolved": 0}
            result[issue_type][status] = count
        
        return jsonify(result), 200
    finally:
        db.close()


@dashboard_bp.route("/issues/stats/by-gate", methods=["GET"])
def issue_stats_by_gate():
    """Get issue counts grouped by gate."""
    db = SessionLocal()
    try:
        stats = (
            db.query(
                Issue.gate,
                Issue.status,
                func.count(Issue.id).label("count")
            )
            .group_by(Issue.gate, Issue.status)
            .all()
        )
        
        result = {}
        for gate, status, count in stats:
            gate_key = f"gate_{gate}"
            if gate_key not in result:
                result[gate_key] = {"open": 0, "resolved": 0}
            result[gate_key][status] = count
        
        return jsonify(result), 200
    finally:
        db.close()


# ── Repo endpoints ────────────────────────────────────────────────────

@dashboard_bp.route("/repos", methods=["GET"])
def list_repos():
    """List all registered repos with summary stats."""
    db = SessionLocal()
    try:
        repos = db.query(Repo).order_by(desc(Repo.registered_at)).all()
        result = []
        for repo in repos:
            open_issues = db.query(Issue).filter(
                Issue.repo == repo.name, Issue.status == "open"
            ).count()
            last_commit = db.query(Commit).filter(
                Commit.repo_id == repo.id
            ).order_by(desc(Commit.timestamp)).first()
            result.append({
                **repo.to_dict(),
                "open_issues": open_issues,
                "last_scored": last_commit.timestamp.isoformat() if last_commit and last_commit.timestamp else None,
                "last_score": last_commit.risk_label if last_commit else None,
            })
        return jsonify(result), 200
    finally:
        db.close()


@dashboard_bp.route("/repos", methods=["POST"])
def create_repo():
    """Register a new repo."""
    data = request.get_json(force=True)
    name = data.get("name") or data.get("remote_url", "").rstrip("/").split("/")[-1].replace(".git", "")
    if not name:
        return jsonify({"error": "name or remote_url required"}), 400

    db = SessionLocal()
    try:
        existing = db.query(Repo).filter(Repo.name == name).first()
        if existing:
            return jsonify({"repo": existing.to_dict()}), 200

        repo = Repo(
            name=name,
            remote_url=data.get("remote_url"),
            default_branch=data.get("default_branch", "main"),
        )
        db.add(repo)
        db.commit()
        db.refresh(repo)
        return jsonify({"repo": repo.to_dict()}), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@dashboard_bp.route("/repos/<int:repo_id>", methods=["GET"])
def get_repo(repo_id):
    """Get repo detail with recent commits and file hotspots."""
    db = SessionLocal()
    try:
        repo = db.query(Repo).filter(Repo.id == repo_id).first()
        if not repo:
            return jsonify({"error": "Repo not found"}), 404

        # Recent commits
        commits = db.query(Commit).filter(
            Commit.repo_id == repo_id
        ).order_by(desc(Commit.timestamp)).limit(50).all()

        # Score distribution
        all_commits = db.query(Commit).filter(Commit.repo_id == repo_id).all()
        band_counts = {"low": 0, "medium": 0, "high": 0}
        for c in all_commits:
            if c.risk_label in band_counts:
                band_counts[c.risk_label] += 1

        # File hotspot data: parse files_touched JSON from all commits
        file_stats = {}
        for c in all_commits:
            if c.files_touched:
                try:
                    files = json.loads(c.files_touched) if isinstance(c.files_touched, str) else c.files_touched
                    for f in files:
                        if f not in file_stats:
                            file_stats[f] = {"changes": 0, "reverts": 0, "authors": set()}
                        file_stats[f]["changes"] += 1
                        if c.author:
                            file_stats[f]["authors"].add(c.author)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Sort files by change count and serialize sets
        top_files = sorted(file_stats.items(), key=lambda x: x[1]["changes"], reverse=True)[:20]
        hotspots = [
            {"file": f, "changes": s["changes"], "authors": len(s["authors"])}
            for f, s in top_files
        ]

        return jsonify({
            "repo": repo.to_dict(),
            "commits": [c.to_dict() for c in commits],
            "band_counts": band_counts,
            "hotspots": hotspots,
            "total_commits": len(all_commits),
        }), 200
    finally:
        db.close()


@dashboard_bp.route("/repos/<int:repo_id>/commits", methods=["POST"])
def log_commit(repo_id):
    """Log a scored commit for a repo."""
    data = request.get_json(force=True)
    db = SessionLocal()
    try:
        repo = db.query(Repo).filter(Repo.id == repo_id).first()
        if not repo:
            return jsonify({"error": "Repo not found"}), 404

        ts = data.get("timestamp")
        if ts and isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts and isinstance(ts, (int, float)):
            ts = datetime.utcfromtimestamp(ts)
        else:
            ts = datetime.utcnow()

        commit = Commit(
            repo_id=repo_id,
            sha=data.get("sha", ""),
            author=data.get("author"),
            timestamp=ts,
            score=data.get("score"),
            band=data.get("band"),
            risk_label=data.get("risk_label"),
            rule_results=json.dumps(data.get("rule_results")) if data.get("rule_results") else None,
            shap_top3=json.dumps(data.get("shap_top3")) if data.get("shap_top3") else None,
            message=data.get("message"),
            files_touched=json.dumps(data.get("files_touched")) if data.get("files_touched") else None,
            lines_added=data.get("lines_added"),
            lines_deleted=data.get("lines_deleted"),
        )
        db.add(commit)
        db.commit()
        db.refresh(commit)
        return jsonify({"commit": commit.to_dict()}), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@dashboard_bp.route("/commits/<int:commit_id>", methods=["GET"])
def get_commit(commit_id):
    """Get full commit detail with SHAP breakdown and rule results."""
    db = SessionLocal()
    try:
        commit = db.query(Commit).filter(Commit.id == commit_id).first()
        if not commit:
            return jsonify({"error": "Commit not found"}), 404

        result = commit.to_dict()
        # Parse JSON fields
        if result.get("rule_results"):
            try:
                result["rule_results"] = json.loads(result["rule_results"])
            except (json.JSONDecodeError, TypeError):
                pass
        if result.get("shap_top3"):
            try:
                result["shap_top3"] = json.loads(result["shap_top3"])
            except (json.JSONDecodeError, TypeError):
                pass
        if result.get("files_touched"):
            try:
                result["files_touched"] = json.loads(result["files_touched"])
            except (json.JSONDecodeError, TypeError):
                pass

        return jsonify(result), 200
    finally:
        db.close()


# ── Drift monitoring endpoints ────────────────────────────────────────

@dashboard_bp.route("/drift", methods=["GET"])
def get_drift():
    """Get per-repo drift status from the latest drift analysis."""
    # Search for drift_results.json in common locations
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", "..", "data", "drift_results.json"),
        os.path.join(base, "..", "..", "..", "data", "drift_results.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return jsonify(json.load(f)), 200
    return jsonify({"error": "No drift results available. Run scripts/drift_per_repo.py first."}), 404

