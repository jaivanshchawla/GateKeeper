#!/usr/bin/env python3
"""
Dashboard routes for the Gatekeeper webhook.
Provides API endpoints for managing issues from Gates 1-3.
"""

from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request
from sqlalchemy import desc, func

# Handle both local development and Docker context
try:
    from webhook.models import Issue, SessionLocal, init_db
except ImportError:
    from models import Issue, SessionLocal, init_db

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
