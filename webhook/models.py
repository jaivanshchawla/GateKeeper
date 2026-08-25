#!/usr/bin/env python3
"""
SQLAlchemy models for the Gatekeeper dashboard.
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://gatekeeper:gatekeeper@localhost:5432/gatekeeper"  # pragma: allowlist secret
)

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all models."""


class Issue(Base):
    """Issue model for tracking problems from Gates 1-3."""
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, index=True)
    gate = Column(Integer, nullable=False)  # 1, 2, or 3
    type = Column(String(50), nullable=False)  # e.g., "high_risk_pr", "secret_detected", "smoke_test_failed"
    repo = Column(String(255), nullable=False, index=True)  # Repository name
    status = Column(String(20), nullable=False, default="open")  # "open" or "resolved"
    details = Column(Text, nullable=True)  # Additional details about the issue
    commit_hash = Column(String(40), nullable=True)  # Associated commit hash
    risk_score = Column(Integer, nullable=True)  # Risk score if applicable (0-100)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "gate": self.gate,
            "type": self.type,
            "repo": self.repo,
            "status": self.status,
            "details": self.details,
            "commit_hash": self.commit_hash,
            "risk_score": self.risk_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Repo(Base):
    """Repository model for per-repo dashboard."""
    __tablename__ = "repos"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    remote_url = Column(String(500), nullable=True)
    default_branch = Column(String(100), nullable=False, default="main")
    registered_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "remote_url": self.remote_url,
            "default_branch": self.default_branch,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
        }


class Commit(Base):
    """Commit model for tracking scored commits per repo."""
    __tablename__ = "commits"

    id = Column(Integer, primary_key=True, index=True)
    repo_id = Column(Integer, nullable=False, index=True)
    sha = Column(String(40), nullable=False)
    author = Column(String(255), nullable=True)
    timestamp = Column(DateTime, nullable=True)
    score = Column(Integer, nullable=True)  # 0-100 scaled
    band = Column(String(10), nullable=True)  # low/medium/high
    risk_label = Column(String(10), nullable=True)
    rule_results = Column(Text, nullable=True)  # JSON
    shap_top3 = Column(Text, nullable=True)  # JSON
    message = Column(Text, nullable=True)
    files_touched = Column(Text, nullable=True)  # JSON list
    lines_added = Column(Integer, nullable=True)
    lines_deleted = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "sha": self.sha,
            "author": self.author,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "score": self.score,
            "band": self.band,
            "risk_label": self.risk_label,
            "rule_results": self.rule_results,
            "shap_top3": self.shap_top3,
            "message": self.message,
            "files_touched": self.files_touched,
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def init_db():
    """Initialize database and create tables."""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized: {DATABASE_URL}")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
