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
    pass


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
