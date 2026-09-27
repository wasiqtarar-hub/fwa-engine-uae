"""Persistence behind a repository interface (build brief §2).

    "SQLite via SQLAlchemy (standing in for PostgreSQL, behind a repository
     interface so Postgres is a config change)"

That is exactly what this is. The ORM models are plain SQLAlchemy 2.0 and the
only thing that decides which engine is used is
:data:`FWA_DATABASE_URL`. Point it at ``postgresql+psycopg://…`` and nothing
else in the repository changes.

What is persisted here is the **operational state that must survive a restart**:
users, sessions, review outcomes, gate evidence and the audit log. The detection
pipeline's own outputs (signals, cases) are recomputed from the immutable source
on each run and are deliberately *not* persisted — recomputing them from the raw
store is what makes the reproducibility claim in §3.10 checkable, and a cached
signal table would be a second source of truth that could drift from the rules
that produced it.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

__all__ = ["Base", "User", "SessionRecord", "ReviewOutcome", "AuditRecord", "GateEvidence",
           "Database", "default_database_url"]


class Base(DeclarativeBase):
    pass


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class User(Base):
    """§13.3's user table, field for field."""

    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(64), primary_key=True)
    password_hash: Mapped[str] = mapped_column(String(255))          # Argon2id, never plaintext
    role: Mapped[str] = mapped_column(String(32))
    tenant_id: Mapped[str] = mapped_column(String(32), default="T001")
    display_name: Mapped[str] = mapped_column(String(128), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_login: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=True)
    extra_permissions: Mapped[str] = mapped_column(Text, default="")   # comma-separated
    assigned_case_ids: Mapped[str] = mapped_column(Text, default="")   # SIU assignment

    def to_row(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "tenant_id": self.tenant_id,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "failed_attempts": self.failed_attempts,
            "must_change_password": self.must_change_password,
            "extra_permissions": self.extra_permissions,
        }


class SessionRecord(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(ForeignKey("users.username"))
    role: Mapped[str] = mapped_column(String(32))
    tenant_id: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_reason: Mapped[str] = mapped_column(String(32), default="")


class ReviewOutcome(Base):
    """The canonical ``review_outcome`` table — the FR7 feedback source."""

    __tablename__ = "review_outcomes"

    review_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    signal_ids: Mapped[str] = mapped_column(Text, default="")
    reviewer: Mapped[str] = mapped_column(String(64))
    reviewer_role: Mapped[str] = mapped_column(String(32))
    disposition: Mapped[str] = mapped_column(String(32))
    validated_category: Mapped[str] = mapped_column(String(64), default="")
    confirmed_amount_aed: Mapped[float] = mapped_column(Float, default=0.0)
    rationale: Mapped[str] = mapped_column(Text)
    decided_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    appeal_raised: Mapped[bool] = mapped_column(Boolean, default=False)
    appeal_result: Mapped[str] = mapped_column(String(64), default="")
    appeal_decided_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    turnaround_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    tenant_id: Mapped[str] = mapped_column(String(32), default="T001")
    was_override: Mapped[bool] = mapped_column(Boolean, default=False)
    original_disposition: Mapped[str] = mapped_column(String(32), default="")

    def to_row(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id, "case_id": self.case_id, "signal_ids": self.signal_ids,
            "reviewer": self.reviewer, "reviewer_role": self.reviewer_role,
            "disposition": self.disposition, "validated_category": self.validated_category,
            "confirmed_amount_aed": self.confirmed_amount_aed, "rationale": self.rationale,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "appeal_raised": self.appeal_raised, "appeal_result": self.appeal_result,
            "turnaround_hours": self.turnaround_hours, "tenant_id": self.tenant_id,
            "was_override": self.was_override, "original_disposition": self.original_disposition,
        }


class AuditRecord(Base):
    """Durable mirror of the in-memory hash-chained audit log."""

    __tablename__ = "audit_log"

    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_time: Mapped[str] = mapped_column(String(48))
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    actor: Mapped[str] = mapped_column(String(64), index=True)
    actor_role: Mapped[str] = mapped_column(String(32), default="")
    tenant_id: Mapped[str] = mapped_column(String(32), default="")
    subject: Mapped[str] = mapped_column(String(128), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    before: Mapped[str | None] = mapped_column(Text, nullable=True)
    after: Mapped[str | None] = mapped_column(Text, nullable=True)
    prev_hash: Mapped[str] = mapped_column(String(64))
    entry_hash: Mapped[str] = mapped_column(String(64))


class GateEvidence(Base):
    """QA's record of the ten-category suite and the Table 6.1 gates."""

    __tablename__ = "gate_evidence"

    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rule_id: Mapped[str] = mapped_column(String(32), index=True)
    category: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24))
    recorded_by: Mapped[str] = mapped_column(String(64))
    recorded_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    note: Mapped[str] = mapped_column(Text, default="")


def default_database_url() -> str:
    """SQLite by default; PostgreSQL is one environment variable away."""
    url = os.environ.get("FWA_DATABASE_URL")
    if url:
        return url
    root = Path(__file__).resolve().parents[3]
    (root / "data").mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / 'data' / 'fwa.db'}"


class Database:
    """Thin repository facade. Swapping the engine is a URL change."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self.url = url or default_database_url()
        connect_args = {"check_same_thread": False} if self.url.startswith("sqlite") else {}
        self.engine = create_engine(self.url, echo=echo, future=True, connect_args=connect_args)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._session_factory()

    # ---- small helpers used by the service layer ------------------------

    def get_user(self, username: str) -> User | None:
        with self.session() as s:
            return s.get(User, username)

    def list_users(self) -> list[User]:
        with self.session() as s:
            return list(s.scalars(select(User).order_by(User.username)))

    def upsert_user(self, user: User) -> User:
        with self.session() as s:
            s.merge(user)
            s.commit()
        return user

    def add(self, obj: Any) -> Any:
        with self.session() as s:
            s.add(obj)
            s.commit()
        return obj

    def review_outcomes(self, tenant_id: str | None = None) -> list[ReviewOutcome]:
        with self.session() as s:
            stmt = select(ReviewOutcome)
            if tenant_id:
                stmt = stmt.where(ReviewOutcome.tenant_id == tenant_id)
            return list(s.scalars(stmt.order_by(ReviewOutcome.decided_at.desc())))

    def outcomes_frame(self, tenant_id: str | None = None):
        import pandas as pd

        rows = [o.to_row() for o in self.review_outcomes(tenant_id)]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=list(ReviewOutcome.__table__.columns.keys()))

    def gate_evidence(self, rule_id: str | None = None) -> list[GateEvidence]:
        with self.session() as s:
            stmt = select(GateEvidence)
            if rule_id:
                stmt = stmt.where(GateEvidence.rule_id == rule_id)
            return list(s.scalars(stmt))

    def mirror_audit(self, audit_log) -> int:
        """Persist the hash-chained log. Append-only: existing rows are never touched."""
        written = 0
        with self.session() as s:
            existing = {row for row in s.scalars(select(AuditRecord.sequence))}
            for event in audit_log:
                if event.sequence in existing:
                    continue
                row = event.to_row()
                s.add(AuditRecord(**{k: row[k] for k in AuditRecord.__table__.columns.keys()}))
                written += 1
            s.commit()
        return written
