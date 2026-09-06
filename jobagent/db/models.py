"""SQLAlchemy 2.0 ORM models.

The canonical ``Job`` table plus ``applications`` and an append-only ``events``
table (one row per status change, so we keep a timeline rather than only the
current state).
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .enums import JobStatus, RemoteType

# Portable JSON: JSONB on Postgres, plain JSON elsewhere (e.g. SQLite in tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")

# Share a single Enum type instance so Postgres creates each type exactly once
# even though several columns reference it.
JOB_STATUS_ENUM = SAEnum(JobStatus, name="job_status")
REMOTE_TYPE_ENUM = SAEnum(RemoteType, name="remote_type")


class Base(DeclarativeBase):
    """Declarative base carrying the shared metadata (used by Alembic)."""


class Job(Base):
    __tablename__ = "jobs"

    # "{source}:{board_job_id}" — stable, globally unique.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, index=True)
    company: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    remote_type: Mapped[RemoteType] = mapped_column(
        REMOTE_TYPE_ENUM, default=RemoteType.unknown, nullable=False
    )
    url: Mapped[str] = mapped_column(String)
    description_html: Mapped[str] = mapped_column(Text, default="")
    description_text: Mapped[str] = mapped_column(Text, default="")
    # Collapsed locations of a near-duplicate cluster (canonical row only).
    locations: Mapped[list[str] | None] = mapped_column(JSONType, nullable=True)

    posted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Age at last poll, in days (NULL if posted_at unknown). Persisted so Phase 2
    # ranking can penalize old postings even when freshness is in 'keep' mode.
    age_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sponsorship_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_breakdown: Mapped[dict[str, Any] | None] = mapped_column(
        JSONType, nullable=True
    )
    # Persisted Stage A embedding (computed once; keyed by text + model hash).
    embedding: Mapped[list[float] | None] = mapped_column(JSONType, nullable=True)
    embedding_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[JobStatus] = mapped_column(
        JOB_STATUS_ENUM, default=JobStatus.new, index=True, nullable=False
    )
    # Why a job was filtered (relevance/sponsorship reason code); NULL when kept.
    filter_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    # When the human marked this job applied (via `jobagent run`). status='applied'
    # is the permanent dedup key; applied_at records when it happened. NULL until.
    applied_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    first_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )

    events: Mapped[list["Event"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="Event.occurred_at",
    )
    applications: Mapped[list["Application"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[JobStatus] = mapped_column(
        JOB_STATUS_ENUM, default=JobStatus.queued, nullable=False
    )
    resume_version: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))

    job: Mapped["Job"] = relationship(back_populates="applications")


class Event(Base):
    """Append-only status-change log. One row per transition == a timeline."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    from_status: Mapped[JobStatus | None] = mapped_column(
        JOB_STATUS_ENUM, nullable=True
    )
    to_status: Mapped[JobStatus] = mapped_column(JOB_STATUS_ENUM, nullable=False)
    note: Mapped[str | None] = mapped_column(String, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    occurred_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )

    job: Mapped["Job"] = relationship(back_populates="events")
