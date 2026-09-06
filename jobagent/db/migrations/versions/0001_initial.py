"""initial schema: jobs, applications, events

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-12
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JOB_STATUS_VALUES = (
    "new",
    "filtered",
    "queued",
    "applied",
    "rejected",
    "interview",
    "offer",
)
REMOTE_TYPE_VALUES = ("remote", "hybrid", "onsite", "unknown")

# create_type=False: we create the enum types once, explicitly, below.
job_status = postgresql.ENUM(*JOB_STATUS_VALUES, name="job_status", create_type=False)
remote_type = postgresql.ENUM(*REMOTE_TYPE_VALUES, name="remote_type", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    job_status.create(bind, checkfirst=True)
    remote_type.create(bind, checkfirst=True)

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("company", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("location", sa.String(), nullable=True),
        sa.Column("remote_type", remote_type, nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sponsorship_ok", sa.Boolean(), nullable=True),
        sa.Column("match_score", sa.Float(), nullable=True),
        sa.Column("score_breakdown", postgresql.JSONB(), nullable=True),
        sa.Column("status", job_status, nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_source", "jobs", ["source"])
    op.create_index("ix_jobs_company", "jobs", ["company"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_last_seen_at", "jobs", ["last_seen_at"])

    op.create_table(
        "applications",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.String(),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", job_status, nullable=False),
        sa.Column("resume_version", sa.String(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_applications_job_id", "applications", ["job_id"])

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.String(),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("from_status", job_status, nullable=True),
        sa.Column("to_status", job_status, nullable=False),
        sa.Column("note", sa.String(), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_events_job_id", "events", ["job_id"])
    op.create_index("ix_events_occurred_at", "events", ["occurred_at"])


def downgrade() -> None:
    op.drop_table("events")
    op.drop_table("applications")
    op.drop_table("jobs")
    bind = op.get_bind()
    remote_type.drop(bind, checkfirst=True)
    job_status.drop(bind, checkfirst=True)
