"""add jobs.applied_at (timestamp the human marked a job applied)

status='applied' is the permanent dedup key for `jobagent run`; applied_at
records when the transition happened. NULL until the human marks it.

Revision ID: 0007_applied_at
Revises: 0006_locations
Create Date: 2026-07-17
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007_applied_at"
down_revision: Union[str, None] = "0006_locations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("jobs", "applied_at")
