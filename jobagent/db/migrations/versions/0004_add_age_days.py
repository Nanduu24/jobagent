"""add jobs.age_days (posting age at last poll, in days)

Revision ID: 0004_age_days
Revises: 0003_filter_reason
Create Date: 2026-07-13
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_age_days"
down_revision: Union[str, None] = "0003_filter_reason"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("age_days", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "age_days")
