"""add jobs.filter_reason for auditable pre-filter cuts

Revision ID: 0003_filter_reason
Revises: 0002_split_description
Create Date: 2026-07-12
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_filter_reason"
down_revision: Union[str, None] = "0002_split_description"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("filter_reason", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "filter_reason")
