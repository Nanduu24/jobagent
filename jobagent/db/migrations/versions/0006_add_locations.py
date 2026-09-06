"""add jobs.locations (collapsed near-duplicate locations)

Revision ID: 0006_locations
Revises: 0005_embedding
Create Date: 2026-07-14
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_locations"
down_revision: Union[str, None] = "0005_embedding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("locations", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "locations")
