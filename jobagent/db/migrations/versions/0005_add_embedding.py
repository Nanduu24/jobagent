"""add jobs.embedding + embedding_hash + embedding_model (Stage A cache)

Revision ID: 0005_embedding
Revises: 0004_age_days
Create Date: 2026-07-13
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_embedding"
down_revision: Union[str, None] = "0004_age_days"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("embedding", postgresql.JSONB(), nullable=True))
    op.add_column("jobs", sa.Column("embedding_hash", sa.String(), nullable=True))
    op.add_column("jobs", sa.Column("embedding_model", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "embedding_model")
    op.drop_column("jobs", "embedding_hash")
    op.drop_column("jobs", "embedding")
