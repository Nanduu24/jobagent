"""split description into html/text; drop ingest-generated events

Revision ID: 0002_split_description
Revises: 0001_initial
Create Date: 2026-07-12

- Replace jobs.description with description_html (raw, for rendering) and
  description_text (normalized, for filtering/scoring).
- Backfill PER ROW: description_html keeps the raw value verbatim; description
  _text is html_to_text(raw); sponsorship_ok is recomputed from the normalized
  text so no stale verdict survives (safe whether the old column held stripped
  text or raw HTML — e.g. a restored dump).
- events is application-scoped only; remove any ingest-generated rows.

This is a data migration and must run online (not `alembic upgrade --sql`).
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Reuse the exact production normalizer + classifier so backfilled rows match
# what a fresh ingest would produce.
from jobagent.filters.sponsorship import classify_sponsorship
from jobagent.text import html_to_text

revision: str = "0002_split_description"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns with a temporary server default so existing rows are valid.
    op.add_column(
        "jobs",
        sa.Column("description_html", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "jobs",
        sa.Column("description_text", sa.Text(), nullable=False, server_default=""),
    )

    # Per-row backfill. Correct regardless of whether the old `description`
    # column held stripped text (Phase 1) or raw HTML (e.g. a restored dump):
    # html_to_text is idempotent on already-plain text, and re-running the
    # sponsorship filter over the normalized text refreshes sponsorship_ok.
    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id, description FROM jobs")).fetchall()
    update = sa.text(
        "UPDATE jobs SET description_html = :html, description_text = :text, "
        "sponsorship_ok = :spons WHERE id = :id"
    )
    for row_id, description in rows:
        raw = description or ""
        text = html_to_text(raw)
        bind.execute(
            update,
            {
                "html": raw,
                "text": text,
                "spons": classify_sponsorship(text),
                "id": row_id,
            },
        )

    op.drop_column("jobs", "description")

    # Drop the temporary defaults now that data is populated.
    op.alter_column("jobs", "description_html", server_default=None)
    op.alter_column("jobs", "description_text", server_default=None)

    # Job status changes are no longer logged as events; remove any
    # ingest-generated rows. (Phase 1 only ever created ingest events.)
    op.execute("DELETE FROM events")


def downgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
    )
    op.execute("UPDATE jobs SET description = description_text")
    op.alter_column("jobs", "description", server_default=None)
    op.drop_column("jobs", "description_text")
    op.drop_column("jobs", "description_html")
