"""Pydantic v2 models at the ingest boundary.

Adapters produce ``Job`` objects; the repository persists them into the ORM
models in ``models.py``. This is the normalized, board-agnostic shape.
"""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict

from .enums import RemoteType


class Job(BaseModel):
    """A normalized job posting as emitted by a board adapter.

    ``id`` is the canonical primary key ``"{source}:{board_job_id}"``.
    Fields that only exist after persistence/scoring (match_score, status,
    seen timestamps) are intentionally *not* part of the ingest boundary.
    """

    model_config = ConfigDict(frozen=False)

    id: str
    source: str
    company: str
    title: str
    location: str | None = None
    remote_type: RemoteType = RemoteType.unknown
    url: str
    # Both forms are persisted: raw HTML for later rendering, normalized plain
    # text for the sponsorship filter and all future scoring.
    description_html: str = ""
    description_text: str = ""
    # Collapsed locations of a near-duplicate cluster (canonical row only).
    locations: list[str] = []
    posted_at: dt.datetime | None = None
    # Populated by the sponsorship filter before upsert; None == unknown.
    sponsorship_ok: bool | None = None
