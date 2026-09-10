"""Persistence operations: idempotent upsert keyed on job id.

Job-level history is captured by ``first_seen_at`` / ``last_seen_at`` only. The
``events`` table is application-scoped (submitted, ack, recruiter reply,
rejection) and is intentionally NOT written during ingest.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy.ext.asyncio import AsyncSession

from . import models
from .enums import JobStatus
from .schemas import Job as JobIn


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


async def upsert_job(
    session: AsyncSession,
    job: JobIn,
    *,
    now: dt.datetime | None = None,
) -> tuple[models.Job, bool]:
    """Insert a new job or refresh an existing one (keyed on ``job.id``).

    On first observation the row is created with status ``new``. On
    re-observation, ``last_seen_at`` is bumped (so stale postings can be
    detected) and mutable fields are refreshed; ``first_seen_at`` is never
    touched. No events are written — job history lives in the seen timestamps.

    Returns ``(model, is_new)``.
    """
    ts = now or _utcnow()
    existing = await session.get(models.Job, job.id)

    if existing is None:
        model = models.Job(
            id=job.id,
            source=job.source,
            company=job.company,
            title=job.title,
            location=job.location,
            remote_type=job.remote_type,
            url=job.url,
            description_html=job.description_html,
            description_text=job.description_text,
            locations=job.locations or None,
            posted_at=job.posted_at,
            sponsorship_ok=job.sponsorship_ok,
            status=JobStatus.new,
            first_seen_at=ts,
            last_seen_at=ts,
        )
        session.add(model)
        return model, True

    # Re-observation: bump last_seen_at and refresh volatile fields.
    existing.last_seen_at = ts
    existing.company = job.company
    existing.title = job.title
    existing.location = job.location
    existing.remote_type = job.remote_type
    existing.url = job.url
    existing.description_html = job.description_html
    existing.description_text = job.description_text
    existing.locations = job.locations or None
    existing.posted_at = job.posted_at
    if job.sponsorship_ok is not None:
        existing.sponsorship_ok = job.sponsorship_ok
    return existing, False
