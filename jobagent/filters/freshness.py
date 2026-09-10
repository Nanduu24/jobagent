"""Freshness filter: cut postings older than MAX_AGE_DAYS."""
from __future__ import annotations

import datetime as dt

REASON_STALE = "stale"


def is_stale(
    posted_at: dt.datetime | None,
    now: dt.datetime,
    max_age_days: int,
) -> bool:
    """Return True if ``posted_at`` is older than ``max_age_days``.

    A missing ``posted_at`` is treated as NOT stale (unknown age -> keep, to
    avoid cutting a good posting on missing data).
    """
    if posted_at is None:
        return False
    if posted_at.tzinfo is None:
        posted_at = posted_at.replace(tzinfo=dt.UTC)
    return posted_at < now - dt.timedelta(days=max_age_days)


def age_in_days(posted_at: dt.datetime | None, now: dt.datetime) -> int | None:
    """Age of a posting in whole days, or None if ``posted_at`` is unknown."""
    if posted_at is None:
        return None
    if posted_at.tzinfo is None:
        posted_at = posted_at.replace(tzinfo=dt.UTC)
    return (now - posted_at).days
