"""Tests for the freshness filter."""
from __future__ import annotations

import datetime as dt

from jobagent.filters.freshness import age_in_days, is_stale

NOW = dt.datetime(2026, 7, 13, tzinfo=dt.timezone.utc)


def test_recent_posting_is_fresh() -> None:
    assert is_stale(NOW - dt.timedelta(days=10), NOW, 30) is False


def test_old_posting_is_stale() -> None:
    assert is_stale(NOW - dt.timedelta(days=31), NOW, 30) is True


def test_boundary_exactly_max_age_is_fresh() -> None:
    # Exactly 30 days old is not yet older-than-30.
    assert is_stale(NOW - dt.timedelta(days=30), NOW, 30) is False


def test_missing_posted_at_is_not_stale() -> None:
    assert is_stale(None, NOW, 30) is False


def test_naive_posted_at_assumed_utc() -> None:
    naive = (NOW - dt.timedelta(days=40)).replace(tzinfo=None)
    assert is_stale(naive, NOW, 30) is True


def test_age_in_days() -> None:
    assert age_in_days(NOW - dt.timedelta(days=45), NOW) == 45
    assert age_in_days(None, NOW) is None
    # Future-dated posting -> negative age (not clamped).
    assert age_in_days(NOW + dt.timedelta(days=2), NOW) == -2
