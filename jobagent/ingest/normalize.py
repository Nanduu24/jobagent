"""Shared normalization helpers for board adapters."""
from __future__ import annotations

import datetime as dt
import html

from ..db.enums import RemoteType
from ..text import collapse_whitespace, html_to_text

_ESCAPED_MARKERS = ("&lt;", "&gt;", "&amp;")


def normalize_description(
    raw_html: str | None, plain: str | None = None
) -> tuple[str, str]:
    """Normalize a board description at the ingest boundary.

    Returns ``(description_html, description_text)``:

    - ``description_html`` — renderable HTML kept for later display. Entity-
      escaped HTML (as Greenhouse returns) is unescaped once so it renders.
    - ``description_text`` — normalized plain text for the sponsorship filter
      and all future scoring. Prefers a board-provided ``plain`` field (e.g.
      Ashby/Lever ``descriptionPlain``) rather than re-deriving it from HTML.
    """
    html_str = raw_html or ""
    if any(marker in html_str for marker in _ESCAPED_MARKERS):
        html_str = html.unescape(html_str)

    if plain and plain.strip():
        text = collapse_whitespace(plain)
    else:
        text = html_to_text(html_str)
    return html_str, text


def parse_iso(value: str | None) -> dt.datetime | None:
    """Parse an ISO-8601 timestamp, tolerating a trailing 'Z'."""
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(raw)
    except ValueError:
        return None


def parse_epoch_ms(value: int | float | None) -> dt.datetime | None:
    """Parse a millisecond epoch (Lever's ``createdAt``)."""
    if value is None:
        return None
    try:
        return dt.datetime.fromtimestamp(float(value) / 1000.0, tz=dt.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def remote_from_text(location: str | None) -> RemoteType:
    """Infer remote type from a free-text location (used when a board gives
    no explicit workplace field)."""
    if not location:
        return RemoteType.unknown
    low = location.lower()
    if "remote" in low:
        return RemoteType.remote
    if "hybrid" in low:
        return RemoteType.hybrid
    return RemoteType.unknown


def remote_from_workplace(value: str | None) -> RemoteType:
    """Map an explicit board workplace field (Lever/Ashby) to RemoteType."""
    if not value:
        return RemoteType.unknown
    low = value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if low == "remote":
        return RemoteType.remote
    if low == "hybrid":
        return RemoteType.hybrid
    if low in {"onsite", "inperson", "office"}:
        return RemoteType.onsite
    return RemoteType.unknown
