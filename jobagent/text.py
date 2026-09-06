"""Neutral text-normalization primitives shared by ingest and filters.

Kept dependency-free so both the ingest boundary (which persists normalized
text) and the sponsorship filter (defense-in-depth) can use the same logic.
"""
from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def collapse_whitespace(text: str | None) -> str:
    """Normalize non-breaking spaces and runs of whitespace; keep line breaks."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("‑", "-")
    text = _INLINE_WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def html_to_text(raw: str | None) -> str:
    """Convert (possibly entity-escaped) HTML to readable plain text.

    Unescapes entities up to twice (Greenhouse double-encodes), strips tags,
    then collapses whitespace. Idempotent on already-plain text.
    """
    if not raw:
        return ""
    text = raw
    for _ in range(2):
        if "&lt;" in text or "&amp;" in text or "&gt;" in text:
            text = html.unescape(text)
        else:
            break
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return collapse_whitespace(text)
