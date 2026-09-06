"""Detect which ATS a job's application form uses, from its source/URL.

We only ingest Greenhouse, Lever, and Ashby, so those are the only boards we
support. Part B ships the Greenhouse adapter first."""
from __future__ import annotations

from enum import Enum


class Board(str, Enum):
    greenhouse = "greenhouse"
    lever = "lever"
    ashby = "ashby"
    unknown = "unknown"


_URL_MARKERS = {
    Board.greenhouse: ("greenhouse.io", "gh_jid="),
    Board.lever: ("jobs.lever.co", "lever.co"),
    Board.ashby: ("jobs.ashbyhq.com", "ashbyhq.com"),
}


def detect_board(source: str | None, url: str | None) -> Board:
    """Prefer the ingest source (authoritative); fall back to URL markers."""
    if source:
        s = source.strip().lower()
        for b in (Board.greenhouse, Board.lever, Board.ashby):
            if s == b.value:
                return b
    if url:
        u = url.lower()
        for board, markers in _URL_MARKERS.items():
            if any(m in u for m in markers):
                return board
    return Board.unknown
