"""Workable board adapter.

Two public endpoints (unauthenticated):
    list:   POST https://apply.workable.com/api/v3/accounts/{token}/jobs
    detail: GET  https://apply.workable.com/api/v1/accounts/{token}/jobs/{shortcode}

The list endpoint paginates at 10 results per page: each response carries a
``nextPage`` cursor which must be POSTed back in the body to get the next page.
Without following it, large accounts are silently truncated to 10 jobs. The
list response also omits the description, so we fetch each posting's detail
(rate-limited by the shared HttpClient). The public URL is constructed as
``https://apply.workable.com/{token}/j/{shortcode}/``.

.. warning::
    Page-1 fetch and the ``nextPage`` cursor SHAPE are confirmed against a live
    account (zego: total=28, 10 results + cursor). Page-2 cursor-FOLLOWING
    (re-POST with ``{"nextPage": <token>}``) is IMPLEMENTED BUT UNVERIFIED
    end-to-end against a live multi-page account — verification was blocked by
    Workable rate-limiting (HTTP 429). Workable is therefore quarantined behind
    ``Settings.enable_workable`` (default False) and excluded from the poll set.
"""
from __future__ import annotations

from typing import Any, cast

from ..db.enums import RemoteType
from ..db.schemas import Job
from .base import BoardAdapter
from .normalize import normalize_description, parse_iso

_LIST = "https://apply.workable.com/api/v3/accounts/{token}/jobs"
_DETAIL = "https://apply.workable.com/api/v1/accounts/{token}/jobs/{shortcode}"
_PUBLIC = "https://apply.workable.com/{token}/j/{shortcode}/"

# Safety cap so a malformed/looping cursor can't spin forever (10/page).
_MAX_PAGES = 1000


def _format_location(location: dict[str, Any] | None) -> str | None:
    if not location:
        return None
    parts = [location.get("city"), location.get("region"), location.get("country")]
    joined = ", ".join(p for p in parts if p)
    return joined or None


class WorkableAdapter(BoardAdapter):
    source = "workable"

    async def _list_all(self, token: str) -> list[dict[str, Any]]:
        """Page through every posting, following the ``nextPage`` cursor."""
        url = _LIST.format(token=token)
        results: list[dict[str, Any]] = []
        body: dict[str, Any] = {}
        for _ in range(_MAX_PAGES):
            page = cast(dict[str, Any], await self._http.post_json(url, json=body))
            batch = page.get("results", [])
            results.extend(batch)
            next_page = page.get("nextPage")
            if not next_page or not batch:
                break
            body = {"nextPage": next_page}
        return results

    async def fetch(self, token: str) -> list[Job]:
        jobs: list[Job] = []
        for result in await self._list_all(token):
            shortcode = result.get("shortcode")
            if not shortcode:
                continue
            detail = cast(
                dict[str, Any],
                await self._http.get_json(
                    _DETAIL.format(token=token, shortcode=shortcode)
                ),
            )
            remote_type = (
                RemoteType.remote if detail.get("remote") else RemoteType.unknown
            )
            html_str, text = normalize_description(detail.get("description"))
            jobs.append(
                Job(
                    id=f"{self.source}:{shortcode}",
                    source=self.source,
                    company=detail.get("name") or token,
                    title=detail.get("title", ""),
                    location=_format_location(detail.get("location")),
                    remote_type=remote_type,
                    url=_PUBLIC.format(token=token, shortcode=shortcode),
                    description_html=html_str,
                    description_text=text,
                    posted_at=parse_iso(detail.get("published")),
                )
            )
        return jobs
