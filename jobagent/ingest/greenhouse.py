"""Greenhouse board adapter.

Public endpoint (unauthenticated):
    GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true

With ``content=true`` each job includes its (HTML-escaped) description inline,
so no per-job request is needed.
"""
from __future__ import annotations

from typing import Any, cast

from ..db.schemas import Job
from .base import BoardAdapter
from .normalize import normalize_description, parse_iso, remote_from_text

_BASE = "https://boards-api.greenhouse.io/v1/boards"


class GreenhouseAdapter(BoardAdapter):
    source = "greenhouse"

    async def fetch(self, token: str) -> list[Job]:
        data = cast(
            dict[str, Any],
            await self._http.get_json(
                f"{_BASE}/{token}/jobs", params={"content": "true"}
            ),
        )
        jobs: list[Job] = []
        for raw in data.get("jobs", []):
            job_id = str(raw["id"])
            location = (raw.get("location") or {}).get("name")
            posted = parse_iso(raw.get("first_published") or raw.get("updated_at"))
            html_str, text = normalize_description(raw.get("content"))
            jobs.append(
                Job(
                    id=f"{self.source}:{job_id}",
                    source=self.source,
                    company=raw.get("company_name") or token,
                    title=raw.get("title", ""),
                    location=location,
                    remote_type=remote_from_text(location),
                    url=raw.get("absolute_url", ""),
                    description_html=html_str,
                    description_text=text,
                    posted_at=posted,
                )
            )
        return jobs
