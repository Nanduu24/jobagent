"""Ashby board adapter.

Public endpoint (unauthenticated):
    GET https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true

Response is ``{"jobs": [...], "apiVersion": ...}``; each job has
``descriptionPlain``/``descriptionHtml``, an explicit ``workplaceType``, and
``publishedAt`` (ISO-8601).
"""
from __future__ import annotations

from typing import Any, cast

from ..db.schemas import Job
from .base import BoardAdapter
from .normalize import normalize_description, parse_iso, remote_from_workplace

_BASE = "https://api.ashbyhq.com/posting-api/job-board"


class AshbyAdapter(BoardAdapter):
    source = "ashby"

    async def fetch(self, token: str) -> list[Job]:
        data = cast(
            dict[str, Any],
            await self._http.get_json(
                f"{_BASE}/{token}", params={"includeCompensation": "true"}
            ),
        )
        jobs: list[Job] = []
        for raw in data.get("jobs", []):
            job_id = str(raw["id"])
            # Ashby already provides descriptionPlain; prefer it over re-deriving.
            html_str, text = normalize_description(
                raw.get("descriptionHtml"), raw.get("descriptionPlain")
            )
            jobs.append(
                Job(
                    id=f"{self.source}:{job_id}",
                    source=self.source,
                    company=token,
                    title=raw.get("title", ""),
                    location=raw.get("location"),
                    remote_type=remote_from_workplace(raw.get("workplaceType")),
                    url=raw.get("jobUrl") or raw.get("applyUrl", ""),
                    description_html=html_str,
                    description_text=text,
                    posted_at=parse_iso(raw.get("publishedAt")),
                )
            )
        return jobs
