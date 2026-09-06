"""Lever board adapter.

Public endpoint (unauthenticated):
    GET https://api.lever.co/v0/postings/{token}?mode=json

Returns a JSON array of postings; ``descriptionPlain`` gives plain text and
``workplaceType`` gives an explicit remote/hybrid/on-site value.
"""
from __future__ import annotations

from typing import Any, cast

from ..db.schemas import Job
from .base import BoardAdapter
from .normalize import normalize_description, parse_epoch_ms, remote_from_workplace

_BASE = "https://api.lever.co/v0/postings"


class LeverAdapter(BoardAdapter):
    source = "lever"

    async def fetch(self, token: str) -> list[Job]:
        data = cast(
            list[dict[str, Any]],
            await self._http.get_json(f"{_BASE}/{token}", params={"mode": "json"}),
        )
        jobs: list[Job] = []
        for raw in data:
            job_id = str(raw["id"])
            categories = raw.get("categories") or {}
            html_str, text = normalize_description(
                raw.get("description"), raw.get("descriptionPlain")
            )
            jobs.append(
                Job(
                    id=f"{self.source}:{job_id}",
                    source=self.source,
                    company=token,
                    title=raw.get("text", ""),
                    location=categories.get("location"),
                    remote_type=remote_from_workplace(raw.get("workplaceType")),
                    url=raw.get("hostedUrl") or raw.get("applyUrl", ""),
                    description_html=html_str,
                    description_text=text,
                    posted_at=parse_epoch_ms(raw.get("createdAt")),
                )
            )
        return jobs
