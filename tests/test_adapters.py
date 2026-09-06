"""respx-mocked adapter tests: one fixture per board."""
from __future__ import annotations

import httpx
import pytest
import respx

from jobagent.db.enums import RemoteType
from jobagent.ingest.ashby import AshbyAdapter
from jobagent.ingest.base import HttpClient
from jobagent.ingest.greenhouse import GreenhouseAdapter
from jobagent.ingest.lever import LeverAdapter
from jobagent.ingest.workable import WorkableAdapter

from .conftest import load_fixture


def _http(client: httpx.AsyncClient) -> HttpClient:
    return HttpClient(client=client, min_interval=0.0, max_retries=0)


@respx.mock
@pytest.mark.asyncio
async def test_greenhouse_adapter() -> None:
    respx.get(
        "https://boards-api.greenhouse.io/v1/boards/acme/jobs"
    ).respond(json=load_fixture("greenhouse.json"))

    async with httpx.AsyncClient() as client:
        jobs = await GreenhouseAdapter(_http(client)).fetch("acme")

    assert len(jobs) == 2
    j = jobs[0]
    assert j.id == "greenhouse:4001"
    assert j.source == "greenhouse"
    assert j.company == "Acme AI"
    assert j.title == "Machine Learning Engineer"
    assert j.location == "San Francisco, CA"
    assert j.url == "https://boards.greenhouse.io/acme/jobs/4001"
    # description_text: entities unescaped and tags stripped.
    assert "<p>" not in j.description_text
    assert "Visa sponsorship is available" in j.description_text
    # description_html: raw markup retained (unescaped once so it renders).
    assert "<p>" in j.description_html
    assert j.posted_at is not None
    # Second job: location "Remote - US" -> remote.
    assert jobs[1].remote_type is RemoteType.remote


@respx.mock
@pytest.mark.asyncio
async def test_lever_adapter() -> None:
    respx.get(
        "https://api.lever.co/v0/postings/acme"
    ).respond(json=load_fixture("lever.json"))

    async with httpx.AsyncClient() as client:
        jobs = await LeverAdapter(_http(client)).fetch("acme")

    assert len(jobs) == 2
    j = jobs[0]
    assert j.id == "lever:abc-123-uuid"
    assert j.company == "acme"
    assert j.title == "Senior Backend Engineer"
    assert j.location == "New York, NY"
    assert j.remote_type is RemoteType.hybrid
    assert j.url == "https://jobs.lever.co/acme/abc-123-uuid"
    # Lever provides descriptionPlain -> used as text; html field retained.
    assert "sponsor the right candidate" in j.description_text
    assert "<p>" not in j.description_text
    assert "<p>" in j.description_html
    assert j.posted_at is not None
    assert jobs[1].remote_type is RemoteType.remote


@respx.mock
@pytest.mark.asyncio
async def test_ashby_adapter() -> None:
    respx.get(
        "https://api.ashbyhq.com/posting-api/job-board/acme"
    ).respond(json=load_fixture("ashby.json"))

    async with httpx.AsyncClient() as client:
        jobs = await AshbyAdapter(_http(client)).fetch("acme")

    assert len(jobs) == 2
    j = jobs[0]
    assert j.id == "ashby:ashby-uuid-1"
    assert j.company == "acme"
    assert j.title == "AI Infrastructure Engineer"
    assert j.location == "San Francisco"
    assert j.remote_type is RemoteType.hybrid
    assert j.url == "https://jobs.ashbyhq.com/acme/ashby-uuid-1"
    # Ashby descriptionPlain preferred for text; descriptionHtml kept as html.
    assert "happy to sponsor" in j.description_text
    assert "<p>" not in j.description_text
    assert "<p>" in j.description_html
    assert j.posted_at is not None
    assert jobs[1].remote_type is RemoteType.remote


@respx.mock
@pytest.mark.asyncio
async def test_workable_adapter() -> None:
    respx.post(
        "https://apply.workable.com/api/v3/accounts/acme/jobs"
    ).respond(json=load_fixture("workable_list.json"))
    respx.get(
        "https://apply.workable.com/api/v1/accounts/acme/jobs/F8427A442D"
    ).respond(json=load_fixture("workable_detail.json"))

    async with httpx.AsyncClient() as client:
        jobs = await WorkableAdapter(_http(client)).fetch("acme")

    assert len(jobs) == 1
    j = jobs[0]
    assert j.id == "workable:F8427A442D"
    assert j.company == "acme"
    assert j.title == "Senior Python Software Engineer"
    assert j.location == "New York, New York, United States"
    assert j.remote_type is RemoteType.remote
    assert j.url == "https://apply.workable.com/acme/j/F8427A442D/"
    assert "open-source team" in j.description_text
    assert "<p>" not in j.description_text
    assert "<p>" in j.description_html
    assert j.posted_at is not None


@respx.mock
@pytest.mark.asyncio
async def test_workable_adapter_follows_pagination() -> None:
    """Large accounts paginate at 10/page via a nextPage cursor; the adapter
    must follow it and not truncate to the first page."""
    list_url = "https://apply.workable.com/api/v3/accounts/acme/jobs"
    detail = load_fixture("workable_detail.json")

    def _list_page(shortcode: str, next_page: str | None) -> dict[str, object]:
        return {
            "total": 2,
            "results": [{"shortcode": shortcode}],
            "nextPage": next_page,
        }

    # Page 1 carries a cursor; page 2 does not (end of results).
    respx.post(list_url).mock(
        side_effect=[
            httpx.Response(200, json=_list_page("PAGE1CODE", "CURSOR_TOKEN")),
            httpx.Response(200, json=_list_page("PAGE2CODE", None)),
        ]
    )
    for code in ("PAGE1CODE", "PAGE2CODE"):
        respx.get(
            f"https://apply.workable.com/api/v1/accounts/acme/jobs/{code}"
        ).respond(json={**detail, "shortcode": code})

    async with httpx.AsyncClient() as client:
        jobs = await WorkableAdapter(_http(client)).fetch("acme")

    # Both pages fetched -> two jobs, not one.
    assert {j.id for j in jobs} == {"workable:PAGE1CODE", "workable:PAGE2CODE"}
    # The second list POST echoed the cursor back in its request body.
    list_posts = [
        c
        for c in respx.calls
        if c.request.method == "POST" and str(c.request.url) == list_url
    ]
    assert len(list_posts) == 2
    assert b"CURSOR_TOKEN" in list_posts[1].request.content
