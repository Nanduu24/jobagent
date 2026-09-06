"""STEP 0 — prove markup fragility of the sponsorship filter.

These feed the filter *raw adapter payloads* (not clean text) for each of the
four boards, in each board's native encoding, including phrases split across
tags. A robust classifier must return False on every one of these; a
markup-fragile one lets them through as None.

The per-board descriptions are lifted from real captured response bodies (see
tests/fixtures/raw/*), with a visa-blocking phrase embedded in the board's own
markup so the regression is realistic.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from jobagent.filters.sponsorship import classify_sponsorship
from jobagent.ingest.ashby import AshbyAdapter
from jobagent.ingest.base import HttpClient
from jobagent.ingest.greenhouse import GreenhouseAdapter
from jobagent.ingest.lever import LeverAdapter
from jobagent.ingest.workable import WorkableAdapter

from .conftest import load_raw

# --- The three required literal cases -------------------------------------
LITERAL_CASES: list[tuple[str, str]] = [
    # blocking phrase split across an inline tag
    ("split-tag", "<p>We are <strong>unable</strong> to sponsor</p>"),
    # HTML-escaped, as Greenhouse returns it
    ("gh-escaped", "&lt;li&gt;No visa sponsorship&lt;/li&gt;"),
    # escaped AND split across tags
    (
        "gh-escaped-split",
        "&lt;p&gt;We are &lt;strong&gt;unable&lt;/strong&gt; to sponsor&lt;/p&gt;",
    ),
]


def _greenhouse_desc() -> str:
    return load_raw("greenhouse_raw.json")["jobs"][0]["content"]


def _lever_desc() -> str:
    # The HTML field (not descriptionPlain) — worst case for the filter.
    return load_raw("lever_raw.json")[0]["description"]


def _ashby_desc() -> str:
    return load_raw("ashby_raw.json")["jobs"][0]["descriptionHtml"]


def _workable_desc() -> str:
    return load_raw("workable_detail_raw.json")["description"]


BOARD_CASES: list[tuple[str, str]] = [
    ("greenhouse", _greenhouse_desc()),
    ("lever", _lever_desc()),
    ("ashby", _ashby_desc()),
    ("workable", _workable_desc()),
]


@pytest.mark.parametrize("name, raw", LITERAL_CASES, ids=[c[0] for c in LITERAL_CASES])
def test_filter_blocks_literal_markup(name: str, raw: str) -> None:
    assert classify_sponsorship(raw) is False


@pytest.mark.parametrize("board, raw", BOARD_CASES, ids=[c[0] for c in BOARD_CASES])
def test_filter_blocks_raw_board_payload(board: str, raw: str) -> None:
    assert classify_sponsorship(raw) is False


# --- Boundary end-to-end: adapter normalization + filter ------------------
# Prove that raw payloads routed through each adapter produce a clean
# description_text (no tags) that the filter blocks, while description_html
# retains renderable markup.


def _http(client: httpx.AsyncClient) -> HttpClient:
    return HttpClient(client=client, min_interval=0.0, max_retries=0)


@respx.mock
@pytest.mark.asyncio
async def test_greenhouse_boundary_normalizes_and_blocks() -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").respond(
        json=load_raw("greenhouse_raw.json")
    )
    async with httpx.AsyncClient() as client:
        jobs = await GreenhouseAdapter(_http(client)).fetch("acme")
    job = jobs[0]
    assert "<" not in job.description_text  # tags stripped, entities unescaped
    assert "&lt;" not in job.description_text
    assert classify_sponsorship(job.description_text) is False
    assert "<p>" in job.description_html  # raw markup retained for rendering


@respx.mock
@pytest.mark.asyncio
async def test_ashby_boundary_normalizes_and_blocks() -> None:
    respx.get("https://api.ashbyhq.com/posting-api/job-board/acme").respond(
        json=load_raw("ashby_raw.json")
    )
    async with httpx.AsyncClient() as client:
        jobs = await AshbyAdapter(_http(client)).fetch("acme")
    job = jobs[0]
    assert "<" not in job.description_text
    assert classify_sponsorship(job.description_text) is False
    assert "<strong>" in job.description_html


@respx.mock
@pytest.mark.asyncio
async def test_lever_boundary_normalizes_and_blocks() -> None:
    respx.get("https://api.lever.co/v0/postings/acme").respond(
        json=load_raw("lever_raw.json")
    )
    async with httpx.AsyncClient() as client:
        jobs = await LeverAdapter(_http(client)).fetch("acme")
    job = jobs[0]
    assert "<" not in job.description_text
    assert classify_sponsorship(job.description_text) is False


@respx.mock
@pytest.mark.asyncio
async def test_workable_boundary_normalizes_and_blocks() -> None:
    respx.post("https://apply.workable.com/api/v3/accounts/acme/jobs").respond(
        json=load_raw("workable_list_raw.json")
    )
    respx.get(
        "https://apply.workable.com/api/v1/accounts/acme/jobs/RAWWORK01"
    ).respond(json=load_raw("workable_detail_raw.json"))
    async with httpx.AsyncClient() as client:
        jobs = await WorkableAdapter(_http(client)).fetch("acme")
    job = jobs[0]
    assert "<" not in job.description_text
    assert classify_sponsorship(job.description_text) is False
