"""Generator pipeline tests — select -> rewrite -> verify -> render.

Zero LLM calls: a ScriptedProvider stands in for the model, including cases where
the "model" fabricates and the pipeline MUST drop the bullet (never ship an
unverified claim — CLAUDE constraint 3). Uses the REAL fact bank so the verifier
is exercised exactly as in production.
"""
from __future__ import annotations

import pytest

from jobagent.factbank import Fact, FactBank, load_fact_bank
from jobagent.llm.base import Message
from jobagent.llm.fake import ScriptedProvider
from jobagent.tailor.generate import (
    JobContext,
    TailorConfig,
    _kw_overlap,
    _term_idf,
    _tokens,
    default_pool,
    generate_bullets,
    generate_summary,
    select_candidates,
    verify_summary,
)
from jobagent.tailor.verify import R_NUMBER, R_ROLE, R_TOOL, verify_bullet

from .conftest import MINIMAL_FACT_BANK, FakeEmbedder

FB = load_fact_bank("tests/fixtures/fact_bank.json")


def _cfg(max_bullets: int = 4, pool: int | None = None, **kw: object) -> TailorConfig:
    return TailorConfig(
        max_bullets=max_bullets,
        candidate_pool=pool if pool is not None else default_pool(max_bullets),
        **kw,  # type: ignore[arg-type]
    )


def _job(**over: object) -> JobContext:
    base: dict[str, object] = {
        "id": "greenhouse:1",
        "title": "AI Engineer, Agents",
        "description_text": (
            "Build production LangGraph agents with stateful checkpointing and "
            "vector retrieval. Python, LangGraph, pgvector. LLM application work."
        ),
        "score_breakdown": {
            "requirements": {"required": ["LangGraph", "Python"], "preferred": ["pgvector"]}
        },
    }
    base.update(over)
    return JobContext(**base)  # type: ignore[arg-type]


# --- helpers to script the "model" ---------------------------------------
def _bullet_json(bullet: str) -> str:
    import json

    return json.dumps({"bullet": bullet})


def _fact_claim_for(messages: list[Message]) -> str:
    """Recover the fact's claim line from the rewrite prompt the pipeline sent."""
    user = messages[-1]["content"]
    for line in user.splitlines():
        if line.startswith("claim: "):
            return line[len("claim: ") :]
    return ""


def _faithful_responder(messages: list[Message]) -> str:
    """A well-behaved model: echoes the fact's claim verbatim (always verifies)."""
    return _bullet_json(_fact_claim_for(messages))


# =========================================================================
# select (deterministic, no LLM)
# =========================================================================
def test_select_is_deterministic_and_capped() -> None:
    job = _job()
    cfg = _cfg(max_bullets=3, pool=5)
    a = select_candidates(job, FB, FakeEmbedder(), cfg)
    b = select_candidates(job, FB, FakeEmbedder(), cfg)
    assert [c.fact.id for c in a] == [c.fact.id for c in b]
    assert len(a) == 5  # capped to the pool
    # Scores are sorted descending.
    assert all(a[i].score >= a[i + 1].score for i in range(len(a) - 1))


def _fact(fid: str, tools: list[str]) -> Fact:
    return Fact(id=fid, project="p", context="research", claim="did a thing", tools=tools)


def test_term_idf_rare_terms_weigh_more() -> None:
    # 'python' in every fact, 'langsmith' in one -> langsmith is more distinctive.
    d = dict(MINIMAL_FACT_BANK)
    d["facts"] = [
        {"id": "f1", "project": "p", "context": "c", "tools": ["Python", "LangSmith"], "claim": "a"},
        {"id": "f2", "project": "p", "context": "c", "tools": ["Python"], "claim": "b"},
        {"id": "f3", "project": "p", "context": "c", "tools": ["Python"], "claim": "c"},
    ]
    idf = _term_idf(FactBank.model_validate(d))
    assert idf["langsmith"] > idf["python"]


def test_kw_overlap_idf_downweights_ubiquitous_tool() -> None:
    # Both facts share the ubiquitous 'python'; only one has the distinctive
    # 'langsmith' that the job actually calls for. The distinctive match must win.
    idf = {"python": 0.1, "langsmith": 3.0, "xgboost": 3.0}
    job_kw = _tokens("build langsmith observability tooling in python")
    langsmith_fact = _fact("ls", ["LangSmith", "Python"])
    geoscience_fact = _fact("geo", ["XGBoost", "Python"])
    ls = _kw_overlap(langsmith_fact, job_kw, idf)
    geo = _kw_overlap(geoscience_fact, job_kw, idf)
    assert ls > geo
    # The geoscience fact matched only the low-IDF 'python' -> near zero.
    assert geo < 0.1 < ls


def test_keyword_signal_lifts_evidencing_fact() -> None:
    # Same (neutral) description -> identical embedding sim for every fact, so any
    # score change is purely the keyword signal. Requiring LangGraph lifts the
    # LangGraph fact and leaves an unrelated fact untouched.
    emb = FakeEmbedder()
    cfg = _cfg(pool=len(FB.facts))
    neutral = "We are hiring a software engineer for a generalist role."
    with_skill = _job(
        description_text=neutral,
        score_breakdown={"requirements": {"required": ["LangGraph"], "preferred": []}},
    )
    no_skill = _job(description_text=neutral, score_breakdown=None)
    a = {c.fact.id: c.score for c in select_candidates(with_skill, FB, emb, cfg)}
    b = {c.fact.id: c.score for c in select_candidates(no_skill, FB, emb, cfg)}
    assert a["mindful-langgraph"] > b["mindful-langgraph"]  # lifted by LangGraph
    assert a["i2ct-paper"] == b["i2ct-paper"]  # unrelated fact unchanged


# =========================================================================
# full pipeline
# =========================================================================
@pytest.mark.asyncio
async def test_faithful_bullets_all_verify_and_carry_fact_ids() -> None:
    provider = ScriptedProvider(_faithful_responder)
    cfg = _cfg(max_bullets=4)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), cfg)

    assert res.report.emitted == 4
    assert res.report.dropped == 0
    assert len(res.bullets) == 4
    for b in res.bullets:
        assert b.verify.ok
        assert b.fact_id  # every bullet traces to exactly one fact
        assert b.verify.fact_id == b.fact_id
    # No regenerations were needed for faithful output.
    assert res.report.regenerations == 0


@pytest.mark.asyncio
async def test_never_ships_over_max_bullets() -> None:
    provider = ScriptedProvider(_faithful_responder)
    cfg = _cfg(max_bullets=2, pool=6)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), cfg)
    assert res.report.emitted == 2
    assert len(res.bullets) == 2


@pytest.mark.asyncio
async def test_fabrication_is_dropped_never_shipped() -> None:
    """The model welds an invented number onto every bullet. Regeneration also
    fabricates. Every bullet must be DROPPED — nothing ships."""

    def liar(messages: list[Message]) -> str:
        claim = _fact_claim_for(messages)
        return _bullet_json(f"{claim} serving 5,000,000 users at 99.99% uptime.")

    provider = ScriptedProvider(liar)
    cfg = _cfg(max_bullets=4)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), cfg)

    assert res.bullets == []
    assert res.report.emitted == 0
    assert res.report.dropped == res.report.attempted
    # Each attempt regenerated exactly once, then dropped (no hunting loop).
    assert res.report.regenerations == res.report.attempted
    # The verifier caught the invented number.
    assert all(R_NUMBER in d.reasons for d in res.report.dropped_bullets)


@pytest.mark.asyncio
async def test_regeneration_recovers_a_bullet() -> None:
    """First attempt fabricates; the retry (which the pipeline signals via the
    violations in the prompt) is faithful. The bullet is kept and marked."""
    seen: dict[str, int] = {}

    def flaky(messages: list[Message]) -> str:
        claim = _fact_claim_for(messages)
        is_retry = "REJECTED" in messages[-1]["content"]
        seen[claim] = seen.get(claim, 0) + 1
        if is_retry:
            return _bullet_json(claim)  # clean on retry
        return _bullet_json(f"{claim} Led a team of 12 engineers.")  # role inflation

    provider = ScriptedProvider(flaky)
    cfg = _cfg(max_bullets=2, pool=2)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), cfg)

    assert res.report.emitted == 2
    assert res.report.dropped == 0
    assert all(b.regenerated for b in res.bullets)
    assert res.report.regenerations == 2
    # Sanity: the first-pass fabrication would have tripped role inflation.
    bad = verify_bullet(
        "x Led a team of 12 engineers.", FB, claimed_fact_id="mindful-langgraph"
    )
    assert R_ROLE in bad.reasons


@pytest.mark.asyncio
async def test_budget_guard_caps_llm_calls() -> None:
    provider = ScriptedProvider(_faithful_responder)
    cfg = _cfg(max_bullets=8, pool=8, max_llm_calls=3)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), cfg)

    assert res.report.budget_tripped == "calls"
    assert provider.usage.calls <= 3
    # Whatever it managed to emit is still fully verified.
    assert all(b.verify.ok for b in res.bullets)


@pytest.mark.asyncio
async def test_ordering_is_by_relevance() -> None:
    provider = ScriptedProvider(_faithful_responder)
    cfg = _cfg(max_bullets=5)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), cfg)
    scores = [b.rank_score for b in res.bullets]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_coverage_reports_unique_facts() -> None:
    provider = ScriptedProvider(_faithful_responder)
    cfg = _cfg(max_bullets=4)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), cfg)
    assert res.report.covered_facts == sorted({b.fact_id for b in res.bullets})
    assert len(res.report.covered_facts) == res.report.emitted  # one bullet per fact


@pytest.mark.asyncio
async def test_invented_tool_dropped() -> None:
    def tool_liar(messages: list[Message]) -> str:
        claim = _fact_claim_for(messages)
        return _bullet_json(f"{claim} Deployed on Kubernetes with Terraform.")

    provider = ScriptedProvider(tool_liar)
    cfg = _cfg(max_bullets=3, pool=3)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), cfg)
    assert res.bullets == []
    assert any(R_TOOL in d.reasons for d in res.report.dropped_bullets)


def _summary_json(text: str) -> str:
    import json

    return json.dumps({"summary": text})


def test_verify_summary_allows_fusion_but_not_fabrication() -> None:
    langgraph = next(f for f in FB.facts if f.id == "mindful-langgraph")
    rag = next(f for f in FB.facts if f.id == "mindful-rag")
    # A summary combining metrics/tools from BOTH facts is allowed (fusion inert).
    good = "Full-stack AI builder shipping 6-node LangGraph agents with sub-200ms RAG retrieval."
    assert verify_summary(good, [langgraph, rag], FB).ok
    # An invented number (99%) not in either fact must fail.
    bad = "Full-stack AI builder shipping LangGraph agents at 99% accuracy."
    assert verify_summary(bad, [langgraph, rag], FB).ok is False


@pytest.mark.asyncio
async def test_generate_summary_faithful_verifies() -> None:
    job = _job()
    good = "Full-stack AI builder shipping 6-node LangGraph agents with sub-200ms RAG retrieval."
    provider = ScriptedProvider(lambda _m: _summary_json(good))
    res = await generate_summary(provider, FakeEmbedder(), FB, job, _cfg())
    assert res.verify.ok and not res.fell_back
    assert res.text == good


@pytest.mark.asyncio
async def test_generate_summary_meta_language_is_rejected() -> None:
    # A summary with a banned meta-word is rejected even though it doesn't fabricate;
    # since the scripted model always returns meta text, it falls back to the skeleton.
    job = _job()
    meta = "Highlights tailored to this role draw on LangGraph and pgvector."
    provider = ScriptedProvider(lambda _m: _summary_json(meta))
    res = await generate_summary(provider, FakeEmbedder(), FB, job, _cfg())
    assert res.fell_back is True
    assert res.verify.ok  # the skeleton verifies
    for w in ("tailored", "highlights", "draw on", "this role"):
        assert w not in res.text.lower()


@pytest.mark.asyncio
async def test_generate_summary_self_assessment_rejected() -> None:
    # "Expert in ..." overreaches on a new-grad resume -> rejected like a meta-word;
    # scripted model keeps returning it, so it falls back to the clean skeleton.
    job = _job()
    provider = ScriptedProvider(lambda _m: _summary_json("Expert in LangGraph and RAG systems."))
    res = await generate_summary(provider, FakeEmbedder(), FB, job, _cfg())
    assert res.fell_back is True
    assert "expert" not in res.text.lower()


@pytest.mark.asyncio
async def test_hollow_opener_bullet_is_soft_regenerated_not_dropped() -> None:
    # A verified bullet that opens with a hollow verb triggers a soft register pass
    # but is NEVER dropped for register alone; a strong-verb retry replaces it.
    def flaky(messages: list[Message]) -> str:
        claim = _fact_claim_for(messages)
        if "REJECTED" in messages[-1]["content"]:
            return _bullet_json("Built " + claim)  # strong verb on the register retry
        return _bullet_json("Leveraged " + claim)  # verifies, but hollow opener

    provider = ScriptedProvider(flaky)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), _cfg(max_bullets=1, pool=1))
    assert res.report.emitted == 1 and res.report.dropped == 0
    assert res.report.regenerations >= 1  # the soft register pass ran
    assert res.bullets[0].text.startswith("Built ")  # improved verb kept


@pytest.mark.asyncio
async def test_hollow_opener_kept_if_retry_does_not_verify() -> None:
    # If the register retry can't verify, the original (verified) bullet ships —
    # register never causes a drop.
    def stubborn(messages: list[Message]) -> str:
        claim = _fact_claim_for(messages)
        if "REJECTED" in messages[-1]["content"]:
            return _bullet_json("Leveraged " + claim + " serving 9,000,000 users.")  # fabricates
        return _bullet_json("Leveraged " + claim)

    provider = ScriptedProvider(stubborn)
    res = await generate_bullets(provider, FakeEmbedder(), FB, _job(), _cfg(max_bullets=1, pool=1))
    assert res.report.emitted == 1 and res.report.dropped == 0
    assert res.bullets[0].text.startswith("Leveraged ")  # original verified bullet kept


@pytest.mark.asyncio
async def test_generate_summary_fabrication_falls_back() -> None:
    job = _job()
    provider = ScriptedProvider(
        lambda _m: _summary_json("Deployed agents serving 5,000,000 users.")
    )
    res = await generate_summary(provider, FakeEmbedder(), FB, job, _cfg())
    assert res.fell_back is True and res.verify.ok  # never ships the fabrication
    assert "5,000,000" not in res.text


@pytest.mark.asyncio
async def test_relevance_floor_makes_cap_not_a_quota() -> None:
    # With a floor set between the 3rd and 4th select scores, exactly 3 facts
    # clear it; the resume emits 3 even though the cap is 8 — never padded up.
    job = _job()
    ranked = select_candidates(
        job, FB, FakeEmbedder(), _cfg(max_bullets=8, pool=13)
    )
    scores = [c.score for c in ranked]  # already sorted desc
    floor = (scores[2] + scores[3]) / 2  # exactly 3 clear
    cfg = _cfg(max_bullets=8, pool=13, min_relevance=floor, min_bullets=1)
    res = await generate_bullets(
        ScriptedProvider(_faithful_responder), FakeEmbedder(), FB, job, cfg
    )
    assert res.report.cleared_floor == 3
    assert res.report.emitted == 3  # cap is 8; only 3 cleared -> emit 3, no padding
    assert res.report.floor_fallback is False
    assert all(b.rank_score >= floor for b in res.bullets)


@pytest.mark.asyncio
async def test_min_bullets_fallback_when_floor_excludes_everything() -> None:
    # An impossibly high floor (scores are in [0,1]) clears nothing, so the
    # min-bullet safety net emits the top-N by score anyway and flags the fallback.
    job = _job()
    cfg = _cfg(max_bullets=8, pool=13, min_relevance=1.01, min_bullets=3)
    res = await generate_bullets(
        ScriptedProvider(_faithful_responder), FakeEmbedder(), FB, job, cfg
    )
    assert res.report.cleared_floor == 0
    assert res.report.floor_fallback is True
    assert res.report.emitted == 3  # took the top-3 by score despite the floor
    # And they are the 3 highest-scoring facts.
    ranked = select_candidates(job, FB, FakeEmbedder(), cfg)
    assert {b.fact_id for b in res.bullets} == {c.fact.id for c in ranked[:3]}
