"""Phase 3 resume-bullet generator — sits BEHIND the proven verifier.

The verifier (:mod:`jobagent.tailor.verify`) is the gate; this generator feeds
it. The pipeline is ``select -> rewrite -> verify -> render``:

1. **select** — deterministically rank the fact bank for THIS job (Stage A
   embedding similarity between the job description and each fact, plus a bonus
   for facts that evidence the job's extracted required/preferred skills) and
   cap the candidate pool. No LLM.
2. **rewrite** — for each candidate the LLM rewrites that ONE fact into a
   job-tailored bullet. The prompt carries exactly one fact's claim + variants +
   tools and is told to introduce NO number, tool, scale, title, or context
   absent from that fact. Structured output.
3. **verify** — ``verify_bullet(bullet, fact_bank, claimed_fact_id=<that
   fact>)``. On failure: regenerate ONCE with the violations fed back as
   constraints; if it still fails the bullet is DROPPED. Never a hunting loop,
   never a shipped unverified claim (CLAUDE constraint 3).
4. **render** — order kept bullets by job relevance, cap at ``max_bullets``,
   return them annotated with their ``fact_id`` and verify result.

Budget-guarded (calls + spend) and per-job idempotent-friendly like the scorer.
Deterministic where it can be; the only nondeterminism is the LLM rewrite, and
the verifier makes that safe.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..factbank import Education, Fact, FactBank
from ..llm.base import BaseLLMProvider, Message
from ..logging import get_logger
from ..scoring.embedder import Embedder
from ..scoring.vectors import cosine, to_vector
from .verify import VerifyResult, verify_bullet

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Inputs / configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class JobContext:
    """The slice of a job the generator needs. Decoupled from the ORM so the
    pipeline is testable without a database."""

    id: str
    title: str
    description_text: str
    score_breakdown: dict[str, Any] | None = None

    def required_skills(self) -> list[str]:
        """Required + preferred skills the scorer extracted for this job (if
        Stage B ran); empty when the job was never scored."""
        bd = self.score_breakdown or {}
        reqs = bd.get("requirements") or {}
        out: list[str] = []
        for key in ("required", "preferred"):
            out.extend(str(s) for s in (reqs.get(key) or []))
        return out


@dataclass
class TailorConfig:
    max_bullets: int  # a CAP, not a quota — a resume may emit fewer
    candidate_pool: int  # max facts to attempt (bounds worst-case LLM calls)
    max_llm_calls: int = 1_000_000  # per-job budget guard
    max_spend_usd: float = 1_000_000.0  # per-job budget guard
    max_desc_chars: int = 1500  # trim the description like Stage B does
    sim_weight: float = 0.6  # embedding similarity share of the select score
    skill_weight: float = 0.4  # keyword/skill-overlap share
    # Relevance floor: a fact is attempted/emitted only if its select score
    # clears this. Default 0.0 = no floor (callers/Settings set the real value);
    # keeps the pipeline permissive unless a threshold is chosen.
    min_relevance: float = 0.0
    min_bullets: int = 1  # if fewer clear the floor, take top-N by score anyway


def default_pool(max_bullets: int) -> int:
    """Attempt up to 2x the target so verifier drops can be back-filled."""
    return max(1, max_bullets * 2)


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    fact: Fact
    score: float  # deterministic select relevance, higher = more relevant


@dataclass
class GeneratedBullet:
    fact_id: str
    text: str
    rank_score: float
    regenerated: bool
    verify: VerifyResult


@dataclass
class DroppedBullet:
    fact_id: str
    text: str
    reasons: list[str]
    violations: list[str]


@dataclass
class TailorReport:
    job_id: str
    candidates: int = 0  # facts ranked into the pool
    attempted: int = 0  # facts we ran the LLM on
    emitted: int = 0
    dropped: int = 0
    regenerations: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    budget_tripped: str | None = None  # "calls" | "spend"
    dropped_bullets: list[DroppedBullet] = field(default_factory=list)
    covered_facts: list[str] = field(default_factory=list)
    floor: float = 0.0  # the relevance threshold applied
    cleared_floor: int = 0  # candidates whose score >= floor
    floor_fallback: bool = False  # too few cleared -> took top-N by score

    def line(self) -> str:
        budget = f" | BUDGET TRIPPED ({self.budget_tripped})" if self.budget_tripped else ""
        fb = " (min-bullet FALLBACK)" if self.floor_fallback else ""
        return (
            f"job {self.job_id}: {self.emitted} bullets emitted / {self.dropped} "
            f"dropped / {self.attempted} attempted ({self.regenerations} regens) "
            f"| floor {self.floor:.2f}: {self.cleared_floor}/{self.candidates} "
            f"cleared{fb} | LLM: {self.llm_calls} calls, {self.input_tokens}+"
            f"{self.output_tokens} tok, ${self.cost_usd:.4f}{budget}"
        )


@dataclass
class TailorResult:
    bullets: list[GeneratedBullet]
    report: TailorReport


# --------------------------------------------------------------------------
# 1. select
# --------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\+\.#-]*", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _fact_blob(fact: Fact) -> str:
    return " ".join([fact.claim, *fact.variants, *fact.tools, *fact.tags, fact.project])


def _skill_bonus(fact: Fact, skills: list[str]) -> float:
    """Fraction of the job's required/preferred skills evidenced by this fact.

    A skill counts as evidenced when every alphanumeric token of the (possibly
    multi-word) skill appears in the fact's text — so "vector search" matches a
    fact mentioning both words. Deterministic; no embeddings needed here."""
    if not skills:
        return 0.0
    blob = _tokens(_fact_blob(fact))
    hits = 0
    for skill in skills:
        skill_toks = _tokens(skill)
        if skill_toks and skill_toks <= blob:
            hits += 1
    return hits / len(skills)


def _job_keywords(job: JobContext) -> set[str]:
    """All content tokens of the job: title + (de-boilerplated) description +
    extracted required skills."""
    return _tokens(" ".join([job.title, job.description_text, *job.required_skills()]))


def _fact_terms(fact: Fact) -> set[str]:
    """The fact's own keyword terms (tools + tags), tokenized."""
    return _tokens(" ".join([*fact.tools, *fact.tags]))


def _term_idf(fact_bank: FactBank) -> dict[str, float]:
    """Inverse document frequency of each fact term across the fact bank.

    A term in many facts ("python", "production") is not distinctive and gets a
    low weight; a term in few facts ("langsmith", "xgboost", "pgvector") is
    distinctive and gets a high one. This is what stops a ubiquitous tool like
    "python" — which appears in almost every job posting — from crediting a
    geoscience fact as much as its genuinely distinctive tools would."""
    n = len(fact_bank.facts)
    df: Counter[str] = Counter()
    for fact in fact_bank.facts:
        for term in _fact_terms(fact):
            df[term] += 1
    return {term: math.log(1.0 + n / freq) for term, freq in df.items()}


def _kw_overlap(fact: Fact, job_keywords: set[str], idf: dict[str, float]) -> float:
    """IDF-weighted fraction of the fact's terms that appear in the job text — the
    fact->job direction that ``_skill_bonus`` misses.

    This is what separates a LangSmith/agent fact from a geoscience fact when the
    extracted ``required[]`` is a generic web stack: the fact's own distinctive
    "LangSmith" tool matches a LangSmith role even though "LangSmith" was never in
    required[]. Embedding cosine alone can't (the MiniLM band is too compressed to
    rank these reliably). Weighting by IDF keeps a ubiquitous match like "python"
    from inflating an off-domain fact."""
    terms = _fact_terms(fact)
    total = sum(idf.get(t, 0.0) for t in terms)
    if total <= 0.0:
        return 0.0
    matched = sum(idf.get(t, 0.0) for t in terms if t in job_keywords)
    return matched / total


def select_candidates(
    job: JobContext,
    fact_bank: FactBank,
    embedder: Embedder,
    config: TailorConfig,
) -> list[Candidate]:
    """Rank the fact bank for this job and return the top ``candidate_pool``.

    Score = ``sim_weight`` * cosine(job, fact) (mapped to [0,1]) +
    ``skill_weight`` * keyword-overlap, where keyword-overlap is the STRONGER of
    two directions: how many of the job's required skills the fact evidences
    (``_skill_bonus``), and how many of the fact's own tools/tags appear in the
    job text (``_kw_overlap``). The second direction is what lets the score
    separate facts the compressed MiniLM cosine band cannot — e.g. a LangSmith
    fact on a LangSmith role even when the extracted required[] is a web stack.
    Deterministic; ties break by fact id so the pool is stable across runs.
    """
    facts = fact_bank.facts
    texts = [job.description_text] + fact_bank.fact_texts()
    matrix = embedder.encode(texts)
    job_vec = to_vector([float(x) for x in matrix[0].tolist()])
    skills = job.required_skills()
    job_kw = _job_keywords(job)
    idf = _term_idf(fact_bank)

    scored: list[Candidate] = []
    for i, fact in enumerate(facts):
        fact_vec = to_vector([float(x) for x in matrix[i + 1].tolist()])
        sim = (cosine(job_vec, fact_vec) + 1.0) / 2.0  # [-1,1] -> [0,1]
        keyword = max(_skill_bonus(fact, skills), _kw_overlap(fact, job_kw, idf))
        score = config.sim_weight * sim + config.skill_weight * keyword
        scored.append(Candidate(fact, round(score, 6)))

    scored.sort(key=lambda c: (-c.score, c.fact.id))
    return scored[: max(0, config.candidate_pool)]


# --------------------------------------------------------------------------
# 2. rewrite (the only LLM step)
# --------------------------------------------------------------------------
class RewriteOut(BaseModel):
    """Structured rewrite output — a single tailored bullet."""

    bullet: str


_REWRITE_SYSTEM = (
    "You are a resume editor. You are given ONE verified accomplishment (the "
    "'fact') and a job. Reword that ONE fact into a single, crisp, "
    "results-first resume bullet tailored to the role's language.\n"
    "ABSOLUTE RULES — the output is machine-verified against the fact and any "
    "violation causes it to be discarded:\n"
    "  1. Use ONLY information present in the given fact. Introduce NO number, "
    "percentage, metric, scale, tool, technology, job title, team size, "
    "timeframe, or production/live-traffic claim that is not already in the "
    "fact.\n"
    "  2. Do NOT borrow details from any other project or invent leadership "
    "('led', 'managed') the fact does not state.\n"
    "  3. You MAY drop details, reorder, and choose synonyms that match the "
    "job's vocabulary, but every concrete claim must trace to the fact.\n"
    "  4. START with a concrete resume verb: Built, Designed, Engineered, Owned, "
    "Implemented, Integrated, Architected, or Developed. Do NOT open with a vague "
    "business verb (Accelerated, Leveraged, Spearheaded, Drove, Empowered) unless "
    "an immediate concrete object follows. Never use self-assessment words "
    "('Expert', 'world-class', 'cutting-edge', 'seasoned').\n"
    "Output ONLY JSON: {\"bullet\": str}. One sentence, no leading dash."
)


def _fact_block(fact: Fact) -> str:
    lines = [f"claim: {fact.claim}"]
    if fact.variants:
        lines.append("approved rephrasings:")
        lines.extend(f"  - {v}" for v in fact.variants)
    if fact.tools:
        lines.append(f"tools (the ONLY tools you may name): {', '.join(fact.tools)}")
    lines.append(f"context: {fact.context}")
    return "\n".join(lines)


def rewrite_messages(
    fact: Fact, job: JobContext, config: TailorConfig, violations: list[str] | None = None
) -> list[Message]:
    user = (
        f"JOB TITLE: {job.title}\n\n"
        f"JOB DESCRIPTION (for tone/keywords only):\n"
        f"{job.description_text[: config.max_desc_chars]}\n\n"
        f"THE FACT (your single source of truth):\n{_fact_block(fact)}"
    )
    if violations:
        user += (
            "\n\nYour previous attempt was REJECTED for these issues:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\nRewrite it to fix every issue while staying strictly inside the "
            "fact (introduce no new number, tool, or claim)."
        )
    return [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user", "content": user},
    ]


async def _rewrite(
    provider: BaseLLMProvider,
    fact: Fact,
    job: JobContext,
    config: TailorConfig,
    violations: list[str] | None = None,
) -> str:
    out = await provider.complete(
        rewrite_messages(fact, job, config, violations), schema=RewriteOut
    )
    return out.bullet.strip()


# --------------------------------------------------------------------------
# 2b. summary — generated through the SAME verify gate as bullets
# --------------------------------------------------------------------------
class SummaryOut(BaseModel):
    """Structured summary output — a 2-3 sentence resume summary."""

    summary: str


@dataclass
class SummaryResult:
    text: str
    verify: VerifyResult
    regenerated: bool = False
    fell_back: bool = False  # both attempts failed -> minimal verified skeleton


_META_WORDS = (
    "tailored", "highlights", "highlight", "draw on", "draws on", "this role",
    "this position", "selected", "curated", "for the role", "summary:",
)
# Unbackable self-assessment — overreaches on a new-grad resume (summary + bullets).
_SELF_ASSESS = ("expert", "world-class", "cutting-edge", "seasoned", "guru", "ninja")
# Vague business verbs that read as hollow when they OPEN a bullet.
_HOLLOW_OPENERS = ("accelerated", "leveraged", "spearheaded", "drove", "empowered")

_SUMMARY_SYSTEM = (
    "You are writing the 2-3 sentence SUMMARY at the top of a resume, in the "
    "candidate's voice. It must read like a person wrote it — capability-forward "
    "and metric-anchored — NOT like a machine describing a resume.\n"
    "HARD RULES (every violation is machine-checked and the summary is discarded):\n"
    "  1. Use ONLY the metrics, tools, and claims present in the FACTS below. "
    "Introduce NO number, tool, scale, or production claim absent from them.\n"
    "  2. BANNED meta-language — never write: 'tailored', 'highlights', 'draw on', "
    "'this role', 'this position', 'selected', 'curated', or any phrase that "
    "narrates the resume's own construction. Never list project NAMES.\n"
    "  3. BANNED self-assessment — this is a new-grad resume: never write 'Expert', "
    "'Expert in', 'world-class', 'cutting-edge', or 'seasoned'. Use grounded "
    "phrasing instead: 'hands-on', 'proven experience', 'experience architecting'.\n"
    "  4. LEAD with the capability and tech that match the JOB (e.g. LangGraph / "
    "LangSmith / RAG / agents for an LLM-platform role), then support it with a "
    "concrete metric from the facts (e.g. sub-200ms retrieval, a 6-node agent).\n"
    "  5. Third person implied (no 'I'), present-tense capability register, like: "
    "'Full-stack AI builder shipping production LangGraph agents with sub-200ms "
    "RAG retrieval.'\n"
    'Output ONLY JSON: {"summary": str}. 2-3 sentences.'
)


def _summary_facts_block(facts: list[Fact]) -> str:
    lines = []
    for f in facts:
        tools = f", tools: {', '.join(f.tools)}" if f.tools else ""
        lines.append(f"- {f.claim}{tools}")
    return "\n".join(lines)


def summary_messages(
    job: JobContext,
    facts: list[Fact],
    education: list[Education],
    violations: list[str] | None = None,
    config: TailorConfig | None = None,
) -> list[Message]:
    max_chars = config.max_desc_chars if config else 1500
    edu = education[0].degree if education else "M.S. Computer Science (AI/ML)"
    user = (
        f"JOB TITLE: {job.title}\n\n"
        f"JOB DESCRIPTION (for keywords/emphasis only):\n"
        f"{job.description_text[:max_chars]}\n\n"
        f"CANDIDATE DEGREE: {edu}\n\n"
        f"FACTS (your only source of metrics/tools/claims):\n"
        f"{_summary_facts_block(facts)}"
    )
    if violations:
        user += (
            "\n\nYour previous summary was REJECTED for introducing content not in "
            "the facts:\n" + "\n".join(f"  - {v}" for v in violations)
            + "\nRewrite it using ONLY the facts above; remove every flagged addition."
        )
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": user},
    ]


def _merge_facts(facts: list[Fact], education: list[Education]) -> Fact:
    """A synthetic fact whose content is the UNION of ``facts`` (+ the degree), so
    a summary may legitimately combine them while every number/tool/claim it makes
    is still checked against that union by the unchanged verifier."""
    claim = " ".join([f.claim for f in facts] + [e.degree for e in education])
    variants = [v for f in facts for v in f.variants]
    tools = list(dict.fromkeys(t for f in facts for t in f.tools))
    tags = list(dict.fromkeys(t for f in facts for t in f.tags))
    # production content is legitimately available if any source fact is production
    context = "production" if any(f.context == "production" for f in facts) else (
        facts[0].context if facts else "research"
    )
    return Fact(
        id="__summary__", project="summary", context=context, kind="summary",
        tags=tags, tools=tools, claim=claim, variants=variants,
    )


def verify_summary(
    summary_text: str, top_facts: list[Fact], fact_bank: FactBank
) -> VerifyResult:
    """Run the summary through the SAME ``verify_bullet`` gate, against a one-fact
    bank whose fact is the union of ``top_facts`` (+ education). Fusion is inert
    (a summary is allowed to combine facts); numbers/tools/role/context are not."""
    merged = _merge_facts(top_facts, fact_bank.education)
    merged_bank = FactBank(
        profile=fact_bank.profile,
        education=fact_bank.education,
        skills=fact_bank.skills,
        facts=[merged],
    )
    return verify_bullet(summary_text, merged_bank, claimed_fact_id="__summary__")


def _fallback_summary(fact_bank: FactBank, top_facts: list[Fact]) -> str:
    """A minimal capability line guaranteed to verify — used only if BOTH generated
    attempts fail. Capability-first, self-owned descriptor, led by the JOB-RELEVANT
    tech (``top_facts`` are already relevance-ranked). No metrics (nothing to
    fabricate), no meta-language, no project names, no geoscience-first dump."""
    edu = fact_bank.education[0] if fact_bank.education else None
    school = f", {edu.school}" if edu and edu.school else ""
    # distinctive tech from the top (most job-relevant) facts, in that order
    tech: list[str] = []
    for f in top_facts:
        for t in f.tools:
            if t not in tech:
                tech.append(t)
    cap = f" with hands-on {', '.join(tech[:4])}" if tech else ""
    line = f"M.S. Computer Science (AI/ML){school} — full-stack AI builder{cap}."
    # belt-and-suspenders: the meta-word ban runs on the fallback too
    if _meta_violations(line):
        line = f"M.S. Computer Science (AI/ML){school} — full-stack AI builder."
    return line


def _contains_words(text: str, terms: tuple[str, ...]) -> list[str]:
    """Whole-word/phrase matches of ``terms`` in ``text`` (so 'expert' matches
    'Expert' but not 'expertise')."""
    low = text.lower()
    return [
        t for t in terms
        if re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", low)
    ]


def _meta_violations(text: str) -> list[str]:
    """Register bar the fabrication verifier doesn't police: self-narrating
    meta-language and unbackable self-assessment. Regenerated (never shipped)."""
    return (
        [f"banned meta-word '{w}'" for w in _contains_words(text, _META_WORDS)]
        + [f"banned self-assessment '{w}'" for w in _contains_words(text, _SELF_ASSESS)]
    )


def _bullet_register_issues(text: str) -> list[str]:
    """Soft register check for a bullet: unbackable self-assessment anywhere, or a
    hollow business verb OPENING the bullet."""
    issues = [f"self-assessment '{w}'" for w in _contains_words(text, _SELF_ASSESS)]
    first = re.match(r"\s*([A-Za-z]+)", text)
    if first and first.group(1).lower() in _HOLLOW_OPENERS:
        issues.append(f"hollow opener '{first.group(1)}'")
    return issues


async def generate_summary(
    provider: BaseLLMProvider,
    embedder: Embedder,
    fact_bank: FactBank,
    job: JobContext,
    config: TailorConfig,
    *,
    top_k: int = 8,
) -> SummaryResult:
    """Generate the resume SUMMARY through generate -> verify -> (regen once) ->
    fall back. Same fabrication gate as bullets, PLUS a meta-language ban; leads
    with job-relevant capability."""
    ranked = select_candidates(job, fact_bank, embedder, config)
    top_facts = [c.fact for c in ranked[:top_k]]

    text = (await provider.complete(
        summary_messages(job, top_facts, fact_bank.education, config=config),
        schema=SummaryOut,
    )).summary.strip()
    result = verify_summary(text, top_facts, fact_bank)
    meta = _meta_violations(text)
    if result.ok and not meta:
        return SummaryResult(text, result)

    # regenerate once with the violations (fabrication + meta) fed back
    text2 = (await provider.complete(
        summary_messages(
            job, top_facts, fact_bank.education, result.violations + meta, config
        ),
        schema=SummaryOut,
    )).summary.strip()
    result2 = verify_summary(text2, top_facts, fact_bank)
    if result2.ok and not _meta_violations(text2):
        return SummaryResult(text2, result2, regenerated=True)

    # both failed -> minimal verified, meta-free skeleton (never ship a bad summary)
    skel = _fallback_summary(fact_bank, top_facts)
    return SummaryResult(skel, verify_summary(skel, top_facts, fact_bank),
                         regenerated=True, fell_back=True)


# --------------------------------------------------------------------------
# 3 + 4. verify, render, orchestration
# --------------------------------------------------------------------------
async def generate_bullets(
    provider: BaseLLMProvider,
    embedder: Embedder,
    fact_bank: FactBank,
    job: JobContext,
    config: TailorConfig,
) -> TailorResult:
    """Run the full select -> rewrite -> verify -> render pipeline for one job."""
    report = TailorReport(job_id=job.id, floor=config.min_relevance)
    ranked = select_candidates(job, fact_bank, embedder, config)
    report.candidates = len(ranked)

    # Relevance floor: attempt only facts whose select score clears the floor, so
    # max_bullets is a CAP (a resume may have fewer) and never a quota we pad to.
    # Safety net: if fewer than min_bullets clear it, take the top-N by score
    # anyway (logged) so a resume is never near-empty.
    cleared = [c for c in ranked if c.score >= config.min_relevance]
    report.cleared_floor = len(cleared)
    if len(cleared) >= config.min_bullets:
        eligible = cleared
    else:
        eligible = ranked[: config.min_bullets]
        report.floor_fallback = True
        log.info(
            "tailor.floor_fallback",
            job=job.id,
            floor=config.min_relevance,
            cleared=len(cleared),
            min_bullets=config.min_bullets,
        )

    kept: list[GeneratedBullet] = []
    for cand in eligible:
        if len(kept) >= config.max_bullets:
            break
        if _budget_tripped(provider, config, report):
            break

        report.attempted += 1
        bullet = await _rewrite(provider, cand.fact, job, config)
        result = verify_bullet(bullet, fact_bank, claimed_fact_id=cand.fact.id)
        regenerated = False

        if not result.ok:
            # ONE regeneration with the violations fed back as constraints, budget
            # permitting. Never a hunting loop (hard invariant).
            if _budget_tripped(provider, config, report):
                _drop(report, cand.fact.id, bullet, result)
                break
            report.regenerations += 1
            bullet = await _rewrite(
                provider, cand.fact, job, config, violations=result.violations
            )
            result = verify_bullet(bullet, fact_bank, claimed_fact_id=cand.fact.id)
            regenerated = True
            if not result.ok:
                _drop(report, cand.fact.id, bullet, result)
                continue  # DROP — never ship an unverified claim
        elif _bullet_register_issues(bullet) and _budget_ok(provider, config):
            # SOFT register pass: try once to improve tone (hollow opener / self-
            # assessment). Keep the improved bullet ONLY if it still verifies;
            # otherwise ship the original — register never causes a DROP.
            report.regenerations += 1
            alt = await _rewrite(
                provider, cand.fact, job, config,
                violations=_bullet_register_issues(bullet),
            )
            alt_result = verify_bullet(alt, fact_bank, claimed_fact_id=cand.fact.id)
            regenerated = True
            if alt_result.ok:
                bullet, result = alt, alt_result

        kept.append(
            GeneratedBullet(
                fact_id=cand.fact.id,
                text=bullet,
                rank_score=cand.score,
                regenerated=regenerated,
                verify=result,
            )
        )

    # render: highest job-relevance first (stable), capped at max_bullets.
    kept.sort(key=lambda b: (-b.rank_score, b.fact_id))
    bullets = kept[: config.max_bullets]

    report.emitted = len(bullets)
    report.covered_facts = sorted({b.fact_id for b in bullets})
    _record_usage(provider, report)
    log.info("tailor.summary", summary=report.line())
    return TailorResult(bullets=bullets, report=report)


def _budget_tripped(
    provider: BaseLLMProvider, config: TailorConfig, report: TailorReport
) -> bool:
    u = provider.usage
    if u.calls >= config.max_llm_calls:
        report.budget_tripped = "calls"
        return True
    if u.cost_usd >= config.max_spend_usd:
        report.budget_tripped = "spend"
        return True
    return False


def _budget_ok(provider: BaseLLMProvider, config: TailorConfig) -> bool:
    """Non-mutating budget check (for the optional soft register pass)."""
    u = provider.usage
    return u.calls < config.max_llm_calls and u.cost_usd < config.max_spend_usd


def _drop(
    report: TailorReport, fact_id: str, bullet: str, result: VerifyResult
) -> None:
    report.dropped += 1
    report.dropped_bullets.append(
        DroppedBullet(fact_id, bullet, list(result.reasons), list(result.violations))
    )


def _record_usage(provider: BaseLLMProvider, report: TailorReport) -> None:
    u = provider.usage
    report.llm_calls = u.calls
    report.input_tokens = u.input_tokens
    report.output_tokens = u.output_tokens
    report.cost_usd = u.cost_usd
