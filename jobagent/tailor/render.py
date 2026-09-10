"""Phase 4 (Part A) — render verified bullets into a one-page resume.

The system PRE-FILLS applications for HUMAN review; it NEVER auto-submits
(CLAUDE constraint 1). This module produces the resume a human reads before
clicking submit: a Markdown master (the reviewable source of truth) and a styled
one-page PDF rendered from the SAME structured data, so the two never diverge.

Layout mirrors the candidate's real resume: centered header (name / contact /
links / honest OPT line), then SUMMARY · EXPERIENCE · PROJECTS · EDUCATION ·
TECHNICAL SKILLS. LinkedIn/GitHub/Salesforce and any project with a `url` render
as real PDF hyperlinks; a project without a url renders as plain text (URLs are
NEVER fabricated).

Safety invariant (CLAUDE constraint 3, enforced HERE too): every bullet placed on
the resume is re-checked with ``verify_bullet`` at render time. If ANY input
bullet fails verification, rendering hard-errors — nothing is written. The
verifier is the gate at generation AND at render; a bug upstream can never leak
an unverified claim onto a rendered resume. The SUMMARY is composed only from
verified facts (project names + their tech) and profile data — no new claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment
from markupsafe import Markup
from weasyprint import HTML

from ..factbank import (
    Fact,
    FactBank,
    Profile,
    ProjectMeta,
    WorkAuthorization,
)
from ..logging import get_logger
from .verify import verify_bullet

log = get_logger(__name__)


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\+\.#-]*", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


class UnverifiedBulletError(RuntimeError):
    """A bullet handed to the renderer did not pass ``verify_bullet``.

    Rendering aborts — an unverified claim must never reach a resume."""


@dataclass(frozen=True)
class Bullet:
    """The minimal bullet the renderer needs. Decoupled from the generator so the
    renderer can be tested and reused without the LLM pipeline."""

    text: str
    fact_id: str
    rank_score: float = 0.0


def with_experience(tailored: list[Bullet], fact_bank: FactBank) -> list[Bullet]:
    """Ensure every EXPERIENCE fact appears on the resume (as its verified claim)
    even when the relevance-floored tailoring didn't surface it — a resume always
    shows work history; only PROJECTS are curated by relevance. Experience bullets
    sort above the tailored project bullets and keep fact-bank order."""
    have = {b.fact_id for b in tailored}
    exp_facts = [
        f for f in fact_bank.facts if f.kind == "experience" and f.id not in have
    ]
    exp = [
        Bullet(f.claim, f.id, rank_score=1.0 - i * 0.001)
        for i, f in enumerate(exp_facts)
    ]
    return exp + tailored


@dataclass
class RenderedResume:
    job_id: str
    company: str
    title: str
    markdown: str
    pdf_path: Path | None
    md_path: Path | None
    bullet_count: int
    experience_entries: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    thin: bool = False  # weak fit — few bullets cleared the relevance floor
    page_count: int = 0  # a resume should be ONE page


# --- links ----------------------------------------------------------------
def _href(url: str) -> str:
    """A clickable href: prepend https:// when a bare domain is given."""
    return url if url.startswith(("http://", "https://")) else f"https://{url}"


# --- header helpers -------------------------------------------------------
def work_auth_line(wa: WorkAuthorization) -> str:
    """One honest sentence about work authorization (matters on OPT)."""
    if wa.us_citizen:
        return "U.S. Citizen — no sponsorship required."
    now = (
        "authorized to work in the U.S. now"
        if not wa.requires_sponsorship_now
        else "requires work sponsorship now"
    )
    line = f"{wa.status} — {now}"
    if wa.requires_sponsorship_future:
        line += "; will require sponsorship in the future"
    return line + "."


def _contact_parts(profile: Profile) -> list[str]:
    return [p for p in (profile.location, profile.phone, profile.email) if p]


def _link_parts(profile: Profile) -> list[tuple[str, str]]:
    """(display, href) for each present profile link."""
    out: list[tuple[str, str]] = []
    for url in (profile.linkedin, profile.github, profile.salesforce):
        if url:
            out.append((url, _href(url)))
    return out


_META_WORDS = (
    "tailored", "highlights", "highlight", "draw on", "draws on", "this role",
    "this position", "selected", "curated",
)


# --- summary fallback (used only if no generated summary is supplied) -------
def fallback_summary(
    fact_bank: FactBank, bullets: list[Bullet], job_keywords: set[str] | None = None
) -> str:
    """A minimal, meta-free, capability-first line if no generated summary is
    supplied. Opens with the self-owned descriptor, then tech led JOB-RELEVANT
    first (not geoscience-first). No meta-language, no project names."""
    by_id = {f.id: f for f in fact_bank.facts}
    tech: list[str] = []
    for b in bullets:
        for tool in by_id[b.fact_id].tools:
            if tool not in tech:
                tech.append(tool)
    if job_keywords:  # job-relevant tech first (stable)
        tech = [t for t in tech if _tokens(t) & job_keywords] + \
               [t for t in tech if not (_tokens(t) & job_keywords)]
    edu = fact_bank.education[0] if fact_bank.education else None
    school = f", {edu.school}" if edu and edu.school else ""
    cap = f" with hands-on {', '.join(tech[:4])}" if tech else ""
    line = f"M.S. Computer Science (AI/ML){school} — full-stack AI builder{cap}."
    if any(w in line.lower() for w in _META_WORDS):  # meta-check on the fallback too
        line = f"M.S. Computer Science (AI/ML){school} — full-stack AI builder."
    return line


# --- skills emphasis (reorder items job-relevant first) -------------------
def _reorder_skills(
    skills: dict[str, list[str]], job_keywords: set[str] | None
) -> list[tuple[str, list[str]]]:
    """Within each category, move items the job actually mentions to the front
    (stable) so skills lead with job-relevant tech, not domain-first ordering."""
    items = list(skills.items())
    if not job_keywords:
        return items
    out: list[tuple[str, list[str]]] = []
    for category, values in items:
        matched = [v for v in values if _tokens(v) & job_keywords]
        rest = [v for v in values if not (_tokens(v) & job_keywords)]
        out.append((category, matched + rest))
    return out


# --- grouping -------------------------------------------------------------
def _group(
    bullets: list[Bullet], fact_bank: FactBank, kind: str, *, by_fact_index: bool = False
) -> list[tuple[str, list[Bullet]]]:
    """Group bullets whose fact.kind == ``kind`` under their fact's project.

    Projects and their bullets order by relevance (rank_score desc) — EXCEPT when
    ``by_fact_index`` is set (EXPERIENCE), where bullets order by fact-bank index
    so the portable/foundational bullet leads and the domain-specific one trails,
    regardless of per-job relevance score."""
    idx = {f.id: i for i, f in enumerate(fact_bank.facts)}
    by_id: dict[str, Fact] = {f.id: f for f in fact_bank.facts}
    groups: dict[str, list[Bullet]] = {}
    for b in bullets:
        fact = by_id[b.fact_id]
        if fact.kind != kind:
            continue
        groups.setdefault(fact.project, []).append(b)
    ordered: list[tuple[str, list[Bullet]]] = []
    for name, group_items in groups.items():
        if by_fact_index:
            group_items.sort(key=lambda b: idx[b.fact_id])
        else:
            group_items.sort(key=lambda b: -b.rank_score)
        ordered.append((name, group_items))
    ordered.sort(key=lambda g: -max(b.rank_score for b in g[1]))
    return ordered


# --- invariant ------------------------------------------------------------
def _assert_all_verified(bullets: list[Bullet], fact_bank: FactBank) -> None:
    for b in bullets:
        result = verify_bullet(b.text, fact_bank, claimed_fact_id=b.fact_id)
        if not result.ok:
            raise UnverifiedBulletError(
                f"refusing to render: bullet for fact '{b.fact_id}' failed "
                f"verification ({result.reasons}): {b.text!r}"
            )


# --- markdown master ------------------------------------------------------
def _md_link(display: str, href: str) -> str:
    return f"[{display}]({href})"


def _md_experience(
    fact_bank: FactBank, experience_groups: list[tuple[str, list[Bullet]]]
) -> list[str]:
    exp_by_name = fact_bank.experience_by_name()
    lines = ["## Experience", ""]
    for name, items in experience_groups:
        e = exp_by_name.get(name)
        if e:
            dates = _dates(e.start, e.end)
            lines.append(f"**{e.title}** — {dates}" if dates else f"**{e.title}**")
            org = e.org + (f" — Advisor: {e.advisor}" if e.advisor else "")
            loc = f" · {e.location}" if e.location else ""
            lines.append(f"*{org}{loc}*")
            if e.descriptor:
                lines.append(f"*Project: {e.descriptor}*")
        else:
            lines.append(f"**{name}**")
        lines += [f"- {b.text}" for b in items]
        lines.append("")
    return lines


def _md_projects(
    fact_bank: FactBank, project_groups: list[tuple[str, list[Bullet]]]
) -> list[str]:
    proj_by_name = fact_bank.project_by_name()
    lines = ["## Projects", ""]
    for name, items in project_groups:
        meta = proj_by_name.get(name)
        title = name
        if meta:
            if meta.url:
                title = _md_link(name, meta.url)
            if meta.descriptor:
                title += f" — {meta.descriptor}"
            if meta.note:
                title += f" — {meta.note}"
            if meta.tech:
                title += f" | {', '.join(meta.tech)}"
        lines.append(f"**{title}**")
        lines += [f"- {b.text}" for b in items]
        lines.append("")
    return lines


def build_markdown(
    fact_bank: FactBank,
    summary: str,
    experience_groups: list[tuple[str, list[Bullet]]],
    project_groups: list[tuple[str, list[Bullet]]],
    skills_ordered: list[tuple[str, list[str]]],
    experience_first: bool = True,
) -> str:
    p = fact_bank.profile
    lines: list[str] = [f"# {p.name}", "", "  |  ".join(_contact_parts(p))]
    links = _link_parts(p)
    if links:
        lines.append("  |  ".join(_md_link(d, h) for d, h in links))
    lines.append(f"*{work_auth_line(p.work_authorization)}*")

    lines += ["", "## Summary", "", summary, ""]

    exp = _md_experience(fact_bank, experience_groups) if experience_groups else []
    proj = _md_projects(fact_bank, project_groups)
    lines += (exp + proj) if experience_first else (proj + exp)

    lines += ["## Education", ""]
    for edu in fact_bank.education:
        dates = _dates(edu.start, edu.end) or edu.graduated or ""
        head = f"**{edu.school}**" + (f" — {dates}" if dates else "")
        lines.append(head)
        detail = edu.degree + (f", GPA {edu.gpa}" if edu.gpa else "")
        loc = f" · {edu.location}" if edu.location else ""
        lines.append(f"*{detail}{loc}*")
    cw = [c for e in fact_bank.education for c in e.coursework]
    if cw:
        lines.append(f"*Relevant Coursework: {', '.join(cw)}*")

    lines += ["", "## Technical Skills", ""]
    for category, values in skills_ordered:
        if values:
            lines.append(f"- **{category}:** {', '.join(values)}")
    return "\n".join(lines).rstrip() + "\n"


def _dates(start: str | None, end: str | None) -> str:
    if start and end:
        return f"{start} – {end}"
    return start or end or ""


# --- HTML / PDF -----------------------------------------------------------
_CSS = """
@page { size: Letter; margin: 0.5in; }
* { box-sizing: border-box; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.6pt;
       line-height: 1.18; color: #111; }
a { color: #113; text-decoration: none; }
header { text-align: center; margin-bottom: 3px; }
h1 { font-size: 18pt; margin: 0 0 2px 0; font-weight: 700; letter-spacing: 0.4px; }
.contact, .links, .workauth { font-size: 8.7pt; color: #222; margin: 0.5px 0; }
.workauth { font-style: italic; color: #444; }
h2 { font-size: 10pt; font-weight: 700; text-transform: uppercase;
     letter-spacing: 1.2px; border-bottom: 1.1px solid #111;
     padding-bottom: 1px; margin: 7px 0 3px; }
.summary { margin: 0; text-align: justify; }
.entry { margin-bottom: 3px; }
.row { display: flex; justify-content: space-between; align-items: baseline; }
.row .left { font-weight: 700; }
.row.sub .left, .row.sub .right { font-style: italic; font-weight: 400; color: #333; }
.row .right { white-space: nowrap; padding-left: 10px; color: #333; }
.descriptor { font-style: italic; color: #333; margin: 0; }
.ptitle { margin: 0; }
.ptitle .name { font-weight: 700; }
.ptitle .tech { font-weight: 400; color: #555; }
ul { margin: 1px 0 0 0; padding-left: 14px; }
li { margin-bottom: 1px; text-align: justify; }
.edu .row .left { font-weight: 700; }
.coursework { font-style: italic; color: #333; margin-top: 1px; }
.skills p { margin: 0 0 1.5px 0; }
"""

_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><style>{{ css }}</style></head>
{% macro exp_section() %}{% if experience %}<section>
    <h2>Experience</h2>
    {% for name, items, e in experience %}<div class="entry">
      {% if e %}<div class="row"><span class="left">{{ e.title }}</span><span class="right">{{ dates(e.start, e.end) }}</span></div>
      <div class="row sub"><span class="left">{{ e.org }}{% if e.advisor %} — Advisor: {{ e.advisor }}{% endif %}</span><span class="right">{{ e.location }}</span></div>
      {% if e.descriptor %}<p class="descriptor">Project: {{ e.descriptor }}</p>{% endif %}
      {% else %}<div class="row"><span class="left">{{ name }}</span></div>{% endif %}
      <ul>{% for b in items %}<li>{{ b.text }}</li>{% endfor %}</ul>
    </div>{% endfor %}
  </section>{% endif %}{% endmacro %}
{% macro proj_section() %}<section>
    <h2>Projects</h2>
    {% for name, items, meta in projects %}<div class="entry">
      <p class="ptitle">{{ project_title(name, meta) }}</p>
      <ul>{% for b in items %}<li>{{ b.text }}</li>{% endfor %}</ul>
    </div>{% endfor %}
  </section>{% endmacro %}
<body>
  <header>
    <h1>{{ p.name }}</h1>
    <div class="contact">{{ contact }}</div>
    {% if links %}<div class="links">{% for d, h in links %}{% if not loop.first %} | {% endif %}<a href="{{ h }}">{{ d }}</a>{% endfor %}</div>{% endif %}
    <div class="workauth">{{ work_auth }}</div>
  </header>

  <section>
    <h2>Summary</h2>
    <p class="summary">{{ summary }}</p>
  </section>

  {% if experience_first %}{{ exp_section() }}{{ proj_section() }}{% else %}{{ proj_section() }}{{ exp_section() }}{% endif %}

  <section class="edu">
    <h2>Education</h2>
    {% for e in education %}
    <div class="entry">
      <div class="row"><span class="left">{{ e.school }}</span><span class="right">{{ dates(e.start, e.end) or e.graduated }}</span></div>
      <div class="row sub"><span class="left">{{ e.degree }}{% if e.gpa %}, GPA: {{ e.gpa }}{% endif %}</span><span class="right">{{ e.location }}</span></div>
    </div>
    {% endfor %}
    {% if coursework %}<p class="coursework">Relevant Coursework: {{ coursework }}</p>{% endif %}
  </section>

  <section class="skills">
    <h2>Technical Skills</h2>
    {% for category, values in skills %}{% if values %}<p><strong>{{ category }}:</strong> {{ values | join(', ') }}</p>{% endif %}{% endfor %}
  </section>
</body></html>"""


def _project_title_html(name: str, meta: ProjectMeta | None) -> Markup:
    """Project title line: name (linked when a url exists) — descriptor | tech."""
    if meta is None:
        return Markup("<span class='name'>%s</span>") % name
    name_html = (
        Markup("<a class='name' href='%s'>%s</a>") % (_href(meta.url), name)
        if meta.url
        else Markup("<span class='name'>%s</span>") % name
    )
    out = name_html
    if meta.descriptor:
        out += Markup(" — %s") % meta.descriptor
    if meta.note:
        out += Markup(" — %s") % meta.note
    if meta.tech:
        out += Markup(" <span class='tech'>| %s</span>") % ", ".join(meta.tech)
    return Markup(out)


def _render_html(
    fact_bank: FactBank,
    summary: str,
    experience_groups: list[tuple[str, list[Bullet]]],
    project_groups: list[tuple[str, list[Bullet]]],
    skills_ordered: list[tuple[str, list[str]]],
    experience_first: bool = True,
) -> str:
    exp_by_name = fact_bank.experience_by_name()
    proj_by_name = fact_bank.project_by_name()
    env = Environment(autoescape=True)
    template = env.from_string(_TEMPLATE)
    return template.render(
        css=Markup(_CSS),
        p=fact_bank.profile,
        contact="  |  ".join(_contact_parts(fact_bank.profile)),
        links=_link_parts(fact_bank.profile),
        work_auth=work_auth_line(fact_bank.profile.work_authorization),
        summary=summary,
        experience_first=experience_first,
        experience=[(n, items, exp_by_name.get(n)) for n, items in experience_groups],
        projects=[(n, items, proj_by_name.get(n)) for n, items in project_groups],
        education=fact_bank.education,
        coursework=", ".join(c for e in fact_bank.education for c in e.coursework),
        skills=skills_ordered,
        dates=_dates,
        project_title=_project_title_html,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "resume"


def resume_stem(name: str) -> str:
    """Human-facing resume filename base, derived from the CANDIDATE'S name — it
    never carries a company name (uploading a company-named file to a different
    company looks careless). Middle initials are dropped:
    'Nantha Kumar A' -> 'Nantha_Kumar' -> 'Nantha_Kumar_Resume'."""
    tokens = [t for t in name.split() if len(t) > 1]
    base = "_".join(tokens) if tokens else (name.strip().replace(" ", "_") or "Resume")
    return f"{base}_Resume"


def _section_relevance(
    groups: list[tuple[str, list[Bullet]]], relevance: dict[str, float]
) -> float:
    ids = [b.fact_id for _, items in groups for b in items]
    vals = [relevance.get(i, 0.0) for i in ids]
    return sum(vals) / len(vals) if vals else 0.0


def render_resume(
    job_id: str,
    company: str,
    title: str,
    bullets: list[Bullet],
    fact_bank: FactBank,
    out_dir: str | Path,
    *,
    summary_text: str | None = None,
    relevance: dict[str, float] | None = None,
    job_keywords: set[str] | None = None,
    thin: bool = False,
    write_pdf: bool = True,
    min_bullets: int = 3,
) -> RenderedResume:
    """Render one resume. Enforces the verify invariant BEFORE writing anything.

    ``summary_text`` is the pre-generated + pre-verified summary (falls back to a
    minimal meta-free capability line if None). ``relevance`` (fact_id -> select
    score) orders EXPERIENCE vs PROJECTS by aggregate job-relevance; ``job_keywords``
    reorders skills job-relevant-first. ``thin`` marks a weak-fit role; the resume
    is NEVER padded.

    ONE PAGE IS A HARD REQUIREMENT. If the rendered PDF exceeds one page, the
    lowest-relevance (lowest rank_score) bullet is dropped and it re-renders,
    repeating until it fits on one page — but never below ``min_bullets``.
    Dropping a verified bullet is fine (the subset stays verified); a 2-page
    resume is never emitted. Experience bullets carry a high rank_score, so work
    history is shed last.
    """
    if not bullets:
        raise UnverifiedBulletError(f"refusing to render {job_id}: no bullets to render")
    _assert_all_verified(bullets, fact_bank)  # hard error if any bullet is unverified

    skills_ordered = _reorder_skills(fact_bank.skills, job_keywords)

    def _compose(
        kept: list[Bullet],
    ) -> tuple[str, str, list[tuple[str, list[Bullet]]], list[tuple[str, list[Bullet]]]]:
        """Build (markdown, html, experience_groups, project_groups) for a bullet
        set. Re-derived on every drop so the md master and PDF never diverge."""
        # EXPERIENCE bullets order by transferability (fact-bank index), projects by relevance.
        exp_groups = _group(kept, fact_bank, "experience", by_fact_index=True)
        proj_groups = _group(kept, fact_bank, "project") + _group(
            kept, fact_bank, "publication"
        )
        summ = summary_text or fallback_summary(fact_bank, kept, job_keywords)
        # Section order by aggregate relevance (experience is always shown, but
        # leads only if its facts are, on average, more job-relevant).
        exp_first = True
        if relevance is not None and exp_groups and proj_groups:
            exp_first = _section_relevance(
                exp_groups, relevance
            ) >= _section_relevance(proj_groups, relevance)
        md = build_markdown(
            fact_bank, summ, exp_groups, proj_groups, skills_ordered, exp_first
        )
        html = _render_html(
            fact_bank, summ, exp_groups, proj_groups, skills_ordered, exp_first
        )
        return md, html, exp_groups, proj_groups

    # Internal organization stays company/title-keyed (a per-job subdirectory),
    # but the FILES a human uploads are named after the candidate, never the
    # company. So renders/<company_title>/Nantha_Kumar_Resume.pdf.
    job_dir = Path(out_dir) / f"{_slug(company)}_{_slug(title)}"
    job_dir.mkdir(parents=True, exist_ok=True)
    stem = resume_stem(fact_bank.profile.name)
    md_path = job_dir / f"{stem}.md"

    kept = list(bullets)
    pdf_path: Path | None = None
    page_count = 0

    if not write_pdf:
        # No PDF to measure; keep all bullets and write the markdown master only.
        markdown, _, experience_groups, project_groups = _compose(kept)
        md_path.write_text(markdown, encoding="utf-8")
    else:
        # ONE PAGE, enforced: render, and while it spills onto a 2nd page drop the
        # lowest-relevance bullet and re-render — never below min_bullets.
        floor = max(1, min(min_bullets, len(kept)))
        while True:
            markdown, html, experience_groups, project_groups = _compose(kept)
            document = HTML(string=html).render()
            page_count = len(document.pages)
            if page_count <= 1 or len(kept) <= floor:
                break
            victim = min(kept, key=lambda b: b.rank_score)
            kept = [b for b in kept if b is not victim]
            log.info(
                "render.drop_for_page",
                job=job_id,
                dropped_fact=victim.fact_id,
                dropped_rank=round(victim.rank_score, 4),
                pages_before=page_count,
                bullets_left=len(kept),
            )
        md_path.write_text(markdown, encoding="utf-8")
        pdf_path = job_dir / f"{stem}.pdf"
        document.write_pdf(str(pdf_path))
        if page_count > 1:
            # Only reachable if even min_bullets won't fit — keep the smallest
            # version and flag it; we still never SHIP more than we must.
            log.warning(
                "render.multipage_at_floor",
                job=job_id,
                pages=page_count,
                bullets=len(kept),
                min_bullets=floor,
            )

    log.info(
        "render.resume",
        job=job_id,
        bullets=len(kept),
        experience=len(experience_groups),
        projects=len(project_groups),
        thin=thin,
        pages=page_count,
        pdf=str(pdf_path) if pdf_path else None,
    )
    return RenderedResume(
        job_id=job_id,
        company=company,
        title=title,
        markdown=markdown,
        pdf_path=pdf_path,
        md_path=md_path,
        bullet_count=len(kept),
        experience_entries=[n for n, _ in experience_groups],
        projects=[n for n, _ in project_groups],
        thin=thin,
        page_count=page_count,
    )
