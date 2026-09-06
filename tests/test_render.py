"""Resume renderer: the verify invariant, resume sections, grouping, hyperlinks,
no-pad. Zero LLM. Uses the REAL (enriched) fact bank so the render-time verifier
runs exactly as in production. Faithful bullets = a fact's own claim verbatim
(always verifies); fabrications must hard-error before anything is written.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from jobagent.factbank import WorkAuthorization, load_fact_bank
from jobagent.tailor.render import (
    Bullet,
    UnverifiedBulletError,
    render_resume,
    resume_stem,
    work_auth_line,
)

FB = load_fact_bank("tests/fixtures/fact_bank.json")


def _faithful(fact_id: str, score: float = 0.5) -> Bullet:
    fact = next(f for f in FB.facts if f.id == fact_id)
    return Bullet(text=fact.claim, fact_id=fact_id, rank_score=score)


def test_invariant_rejects_unverified_bullet(tmp_path: Path) -> None:
    # An invented scale ("9,000,000 users") not in the fact -> hard error, and
    # NOTHING is written to disk.
    bad = Bullet(
        text="Built a 6-node LangGraph agent serving 9,000,000 users in production.",
        fact_id="mindful-langgraph",
        rank_score=0.9,
    )
    with pytest.raises(UnverifiedBulletError):
        render_resume("j", "LangChain", "AI Engineer", [bad], FB, tmp_path)
    assert list(tmp_path.iterdir()) == []  # no partial resume left behind


def test_empty_bullets_hard_errors(tmp_path: Path) -> None:
    with pytest.raises(UnverifiedBulletError):
        render_resume("j", "Co", "Role", [], FB, tmp_path)


def test_renders_all_sections_markdown_and_pdf(tmp_path: Path) -> None:
    # one experience fact + two project facts -> all sections populated
    bullets = [
        _faithful("exp-modeling", 0.8),
        _faithful("mindful-langgraph", 0.7),
        _faithful("jobagent", 0.6),
    ]
    r = render_resume(
        "greenhouse:1", "LangChain", "Python OSS Engineer", bullets, FB, tmp_path
    )
    md = r.markdown
    assert md.startswith(f"# {FB.profile.name}")
    assert FB.profile.email in md
    assert "OPT" in md and "sponsorship" in md  # honest work-auth line
    for section in ("## Summary", "## Experience", "## Projects",
                    "## Education", "## Technical Skills"):
        assert section in md
    # skills use the real resume's category labels (not mangled)
    assert "**Languages & Core CS:**" in md
    # both artifacts exist; the PDF is a real, one-page file
    assert r.md_path is not None and r.md_path.exists()
    assert r.pdf_path is not None and r.pdf_path.exists()
    assert r.pdf_path.read_bytes()[:4] == b"%PDF"
    assert r.pdf_path.stat().st_size > 1500
    assert r.page_count == 1


def test_resume_stem_drops_middle_initial() -> None:
    assert resume_stem("Nantha Kumar A") == "Nantha_Kumar_Resume"
    assert resume_stem("Ada Lovelace") == "Ada_Lovelace_Resume"


def test_upload_filename_has_no_company_name(tmp_path: Path) -> None:
    # The file a human uploads must be candidate-named, never company-named — but
    # the per-job parent directory keeps the company/title for internal tracking.
    r = render_resume(
        "greenhouse:1", "Cartesia", "Software Engineer, Platform",
        [_faithful("jobagent", 0.6)], FB, tmp_path,
    )
    assert r.pdf_path is not None and r.md_path is not None
    assert r.pdf_path.name == "Nantha_Kumar_Resume.pdf"
    assert r.md_path.name == "Nantha_Kumar_Resume.md"
    # no company name leaks into the uploaded filename
    assert "cartesia" not in r.pdf_path.name.lower()
    # internal organization: company/title lives in the parent directory
    assert "cartesia" in r.pdf_path.parent.name.lower()


def test_experience_and_projects_split_by_kind(tmp_path: Path) -> None:
    bullets = [_faithful("exp-pipelines", 0.9), _faithful("mindful-rag", 0.7)]
    r = render_resume("j", "Co", "Role", bullets, FB, tmp_path, write_pdf=False)
    assert r.experience_entries == ["iResearchE3 Lab"]  # experience-kind fact
    assert "Mindful AI" in r.projects  # project-kind fact
    # experience header carries the entry metadata (title/advisor)
    assert "AI / Machine Learning Research Assistant" in r.markdown
    assert "Advisor: Dr. A. Mentor" in r.markdown


def test_project_url_renders_as_hyperlink(tmp_path: Path) -> None:
    # Mindful AI has a url -> markdown link + PDF anchor; a project without a url
    # stays plain text (URLs are never fabricated).
    r = render_resume(
        "j", "Co", "Role",
        [_faithful("mindful-langgraph", 0.9), _faithful("jobagent", 0.5)],
        FB, tmp_path,
    )
    assert "[Mindful AI](https://mindful-ai-omega.vercel.app)" in r.markdown
    assert "](https://mindful-ai-omega.vercel.app)" in r.markdown
    # JobAgent has no url -> plain, no link syntax around the name
    assert "[JobAgent]" not in r.markdown
    # header links are clickable
    assert "linkedin.com/in/nanthakumarashokanand" in r.markdown
    assert "github.com/Nanduu24" in r.markdown


def test_no_padding_and_thin_flag(tmp_path: Path) -> None:
    bullets = [_faithful("mindful-langgraph"), _faithful("jobagent"), _faithful("fleetiq")]
    r = render_resume(
        "j", "Co", "Role", bullets, FB, tmp_path, thin=True, write_pdf=False
    )
    assert r.bullet_count == 3  # exactly the input; never padded to a cap
    body = r.markdown.split("## Projects")[1].split("## Education")[0]
    assert body.count("\n- ") == 3  # three project bullets, no filler
    assert r.thin is True


def test_fallback_summary_is_meta_free_and_capability_forward(tmp_path: Path) -> None:
    # With no generated summary supplied, the fallback is a clean capability line:
    # degree + tech, NO meta-language, NO project names.
    r = render_resume(
        "j", "Co", "Role", [_faithful("mindful-rag", 0.9)], FB, tmp_path, write_pdf=False
    )
    summary = r.markdown.split("## Summary")[1].split("## ")[0]
    assert "pgvector" in summary  # tech from the selected fact
    for meta in ("tailored", "highlights", "draw on", "this role", "selected"):
        assert meta not in summary.lower()
    assert "Mindful AI" not in summary  # never lists project names


def test_generated_summary_is_used_verbatim(tmp_path: Path) -> None:
    # A supplied (pre-verified) summary is rendered as-is.
    r = render_resume(
        "j", "Co", "Role", [_faithful("mindful-rag", 0.9)], FB, tmp_path,
        summary_text="Full-stack AI builder shipping sub-200ms RAG retrieval.",
        write_pdf=False,
    )
    assert "Full-stack AI builder shipping sub-200ms RAG retrieval." in r.markdown


def test_work_auth_line_is_honest() -> None:
    opt = WorkAuthorization(
        status="F-1 OPT",
        requires_sponsorship_now=False,
        requires_sponsorship_future=True,
        us_citizen=False,
        clearance_eligible=False,
    )
    line = work_auth_line(opt)
    assert "authorized to work in the U.S. now" in line
    assert "require sponsorship in the future" in line

    citizen = WorkAuthorization(
        status="Citizen",
        requires_sponsorship_now=False,
        requires_sponsorship_future=False,
        us_citizen=True,
        clearance_eligible=True,
    )
    assert "Citizen" in work_auth_line(citizen)


# --- one page is a HARD requirement ---------------------------------------
def test_never_exceeds_one_page_drops_to_fit(tmp_path: Path) -> None:
    # Far more bullets than fit (all 14 facts) — the Mercor bug was 9 bullets on
    # 2 pages. The renderer must drop the lowest-relevance bullets until it fits
    # on ONE page, never emitting a 2-page resume.
    bullets = [
        _faithful(f.id, score=0.9 - i * 0.05)
        for i, f in enumerate(FB.facts)
    ]
    assert len(bullets) > 9  # genuinely overflows
    r = render_resume(
        "greenhouse:mercor", "Mercor", "Software Engineer, Marketplace",
        bullets, FB, tmp_path, min_bullets=3,
    )
    assert r.pdf_path is not None and r.pdf_path.exists()
    assert r.page_count == 1                       # the hard requirement
    assert r.bullet_count < len(bullets)           # some were dropped to fit
    assert r.bullet_count >= 3                      # but never below the floor


def test_fitting_resume_keeps_all_bullets(tmp_path: Path) -> None:
    # A small bullet set already fits on one page -> nothing is dropped.
    bullets = [_faithful("exp-modeling", 0.8), _faithful("jobagent", 0.6)]
    r = render_resume("j", "Co", "Role", bullets, FB, tmp_path, min_bullets=3)
    assert r.page_count == 1
    assert r.bullet_count == len(bullets)          # untouched
