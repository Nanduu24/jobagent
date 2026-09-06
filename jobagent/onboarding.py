"""Interactive onboarding: build a user's ``fact_bank.json`` from prompts.

`jobagent setup` walks a new user through creating their own fact bank so the
tool is usable by anyone — no code editing, no hand-writing JSON. The collection
logic is decoupled from the terminal via the ``Prompter`` protocol, so it runs
under test with scripted answers (zero real I/O).

GOLDEN RULE surfaced to the user: every ``claim`` must be literally true and
self-contained. The resume tailor never invents numbers, tools, titles, or scale
beyond the source fact, and the verifier drops any bullet that drifts from it —
so a padded claim here just yields a weaker resume, never a fabricated one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from .factbank import FactBank


class Prompter(Protocol):
    """Injected I/O seam. TyperPrompter drives the terminal; tests script it."""

    def text(self, message: str, *, default: str | None = None,
             allow_empty: bool = False) -> str: ...
    def boolean(self, message: str, *, default: bool = False) -> bool: ...
    def integer(self, message: str, *, default: int = 0) -> int: ...
    def note(self, message: str) -> None: ...


def _csv(value: str) -> list[str]:
    """Split a comma-separated answer into a clean list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32] or "item"


# --- section collectors ----------------------------------------------------
def _collect_profile(p: Prompter) -> dict[str, Any]:
    p.note("\n== Your profile ==")
    name = p.text("Full name")
    email = p.text("Email")
    phone = p.text("Phone (optional)", allow_empty=True)
    linkedin = p.text("LinkedIn URL (optional)", allow_empty=True)
    github = p.text("GitHub URL (optional)", allow_empty=True)
    location = p.text("Location (e.g. 'Boston, MA'; optional)", allow_empty=True)
    relocation = p.boolean("Open to relocation?", default=True)

    p.note("\n-- Work authorization (used to answer only the honest, factual "
           "questions on a form; everything sensitive is always left for you) --")
    status = p.text("Work-authorization status (e.g. 'U.S. Citizen', 'F-1 OPT', "
                    "'H-1B', 'Permanent Resident')", default="U.S. Citizen")
    us_citizen = p.boolean("Are you a U.S. citizen?", default=True)
    spon_now = p.boolean("Do you require visa sponsorship NOW?", default=False)
    spon_future = p.boolean("Will you require sponsorship in the FUTURE?",
                            default=False)
    clearance = p.boolean("Eligible for a U.S. security clearance?",
                          default=us_citizen)
    years = p.integer("Years of professional experience", default=0)
    seniority = _csv(p.text("Target seniority (comma-separated)",
                            default="new grad, entry, associate"))

    profile: dict[str, Any] = {
        "name": name,
        "email": email,
        "relocation": relocation,
        "work_authorization": {
            "status": status,
            "requires_sponsorship_now": spon_now,
            "requires_sponsorship_future": spon_future,
            "us_citizen": us_citizen,
            "clearance_eligible": clearance,
        },
        "years_professional_experience": max(0, years),
        "target_seniority": seniority or ["new grad"],
    }
    for key, val in (("phone", phone), ("linkedin", linkedin),
                     ("github", github), ("location", location)):
        if val:
            profile[key] = val
    return profile


def _collect_education(p: Prompter, index: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": f"edu-{index}",
        "degree": p.text("Degree (e.g. 'M.S. in Computer Science')"),
        "school": p.text("School"),
    }
    for key, msg in (("location", "Location (optional)"),
                     ("start", "Start (e.g. 'Aug 2023'; optional)"),
                     ("end", "End (e.g. 'May 2025'; optional)"),
                     ("gpa", "GPA (optional)")):
        val = p.text(msg, allow_empty=True)
        if val:
            entry[key] = val
    coursework = _csv(p.text("Relevant coursework (comma-separated; optional)",
                             allow_empty=True))
    if coursework:
        entry["coursework"] = coursework
    return entry


def _collect_experience(p: Prompter, index: int) -> dict[str, Any]:
    name = p.text("Short label for this role (used to group its bullets, "
                  "e.g. 'Acme Corp')")
    entry: dict[str, Any] = {
        "name": name,
        "title": p.text("Job title"),
        "org": p.text("Organization", default=name),
    }
    for key, msg in (("location", "Location (optional)"),
                     ("start", "Start (optional)"), ("end", "End (optional)"),
                     ("advisor", "Advisor/manager (optional)"),
                     ("descriptor", "One-line descriptor (optional)")):
        val = p.text(msg, allow_empty=True)
        if val:
            entry[key] = val
    return entry


def _collect_project(p: Prompter, index: int) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": p.text("Project name")}
    descriptor = p.text("One-line descriptor (optional)", allow_empty=True)
    if descriptor:
        entry["descriptor"] = descriptor
    tech = _csv(p.text("Tech stack (comma-separated; optional)", allow_empty=True))
    if tech:
        entry["tech"] = tech
    url = p.text("Public URL (optional; never fabricated on the resume)",
                 allow_empty=True)
    if url:
        entry["url"] = url
    return entry


def _collect_skills(p: Prompter) -> dict[str, list[str]]:
    p.note("\n== Skills ==  Enter one category at a time (blank category name to "
           "finish). Example category: 'Languages' -> 'Python, SQL, Go'.")
    skills: dict[str, list[str]] = {}
    while True:
        category = p.text("Skill category (blank to finish)", allow_empty=True)
        if not category:
            break
        items = _csv(p.text(f"Skills in '{category}' (comma-separated)"))
        if items:
            skills[category] = items
    return skills


def _collect_facts(
    p: Prompter, labels: list[str]
) -> list[dict[str, Any]]:
    p.note(
        "\n== Facts ==  Each fact is one verified accomplishment the tailor may "
        "draw a resume bullet from. Add at least one. GOLDEN RULE: the claim must "
        "be literally true and self-contained — state the real numbers/tools; the "
        "tailor will never add any you omit."
    )
    if labels:
        p.note(f"Known labels you can attach a fact to: {', '.join(labels)}")
    facts: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    while True:
        if facts and not p.boolean("Add another fact?", default=True):
            break
        if not facts:
            p.note("\n-- Fact #1 --")
        else:
            p.note(f"\n-- Fact #{len(facts) + 1} --")
        project = p.text("Which project/role does this belong to? (its label)")
        kind = p.text("Kind: project | experience | publication",
                      default="project")
        if kind not in ("project", "experience", "publication"):
            kind = "project"
        claim = p.text("The claim (one sentence, literally true, self-contained)")
        while not claim.strip():
            p.note("A claim can't be empty — it's the source of a resume bullet.")
            claim = p.text("The claim")
        base = _slug(project)
        fid = base
        n = 1
        while fid in used_ids:
            n += 1
            fid = f"{base}-{n}"
        used_ids.add(fid)
        fact: dict[str, Any] = {
            "id": fid,
            "project": project,
            "kind": kind,
            "context": p.text("Context (e.g. 'production', 'research'; optional)",
                              default=kind),
            "tags": _csv(p.text("Tags (comma-separated; optional)",
                                allow_empty=True)),
            "tools": _csv(p.text("Tools/technologies in this claim "
                                 "(comma-separated)", allow_empty=True)),
            "claim": claim.strip(),
        }
        variant = p.text("A shorter paraphrase of the same claim (optional)",
                         allow_empty=True)
        if variant:
            fact["variants"] = [variant]
        facts.append(fact)
    return facts


def build_fact_bank(p: Prompter) -> dict[str, Any]:
    """Collect a complete fact bank interactively and return it as a dict."""
    p.note(
        "This creates your personal fact bank (data/fact_bank.json) — the single "
        "source of truth the tool tailors resumes from. Nothing here is uploaded; "
        "it stays on your machine and is git-ignored."
    )
    profile = _collect_profile(p)

    education: list[dict[str, Any]] = []
    p.note("\n== Education ==")
    while p.boolean("Add an education entry?", default=not education):
        education.append(_collect_education(p, len(education) + 1))

    experience: list[dict[str, Any]] = []
    p.note("\n== Work experience ==")
    while p.boolean("Add a work-experience entry?", default=not experience):
        experience.append(_collect_experience(p, len(experience) + 1))

    projects: list[dict[str, Any]] = []
    p.note("\n== Projects ==")
    while p.boolean("Add a project?", default=not projects):
        projects.append(_collect_project(p, len(projects) + 1))

    skills = _collect_skills(p)
    labels = [e["name"] for e in experience] + [pr["name"] for pr in projects]
    facts = _collect_facts(p, labels)

    return {
        "profile": profile,
        "education": education,
        "experience": experience,
        "projects": projects,
        "skills": skills,
        "facts": facts,
    }


def write_fact_bank(data: dict[str, Any], path: Path) -> FactBank:
    """Validate the collected data and write it to ``path``. Raises pydantic
    ValidationError (surfaced by the caller) if the fact bank is malformed."""
    fact_bank = FactBank.model_validate(data)  # hard fail on malformed input
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return fact_bank
