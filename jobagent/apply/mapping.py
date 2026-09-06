"""Decide, per form field, whether to FILL (from verified data), FLAG (never
auto-answer — the human decides), or leave BLANK (couldn't confidently map).

Guardrails (CLAUDE constraint 1 + the human-in-the-loop rule):
  - EEO/demographic, citizenship, salary, and free-text "why us"/cover-letter
    fields are ALWAYS flagged, never guessed into.
  - Work-authorization questions get ONLY the honest OPT pre-set answer, and only
    for high-confidence phrasings; anything ambiguous is flagged.
  - Anything we can't confidently map is left BLANK and listed for the human — we
    never guess into a form field.
"""
from __future__ import annotations

from .models import (
    CandidateData,
    FieldType,
    FormField,
    Outcome,
    PlannedField,
    PrefillPlan,
)

_DEMOGRAPHIC = (
    "race", "ethnic", "gender", "veteran", "disab", "lgbtq", "sexual orientation",
    "orientation", "pronoun", "hispanic", "latino", "transgender",
    "self-identification", "self identification", "citizen", "citizenship",
)
_SALARY = (
    "salary", "compensation", "desired pay", "expected pay", "pay expectation",
    "rate expectation", "expected comp", "comp expectation",
)
_MOTIVATION = (
    "why do you want", "why are you interested", "why do you", "what interests you",
    "cover letter", "tell us about", "what excites you", "motivat",
    "why work", "why join", "in your own words",
)


def _has(label: str, keys: tuple[str, ...]) -> bool:
    low = label.lower()
    return any(k in low for k in keys)


def _yesno_value(field: FormField, want_yes: bool) -> str | None:
    """The option text for a Yes/No answer, or None if the field has options that
    don't clearly include the intended answer (then we flag rather than guess)."""
    want = "yes" if want_yes else "no"
    if not field.options:
        return "Yes" if want_yes else "No"
    for opt in field.options:
        if opt.strip().lower().startswith(want):
            return opt
    return None


def _classify(field: FormField, c: CandidateData) -> PlannedField:
    label = field.label.strip()
    low = label.lower()

    # 0. resume file input is handled separately (attach step), not here.
    if field.type is FieldType.file:
        if _has(low, ("resume", "cv", "curriculum")):
            return PlannedField(field, Outcome.blank, reason="resume file (attach step)")
        return PlannedField(field, Outcome.flag, reason="file upload — human decides (e.g. cover letter)")

    # 1. NEVER auto-answer: work-auth FIRST so 'authorized to work' isn't caught
    #    by the 'citizen' demographic key, then demographic / salary / motivation.
    if _has(low, ("authorized to work", "legally authorized", "work authorization")):
        val = _yesno_value(field, c.authorized_to_work_now)
        if val is not None:
            return PlannedField(field, Outcome.fill, value=val,
                                reason="honest work-auth pre-set")
        return PlannedField(field, Outcome.flag, reason="work-auth phrasing — human confirms")
    if "sponsorship" in low or "sponsor" in low:
        if _has(low, ("future", "now or in the future", "at any point")):
            val = _yesno_value(field, c.requires_future_sponsorship)
            if val is not None:
                return PlannedField(field, Outcome.fill, value=val,
                                    reason="honest work-auth pre-set (future sponsorship)")
        if "now" in low and "future" not in low:
            val = _yesno_value(field, not c.authorized_to_work_now)
            if val is not None:
                return PlannedField(field, Outcome.fill, value=val,
                                    reason="honest work-auth pre-set (sponsorship now)")
        return PlannedField(field, Outcome.flag,
                            reason="sponsorship phrasing beyond the pre-set — human answers")
    if _has(low, _DEMOGRAPHIC):
        return PlannedField(field, Outcome.flag, reason="EEO/demographic — human only")
    if _has(low, _SALARY):
        return PlannedField(field, Outcome.flag, reason="salary/compensation — human only")
    if _has(low, _MOTIVATION) or field.type is FieldType.textarea:
        return PlannedField(field, Outcome.flag,
                            reason="free-text / motivation — human writes this")

    # 2. Core identity fields (fill from verified profile).
    if _has(low, ("first name", "given name")):
        return PlannedField(field, Outcome.fill, value=c.first_name)
    if _has(low, ("last name", "surname", "family name")):
        return PlannedField(field, Outcome.fill, value=c.last_name)
    if _has(low, ("full name", "legal name")) or low == "name":
        return PlannedField(field, Outcome.fill, value=f"{c.first_name} {c.last_name}".strip())
    if field.type is FieldType.email or "email" in low:
        return PlannedField(field, Outcome.fill, value=c.email)
    if field.type is FieldType.tel or _has(low, ("phone", "mobile")):
        if c.phone:
            return PlannedField(field, Outcome.fill, value=c.phone)
        return PlannedField(field, Outcome.blank, reason="no phone on file")

    # 3. Labeled links (fill only when we hold the value).
    if "linkedin" in low and c.linkedin:
        return PlannedField(field, Outcome.fill, value=c.linkedin)
    if _has(low, ("github", "git hub")) and c.github:
        return PlannedField(field, Outcome.fill, value=c.github)
    if _has(low, ("location", "city", "where are you")) and c.location:
        return PlannedField(field, Outcome.fill, value=c.location)
    if _has(low, ("website", "portfolio", "personal site")) and c.github:
        return PlannedField(field, Outcome.fill, value=c.github)

    # 4. Anything else -> blank, listed for the human. Never guessed.
    return PlannedField(field, Outcome.blank, reason="unmapped — human fills")


def plan_prefill(fields: list[FormField], candidate: CandidateData) -> PrefillPlan:
    """Pure planner: classify every field. No browser, no side effects."""
    planned = [_classify(f, candidate) for f in fields]
    resume_field = next(
        (f for f in fields
         if f.type is FieldType.file and _has(f.label.lower(), ("resume", "cv", "curriculum"))),
        None,
    )
    return PrefillPlan(planned=planned, resume_field=resume_field)
