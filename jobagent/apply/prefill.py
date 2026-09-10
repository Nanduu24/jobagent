"""Orchestrate one pre-fill-and-STOP against an application form.

Flow: goto -> introspect fields -> plan (fill/flag/blank) -> apply fills ->
optionally attach resume -> LOCATE the submit button (never click) -> screenshot
-> report. The browser is left open on the filled form for the human to review
and submit. There is no submit step in this code, by construction.
"""
from __future__ import annotations

from ..logging import get_logger
from .driver import BrowserDriver
from .mapping import plan_prefill
from .models import CandidateData, FieldType, PrefillReport

log = get_logger(__name__)


def prefill_application(
    driver: BrowserDriver,
    job_id: str,
    apply_url: str,
    candidate: CandidateData,
    *,
    screenshot_path: str | None = None,
    attach_resume: bool = False,
) -> PrefillReport:
    """Pre-fill the form at ``apply_url`` and stop at submit.

    ``attach_resume`` defaults to False: attaching uploads the PDF to the
    employer's ATS, which transmits the candidate's resume before they've decided
    to apply — so it is opt-in, never automatic.
    """
    driver.goto(apply_url)
    fields = driver.list_fields()
    plan = plan_prefill(fields, candidate)

    for p in plan.fills:
        if p.value is None:
            continue
        if p.field.type in (FieldType.select, FieldType.radio):
            driver.choose(p.field.selector, p.value)
        else:
            driver.fill_text(p.field.selector, p.value)

    resume_attached = False
    if attach_resume and plan.resume_field and candidate.resume_path:
        driver.attach_file(plan.resume_field.selector, str(candidate.resume_path))
        resume_attached = True

    # LOCATE the submit button for the human — do NOT click it.
    plan.submit = driver.locate_submit()

    if screenshot_path:
        driver.screenshot(screenshot_path)

    log.info(
        "apply.prefill",
        job=job_id,
        filled=len(plan.fills),
        flagged=len(plan.flagged),
        blank=len(plan.blanks),
        resume_attached=resume_attached,
        submit_located=plan.submit is not None,
    )
    return PrefillReport(
        job_id=job_id, apply_url=apply_url, plan=plan,
        resume_attached=resume_attached, screenshot_path=screenshot_path,
    )


def render_report(report: PrefillReport) -> str:
    """Human-readable pre-fill report ending with 'review and submit manually'."""
    p = report.plan
    lines = [
        f"PRE-FILLED (not submitted): {report.apply_url}",
        f"  resume attached: {report.resume_attached}",
        "",
        f"FILLED {len(p.fills)} field(s) from your verified profile:",
    ]
    lines += [f"    ✓ {pf.field.label!r} = {pf.value!r}" for pf in p.fills]
    lines += ["", f"FLAGGED {len(p.flagged)} field(s) — you must answer these:"]
    lines += [f"    ⚑ {pf.field.label!r}  ({pf.reason})" for pf in p.flagged]
    lines += ["", f"LEFT BLANK {len(p.blanks)} field(s) for you:"]
    lines += [f"    · {pf.field.label!r}  ({pf.reason})" for pf in p.blanks]
    if p.submit:
        lines += ["", f"SUBMIT button located: {p.submit.label!r} — NOT clicked."]
    lines += ["", "REVIEW AND SUBMIT MANUALLY. The agent never submits."]
    return "\n".join(lines)
