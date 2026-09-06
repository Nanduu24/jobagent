"""Prompt builders for Stage B. The LLM only READS and scores — it never writes
or paraphrases any claim about the candidate."""
from __future__ import annotations

from ..factbank import FactBank
from ..llm.base import Message

_EXTRACT_SYSTEM = (
    "You are a precise recruiting analyst. Extract the hard facts from a job "
    "description. Output ONLY JSON matching this schema:\n"
    '{"required": [str], "preferred": [str], "years_experience": int|null, '
    '"seniority": str|null, "must_have_citizenship": bool, '
    '"requires_clearance": bool}\n'
    "Rules: required = must-have skills/technologies/qualifications; "
    "years_experience = the minimum years explicitly required (null if none); "
    "must_have_citizenship = true only if US citizenship/US-person is mandatory; "
    "requires_clearance = true only if an active security clearance is required. "
    "Do not infer beyond the text."
)

_RUBRIC_SYSTEM = (
    "You are a STRICT, discriminating hiring rater scoring how well a candidate "
    "fits a job on a 0-10 scale per axis, using ONLY the candidate profile and "
    "job description provided. Do not invent candidate facts. Most jobs are a "
    "MEDIOCRE fit — reserve 8-10 for genuinely strong matches and use the full "
    "range. Output ONLY JSON matching:\n"
    '{"skill_overlap": {"score": float, "justification": str}, '
    '"seniority_fit": {"score": float, "justification": str}, '
    '"domain_fit": {"score": float, "justification": str}}\n'
    "Each justification is ONE sentence naming the specific evidence.\n"
    "ANCHORS (apply to every axis):\n"
    "  0-2 = no/there-is-a-hard-mismatch;\n"
    "  3-5 = partial/tangential overlap, notable gaps;\n"
    "  6-7 = solid fit with minor gaps;\n"
    "  8-10 = strong, specific, direct match.\n"
    "skill_overlap = candidate skills vs the role's required/preferred stack "
    "(a 3 means only generic overlap like 'Python'; an 8 means most required "
    "tools are directly evidenced). seniority_fit = the candidate is a new-grad "
    "MS with 0 years professional experience (a 2 = the role needs a senior/"
    "5+yrs; a 9 = explicitly new-grad/entry). domain_fit = candidate's actual "
    "projects vs the role's domain (a 3 = adjacent, an 8 = same domain, e.g. "
    "LLM/agents/ML infra)."
)

# Trimmed hard so each call stays comfortably under ~2K input tokens (keeps us
# within the free-tier 10K tok/min at 5 req/min). Requirements + level signals
# are almost always in the first part of a posting.
_MAX_DESC_CHARS = 1500


def _candidate_profile(fact_bank: FactBank) -> str:
    skills = sorted(fact_bank.all_skills())
    claims = [f"- {f.claim}" for f in fact_bank.facts]
    prof = fact_bank.profile
    return (
        f"Candidate: {prof.name}, target seniority {prof.target_seniority}, "
        f"{prof.years_professional_experience} years professional experience.\n"
        f"Skills: {', '.join(skills)}\n"
        f"Evidence (projects):\n" + "\n".join(claims)
    )


def extract_messages(title: str, description_text: str) -> list[Message]:
    return [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Job title: {title}\n\nJob description:\n"
                f"{description_text[:_MAX_DESC_CHARS]}"
            ),
        },
    ]


def rubric_messages(
    title: str, description_text: str, fact_bank: FactBank
) -> list[Message]:
    return [
        {"role": "system", "content": _RUBRIC_SYSTEM},
        {
            "role": "user",
            "content": (
                f"CANDIDATE PROFILE:\n{_candidate_profile(fact_bank)}\n\n"
                f"JOB TITLE: {title}\n\nJOB DESCRIPTION:\n"
                f"{description_text[:_MAX_DESC_CHARS]}"
            ),
        },
    ]
