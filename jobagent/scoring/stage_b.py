"""Stage B: LLM requirement extraction (b1) + fit rubric (b2)."""
from __future__ import annotations

from ..factbank import FactBank
from ..llm.base import BaseLLMProvider
from .prompts import extract_messages, rubric_messages
from .schemas import Requirements, Rubric


async def extract_requirements(
    provider: BaseLLMProvider, title: str, description_text: str
) -> Requirements:
    return await provider.complete(
        extract_messages(title, description_text), schema=Requirements
    )


async def score_rubric(
    provider: BaseLLMProvider,
    title: str,
    description_text: str,
    fact_bank: FactBank,
) -> Rubric:
    return await provider.complete(
        rubric_messages(title, description_text, fact_bank), schema=Rubric
    )
