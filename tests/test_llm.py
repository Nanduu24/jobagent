"""LLM abstraction: parsing, fence-stripping, one-retry, accounting, factory."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from jobagent.config import Settings
from jobagent.llm.base import LLMError, parse_structured, strip_fences
from jobagent.llm.factory import build_provider
from jobagent.llm.fake import ScriptedProvider


class Out(BaseModel):
    a: int
    b: str


def test_strip_fences() -> None:
    assert strip_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_fences('```\n{"a":1}\n```') == '{"a":1}'
    assert strip_fences('{"a":1}') == '{"a":1}'


def test_parse_structured_variants() -> None:
    assert parse_structured('{"a":1,"b":"x"}', Out) == Out(a=1, b="x")
    assert parse_structured('```json\n{"a":1,"b":"x"}\n```', Out) == Out(a=1, b="x")
    # leading prose before the JSON
    assert parse_structured('Here you go:\n{"a":2,"b":"y"}', Out) == Out(a=2, b="y")


def test_parse_structured_failure() -> None:
    with pytest.raises(LLMError):
        parse_structured("not json at all", Out)


@pytest.mark.asyncio
async def test_complete_retries_once_on_parse_failure() -> None:
    replies = iter(["garbage, no json", '{"a":1,"b":"ok"}'])
    provider = ScriptedProvider(lambda _m: next(replies))
    result = await provider.complete([{"role": "user", "content": "hi"}], schema=Out)
    assert result == Out(a=1, b="ok")
    assert provider.usage.calls == 2  # first parse failed -> one retry


@pytest.mark.asyncio
async def test_complete_raw_and_usage_accounting() -> None:
    provider = ScriptedProvider(lambda _m: "plain text reply")
    text = await provider.complete([{"role": "user", "content": "hello there"}])
    assert text == "plain text reply"
    assert provider.usage.calls == 1
    assert provider.usage.input_tokens > 0
    assert provider.usage.cost_usd >= 0.0


def test_factory_unknown_provider() -> None:
    with pytest.raises(LLMError):
        build_provider(Settings(LLM_PROVIDER="does-not-exist"))


def test_factory_groq_requires_key() -> None:
    with pytest.raises(LLMError):
        build_provider(Settings(LLM_PROVIDER="groq", GROQ_API_KEY=""))
