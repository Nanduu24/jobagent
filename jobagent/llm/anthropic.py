"""Anthropic provider (Messages API), paced for the free tier."""
from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import httpx

from .base import (
    BaseLLMProvider,
    LLMError,
    Message,
    RawCompletion,
    request_with_retry,
)

_URL = "https://api.anthropic.com/v1/messages"
_VERSION = "2023-06-01"


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
        min_interval: float = 12.0,
    ) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._api_key = api_key
        self._max_tokens = max_tokens
        # Free-tier pacing: keep >= min_interval seconds between requests so we
        # stay under 5 req/min and never burst into a 429.
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def _pace(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval

    async def _raw_complete(self, messages: list[Message]) -> RawCompletion:
        await self._pace()
        system, rest = _split_system(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": rest,
        }
        if system:
            body["system"] = system
        resp = await request_with_retry(
            self._client,
            "POST",
            _URL,
            headers={"x-api-key": self._api_key, "anthropic-version": _VERSION},
            json=body,
        )
        resp.raise_for_status()
        data = cast(dict[str, Any], resp.json())
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return RawCompletion(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
