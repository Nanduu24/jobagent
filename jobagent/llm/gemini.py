"""Google Gemini provider (generateContent), paced for the free tier."""
from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import httpx

from .anthropic import _split_system
from .base import BaseLLMProvider, LLMError, Message, RawCompletion, request_with_retry

_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        min_interval: float = 5.0,
    ) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMError("GEMINI_API_KEY / GOOGLE_API_KEY is not set")
        self._api_key = api_key
        # Free-tier pacing: keep >= min_interval seconds between requests.
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
        contents = [
            {
                "role": "model" if m.get("role") == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in rest
        ]
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        resp = await request_with_retry(
            self._client,
            "POST",
            _URL.format(model=self.model),
            params={"key": self._api_key},
            json=body,
        )
        resp.raise_for_status()
        data = cast(dict[str, Any], resp.json())
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        return RawCompletion(
            text=text,
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=int(usage.get("candidatesTokenCount", 0)),
        )
