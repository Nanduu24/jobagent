"""Groq provider (OpenAI-compatible chat completions). Default provider."""
from __future__ import annotations

from typing import Any, cast

import httpx

from .base import BaseLLMProvider, LLMError, Message, RawCompletion, request_with_retry

_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def _raw_complete(self, messages: list[Message]) -> RawCompletion:
        resp = await request_with_retry(
            self._client,
            "POST",
            _URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        data = cast(dict[str, Any], resp.json())
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return RawCompletion(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
