"""A scripted, offline provider for tests and keyless runs.

Makes NO network calls. A ``responder`` callable maps the outgoing messages to
the raw text the "model" returns, so tests can assert on prompts and drive
structured output deterministically. Call/token accounting works exactly like a
real provider, so cost-discipline and idempotency assertions are meaningful.
"""
from __future__ import annotations

from collections.abc import Callable

from .base import BaseLLMProvider, Message, RawCompletion

Responder = Callable[[list[Message]], str]


class ScriptedProvider(BaseLLMProvider):
    def __init__(self, responder: Responder, model: str = "fake-model") -> None:
        super().__init__(model)
        self._responder = responder
        self.prompts: list[list[Message]] = []

    async def _raw_complete(self, messages: list[Message]) -> RawCompletion:
        self.prompts.append(messages)
        text = self._responder(messages)
        # Rough token estimate (~4 chars/token) for realistic accounting.
        in_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        return RawCompletion(text=text, input_tokens=in_tokens, output_tokens=len(text) // 4)
