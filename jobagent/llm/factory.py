"""Build the configured LLM provider from settings/env."""
from __future__ import annotations

from ..config import Settings, get_settings
from .anthropic import AnthropicProvider
from .base import BaseLLMProvider, LLMError
from .gemini import GeminiProvider
from .groq import GroqProvider


def build_provider(settings: Settings | None = None) -> BaseLLMProvider:
    """Instantiate the provider named by ``LLM_PROVIDER`` (default groq)."""
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "groq":
        return GroqProvider(settings.groq_api_key, settings.groq_model)
    if provider == "anthropic":
        return AnthropicProvider(
            settings.anthropic_api_key,
            settings.anthropic_model,
            min_interval=settings.anthropic_min_interval,
        )
    if provider == "gemini":
        return GeminiProvider(
            settings.gemini_api_key,
            settings.gemini_model,
            min_interval=settings.gemini_min_interval,
        )
    raise LLMError(
        f"unknown LLM_PROVIDER {settings.llm_provider!r} "
        "(expected: groq | anthropic | gemini)"
    )
