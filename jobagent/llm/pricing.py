"""Per-model pricing (USD per 1M tokens) for cost accounting.

Estimates for reporting, not billing. A model absent from the table is treated
as UNPRICED and contributes $0.00 (we never fabricate a cost from a default
rate). Groq's llama models are on the free tier here, so they price at $0.00 —
paid list prices are noted in comments for reference.
"""
from __future__ import annotations

# model -> (input_usd_per_mtok, output_usd_per_mtok)
PRICES: dict[str, tuple[float, float]] = {
    # Groq free tier: no per-token charge (rate-limited). Paid list ~0.59/0.79.
    "llama-3.3-70b-versatile": (0.0, 0.0),
    "llama-3.1-8b-instant": (0.0, 0.0),
    # Anthropic (paid).
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    # Google Gemini free tier: no per-token charge.
    "gemini-3.1-flash-lite": (0.0, 0.0),
    "gemini-2.5-flash-lite": (0.0, 0.0),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
}


def is_priced(model: str) -> bool:
    return model in PRICES


def price_for(model: str) -> tuple[float, float] | None:
    return PRICES.get(model)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost of one call. Unknown model -> $0.00 (never a default
    paid rate)."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    in_price, out_price = price
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0
