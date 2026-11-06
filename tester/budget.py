"""Process-level cost meter (safety rule 6).

Lives outside graph state, behind a lock, so parallel workers (M4) can't each
overshoot the ceiling before a state merge.
"""

from __future__ import annotations

import threading

# USD per million tokens: (input, output, cache read, cache write). Anthropic
# first-party list prices as of 2026-10; unknown models are priced as Opus so the
# ceiling errs on the safe side.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
FALLBACK = PRICES["claude-opus-5-5"]


def price_for(model: str | None) -> tuple[float, float, float, float]:
    name = (model or "").split(":", 1)[-1]
    return PRICES.get(name, FALLBACK)


def cost_of(model: str | None, usage: dict | None) -> float:
    if not usage:
        return 0.0
    p_in, p_out, p_read, p_write = price_for(model)
    details = usage.get("input_token_details") or {}
    read = details.get("cache_read") or 0
    write = details.get("cache_creation") or 0
    fresh = max((usage.get("input_tokens") or 0) - read - write, 0)
    out = usage.get("output_tokens") or 0
    return (fresh * p_in + read * p_read + write * p_write + out * p_out) / 1_000_000


class CostMeter:
    def __init__(self, ceiling_usd: float, model: str | None):
        self.ceiling = ceiling_usd
        self.model = model
        self.spent = 0.0
        self._lock = threading.Lock()

    def add(self, usage: dict | None) -> float:
        c = cost_of(self.model, usage)
        with self._lock:
            self.spent += c
        return c

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent >= self.ceiling
