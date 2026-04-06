"""Tests for pricing.py — cost estimation from litellm pricing data.

Covers:
- SC-7: correct cost calculation for known model (bundled JSON)
- SC-7: versioned model name resolves correctly
- Empty pricing data returns None (never raises)
- Unknown model returns None
"""

from __future__ import annotations

import pytest

from llm_core import pricing
from llm_core.pricing import estimate_cost


def test_known_model_returns_correct_cost() -> None:
    """SC-7: gpt-4.1-mini pricing from bundled litellm data."""
    cost = estimate_cost("gpt-4.1-mini", 1000, 500)

    # input_cost_per_token: 4e-07, output_cost_per_token: 1.6e-06
    # 1000 * 4e-07 + 500 * 1.6e-06 = 0.0004 + 0.0008 = 0.0012
    assert cost is not None
    assert cost == pytest.approx(0.0012)


def test_versioned_model_name_resolves() -> None:
    """SC-7: Provider-returned versioned name resolves in pricing data."""
    cost = estimate_cost("gpt-4.1-mini-2025-04-14", 1000, 500)

    assert cost is not None
    assert cost == pytest.approx(0.0012)


def test_empty_pricing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-7: Empty pricing data returns None, never raises."""
    monkeypatch.setattr(pricing, "_cache", {})

    cost = estimate_cost("gpt-4.1-mini", 1000, 500)

    assert cost is None


def test_unknown_model_returns_none() -> None:
    """Unknown model not in pricing data returns None."""
    cost = estimate_cost("unknown-model-xyz-999", 1000, 500)

    assert cost is None
