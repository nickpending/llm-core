"""Tests for pricing.py -- cost estimation from pricing.toml.

Covers:
- SC-1: correct cost calculation for known model (pricing.toml)
- Empty pricing data returns None (never raises)
- Unknown model returns None
- Missing pricing.toml returns None (no network call, no error)
"""

from __future__ import annotations

import pytest

from llm_core import pricing
from llm_core.pricing import estimate_cost


def test_known_model_returns_correct_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-1: gpt-4.1-mini pricing from pricing.toml.

    pricing.toml: input=0.40, output=1.60 (per 1M tokens)
    1000/1M * 0.40 + 500/1M * 1.60 = 0.0004 + 0.0008 = 0.0012
    """
    # Reset cache so it re-reads from real pricing.toml
    monkeypatch.setattr(pricing, "_cache", None)
    cost = estimate_cost("gpt-4.1-mini", 1000, 500)

    assert cost is not None
    assert cost == pytest.approx(0.0012)


def test_empty_pricing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty pricing data returns None, never raises."""
    monkeypatch.setattr(pricing, "_cache", {})

    cost = estimate_cost("gpt-4.1-mini", 1000, 500)

    assert cost is None


def test_unknown_model_returns_none() -> None:
    """Unknown model not in pricing data returns None."""
    cost = estimate_cost("unknown-model-xyz-999", 1000, 500)

    assert cost is None


def test_missing_toml_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """SC-2: Missing pricing.toml returns None without network calls or errors."""
    monkeypatch.setattr(pricing, "_cache", None)
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path / "nonexistent"))

    cost = estimate_cost("gpt-4.1-mini", 1000, 500)

    assert cost is None
