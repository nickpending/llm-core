"""Tests for pricing.py — cost estimation from pricing.toml.

Covers:
- SC-2: correct cost calculation with per-1M-token rates
- SC-2: versioned model name resolves correctly
- Empty pricing data returns None (never raises)
- Unknown model returns None
"""

from __future__ import annotations

import textwrap

import pytest

from llm_core import pricing
from llm_core.pricing import estimate_cost

SAMPLE_TOML = textwrap.dedent("""\
    [models."gpt-4.1-mini"]
    input = 0.4
    output = 1.6

    [models."gpt-4.1-mini-2025-04-14"]
    input = 0.4
    output = 1.6
""")


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """Reset pricing cache and point config dir to tmp with sample TOML."""
    pricing._cache = None
    config_dir = tmp_path / "config"  # type: ignore[operator]
    config_dir.mkdir()
    toml_path = config_dir / "pricing.toml"
    toml_path.write_text(SAMPLE_TOML)
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(config_dir))


def test_known_model_returns_correct_cost() -> None:
    """SC-2: gpt-4.1-mini pricing with per-1M-token rates."""
    cost = estimate_cost("gpt-4.1-mini", 1000, 500)

    # input: (1000 / 1_000_000) * 0.4 = 0.0004
    # output: (500 / 1_000_000) * 1.6 = 0.0008
    # total: 0.0012
    assert cost is not None
    assert cost == pytest.approx(0.0012)


def test_versioned_model_name_resolves() -> None:
    """SC-2: Provider-returned versioned name resolves in pricing data."""
    cost = estimate_cost("gpt-4.1-mini-2025-04-14", 1000, 500)

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
