"""Tests for pricing.py — cost estimation from pricing.toml.

Covers:
- SC-7: correct cost calculation for known model
- SC-7: missing pricing file returns None (never raises)
- Unknown model returns None
- LLM_CORE_CONFIG_DIR env var isolation
"""

import tempfile
from pathlib import Path

import pytest

from llm_core.pricing import estimate_cost

PRICING_TOML_CONTENT = """\
[models."gpt-4.1-mini"]
input = 0.40
output = 1.60
"""


def test_known_model_returns_correct_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-7: gpt-4.1-mini, 1000 input + 500 output tokens = 0.0012 USD."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pricing_file = Path(tmp_dir) / "pricing.toml"
        pricing_file.write_text(PRICING_TOML_CONTENT)

        monkeypatch.setenv("LLM_CORE_CONFIG_DIR", tmp_dir)

        cost = estimate_cost("gpt-4.1-mini", 1000, 500)

    # 1000/1_000_000 * 0.40 + 500/1_000_000 * 1.60 = 0.0004 + 0.0008 = 0.0012
    assert cost == pytest.approx(0.0012)


def test_missing_pricing_file_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-7: No pricing.toml (write blocked) → estimate_cost returns None."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Make dir read-only so pricing.toml can't be written
        Path(tmp_dir).chmod(0o444)
        monkeypatch.setenv("LLM_CORE_CONFIG_DIR", tmp_dir)

        try:
            cost = estimate_cost("gpt-4.1-mini", 1000, 500)
        finally:
            Path(tmp_dir).chmod(0o755)  # Restore for cleanup

    assert cost is None, "Missing pricing file must return None, not raise"


def test_unknown_model_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown model not in pricing.toml returns None."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pricing_file = Path(tmp_dir) / "pricing.toml"
        pricing_file.write_text(PRICING_TOML_CONTENT)

        monkeypatch.setenv("LLM_CORE_CONFIG_DIR", tmp_dir)

        cost = estimate_cost("unknown-model-xyz", 1000, 500)

    assert cost is None
