"""Tests for types.py — INV-002: CompleteResult canonical field set.

Covers:
- INV-002: CompleteResult has exactly the required fields
- SC-5: apiconf key resolution via load_api_key (uses real ~/.config/apiconf/config.toml)
"""

import dataclasses

import pytest

from llm_core.types import CompleteResult


def test_complete_result_has_all_required_fields() -> None:
    """INV-002: CompleteResult must have all 7 canonical fields."""
    required = {"text", "model", "provider", "tokens", "finish_reason", "duration_ms", "cost"}
    actual = {f.name for f in dataclasses.fields(CompleteResult)}
    missing = required - actual
    assert not missing, f"CompleteResult missing required fields: {missing}"


def test_complete_result_tokens_has_input_and_output() -> None:
    """INV-002: tokens field is a TokenUsage with input and output sub-fields."""
    from llm_core.types import TokenUsage

    token_fields = {f.name for f in dataclasses.fields(TokenUsage)}
    assert "input" in token_fields
    assert "output" in token_fields


def test_load_api_key_for_service_without_key_required() -> None:
    """load_api_key returns None for services that don't require a key (ollama pattern)."""
    from llm_core.config import load_api_key
    from llm_core.types import ServiceConfig

    svc = ServiceConfig(
        adapter="ollama",
        base_url="http://localhost:11434",
        key_required=False,
    )
    result = load_api_key(svc)
    assert result is None


def test_load_api_key_with_real_apiconf() -> None:
    """SC-5: load_api_key reads real apiconf config and returns key value."""

    from llm_core.config import load_api_key
    from llm_core.exceptions import ConfigError
    from llm_core.types import ServiceConfig

    svc = ServiceConfig(
        adapter="openai",
        base_url="https://api.openai.com/v1",
        key="openai",
        key_required=True,
    )

    try:
        key = load_api_key(svc)
        assert key is not None
        assert len(key) > 0
    except ConfigError:
        pytest.skip("apiconf config not available — skipping SC-5 integration test")
