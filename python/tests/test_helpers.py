"""Tests for helpers.py — extract_json and is_truncated.

Covers:
- SC-9: extract_json strips markdown code blocks and parses JSON
- SC-9: is_truncated returns True for finish_reason="max_tokens"
- extract_json on plain JSON (no code block)
- extract_json on invalid JSON returns None
"""

from llm_core.helpers import extract_json, is_truncated
from llm_core.types import CompleteResult, TokenUsage


def _make_result(finish_reason: str) -> CompleteResult:
    return CompleteResult(
        text="hello",
        model="gpt-4.1-mini",
        provider="openai",
        tokens=TokenUsage(input=10, output=5),
        finish_reason=finish_reason,
        duration_ms=100,
        cost=None,
    )


def test_extract_json_from_markdown_code_block() -> None:
    """SC-9: JSON wrapped in ```json ... ``` is extracted and parsed."""
    text = '```json\n{"key": "value"}\n```'
    result = extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_from_plain_json() -> None:
    """Plain JSON string with no code block is parsed directly."""
    result = extract_json('{"answer": 42}')
    assert result == {"answer": 42}


def test_extract_json_invalid_returns_none() -> None:
    """Non-JSON text returns None, never raises."""
    result = extract_json("this is not json at all")
    assert result is None


def test_is_truncated_true_for_max_tokens() -> None:
    """SC-9: finish_reason='max_tokens' → is_truncated returns True."""
    assert is_truncated(_make_result("max_tokens")) is True


def test_is_truncated_false_for_stop() -> None:
    """SC-9: finish_reason='stop' → is_truncated returns False."""
    assert is_truncated(_make_result("stop")) is False
