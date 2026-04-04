"""Tests for provider adapters — Anthropic, OpenAI, Ollama.

Covers:
- INV-002: AdapterResponse has all 5 required fields (text, model, tokens_input,
  tokens_output, finish_reason) from every adapter
- INV-004: HTTP error paths raise ProviderError with status_code set
  (4xx non-transient, 5xx transient — classified correctly by retry.py)
- INV-005 (discovered): All response-shape failures raise ProviderError, never KeyError —
  unclassified exceptions bypass retry.py and surface as crashes to callers
- Edge cases: Anthropic system field gating, OpenAI response_format gating,
  Ollama options block with None values excluded

All HTTP calls are intercepted via monkeypatch.setattr on httpx.Client.post.
No real network calls. No MagicMock (claudex-guard blocks it).
"""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest

from llm_core.exceptions import ProviderError
from llm_core.retry import is_transient_error
from llm_core.types import AdapterRequest, AdapterResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal httpx.Response stand-in for adapter tests.

    Adapters access: .status_code, .text, .json()
    """

    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body
        # .text is used in error message construction — serialize body for it
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self) -> object:
        return self._body


def _make_adapter_request(
    base_url: str = "https://example.com",
    api_key: str | None = "test-key",
    model: str = "test-model",
    prompt: str = "hello",
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> AdapterRequest:
    return AdapterRequest(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        json=json_mode,
    )


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


def test_anthropic_adapter_response_has_all_required_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-002: Anthropic complete() returns AdapterResponse with all 5 required fields set."""
    fake_body = {
        "content": [{"type": "text", "text": "Hello from Anthropic"}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, **kwargs: _FakeResponse(200, fake_body),
    )

    from llm_core.providers import anthropic

    result = anthropic.complete(_make_adapter_request())

    # Verify all required fields are present and typed correctly
    required_fields = {"text", "model", "tokens_input", "tokens_output", "finish_reason"}
    actual_fields = {f.name for f in dataclasses.fields(AdapterResponse)}
    assert required_fields <= actual_fields, (
        f"AdapterResponse missing fields: {required_fields - actual_fields}"
    )

    assert isinstance(result, AdapterResponse)
    assert result.text == "Hello from Anthropic"
    assert result.model == "claude-3-opus-20240229"
    assert result.tokens_input == 10
    assert result.tokens_output == 5
    assert result.finish_reason == "stop"  # end_turn → stop


def test_anthropic_http_error_raises_provider_error_with_status_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-004: Anthropic HTTP errors raise ProviderError with status_code set.

    Verifies both a non-transient (401) and a transient (503) code to confirm
    retry.py can classify errors from this adapter correctly.
    """
    from llm_core.providers import anthropic

    def make_fake_post(code: int):
        def fake_post(self: httpx.Client, url: str, **kwargs: object) -> _FakeResponse:
            return _FakeResponse(code, "error")

        return fake_post

    for status_code in (401, 503):
        monkeypatch.setattr(httpx.Client, "post", make_fake_post(status_code))
        with pytest.raises(ProviderError) as exc_info:
            anthropic.complete(_make_adapter_request())

        err = exc_info.value
        assert err.status_code == status_code, (
            f"Expected status_code={status_code}, got {err.status_code}"
        )

        if status_code == 401:
            assert not is_transient_error(err), "401 must be non-transient"
        elif status_code == 503:
            assert is_transient_error(err), "503 must be transient"


def test_anthropic_system_field_excluded_when_no_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge case: Anthropic body must NOT include 'system' key when system_prompt is None."""
    captured_body: list[dict] = []

    def fake_post(self: httpx.Client, url: str, **kwargs: object) -> _FakeResponse:
        captured_body.append(kwargs.get("json", {}))  # type: ignore[arg-type]
        return _FakeResponse(
            200,
            {
                "content": [{"type": "text", "text": "ok"}],
                "model": "test-model",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    from llm_core.providers import anthropic

    # No system_prompt — body should NOT contain "system" key
    anthropic.complete(_make_adapter_request(system_prompt=None))
    assert "system" not in captured_body[0], (
        "system field must be absent when system_prompt is None"
    )

    captured_body.clear()

    # With system_prompt — body SHOULD contain "system" key
    anthropic.complete(_make_adapter_request(system_prompt="You are helpful"))
    assert "system" in captured_body[0], "system field must be present when system_prompt is set"
    assert captured_body[0]["system"] == "You are helpful"


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


def test_openai_adapter_response_has_all_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """INV-002: OpenAI complete() returns AdapterResponse with all 5 required fields set."""
    fake_body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from OpenAI"},
                "finish_reason": "stop",
            }
        ],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 8, "completion_tokens": 6},
    }
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, **kwargs: _FakeResponse(200, fake_body),
    )

    from llm_core.providers import openai

    result = openai.complete(_make_adapter_request())

    assert isinstance(result, AdapterResponse)
    assert result.text == "Hello from OpenAI"
    assert result.model == "gpt-4o"
    assert result.tokens_input == 8
    assert result.tokens_output == 6
    assert result.finish_reason == "stop"


def test_openai_http_error_raises_provider_error_with_status_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-004: OpenAI HTTP errors raise ProviderError with status_code set."""
    from llm_core.providers import openai

    def make_fake_post(code: int):
        def fake_post(self: httpx.Client, url: str, **kwargs: object) -> _FakeResponse:
            return _FakeResponse(code, "error")

        return fake_post

    for status_code in (401, 503):
        monkeypatch.setattr(httpx.Client, "post", make_fake_post(status_code))
        with pytest.raises(ProviderError) as exc_info:
            openai.complete(_make_adapter_request())

        err = exc_info.value
        assert err.status_code == status_code

        if status_code == 401:
            assert not is_transient_error(err)
        elif status_code == 503:
            assert is_transient_error(err)


def test_openai_response_format_only_included_when_json_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge case: OpenAI body must NOT include response_format when json=False."""
    captured_body: list[dict] = []

    def fake_post(self: httpx.Client, url: str, **kwargs: object) -> _FakeResponse:
        captured_body.append(kwargs.get("json", {}))  # type: ignore[arg-type]
        return _FakeResponse(
            200,
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    from llm_core.providers import openai

    # json=False (default) — response_format must be absent
    openai.complete(_make_adapter_request(json_mode=False))
    assert "response_format" not in captured_body[0], (
        "response_format must be absent when json=False"
    )

    captured_body.clear()

    # json=True — response_format must be present with correct value
    openai.complete(_make_adapter_request(json_mode=True))
    assert "response_format" in captured_body[0], "response_format must be present when json=True"
    assert captured_body[0]["response_format"] == {"type": "json_object"}


def test_openai_missing_content_raises_provider_error_not_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-005: OpenAI response with message dict but absent content key must raise
    ProviderError, not KeyError.

    The shape validator confirms choices[0].message is a dict, but without checking
    the content key inside it. A message dict with no content key (e.g., tool-call
    response or malformed response) previously caused an unclassified KeyError that
    bypassed retry.py classification and surfaced as a crash to callers.
    """
    # message is a dict (passes old shape check) but has no "content" key
    fake_body = {
        "choices": [
            {
                "message": {"role": "assistant"},  # content key absent
                "finish_reason": "stop",
            }
        ],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 1, "completion_tokens": 0},
    }
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, **kwargs: _FakeResponse(200, fake_body),
    )

    from llm_core.providers import openai

    with pytest.raises(ProviderError):
        openai.complete(_make_adapter_request())


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------


def test_ollama_adapter_response_has_all_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """INV-002: Ollama complete() returns AdapterResponse with all 5 required fields set."""
    fake_body = {
        "response": "Hello from Ollama",
        "model": "llama3",
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 7,
    }
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, **kwargs: _FakeResponse(200, fake_body),
    )

    from llm_core.providers import ollama

    result = ollama.complete(_make_adapter_request(api_key=None))

    assert isinstance(result, AdapterResponse)
    assert result.text == "Hello from Ollama"
    assert result.model == "llama3"
    assert result.tokens_input == 12
    assert result.tokens_output == 7
    assert result.finish_reason == "stop"


def test_ollama_http_error_raises_provider_error_with_status_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-004: Ollama HTTP errors raise ProviderError with status_code set."""
    from llm_core.providers import ollama

    def make_fake_post(code: int):
        def fake_post(self: httpx.Client, url: str, **kwargs: object) -> _FakeResponse:
            return _FakeResponse(code, "error")

        return fake_post

    for status_code in (401, 503):
        monkeypatch.setattr(httpx.Client, "post", make_fake_post(status_code))
        with pytest.raises(ProviderError) as exc_info:
            ollama.complete(_make_adapter_request(api_key=None))

        err = exc_info.value
        assert err.status_code == status_code

        if status_code == 401:
            assert not is_transient_error(err)
        elif status_code == 503:
            assert is_transient_error(err)


def test_ollama_options_block_excludes_none_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edge case: Ollama options dict must omit keys whose values are None."""
    captured_body: list[dict] = []

    def fake_post(self: httpx.Client, url: str, **kwargs: object) -> _FakeResponse:
        captured_body.append(kwargs.get("json", {}))  # type: ignore[arg-type]
        return _FakeResponse(
            200,
            {
                "response": "ok",
                "model": "llama3",
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    from llm_core.providers import ollama

    # No temperature/max_tokens — options must be empty dict (not contain None values)
    ollama.complete(_make_adapter_request(api_key=None, temperature=None, max_tokens=None))
    options = captured_body[0].get("options", {})
    assert "temperature" not in options, "temperature must be absent when None"
    assert "num_predict" not in options, "num_predict must be absent when max_tokens is None"

    captured_body.clear()

    # With temperature and max_tokens — options must include both
    ollama.complete(_make_adapter_request(api_key=None, temperature=0.7, max_tokens=512))
    options = captured_body[0].get("options", {})
    assert options.get("temperature") == 0.7
    assert options.get("num_predict") == 512
