"""Tests for core.py — complete() and health_check() orchestration.

Covers:
- INV-002: complete() returns CompleteResult with all 7 required fields
- INV-005: __init__.py exports all 9 required public functions
- SC-1/SC-8: complete() cost field wired correctly from estimate_cost()
- SC-8: health_check() raises ProviderError on non-200/204 responses
- SC-8: health_check() returns None on 200 (no exception)
- Error cases: no model specified → ValueError; unknown model → cost=None
- Edge case: caller model overrides service default_model

All HTTP calls intercepted via monkeypatch.setattr on httpx.Client.post / httpx.Client.get.
No real network calls. No MagicMock (claudex-guard blocks it).
load_api_key monkeypatched to bypass apiconf dependency.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest

from llm_core.exceptions import ProviderError
from llm_core.types import CompleteResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal httpx.Response stand-in for adapter tests.

    Adapters access: .status_code, .text, .json()
    health_check accesses: .status_code, .text
    """

    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self) -> object:
        return self._body


def _openai_success_body(
    text: str = "Hello",
    model: str = "gpt-4.1-mini",
    tokens_in: int = 10,
    tokens_out: int = 5,
) -> dict:
    """Return a well-formed OpenAI completion response body."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "model": model,
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }


# ---------------------------------------------------------------------------
# INV-005: Public API exports
# ---------------------------------------------------------------------------


def test_public_api_exports_all_required_functions() -> None:
    """INV-005: llm_core package exports all 8 required public functions.

    __init__.py exports 8 names: complete, health_check, resolve_service,
    load_services, list_services, estimate_cost, extract_json, is_truncated.
    (update_pricing removed — pricing.py no longer fetches from GitHub.)
    """
    import llm_core

    required = [
        "complete",
        "health_check",
        "resolve_service",
        "load_services",
        "list_services",
        "estimate_cost",
        "extract_json",
        "is_truncated",
    ]
    missing = [name for name in required if not hasattr(llm_core, name)]
    assert not missing, f"llm_core missing exports: {missing}"


# ---------------------------------------------------------------------------
# INV-002: complete() returns normalized CompleteResult
# ---------------------------------------------------------------------------


def test_complete_returns_normalized_complete_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """INV-002: complete() returns CompleteResult with all 7 required fields set.

    Uses a mocked OpenAI adapter response. Verifies field presence and types,
    not just that CompleteResult is a dataclass — this catches wiring failures
    where fields are None due to broken orchestration.
    """
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    fake_body = _openai_success_body(
        text="Hello from openai",
        model="gpt-4.1-mini",
        tokens_in=14,
        tokens_out=5,
    )

    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, **kwargs: _FakeResponse(200, fake_body),
    )

    import llm_core.core as core_mod

    monkeypatch.setattr(core_mod, "load_api_key", lambda svc: "fake-key")

    from llm_core import complete

    result = complete("Say hello", service="openai", model="gpt-4.1-mini")

    # All 7 required fields must be present and non-empty/non-None where required
    required_fields = [
        "text",
        "model",
        "provider",
        "tokens",
        "finish_reason",
        "duration_ms",
        "cost",
    ]
    actual_fields = {f.name for f in dataclasses.fields(CompleteResult)}
    missing_fields = [f for f in required_fields if f not in actual_fields]
    assert not missing_fields, f"CompleteResult missing fields: {missing_fields}"

    assert isinstance(result, CompleteResult)
    assert result.text == "Hello from openai"
    assert result.model == "gpt-4.1-mini"
    assert result.provider == "openai"
    assert result.tokens.input == 14
    assert result.tokens.output == 5
    assert result.finish_reason == "stop"
    assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Cost wiring
# ---------------------------------------------------------------------------


def test_complete_populates_cost_for_known_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """HIGH: complete() cost field is populated from estimate_cost() for known model.

    gpt-4.1-mini pricing from pricing.toml. If cost wiring breaks
    (e.g., estimate_cost() not called, or wrong model passed), cost is silently
    None and cost observability regresses. This test catches that regression.
    """
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    # Write pricing.toml so estimate_cost can find the model
    (tmp_path / "pricing.toml").write_text('[models."gpt-4.1-mini"]\ninput = 0.40\noutput = 1.60\n')

    # Reset pricing cache so it reads from tmp_path
    import llm_core.pricing as pricing_mod

    monkeypatch.setattr(pricing_mod, "_cache", None)

    fake_body = _openai_success_body(
        text="Hello",
        model="gpt-4.1-mini",
        tokens_in=1000,
        tokens_out=500,
    )

    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, **kwargs: _FakeResponse(200, fake_body),
    )

    import llm_core.core as core_mod

    monkeypatch.setattr(core_mod, "load_api_key", lambda svc: "fake-key")

    from llm_core import complete

    result = complete("Hello", service="openai", model="gpt-4.1-mini")

    # gpt-4.1-mini: input=4e-07/token, output=1.6e-06/token
    # 1000 * 4e-07 + 500 * 1.6e-06 = 0.0004 + 0.0008 = 0.0012
    assert result.cost is not None, (
        "cost must be populated for gpt-4.1-mini — "
        "None here means estimate_cost() was not called or received wrong model name"
    )
    assert result.cost == pytest.approx(0.0012)


def test_complete_cost_is_none_for_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Edge case: Unknown model returns cost=None, not an error.

    Pricing is best-effort. Unknown models must not raise.
    """
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    fake_body = _openai_success_body(
        text="Hello",
        model="unknown-model-xyz",
        tokens_in=10,
        tokens_out=5,
    )

    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, **kwargs: _FakeResponse(200, fake_body),
    )

    import llm_core.core as core_mod

    monkeypatch.setattr(core_mod, "load_api_key", lambda svc: "fake-key")

    from llm_core import complete

    result = complete("Hello", service="openai", model="unknown-model-xyz")

    assert result.cost is None, "Unknown model must return cost=None without raising"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_complete_raises_value_error_when_no_model_specified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Error case: No model in call args and no default_model in service config → ValueError.

    The default bootstrapped services.toml has no default_model set.
    Callers that forget model= must get an informative ValueError, not a
    cryptic provider error.
    """
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    import llm_core.core as core_mod

    monkeypatch.setattr(core_mod, "load_api_key", lambda svc: "fake-key")

    from llm_core import complete

    with pytest.raises(ValueError, match="Model name required"):
        complete("Hello", service="openai")


def test_complete_caller_model_overrides_service_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Edge case: model= passed to complete() is used, not service default_model.

    Writes a services.toml with default_model='default-model' for openai,
    then verifies the adapter receives the caller-specified model instead.
    """
    # Write a services.toml that has a default_model
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "services.toml").write_text(
        """
default_service = "openai"

[services.openai]
adapter = "openai"
key = "openai"
base_url = "https://api.openai.com/v1"
default_model = "default-model"
"""
    )
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(config_dir))

    captured_bodies: list[dict] = []

    def fake_post(self: httpx.Client, url: str, **kwargs: object) -> _FakeResponse:
        body = kwargs.get("json", {})
        captured_bodies.append(body)  # type: ignore[arg-type]
        model = body.get("model", "default-model")  # type: ignore[union-attr]
        return _FakeResponse(
            200,
            _openai_success_body(model=model),
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    import llm_core.core as core_mod

    monkeypatch.setattr(core_mod, "load_api_key", lambda svc: "fake-key")

    from llm_core import complete

    result = complete("Hello", service="openai", model="caller-model")

    assert captured_bodies, "Expected at least one POST request"
    assert captured_bodies[0].get("model") == "caller-model", (
        f"Adapter received model={captured_bodies[0].get('model')!r}, expected 'caller-model'"
    )
    assert result.provider == "openai"


# ---------------------------------------------------------------------------
# SC-8: health_check()
# ---------------------------------------------------------------------------


def test_health_check_returns_none_on_200(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SC-8: health_check() returns None (no exception) when provider responds 200."""
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda self, url, **kwargs: _FakeResponse(200, {"models": []}),
    )

    import llm_core.core as core_mod

    monkeypatch.setattr(core_mod, "load_api_key", lambda svc: "fake-key")

    from llm_core import health_check

    result = health_check(service="openai")
    assert result is None, "health_check() must return None on success"


def test_health_check_raises_provider_error_on_non_200(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SC-8: health_check() raises ProviderError with status_code when provider returns 401.

    Verifies that the status_code is propagated so retry.py can classify it
    as non-transient (no retry on auth failure).
    """
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda self, url, **kwargs: _FakeResponse(401, "unauthorized"),
    )

    import llm_core.core as core_mod

    monkeypatch.setattr(core_mod, "load_api_key", lambda svc: "fake-key")

    from llm_core import health_check

    with pytest.raises(ProviderError) as exc_info:
        health_check(service="openai")

    err = exc_info.value
    assert err.status_code == 401, f"Expected status_code=401, got {err.status_code}"


def test_health_check_raises_value_error_for_unknown_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Edge case: health_check() raises ValueError for unknown adapter name.

    Guards against future misconfiguration where a new service uses an
    adapter name that health_check() doesn't recognize.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "services.toml").write_text(
        """
default_service = "custom"

[services.custom]
adapter = "unknown-adapter"
key = "custom"
base_url = "https://custom.example.com/v1"
"""
    )
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(config_dir))

    import llm_core.core as core_mod

    monkeypatch.setattr(core_mod, "load_api_key", lambda svc: "fake-key")

    from llm_core import health_check

    with pytest.raises(ValueError, match="Unknown adapter"):
        health_check(service="custom")
