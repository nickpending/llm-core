"""Core orchestration functions for llm-core.

complete() orchestrates the full completion flow: service resolution,
key loading, adapter dispatch with retry, cost estimation.

health_check() validates provider connectivity without consuming tokens.
"""

from __future__ import annotations

import time

import httpx

from .config import load_api_key
from .exceptions import ProviderError
from .pricing import estimate_cost
from .providers import get_adapter
from .retry import with_retry
from .services import resolve_service
from .types import AdapterRequest, CompleteResult, TokenUsage


def complete(
    prompt: str,
    *,
    service: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json: bool = False,
) -> CompleteResult:
    """Execute a single-turn LLM completion.

    Args:
        prompt: The user prompt text.
        service: Named service from services.toml (defaults to default_service).
        model: Model override (defaults to service's default_model).
        system_prompt: Optional system prompt.
        temperature: Sampling temperature override.
        max_tokens: Max tokens override.
        json: Request JSON output format from the provider.

    Returns:
        CompleteResult with normalized fields across all providers.

    Raises:
        ValueError: If no model specified and service has no default_model.
        ConfigError: If service or API key configuration is invalid.
        ProviderError: If the provider returns an HTTP error.
    """
    start_time = time.monotonic()

    # 1. Resolve service configuration
    svc = resolve_service(service)

    # 2. Load API key (None for key_required=False services)
    api_key = load_api_key(svc)

    # 3. Get provider adapter
    adapter = get_adapter(svc.adapter)

    # 4. Resolve model: caller model > service default_model
    resolved_model = model or svc.default_model
    if not resolved_model:
        raise ValueError(
            "Model name required: pass model= in complete() or set default_model in services.toml"
        )

    # 5. Build adapter request
    request = AdapterRequest(
        base_url=svc.base_url,
        api_key=api_key,
        model=resolved_model,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        json=json,
    )

    # 6. Call provider with retry
    response = with_retry(lambda: adapter.complete(request))

    # 7. Estimate cost
    cost = estimate_cost(response.model, response.tokens_input, response.tokens_output)

    # 8. Return normalized envelope
    duration_ms = int((time.monotonic() - start_time) * 1000)

    return CompleteResult(
        text=response.text,
        model=response.model,
        provider=svc.adapter,
        tokens=TokenUsage(input=response.tokens_input, output=response.tokens_output),
        finish_reason=response.finish_reason,
        duration_ms=duration_ms,
        cost=cost,
    )


def health_check(service: str | None = None) -> None:
    """Validate provider connectivity and auth.

    Uses lightweight provider-specific endpoints (not a completion call)
    to verify the provider is reachable and credentials are valid.

    Args:
        service: Named service from services.toml (defaults to default_service).

    Raises:
        ProviderError: If provider is unreachable or auth fails.
        ConfigError: If service config is invalid.
        ValueError: If adapter name is unknown.
    """
    svc = resolve_service(service)
    api_key = load_api_key(svc)

    # Provider-specific lightweight endpoint (delegated to adapter)
    adapter = get_adapter(svc.adapter)
    url, headers = adapter.health_check_config(svc.base_url, api_key)

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
    except httpx.ConnectError as exc:
        raise ProviderError(
            f"{svc.adapter} health check failed: {exc}",
            status_code=None,
        ) from exc

    if response.status_code not in {200, 204}:
        raise ProviderError(
            f"{svc.adapter} health check failed ({response.status_code}): {response.text}",
            status_code=response.status_code,
        )
