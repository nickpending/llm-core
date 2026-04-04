"""Anthropic Messages API adapter.

Converts AdapterRequest to Anthropic's /messages endpoint format
and normalizes the response into AdapterResponse.

Reference: llmcli-tools/packages/llm-core/lib/providers/anthropic.ts
"""

from __future__ import annotations

import httpx

from ..exceptions import ProviderError
from ..types import AdapterRequest, AdapterResponse

# Anthropic API version — pinned per https://docs.anthropic.com/en/api/versioning
ANTHROPIC_API_VERSION = "2023-06-01"

# Anthropic requires max_tokens (unlike OpenAI/Ollama). Default if caller doesn't specify.
DEFAULT_MAX_TOKENS = 8192


def complete(request: AdapterRequest) -> AdapterResponse:
    """Execute a completion request against the Anthropic Messages API."""
    body: dict[str, object] = {
        "model": request.model,
        "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": request.prompt}],
    }

    if request.system_prompt is not None:
        body["system"] = request.system_prompt
    if request.temperature is not None:
        body["temperature"] = request.temperature

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_API_VERSION,
    }
    if request.api_key is not None:
        headers["x-api-key"] = request.api_key

    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{request.base_url}/messages", headers=headers, json=body)

        if response.status_code != 200:
            raise ProviderError(
                f"Anthropic API error ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

        data = response.json()

    # Validate response shape
    if (
        not isinstance(data.get("content"), list)
        or len(data["content"]) == 0
        or not isinstance(data["content"][0].get("text"), str)
    ):
        raise ProviderError(
            "Anthropic API returned unexpected response shape: missing content[0].text"
        )

    # Map stop_reason to normalized finish_reason
    stop_reason = data.get("stop_reason")
    finish_reason = "max_tokens" if stop_reason == "max_tokens" else "stop"

    return AdapterResponse(
        text=data["content"][0]["text"],
        model=data["model"],
        tokens_input=data.get("usage", {}).get("input_tokens", 0),
        tokens_output=data.get("usage", {}).get("output_tokens", 0),
        finish_reason=finish_reason,
    )
