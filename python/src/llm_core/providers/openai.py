"""OpenAI Chat Completions API adapter.

Converts AdapterRequest to OpenAI's /chat/completions endpoint format
and normalizes the response into AdapterResponse.

Reference: llmcli-tools/packages/llm-core/lib/providers/openai.ts
"""

from __future__ import annotations

import httpx

from ..exceptions import ProviderError
from ..types import AdapterRequest, AdapterResponse


def health_check_config(base_url: str, api_key: str | None) -> tuple[str, dict[str, str]]:
    """Return the health check endpoint URL and headers for OpenAI.

    Uses the /models endpoint with Bearer token auth.
    """
    headers: dict[str, str] = {"Authorization": f"Bearer {api_key or ''}"}
    return f"{base_url}/models", headers


def complete(request: AdapterRequest) -> AdapterResponse:
    """Execute a completion request against the OpenAI Chat Completions API."""
    messages: list[dict[str, str]] = []
    if request.system_prompt is not None:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.prompt})

    body: dict[str, object] = {
        "model": request.model,
        "messages": messages,
    }

    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.json:
        body["response_format"] = {"type": "json_object"}

    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if request.api_key is not None:
        headers["Authorization"] = f"Bearer {request.api_key}"
    if request.app_title is not None:
        headers["X-OpenRouter-Title"] = request.app_title
    if request.app_url is not None:
        headers["HTTP-Referer"] = request.app_url

    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{request.base_url}/chat/completions", headers=headers, json=body)

        if response.status_code != 200:
            raise ProviderError(
                f"OpenAI API error ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

        data = response.json()

    # Validate response shape
    if (
        not isinstance(data.get("choices"), list)
        or len(data["choices"]) == 0
        or not isinstance(data["choices"][0].get("message"), dict)
        or not isinstance(data["choices"][0]["message"].get("content"), str)
    ):
        raise ProviderError(
            "OpenAI API returned unexpected response shape: missing choices[0].message.content"
        )

    choice = data["choices"][0]

    # Map finish_reason to normalized finish_reason
    raw_reason = choice.get("finish_reason")
    finish_reason = "max_tokens" if raw_reason == "length" else "stop"

    return AdapterResponse(
        text=choice["message"]["content"],
        model=data["model"],
        tokens_input=data.get("usage", {}).get("prompt_tokens", 0),
        tokens_output=data.get("usage", {}).get("completion_tokens", 0),
        finish_reason=finish_reason,
    )
