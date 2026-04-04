"""Ollama Generate API adapter.

Converts AdapterRequest to Ollama's /api/generate endpoint format
and normalizes the response into AdapterResponse.

Note: Uses /api/generate (prompt + system fields), NOT /api/chat
(messages array). The generate endpoint maps directly to
AdapterRequest's prompt/systemPrompt structure.

Reference: llmcli-tools/packages/llm-core/lib/providers/ollama.ts
"""

from __future__ import annotations

import httpx

from ..exceptions import ProviderError
from ..types import AdapterRequest, AdapterResponse


def complete(request: AdapterRequest) -> AdapterResponse:
    """Execute a completion request against the Ollama Generate API."""
    body: dict[str, object] = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False,
    }

    if request.system_prompt is not None:
        body["system"] = request.system_prompt

    options: dict[str, object] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    body["options"] = options

    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{request.base_url}/api/generate", headers=headers, json=body)

        if response.status_code != 200:
            raise ProviderError(
                f"Ollama API error ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

        data = response.json()

    # Validate response shape
    if not isinstance(data.get("response"), str):
        raise ProviderError(
            "Ollama API returned unexpected response shape: missing or non-string response field"
        )

    # Map done_reason to normalized finish_reason
    # done_reason may be absent or null — default to "stop"
    done_reason = data.get("done_reason")
    finish_reason = "max_tokens" if done_reason == "length" else "stop"

    return AdapterResponse(
        text=data["response"],
        model=data["model"],
        tokens_input=data.get("prompt_eval_count", 0),
        tokens_output=data.get("eval_count", 0),
        finish_reason=finish_reason,
    )
