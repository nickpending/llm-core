"""Provider adapter registry.

Maps adapter names to their provider modules.
Used by the core orchestration layer to route requests.
"""

from __future__ import annotations

from typing import Any

from . import anthropic, ollama, openai

_ADAPTERS: dict[str, Any] = {
    "anthropic": anthropic,
    "openai": openai,
    "ollama": ollama,
}


def get_adapter(name: str) -> Any:
    """Look up a provider adapter module by name.

    Returns a module with a complete(AdapterRequest) -> AdapterResponse function.

    Raises:
        ValueError: If the adapter name is not registered.
    """
    adapter = _ADAPTERS.get(name)
    if not adapter:
        available = ", ".join(_ADAPTERS.keys())
        raise ValueError(f'Unknown adapter: "{name}". Available: {available}')
    return adapter
