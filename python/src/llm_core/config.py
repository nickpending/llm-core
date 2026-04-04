"""API key loading via apiconf.

Wraps apiconf's get_key() with service-aware error handling.
Translates apiconf errors into actionable ConfigError messages.
"""

from __future__ import annotations

from apiconf import ConfigNotFoundError, KeyNotFoundError, get_key

from .exceptions import ConfigError
from .types import ServiceConfig


def load_api_key(service: ServiceConfig) -> str | None:
    """Load the API key for a service using apiconf.

    Returns None if the service does not require a key (e.g., ollama).
    Raises ConfigError with actionable messages for missing keys or config.
    """
    if service.key_required is False:
        return None

    if not service.key:
        raise ConfigError("Service requires an API key but no 'key' field configured.")

    try:
        return get_key(service.key)
    except KeyNotFoundError as e:
        raise ConfigError(
            f"API key '{service.key}' not found in apiconf. Add it to ~/.config/apiconf/config.toml"
        ) from e
    except ConfigNotFoundError as e:
        raise ConfigError("apiconf config not found. Create ~/.config/apiconf/config.toml") from e
