"""Service resolution from services.toml.

Loads service configuration, generates defaults on first run,
and resolves named services to their config.

Config dir resolution order:
1. LLM_CORE_CONFIG_DIR env var (for test isolation)
2. XDG_CONFIG_HOME env var / "llm-core"
3. ~/.config/llm-core/

The config path is computed inside load_services() on each call (before cache check)
so that LLM_CORE_CONFIG_DIR overrides work for test isolation.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import TypedDict

from .exceptions import ConfigError
from .types import ServiceConfig

DEFAULT_SERVICES_TOML = """\
default_service = "anthropic"

[services.anthropic]
adapter = "anthropic"
key = "anthropic"
base_url = "https://api.anthropic.com/v1"

[services.openai]
adapter = "openai"
key = "openai"
base_url = "https://api.openai.com/v1"

[services.ollama]
adapter = "ollama"
base_url = "http://localhost:11434"
key_required = false
"""


class ServiceMap(TypedDict):
    default_service: str
    services: dict[str, ServiceConfig]


_cache: ServiceMap | None = None
_cached_config_dir: str | None = None


def _get_config_dir() -> Path:
    """Compute config directory from environment."""
    override = os.environ.get("LLM_CORE_CONFIG_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "llm-core"
    return Path.home() / ".config" / "llm-core"


def load_services() -> ServiceMap:
    """Load and parse services.toml. Generates default config on first run.

    Caches the result for subsequent calls. Cache is invalidated if
    LLM_CORE_CONFIG_DIR changes between calls.
    """
    global _cache, _cached_config_dir

    config_dir = _get_config_dir()
    config_dir_str = str(config_dir)

    # Invalidate cache if config dir changed (test isolation support)
    if _cache is not None and _cached_config_dir == config_dir_str:
        return _cache

    services_path = config_dir / "services.toml"

    # Generate default config if missing
    if not services_path.exists():
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            services_path.write_text(DEFAULT_SERVICES_TOML)
        except OSError as e:
            raise ConfigError(
                f"Failed to create default services.toml at {services_path}: {e}"
            ) from e

    # Read and parse
    try:
        raw = services_path.read_text()
    except OSError as e:
        raise ConfigError(f"Failed to read {services_path}: {e}") from e

    try:
        parsed = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Failed to parse {services_path}: {e}") from e

    # Validate required fields
    if not isinstance(parsed.get("default_service"), str):
        raise ConfigError(
            f'Invalid config: missing or non-string "default_service" in {services_path}'
        )

    if not isinstance(parsed.get("services"), dict):
        raise ConfigError(
            f"Invalid config: missing or invalid [services] section in {services_path}"
        )

    services_raw = parsed["services"]

    # Validate each service entry
    for name, entry in services_raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(
                f'Invalid config: service "{name}" must be a table in {services_path}'
            )
        if not isinstance(entry.get("adapter"), str):
            raise ConfigError(
                f'Invalid config: service "{name}" missing "adapter" field in {services_path}'
            )
        if not isinstance(entry.get("base_url"), str):
            raise ConfigError(
                f'Invalid config: service "{name}" missing "base_url" field in {services_path}'
            )

    # Validate default_service references a known service
    default_service = parsed["default_service"]
    if default_service not in services_raw:
        available = ", ".join(services_raw.keys())
        raise ConfigError(
            f'Invalid config: default_service "{default_service}" not found in [services]. '
            f"Available: [{available}]"
        )

    # Build ServiceConfig objects
    services: dict[str, ServiceConfig] = {}
    for name, entry in services_raw.items():
        services[name] = ServiceConfig(
            adapter=entry["adapter"],
            base_url=entry["base_url"],
            key=entry.get("key"),
            key_required=entry.get("key_required", True),
            default_model=entry.get("default_model"),
        )

    result: ServiceMap = {
        "default_service": default_service,
        "services": services,
    }

    _cache = result
    _cached_config_dir = config_dir_str
    return result


def resolve_service(name: str | None = None) -> ServiceConfig:
    """Resolve a service by name. If name is None, returns the default service."""
    service_map = load_services()
    service_name = name if name is not None else service_map["default_service"]

    service = service_map["services"].get(service_name)
    if not service:
        available = ", ".join(service_map["services"].keys())
        raise ConfigError(f'Unknown service: "{service_name}". Available: [{available}]')

    return service


def list_services() -> list[str]:
    """List all configured service names."""
    service_map = load_services()
    return list(service_map["services"].keys())
