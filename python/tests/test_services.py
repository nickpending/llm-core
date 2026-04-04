"""Tests for services.py — service loading and resolution.

Covers:
- SC-4: services.toml bootstrap on first run
- INV-003: bootstrap creates valid defaults
- resolve_service() by name and default
- ConfigError on unknown service
- LLM_CORE_CONFIG_DIR env var isolation
"""

import tempfile
from pathlib import Path

import pytest

from llm_core.exceptions import ConfigError
from llm_core.services import list_services, load_services, resolve_service
from llm_core.types import ServiceConfig


def _isolated_config_dir(tmp_path: Path) -> str:
    """Return a temp dir path string for LLM_CORE_CONFIG_DIR."""
    return str(tmp_path)


def test_bootstrap_creates_services_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-4: No services.toml exists → resolve_service() creates it with defaults."""
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    services_file = tmp_path / "services.toml"
    assert not services_file.exists(), "Precondition: file must not exist"

    result = resolve_service()

    assert services_file.exists(), "services.toml must be created on first run"
    assert isinstance(result, ServiceConfig)
    assert result.adapter, "Bootstrap result must have adapter"
    assert result.base_url, "Bootstrap result must have base_url"


def test_bootstrap_includes_anthropic_openai_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SC-4: Bootstrapped file contains anthropic, openai, ollama services."""
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    names = list_services()

    assert "anthropic" in names
    assert "openai" in names
    assert "ollama" in names


def test_resolve_named_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_service('openai') returns ServiceConfig with correct adapter and base_url."""
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    svc = resolve_service("openai")

    assert svc.adapter == "openai"
    assert svc.base_url == "https://api.openai.com/v1"


def test_resolve_unknown_service_raises_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve_service('nonexistent') raises ConfigError with service name in message."""
    monkeypatch.setenv("LLM_CORE_CONFIG_DIR", str(tmp_path))

    with pytest.raises(ConfigError, match="nonexistent"):
        resolve_service("nonexistent")


def test_llm_core_config_dir_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_CORE_CONFIG_DIR env var redirects config, enabling test isolation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        monkeypatch.setenv("LLM_CORE_CONFIG_DIR", tmp_dir)

        services_file = Path(tmp_dir) / "services.toml"
        assert not services_file.exists()

        load_services()

        assert services_file.exists(), "Config dir override must write services.toml to tmp_dir"
