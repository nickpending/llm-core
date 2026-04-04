"""Exception classes for llm-core."""


class LLMCoreError(Exception):
    """Base exception for llm-core errors."""


class ProviderError(LLMCoreError):
    """Raised by provider adapters on HTTP errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ConfigError(LLMCoreError):
    """Raised for configuration errors (missing files, bad keys)."""
