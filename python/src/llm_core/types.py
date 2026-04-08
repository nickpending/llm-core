"""Type definitions for llm-core.

Dataclass equivalents of the TypeScript interfaces in lib/types.ts.
Snake_case field names per Python convention.
"""

from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Token counts for a completion."""

    input: int
    output: int


@dataclass
class CompleteOptions:
    """Options for a completion request."""

    prompt: str
    service: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    json: bool = False


@dataclass
class CompleteResult:
    """Result from a completion request.

    Fields are the canonical set verified by INV-002.
    """

    text: str
    model: str
    provider: str
    tokens: TokenUsage
    finish_reason: str  # "stop" | "max_tokens" | "error"
    duration_ms: int
    cost: float | None


@dataclass
class ServiceConfig:
    """Configuration for a named service from services.toml."""

    adapter: str
    base_url: str
    key: str | None = None
    key_required: bool = True
    default_model: str | None = None
    app_title: str | None = None
    app_url: str | None = None


@dataclass
class AdapterRequest:
    """Request passed to a provider adapter."""

    base_url: str
    api_key: str | None
    model: str
    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    json: bool = False
    app_title: str | None = None
    app_url: str | None = None


@dataclass
class AdapterResponse:
    """Normalized response from a provider adapter."""

    text: str
    model: str
    tokens_input: int
    tokens_output: int
    finish_reason: str  # "stop" | "max_tokens" | "error"
