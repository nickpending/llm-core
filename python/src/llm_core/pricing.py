"""Cost estimation from static pricing.toml configuration.

Loads pricing rates (cost per 1M tokens) from ~/.config/llm-core/pricing.toml.
Estimates cost from token counts. Pricing is best-effort: unknown model returns None.

The pricing.toml format uses per-1M-token rates:

    [models."gpt-4.1-mini"]
    input = 0.40    # USD per 1M tokens
    output = 1.60   # USD per 1M tokens
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

_cache: dict[str, dict[str, float]] | None = None

_PRICING_FILENAME = "pricing.toml"


def _get_config_dir() -> Path:
    """Compute config directory from environment."""
    override = os.environ.get("LLM_CORE_CONFIG_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "llm-core"
    return Path.home() / ".config" / "llm-core"


def _get_pricing_path() -> Path:
    """Return the path to the local pricing TOML file."""
    return _get_config_dir() / _PRICING_FILENAME


def _load_pricing() -> dict[str, dict[str, float]]:
    """Load pricing data from pricing.toml.

    Caches the result for subsequent calls. Returns empty dict if file
    missing or malformed. Never makes network calls.
    """
    global _cache

    if _cache is not None:
        return _cache

    pricing_path = _get_pricing_path()

    try:
        if pricing_path.exists():
            with pricing_path.open("rb") as f:
                data = tomllib.load(f)
            models = data.get("models", {})
            _cache = models if isinstance(models, dict) else {}
            return _cache
    except (OSError, tomllib.TOMLDecodeError):
        pass

    _cache = {}
    return _cache


def estimate_cost(model: str, tokens_input: int, tokens_output: int) -> float | None:
    """Estimate cost in USD from token counts and model name.

    Returns None if pricing data unavailable for the model.
    Rates in pricing.toml are per-1M-tokens; converted to per-token here.
    """
    pricing = _load_pricing()
    entry = pricing.get(model)

    if not entry:
        return None

    try:
        input_rate = float(entry.get("input", 0))
        output_rate = float(entry.get("output", 0))
    except (TypeError, ValueError):
        return None

    if input_rate == 0 and output_rate == 0:
        return None

    # Convert per-1M-token rates to per-token
    return tokens_input * (input_rate / 1_000_000) + tokens_output * (output_rate / 1_000_000)
