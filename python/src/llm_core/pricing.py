"""Cost estimation from litellm model pricing data.

Loads pricing rates (cost per token) from a local JSON file sourced from
litellm's model_prices_and_context_window.json. Estimates cost from token counts.
Pricing is best-effort: unknown model returns None.

On first use, fetches pricing data from GitHub if no local copy exists.
Call update_pricing() to refresh with the latest data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

_cache: dict[str, dict[str, object]] | None = None

_PRICING_FILENAME = "model_prices.json"
_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)


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
    """Return the path to the local pricing JSON file."""
    return _get_config_dir() / _PRICING_FILENAME


def _fetch_pricing(pricing_path: Path) -> dict[str, dict[str, object]]:
    """Fetch pricing data from GitHub and save locally.

    Returns the parsed data on success, empty dict on any failure.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(_PRICING_URL)
            response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict):
            return {}

        pricing_path.parent.mkdir(parents=True, exist_ok=True)
        pricing_path.write_text(json.dumps(data))
        return data
    except (httpx.HTTPError, OSError, ValueError):
        return {}


def _load_pricing() -> dict[str, dict[str, object]]:
    """Load pricing data from local JSON, fetching from GitHub if missing.

    Caches the result for subsequent calls. Returns empty dict on any error.
    """
    global _cache

    if _cache is not None:
        return _cache

    pricing_path = _get_pricing_path()

    # Try reading local file first
    try:
        if pricing_path.exists():
            raw = pricing_path.read_text()
            data = json.loads(raw)
            _cache = data if isinstance(data, dict) else {}
            return _cache
    except (OSError, json.JSONDecodeError):
        pass

    # No local file or read failed — fetch from GitHub
    data = _fetch_pricing(pricing_path)
    _cache = data if isinstance(data, dict) else {}
    return _cache


def estimate_cost(model: str, tokens_input: int, tokens_output: int) -> float | None:
    """Estimate cost in USD from token counts and model name.

    Returns None if pricing data unavailable for the model.
    Uses per-token rates from litellm pricing data.
    """
    pricing = _load_pricing()
    entry = pricing.get(model)

    if not entry:
        return None

    try:
        input_rate = float(entry.get("input_cost_per_token", 0))
        output_rate = float(entry.get("output_cost_per_token", 0))
    except (TypeError, ValueError):
        return None

    if input_rate == 0 and output_rate == 0:
        return None

    return tokens_input * input_rate + tokens_output * output_rate


def update_pricing() -> None:
    """Fetch the latest pricing data from litellm's GitHub repository.

    Overwrites the local pricing file and invalidates the cache.
    Raises on network or write errors.
    """
    global _cache

    with httpx.Client(timeout=30.0) as client:
        response = client.get(_PRICING_URL)
        response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        msg = "Pricing data is not a JSON object"
        raise ValueError(msg)

    pricing_path = _get_pricing_path()
    pricing_path.parent.mkdir(parents=True, exist_ok=True)
    pricing_path.write_text(json.dumps(data))

    _cache = None
