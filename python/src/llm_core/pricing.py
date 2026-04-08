"""Cost estimation from ~/.config/llm-core/pricing.toml.

Loads pricing rates ($/1M tokens) and estimates cost from token counts.
Pricing is best-effort: missing file or unknown model returns None.

pricing.toml format:
    [models."claude-3-5-sonnet-20241022"]
    input = 3.00
    output = 15.00

Call update_pricing() to fetch litellm data and write pricing.toml.
"""

from __future__ import annotations

import math
import os
import tomllib
from pathlib import Path

import httpx

_cache: dict[str, dict[str, float]] | None = None

_PRICING_FILENAME = "pricing.toml"
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
    """Return the path to the local pricing TOML file."""
    return _get_config_dir() / _PRICING_FILENAME


def _load_pricing() -> dict[str, dict[str, float]]:
    """Load pricing data from local TOML file.

    Caches the result for subsequent calls. Returns empty dict if file
    missing or corrupt. No auto-fetch — pricing is best-effort.
    """
    global _cache

    if _cache is not None:
        return _cache

    pricing_path = _get_pricing_path()

    if not pricing_path.exists():
        _cache = {}
        return _cache

    try:
        with pricing_path.open("rb") as f:
            data = tomllib.load(f)
        models = data.get("models", {})
        _cache = models if isinstance(models, dict) else {}
        return _cache
    except (OSError, tomllib.TOMLDecodeError):
        _cache = {}
        return _cache


def estimate_cost(model: str, tokens_input: int, tokens_output: int) -> float | None:
    """Estimate cost in USD from token counts and model name.

    Returns None if pricing data unavailable for the model.
    Rates in pricing.toml are per 1M tokens.
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

    return (tokens_input / 1_000_000) * input_rate + (tokens_output / 1_000_000) * output_rate


def update_pricing() -> int:
    """Fetch pricing data from litellm and write pricing.toml.

    Converts per-token rates to per-1M-token rates and writes TOML
    with the same format TypeScript uses. Invalidates cache.
    Returns count of models written.
    """
    global _cache

    with httpx.Client(timeout=30.0) as client:
        response = client.get(_PRICING_URL)
        response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        msg = "Pricing data is not a JSON object"
        raise ValueError(msg)

    lines: list[str] = []
    count = 0

    for model, entry in data.items():
        if not entry or not isinstance(entry, dict):
            continue

        raw_input = entry.get("input_cost_per_token")
        raw_output = entry.get("output_cost_per_token")

        try:
            input_rate = float(raw_input)
            output_rate = float(raw_output)
        except (TypeError, ValueError):
            continue

        if math.isnan(input_rate) or math.isnan(output_rate) or input_rate <= 0 or output_rate <= 0:
            continue

        input_per_1m = round(input_rate * 1_000_000, 6)
        output_per_1m = round(output_rate * 1_000_000, 6)

        safe_name = model.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'[models."{safe_name}"]')
        lines.append(f"input = {input_per_1m}")
        lines.append(f"output = {output_per_1m}")
        lines.append("")
        count += 1

    pricing_path = _get_pricing_path()
    pricing_path.parent.mkdir(parents=True, exist_ok=True)
    pricing_path.write_text("\n".join(lines), encoding="utf-8")

    _cache = None

    return count
