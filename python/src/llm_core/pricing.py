"""Cost estimation from pricing.toml.

Loads pricing rates ($/1M tokens) and estimates cost from token counts.
Pricing is best-effort: missing file or unknown model returns None.

Config dir resolution uses the same LLM_CORE_CONFIG_DIR / XDG_CONFIG_HOME
/ ~/.config/llm-core/ precedence as services.py.

pricing.toml format:
    [models."gpt-4.1-mini"]
    input = 0.40
    output = 1.60
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULT_PRICING_TOML = """\
[models."gpt-4.1-mini"]
input = 0.40
output = 1.60

[models."gpt-4.1"]
input = 2.00
output = 8.00

[models."gpt-4o"]
input = 2.50
output = 10.00

[models."gpt-4o-mini"]
input = 0.15
output = 0.60

[models."gpt-5-mini"]
input = 1.25
output = 5.00

[models."o3-mini"]
input = 1.10
output = 4.40

[models."claude-sonnet-4-5"]
input = 3.00
output = 15.00

[models."claude-haiku-4-5"]
input = 0.80
output = 4.00

[models."claude-opus-4-5"]
input = 15.00
output = 75.00
"""

_cache: dict[str, dict[str, float]] | None = None
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


def _load_pricing() -> dict[str, dict[str, float]]:
    """Load pricing.toml if it exists, write default if missing.

    Pricing is best-effort: missing file or corrupt data returns empty dict.
    Default file write is also best-effort (never raises).
    Caches the result for subsequent calls.
    """
    global _cache, _cached_config_dir

    config_dir = _get_config_dir()
    config_dir_str = str(config_dir)

    # Return cache if config dir unchanged
    if _cache is not None and _cached_config_dir == config_dir_str:
        return _cache

    pricing_path = config_dir / "pricing.toml"

    # Write default pricing.toml if missing (best-effort, never raises)
    try:
        file_exists = pricing_path.exists()
    except OSError:
        # Can't even stat the path (permission denied) — return empty
        _cache = {}
        _cached_config_dir = config_dir_str
        return _cache

    if not file_exists:
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            pricing_path.write_text(DEFAULT_PRICING_TOML)
        except OSError:
            # Best-effort write failed — continue with empty pricing
            _cache = {}
            _cached_config_dir = config_dir_str
            return _cache

    # Read and parse
    try:
        raw = pricing_path.read_text()
        parsed = tomllib.loads(raw)
        models = parsed.get("models", {})
        _cache = models if isinstance(models, dict) else {}
    except (OSError, tomllib.TOMLDecodeError):
        _cache = {}

    _cached_config_dir = config_dir_str
    return _cache


def estimate_cost(model: str, tokens_input: int, tokens_output: int) -> float | None:
    """Estimate cost in USD from token counts and model name.

    Returns None if pricing data unavailable for the model.
    Rates in pricing.toml are per 1M tokens.
    """
    pricing = _load_pricing()
    rates = pricing.get(model)

    if not rates:
        return None

    input_cost = (tokens_input / 1_000_000) * rates.get("input", 0)
    output_cost = (tokens_output / 1_000_000) * rates.get("output", 0)

    return input_cost + output_cost
