`llm-core` is a dual-language (TS/Python) library providing a unified API for Anthropic, OpenAI, and Ollama LLMs. It routes requests via named services defined in `~/.config/llm-core/services.toml`, handles retries, tracks costs, and returns a normalized `CompleteResult` envelope. It uses raw HTTP for providers, except for the TypeScript `claude-cli` adapter which uses a subprocess.

Key data flows: `complete()`/`healthCheck()` -> Service Configuration (`resolveService()`) -> API Key Management (`getKey()`) -> Retry Logic (wraps) -> Provider Adapters (`complete()`/`healthCheck()`) -> Pricing (`estimate_cost()`). `embed()` (TS only) directly calls an "embed" service.

Contracts:
- **Consumer API:** `complete()`, `healthCheck()`, `embed()` (TS only), `extractJson()`, `isTruncated()`, `listServices()`, `loadServices()`, `resolveService()` returning `CompleteResult` (text, model, provider, tokens, finishReason, durationMs, cost). `finishReason` is `{stop, max_tokens, error}`.
- **Services Configuration:** `~/.config/llm-core/services.toml` defines `[settings]` (default_service) and `[services.<name>]` with `adapter`, `base_url`, optional `key`, `key_required`, `default_model`, `app_title`, `app_url`.
- **Adapter Interface:** `complete(AdapterRequest) -> AdapterResponse` (text, model, tokens, finishReason). TS adapters also `healthCheck(baseUrl, apiKey?)`. Python adapters `health_check_config(service) -> (url, headers)`. Adapters must normalize `finishReason`.
- **apiconf:** `getKey(keyName)`/`get_key(key_name)` for API keys; translates exceptions to `llm-core` errors.

Gotchas:
- The `claude-cli` adapter (TS only) is a subprocess, not HTTP. It ignores `apiKey`/`baseUrl`, authenticates via local OAuth, and throws plain `Error`s (not HTTP status codes) for retry classification.
- `pricing.toml` is manually serialized/deserialized; `updatePricing()` must be explicitly called to populate pricing data.
- `LLM_CORE_CONFIG_DIR` env var can override the config directory for Python.
- `app_title`/`app_url` in `services.toml` are only used by the OpenAI adapter for OpenRouter attribution.
- Python `ProviderError` uses a typed `status_code` for retry classification, unlike TypeScript's error message parsing.
- `embed()` is TypeScript-only and not routed through the adapter pattern.
