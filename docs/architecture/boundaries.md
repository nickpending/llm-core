---
type: architecture
subtype: boundaries
project: "llm-core"
status: active
created: "2026-04-08"
updated: "2026-06-12"
tags: [architecture, boundaries]
---

# Boundaries

Interface contracts between components and external systems.

## Consumer API

**Between:** llm-core ↔ Consuming applications
**Contract:** TypeScript exports `complete()`, `healthCheck()`, `embed()`, `extractJson()`, `isTruncated()`, `updatePricing()`, `listServices()`, `loadServices()`, `resolveService()` from `@voidwire/llm-core` (package entry `typescript/index.ts`, also exported types `CompleteOptions`, `CompleteResult`, `EmbedOptions`, `EmbedResult`, `ServiceConfig`, `ServiceMap`). Python exports the same surface (snake_case) minus `embed()` from `llm_core` (`python/src/llm_core/__init__.py`). Both return `CompleteResult` with fields: text, model, provider, tokens (input/output), finishReason, durationMs, cost.
**Constraints:** Response envelope shape is the public contract. Adding fields is safe; removing or renaming fields is breaking. `finishReason` is always one of `{stop, max_tokens, error}`.

## Services Configuration

**Between:** llm-core ↔ Filesystem (`~/.config/llm-core/services.toml`)
**Contract:** TOML file with `[settings]` section (default_service) and `[services.<name>]` sections. Each service requires `adapter` (anthropic|openai|ollama) and `base_url`. Optional: `key`, `key_required` (default true), `default_model`, `app_title`, `app_url`.
**Constraints:** Auto-generated on first run with Anthropic, OpenAI, Ollama defaults. Validation enforces: default_service must reference an existing service, all services must have adapter and base_url. Python supports `LLM_CORE_CONFIG_DIR` env var override for test isolation.

## Adapter Interface

**Between:** Core Orchestration ↔ Provider Adapters
**Contract:** Adapters implement `complete(request: AdapterRequest) -> AdapterResponse`. Request contains: model, prompt, systemPrompt, temperature, maxTokens, json, apiKey, baseUrl. Response contains: text, model, tokens (input/output), finishReason. TypeScript adapters also implement `healthCheck(baseUrl, apiKey?)`. Python adapters implement `health_check_config(service) -> (url, headers)`.
**Constraints:** Adapters must normalize `finishReason` to `{stop, max_tokens, error}`. New adapters must be registered in the adapter registry (`providers/index`). HTTP adapters throw with status codes accessible for retry classification. **Subprocess adapters (claude-cli) are exempt from the HTTP-status constraint** — they throw plain `Error`s with subprocess exit codes, so withRetry can't classify them by status; an adapter may also satisfy the contract without consuming `apiKey`/`baseUrl` (claude-cli ignores both and authenticates via the local CLI's OAuth).

## apiconf Integration

**Between:** llm-core ↔ apiconf (`@voidwire/apiconf` / `apiconf`)
**Contract:** Calls `getKey(keyName)` / `get_key(key_name)` where key name comes from the service's `key` field. Returns the API key string. Throws `KeyNotFoundError` or `ConfigNotFoundError` on failure.
**Constraints:** apiconf manages its own config at `~/.config/apiconf/config.toml`. llm-core translates apiconf exceptions into its own error types (`ConfigError` in Python, descriptive error messages in TypeScript). Services with `key_required: false` skip apiconf entirely.

## CLI Interface (TypeScript only)

**Between:** llm-core CLI ↔ Shell
**Contract:** `llm-core "prompt" [--service name] [--model id] [--system text] [--temperature n] [--max-tokens n] [--json] [--list-services]`. JSON output to stdout, diagnostics to stderr. Exit 0 = success, 1 = API/runtime error, 2 = client error.
**Constraints:** Output format is the JSON-serialized `CompleteResult`. Changing field names or structure is a breaking change for scripts consuming the output.
