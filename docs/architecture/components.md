---
type: architecture
subtype: components
project: "llm-core"
status: active
created: "2026-04-08"
updated: "2026-06-12"
tags: [architecture, components]
---

# Components

Registry of all system components. Each entry links to a detail doc when the component has enough substance to warrant one.

## Core Orchestration

**Purpose:** Wires together service resolution, adapter dispatch, retry wrapping, cost estimation, and response normalization into the `complete()` and `healthCheck()` entry points.
**Key files:** `typescript/lib/core.ts`, `python/src/llm_core/core.py`
**Connections:** Calls into Service Configuration, API Key Management, Provider Adapters, Retry Logic, and Pricing. Consumed by callers via public API.
**Detail:** [components/core.md](components/core.md)

## Provider Adapters

**Purpose:** Translate between the normalized `AdapterRequest`/`AdapterResponse` interface and provider-specific backends (Anthropic Messages, OpenAI Chat Completions, Ollama Generate over HTTP; claude-cli over subprocess).
**Key files:** `typescript/lib/providers/`, `python/src/llm_core/providers/`
**Connections:** Called by Core Orchestration. Each adapter is independent — no cross-adapter dependencies. Registry in `providers/index.ts` / `providers/__init__.py` maps adapter names to modules. **Asymmetry:** TypeScript has 4 adapters (anthropic, openai, ollama, claude-cli), Python has 3 (no claude-cli). `claude-cli` is a subprocess adapter — see detail doc.
**Detail:** [components/adapters.md](components/adapters.md)

## Service Configuration

**Purpose:** Load, validate, and cache named service definitions from `~/.config/llm-core/services.toml`. Auto-generates defaults on first run.
**Key files:** `typescript/lib/services.ts`, `python/src/llm_core/services.py`
**Connections:** Called by Core Orchestration for service resolution. Reads TOML config from disk. Exposes `listServices()`, `loadServices()`, `resolveService()` to consumers.
**Detail:** [components/services.md](components/services.md)

## API Key Management

**Purpose:** Load API credentials via apiconf with service-aware error handling. Returns null for services that don't require keys.
**Key files:** `typescript/lib/config.ts`, `python/src/llm_core/config.py`
**Connections:** Called by Core Orchestration. Wraps `@voidwire/apiconf` / `apiconf` package. Translates apiconf exceptions into llm-core error types.

## Retry Logic

**Purpose:** Classify errors as transient or permanent, wrap adapter calls with exponential backoff (3 attempts, delays 1s/2s/4s).
**Key files:** `typescript/lib/retry.ts`, `python/src/llm_core/retry.py`
**Connections:** Called by Core Orchestration to wrap adapter dispatch. TypeScript classifies via status code parsing from error messages. Python uses typed `ProviderError.status_code`. Transient: 429, 5xx, network errors. Non-transient: 400, 401, 403, 404.

## Pricing

**Purpose:** Estimate USD cost per completion based on token usage and model pricing data.
**Key files:** `typescript/lib/pricing.ts`, `python/src/llm_core/pricing.py`
**Connections:** Called by Core Orchestration after adapter response. Both languages read/write `pricing.toml` with per-1M-token rates. Each has its own `updatePricing()` that fetches from litellm's GitHub and writes TOML (manual serialization — no TOML writer in either runtime). Both cache after first load.

## Embedding

**Purpose:** Direct HTTP calls to an embedding service for vector generation. TypeScript only.
**Key files:** `typescript/lib/embed.ts`
**Connections:** Resolves "embed" service from Service Configuration. POSTs to `/embed` endpoint. Not routed through the adapter pattern — direct HTTP with 5-second timeout.

## Helpers

**Purpose:** Utility functions for common post-processing: JSON extraction from LLM output, truncation detection.
**Key files:** `typescript/lib/helpers.ts`, `python/src/llm_core/helpers.py`
**Connections:** Standalone — no dependencies on other components. Consumed directly by callers.

## CLI (TypeScript only)

**Purpose:** Command-line entry point that parses flags, calls the public `complete()`/`listServices()` API, and emits the JSON-serialized `CompleteResult` to stdout.
**Key files:** `typescript/cli.ts` (registered as the `llm-core` bin in `typescript/package.json`)
**Connections:** Consumes the Consumer API. Flags: `--service`, `--model`, `--system`, `--temperature`, `--max-tokens`, `--json`, `--list-services`. Exit codes: 0 success, 1 API/runtime error, 2 client error. Diagnostics to stderr. No Python equivalent.
