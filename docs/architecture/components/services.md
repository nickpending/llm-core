---
type: architecture
subtype: component-detail
project: "llm-core"
component: "Service Configuration"
status: active
created: "2026-04-08"
updated: "2026-04-08"
tags: [architecture, components]
---

# Service Configuration

**Purpose:** Load, validate, and cache named service definitions from TOML. Auto-generates defaults on first run.

## Current State

**Config path:** `~/.config/llm-core/services.toml`

**TOML structure:**
```toml
[settings]
default_service = "anthropic"

[services.anthropic]
adapter = "anthropic"
base_url = "https://api.anthropic.com/v1"
key = "anthropic"
default_model = "claude-sonnet-4-20250514"

[services.openai]
adapter = "openai"
base_url = "https://api.openai.com/v1"
key = "openai"
default_model = "gpt-4o"

[services.ollama]
adapter = "ollama"
base_url = "http://localhost:11434"
key_required = false
default_model = "llama3.2"
```

**TypeScript** (`typescript/lib/services.ts`):
- Parses with `Bun.TOML.parse()`
- Caches after first load, `_resetServicesCache()` for test isolation
- `resolveService()` resolves by name or falls back to default

**Python** (`python/src/llm_core/services.py`):
- Parses with `tomllib`
- Supports `LLM_CORE_CONFIG_DIR` env var and `XDG_CONFIG_HOME` for config path override
- Cache invalidation on config dir change
- Same resolution and validation logic

**Validation (both):**
- `default_service` must reference an existing service
- All services must have `adapter` and `base_url`

## Design

Services are the routing layer. Instead of hardcoding provider URLs and keys, consumers name a service and the library resolves everything else. This enables multiple configurations for the same provider (e.g., different OpenAI-compatible endpoints) without code changes.

Auto-generation on first run means zero-config for common setups. The defaults cover the three supported providers with sensible models.

Caching is important because `resolveService()` is called on every `complete()` call. File I/O happens once.

## Connections

- **Called by:** Core Orchestration (`resolveService()`), Consumer API (`listServices()`, `loadServices()`)
- **Reads:** `~/.config/llm-core/services.toml`
- **Returns:** `ServiceConfig` (adapter, base_url, key, key_required, default_model)
