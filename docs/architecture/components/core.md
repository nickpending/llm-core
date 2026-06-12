---
type: architecture
subtype: component-detail
project: "llm-core"
component: "Core Orchestration"
status: active
created: "2026-04-08"
updated: "2026-04-08"
tags: [architecture, components]
---

# Core Orchestration

**Purpose:** The `complete()` entry point that wires together the full request lifecycle: service resolution, key loading, adapter dispatch, retry wrapping, cost estimation, and response normalization.

## Current State

**TypeScript** (`typescript/lib/core.ts`):
- `complete(options)` — Async. Resolves service, loads key, dispatches to adapter via `withRetry()`, estimates cost, returns `CompleteResult`.
- `healthCheck(serviceName?)` — Delegates to provider-specific health check functions in adapter modules.
- Adapter lookup via `getAdapter()` from `providers/index.ts`.

**Python** (`python/src/llm_core/core.py`):
- `complete(prompt, *, service, model, ...)` — Synchronous. Keyword-only args after prompt. Same pipeline as TypeScript.
- `health_check(service_name?)` — Calls adapter-specific `health_check_config()` to get URL + headers, then makes the HTTP request itself.
- Adapter lookup via `get_adapter()` from `providers/__init__.py`.

Both expose the same public surface: `complete()` and `healthCheck()`/`health_check()`.

## Design

The core module is deliberately thin — it orchestrates but doesn't implement. Each concern (service resolution, key loading, retry, pricing) lives in its own module. Core's job is sequencing.

Key design choice: Core doesn't catch adapter errors except through the retry wrapper. If an error is non-transient, it propagates directly to the caller. The retry module owns the classification logic.

Health check is intentionally NOT wrapped by retry — transient connection errors surface immediately rather than retrying, since health checks are diagnostic tools.

## Connections

- **Inputs:** `CompleteOptions` from caller
- **Calls:** Service Configuration (`resolveService`), API Key Management (`loadApiKey`), Provider Adapters (via registry), Retry Logic (`withRetry`), Pricing (`estimateCost`)
- **Returns:** `CompleteResult` to caller
