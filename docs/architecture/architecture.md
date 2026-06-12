---
type: architecture
subtype: overview
project: "llm-core"
status: active
created: "2026-04-08"
updated: "2026-06-09"
tags: [architecture]
---

# llm-core Architecture

A dual-language (TypeScript + Python) LLM abstraction library that normalizes Anthropic, OpenAI, and Ollama APIs into a unified interface. Routes requests through named service definitions, handles retries with transient error classification, tracks costs, and returns a normalized response envelope. No provider SDKs — raw HTTP only.

## Principles

- **Raw HTTP over SDKs** — Provider adapters use native HTTP clients (Bun fetch, httpx) instead of official SDKs. Reduces dependency surface, keeps the library lightweight. Exception: the `claude-cli` adapter (TypeScript only) integrates via a `claude --print` subprocess rather than HTTP — required because Claude Code subscription auth is OAuth-bound to the local CLI, not the HTTP API.
- **Service-based routing** — Consumers define named services in TOML. Routing is resolved at runtime, not compile time. No hardcoded provider defaults.
- **Normalized response envelope** — Every provider returns the same `CompleteResult` shape. Callers never deal with provider-specific response formats.
- **Dual-language parity** — TypeScript and Python implementations share the same architecture, service model, adapter pattern, and retry strategy. Language-idiomatic where appropriate.
- **Zero runtime dependencies** — TypeScript has no runtime deps beyond apiconf. Python depends only on httpx and apiconf.
- **Best-effort enrichment** — Pricing returns null for unknown models instead of failing. Health checks validate connectivity without spending tokens.

## Components

| Component | Purpose | Detail |
|-----------|---------|--------|
| Core Orchestration | Wires service resolution, adapter dispatch, retry, and cost estimation | [components/core.md](components/core.md) |
| Provider Adapters | Translate between normalized interface and provider-specific HTTP APIs | [components/adapters.md](components/adapters.md) |
| Service Configuration | Load and validate named service definitions from TOML | [components/services.md](components/services.md) |
| API Key Management | Load credentials via apiconf with service-aware error handling | — |
| Retry Logic | Classify transient vs permanent errors, exponential backoff | — |
| Pricing | Estimate cost per completion from static rate data | — |
| Embedding | Direct HTTP embedding calls (TypeScript only) | — |
| Helpers | extractJson, isTruncated utilities | — |

## Key Decisions

See [decisions.md](decisions.md) for the full decision log.

## Boundaries

See [boundaries.md](boundaries.md) for interface contracts.
