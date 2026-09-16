---
type: manifest
project: llm-core
generated: 2026-09-16
source: /Users/rudy/development/projects/llm-core/docs/architecture
reconciled_at: 33ea78972cb946655285157fbc1558ee3f8bf196
---

# llm-core Manifest

## Components

- **Core Orchestration** — Service resolution, dispatch, retry, cost → components/core.md
- **Provider Adapters** — Anthropic/OpenAI/Ollama HTTP + claude-cli subprocess → components/adapters.md
- **Service Configuration** — Load/validate/cache services.toml → components/services.md
- **API Key Management** — Credentials via apiconf
- **Retry Logic** — Transient vs permanent, backoff
- **Pricing** — Cost from pricing.toml rates
- **Embedding** — HTTP embedding (TypeScript only)
- **Helpers** — JSON extraction, truncation detection
- **CLI** — Command-line entry (TypeScript only)

## Where to look

- Decisions: /Users/rudy/development/projects/llm-core/docs/architecture/decisions.md
- Contracts: /Users/rudy/development/projects/llm-core/docs/architecture/boundaries.md
