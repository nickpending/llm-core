---
type: manifest
project: llm-core
generated: 2026-06-12
source: /Users/rudy/development/projects/llm-core/docs/architecture
reconciled_at: 9542b3065ab2ba845e279e4bb92907d3dc33e6b1
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
