# llm-core

<div align="center">

**Lightweight LLM transport — raw HTTP, no SDKs, no bloat**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

Service-based routing to provider APIs with normalized response envelope. Configure named services in `services.toml`, call `complete()` from anywhere. No provider SDKs — just httpx making HTTP calls.

## 📦 Packages

| Package | Language | Status |
|---------|----------|--------|
| [python/](python/) | Python 3.11+ | ✅ Stable |
| TypeScript | Bun | 📍 Lives in [llmcli-tools](https://github.com/nickpending/llmcli-tools) (merging later) |

## ✨ Features

- 🔀 **Service Routing** — Named services in `services.toml`, resolved at call time
- 🔌 **Provider Adapters** — OpenAI, Anthropic, Ollama via raw httpx
- 📦 **Normalized Responses** — `CompleteResult` with text, tokens, cost, timing regardless of provider
- 🔑 **Key Management** — API keys via [apiconf](https://github.com/nickpending/apiconf), separate from routing config
- 🔄 **Smart Retry** — Transient errors (429, 5xx, network) retried; auth errors fail fast
- 💰 **Cost Estimation** — Per-call cost tracking with `update_pricing()` to refresh rates
- 🏥 **Health Checks** — Per-adapter lightweight connectivity validation (no token spend)
- 🛡️ **No SDKs** — Zero dependency on openai, anthropic, or ollama packages

## 🎬 Quick Start

```python
from llm_core import complete, health_check

# Verify connectivity
health_check(service="openai")

# Single-turn completion
result = complete(
    prompt="Summarize this in one sentence",
    system_prompt="You are a concise assistant",
    service="openai",
    temperature=0.3,
)

print(result.text)
print(f"Tokens: {result.tokens.input}in / {result.tokens.output}out")
print(f"Cost: ${result.cost}")
print(f"Duration: {result.duration_ms}ms")
```

See [python/README.md](python/README.md) for full API docs, configuration, and development setup.

## 🔧 Configuration

Services defined in `~/.config/llm-core/services.toml`, API keys managed by [apiconf](https://github.com/nickpending/apiconf) at `~/.config/apiconf/config.toml`. Created automatically on first use.

## License

MIT
