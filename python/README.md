# llm-core

<div align="center">

**Lightweight LLM transport — raw HTTP, no SDKs, no bloat**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

**llm-core** is a shared LLM completion library that routes requests to provider APIs through named services. Configure once in `services.toml`, call `complete()` from anywhere. No provider SDKs — just httpx making HTTP calls.

## ✨ Features

- 🔀 **Service Routing** — Named services in `services.toml`, resolved at call time
- 🔌 **Provider Adapters** — OpenAI, Anthropic, Ollama via raw httpx
- 📦 **Normalized Responses** — `CompleteResult` with text, tokens, cost, timing regardless of provider
- 🔑 **Key Management** — API keys via [apiconf](https://github.com/nickpending/apiconf), separate from routing config
- 🔄 **Smart Retry** — Transient errors (429, 5xx, network) retried; auth errors fail fast
- 💰 **Cost Estimation** — Per-call cost from litellm pricing data; `update_pricing()` to refresh
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

## ⚙️ Configuration

### services.toml

`~/.config/llm-core/services.toml` — created automatically on first use:

```toml
default_service = "openai"

[services.openai]
adapter = "openai"
key = "openai"
base_url = "https://api.openai.com/v1"
default_model = "gpt-4.1-mini"

[services.anthropic]
adapter = "anthropic"
key = "anthropic"
base_url = "https://api.anthropic.com/v1"
default_model = "claude-sonnet-4-5"

[services.ollama]
adapter = "ollama"
base_url = "http://localhost:11434"
key_required = false
```

### API Keys

Managed by [apiconf](https://github.com/nickpending/apiconf) at `~/.config/apiconf/config.toml`:

```toml
[keys.openai]
value = "sk-..."

[keys.anthropic]
value = "sk-ant-..."
```

## 📖 API

| Function | Description |
|----------|-------------|
| `complete(prompt, service, ...)` | Single-turn LLM completion |
| `health_check(service)` | Validate provider connectivity |
| `estimate_cost(model, in, out)` | Estimate USD cost from token counts |
| `update_pricing()` | Refresh pricing data from litellm |
| `resolve_service(name)` | Resolve service config from services.toml |
| `list_services()` | List available service names |
| `extract_json(text)` | Extract JSON from markdown code blocks |
| `is_truncated(result)` | Check if response hit max_tokens |

### CompleteResult

```python
@dataclass
class CompleteResult:
    text: str              # Response content
    model: str             # Model that responded
    provider: str          # Adapter name (openai, anthropic, ollama)
    tokens: TokenUsage     # .input, .output
    finish_reason: str     # "stop", "max_tokens", "error"
    duration_ms: int       # Wall-clock milliseconds
    cost: float | None     # Estimated USD cost (None if unknown model)
```

## 🔧 Development

```bash
cd python
uv sync
uv run pytest tests/ -v
uv run ruff check src/
uv run ruff format --check src/
```

44 tests covering services, providers, retry, pricing, types, and core orchestration.

## 📋 Requirements

- Python 3.11+ (uses `tomllib` from stdlib)
- [apiconf](https://github.com/nickpending/apiconf) for API key resolution
- [httpx](https://www.python-httpx.org/) for HTTP calls

## License

MIT
