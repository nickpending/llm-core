# @voidwire/llm-core (TypeScript)

Lightweight LLM transport layer for Bun. Service-based routing to provider APIs with normalized response envelope. No provider SDKs — just `fetch()` making HTTP calls.

## Install

```bash
bun add @voidwire/llm-core
```

## Usage

```typescript
import { complete, healthCheck } from "@voidwire/llm-core";

// Verify connectivity
await healthCheck("openai");

// Single-turn completion
const result = await complete({
  prompt: "Summarize this in one sentence",
  systemPrompt: "You are a concise assistant",
  service: "openai",
  temperature: 0.3,
});

console.log(result.text);
console.log(`Tokens: ${result.tokens.input}in / ${result.tokens.output}out`);
console.log(`Cost: $${result.cost}`);
console.log(`Duration: ${result.durationMs}ms`);
```

### Embeddings

```typescript
import { embed } from "@voidwire/llm-core";

const result = await embed({
  text: "Some text to embed",
  prefix: "search_document",
  service: "embed",
});

console.log(result.embedding.length); // vector dimensions
```

### CLI

```bash
echo "Hello" | llm-core --service anthropic
llm-core --list-services
llm-core --service openai --model gpt-4.1-mini "Explain REST APIs"
```

## API

### Functions

| Function | Description |
|----------|-------------|
| `complete(options)` | Send a completion request, get a normalized `CompleteResult` |
| `healthCheck(service)` | Lightweight connectivity check (no token spend) |
| `embed(options)` | Get embeddings from an embed server |
| `extractJson(text)` | Extract JSON from model output (handles markdown fences) |
| `isTruncated(result)` | Check if a completion was cut off by `max_tokens` |
| `loadServices()` | Load all services from `services.toml` |
| `listServices()` | Get service names as a string array |
| `resolveService(name)` | Resolve a named service to its config |

### Types

| Type | Description |
|------|-------------|
| `CompleteOptions` | Input: prompt, service, model, systemPrompt, temperature, maxTokens, json |
| `CompleteResult` | Output: text, model, provider, tokens, finishReason, durationMs, cost |
| `EmbedOptions` | Input: text, prefix, service |
| `EmbedResult` | Output: embedding, dims, durationMs |
| `ServiceConfig` | Service definition: adapter, key, base_url, default_model |
| `ServiceMap` | Full config: default_service + services record |

## Configuration

Services: `~/.config/llm-core/services.toml` (created on first use)

```toml
default_service = "anthropic"

[services.anthropic]
adapter = "anthropic"
key = "anthropic"
base_url = "https://api.anthropic.com/v1"
default_model = "claude-sonnet-4-20250514"

[services.openai]
adapter = "openai"
key = "openai"
base_url = "https://api.openai.com/v1"
default_model = "gpt-4.1-mini"

[services.ollama]
adapter = "ollama"
base_url = "http://localhost:11434"
key_required = false
default_model = "llama3"
```

API keys: managed by [@voidwire/apiconf](https://github.com/nickpending/apiconf) at `~/.config/apiconf/config.toml`.

## Provider Adapters

All adapters use native `fetch()` — zero SDK dependencies.

| Adapter | Endpoint | JSON mode |
|---------|----------|-----------|
| `anthropic` | `/v1/messages` | Ignored silently |
| `openai` | `/v1/chat/completions` | `response_format: { type: "json_object" }` |
| `ollama` | `/api/generate` | Not supported |

## Development

```bash
cd typescript
bun install
bun test                  # run tests
bun run typecheck         # tsc --noEmit
bunx biome check .        # lint + format
bunx biome check --write . # auto-fix
```

## License

MIT
