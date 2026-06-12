---
type: architecture
subtype: component-detail
project: "llm-core"
component: "Provider Adapters"
status: active
created: "2026-04-08"
updated: "2026-06-09"
tags: [architecture, components]
---

# Provider Adapters

**Purpose:** Translate between the normalized `AdapterRequest`/`AdapterResponse` interface and provider-specific HTTP APIs.

## Current State

Three HTTP adapters in both TypeScript and Python, plus one subprocess adapter (claude-cli) in TypeScript only:

**Anthropic** (`anthropic.ts` / `anthropic.py`):
- Endpoint: `POST /messages` (Messages API)
- Auth: `x-api-key` header + `anthropic-version: 2023-06-01`
- Defaults `max_tokens` to 8192 if not specified
- Normalizes `stop_reason` → `finishReason` (`end_turn` → `stop`, `max_tokens` → `max_tokens`)
- Health check: `GET /models`

**OpenAI** (`openai.ts` / `openai.py`):
- Endpoint: `POST /chat/completions`
- Auth: `Authorization: Bearer {key}`
- Optional JSON mode: `response_format: { type: "json_object" }`
- Normalizes `finish_reason` (`length` → `max_tokens`)
- Health check: `GET /models`

**Ollama** (`ollama.ts` / `ollama.py`):
- Endpoint: `POST /api/generate` (prompt-based, NOT /api/chat)
- No auth required (`key_required: false`)
- Uses `options: { temperature, num_predict }` for parameters
- Normalizes `done_reason` (`length` → `max_tokens`)
- Health check: `GET /api/tags`

**claude-cli** (`claude-cli.ts`, TypeScript only):
- Backend: spawns `claude --print --output-format json` as a subprocess — NOT HTTP
- Auth: none passed by llm-core; relies on Claude Code's local OAuth credentials. Child env scrubs `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDECODE` so the CLI falls through to subscription auth
- Cost: zero per-call API spend — billed against the subscription cap
- Flags locked down: `--tools ''`, `--setting-sources ''`, `--exclude-dynamic-system-prompt-sections`, system prompt replaces rather than appends
- Ignores `maxTokens`, `temperature`, `json` (not exposed by `claude --print`)
- Resolved model read from the JSON envelope's `modelUsage` first key; real token counts from `usage`
- Normalizes `stop_reason` (`max_tokens` → `max_tokens`, else `stop`)
- Health check: `claude --version` (subprocess) — confirms the binary responds, does NOT validate subscription auth
- Errors: throws plain `Error` with subprocess exit codes / stderr; ENOENT → "claude CLI not found on PATH" remediation hint. No HTTP status code
- 120s subprocess timeout (`claude --print` can stall on auth or large prompts)
- Opt-in: not in the default services.toml; requires explicit `[services.claude-cli]`
- `parseClaudeJsonResponse()` extracted as a pure function for unit testing without spawn

**Adapter Registry:**
- TypeScript: `providers/index.ts` — lookup table mapping adapter name strings to modules (4 entries)
- Python: `providers/__init__.py` — same pattern (3 entries, no claude-cli)

## Design

Each adapter is self-contained. No cross-adapter dependencies, no shared base class. The contract is the `AdapterRequest`/`AdapterResponse` type — implement the shape and you're an adapter.

Raw HTTP instead of SDKs keeps the dependency surface at zero for providers. The adapter layer is thin enough (request mapping + response normalization) that SDK abstraction adds more weight than value.

Error handling: Adapters throw errors with HTTP status codes embedded. TypeScript encodes in message string as `(status)`. Python sets `ProviderError.status_code` attribute. This enables the retry module to classify without parsing provider-specific error formats.

Health checks differ between languages: TypeScript adapters implement `healthCheck()` directly. Python adapters implement `health_check_config()` that returns URL + headers, with core.py making the actual request.

**Subprocess exception (claude-cli):** The "raw HTTP, status codes embedded in errors" framing holds for anthropic/openai/ollama but not claude-cli. It's a subprocess adapter: no HTTP, no status code, health check via `claude --version` rather than a provider endpoint. It still satisfies the `AdapterRequest`/`AdapterResponse` contract and the `finishReason` normalization — that's what makes it a valid adapter — but retry classification can't key off an HTTP status for it.

## Connections

- **Called by:** Core Orchestration (via adapter registry lookup)
- **Inputs:** `AdapterRequest` (model, prompt, systemPrompt, temperature, maxTokens, json, apiKey, baseUrl)
- **Returns:** `AdapterResponse` (text, model, tokens, finishReason)
- **External:** Provider HTTP APIs (Anthropic, OpenAI, Ollama)
