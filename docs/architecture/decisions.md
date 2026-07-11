---
type: architecture
subtype: decisions
project: "llm-core"
status: active
created: "2026-04-08"
updated: "2026-07-11"
tags: [architecture, decisions]
---

# Decisions

Architectural decisions and their rationale. Most recent first.

## Anthropic adapter finds the text block instead of indexing content[0] (TypeScript v0.7.1)

**Context:** Extended-thinking models (claude-sonnet-5, claude-opus-4, etc.) return a `type: "thinking"` content block first, followed by the `type: "text"` block. The TypeScript Anthropic adapter indexed `content[0].text` directly, so calls to these models returned the thinking trace as `CompleteResult.text` instead of the actual answer — a silent correctness bug, not a thrown error, since `content[0]` still had *a* string in some shapes.
**Choice:** `typescript/lib/providers/anthropic.ts` now uses `data.content?.find(block => block.type === "text" && typeof block.text === "string")` to locate the answer block regardless of position, and throws `"no type:\"text\" block in content"` if none is found. Shipped as `@voidwire/llm-core` v0.7.1. Test coverage added in `typescript/tests/providers.test.ts` for the thinking-block-then-text-block shape.
**Why:** INV-001 (verbatim text extraction) is only meaningful if the adapter extracts the *right* block. Scanning by `type` is robust to Anthropic adding more block types or reordering them; positional indexing isn't. **Not yet ported to Python** — `python/src/llm_core/providers/anthropic.py` still indexes `content[0].text` and will mis-extract on extended-thinking responses; tracked as a cross-language parity gap, not fixed in this change.

## claude-cli as its own subprocess adapter, not anthropic with a different base_url

**Context:** Routing inference through a Claude Code subscription (zero per-call API cost) instead of the metered Anthropic Messages API. The obvious shortcut — point the anthropic adapter at a different `base_url` — doesn't work.
**Choice:** A dedicated `claude-cli` adapter (`typescript/lib/providers/claude-cli.ts`) that spawns `claude --print --output-format json` as a subprocess and parses the JSON envelope for real token counts. TypeScript only. Opt-in via `[services.claude-cli]` — not in the default services.toml. Registered in `providers/index.ts`; dispatched explicitly in `core.ts` healthCheck.
**Why:** Claude Code's subscription auth is OAuth-bound to the local CLI install and is only honored by the `claude` binary, not by HTTP requests to api.anthropic.com. The child env scrubs `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `CLAUDECODE` so the CLI falls through to OAuth credentials and doesn't trip its nested-session guard. Flags are locked down (`--tools ''`, `--setting-sources ''`, system prompt replaces rather than appends) for deterministic, pure-inference behavior. This adapter deliberately breaks three library invariants because it's a different integration shape: it's a subprocess (not raw HTTP), its health check is `claude --version` (not an HTTP endpoint), and it throws plain Errors with subprocess exit codes (not HTTP status codes for retry classification).

## App metadata via services.toml for OpenRouter attribution

**Context:** OpenRouter shows "Unknown" for apps that don't send identification headers. Other providers don't support app identification.
**Choice:** Optional `app_title` and `app_url` fields on `ServiceConfig` in services.toml. The OpenAI adapter sends `X-OpenRouter-Title` and `HTTP-Referer` headers when present. Other adapters ignore these fields.
**Why:** Per-service config — set once, no code changes for callers. OpenRouter requires `HTTP-Referer` as the primary identifier; `X-OpenRouter-Title` sets the display name. The legacy `X-Title` header doesn't work.

## Unified pricing.toml across both languages

**Context:** TypeScript wrote pricing.toml with per-1M-token rates. Python wrote model_prices.json with per-token rates. Two files, two formats, divergent cost estimates.
**Choice:** Both languages read/write the same pricing.toml format. Each has its own self-sufficient `update_pricing()` that fetches litellm JSON, converts to per-1M-token TOML, and writes pricing.toml. Manual TOML serialization in both (no TOML writer in Bun or Python stdlib).
**Why:** Cross-language consistency. Either language can be used independently. Running update_pricing() in one language populates the data for both.

## Drop smol-toml for Bun.TOML.parse()

**Context:** TypeScript side used smol-toml as a TOML parser. Bun added native TOML parsing.
**Choice:** Replaced smol-toml with `Bun.TOML.parse()` — zero runtime dependencies now (apiconf aside).
**Why:** Platform capability caught up. No reason to carry a dependency when the runtime provides it natively.

## No auto-fetch on first use

**Context:** Python previously auto-fetched litellm JSON on first `estimate_cost()` call if no local file existed. Surprise network calls.
**Choice:** Both languages return None if pricing.toml is missing. Users must explicitly call `update_pricing()`.
**Why:** No surprise network calls. Pricing is best-effort — missing file means no cost estimate, not a network request.

## Typed ProviderError.status_code (Python)

**Context:** TypeScript classifies transient errors by parsing `(status)` patterns from error message strings.
**Choice:** Python uses a typed `status_code` attribute on `ProviderError` instead of string matching.
**Why:** More Pythonic. Enables retry.py to classify transient vs non-transient errors without parsing message text. All adapter HTTP error paths must set `status_code` on `ProviderError`.

## Per-adapter health check functions (Python)

**Context:** Adding a new provider required editing core.py to add health check logic.
**Choice:** Moved health check config to per-adapter `health_check_config()` functions that return URL + headers.
**Why:** Adding a new provider no longer requires editing core.py. Each adapter is self-contained.

## Raw HTTP over provider SDKs

**Context:** Official SDKs exist for Anthropic and OpenAI. Using them would simplify some adapter code.
**Choice:** Raw HTTP calls via native fetch (Bun) and httpx (Python). No SDK dependencies.
**Why:** Keeps dependency surface minimal. The adapter layer is thin enough that SDK abstraction adds more weight than value.

## Normalized finish_reason values

**Context:** Each provider returns different completion reason strings (Anthropic: end_turn/length, OpenAI: stop/length, Ollama: done_reason field).
**Choice:** All adapters normalize to exactly `{stop, max_tokens, error}`.
**Why:** Callers can safely compare without handling provider-specific values.

## Restore update_pricing() as public API

**Context:** `update_pricing()` was removed entirely as an overcorrection during a refactor.
**Choice:** Restored as public API. The fetch mechanism was sound — only the source needed fixing.
**Why:** Removing the capability was worse than rewriting it. The function fetches from a clean source and writes TOML.
