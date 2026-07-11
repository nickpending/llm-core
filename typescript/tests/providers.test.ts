/**
 * tests/providers.test.ts
 *
 * Unit tests for provider adapters: anthropic.ts, openai.ts, ollama.ts, index.ts
 *
 * Strategy: mock global.fetch to test adapters without real API calls.
 * Adapters are pure functions with no module-level state — no mock.module()
 * for os/homedir needed (unlike services.ts).
 *
 * Invariants protected:
 *   INV-001: text field is verbatim extraction from provider response, no modification
 *   INV-002: adapters use req.baseUrl — no hardcoded provider URLs
 *   HTTP errors throw with status code so callers can distinguish failure types
 *   finish_reason mapping is correct (provider-specific → normalized)
 *   Ollama missing token counts fall back to 0, not undefined/NaN
 *   getAdapter() throws on unknown name (not silently returning undefined)
 */

import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { join } from "node:path";
import type { AdapterRequest } from "../lib/types.ts";

// Import adapters directly — they are pure functions, no module-level side effects
const { complete: anthropicComplete } = await import(
  join(import.meta.dir, "../lib/providers/anthropic.ts")
);
const { complete: openaiComplete } = await import(
  join(import.meta.dir, "../lib/providers/openai.ts")
);
const { complete: ollamaComplete } = await import(
  join(import.meta.dir, "../lib/providers/ollama.ts")
);
const { parseClaudeJsonResponse } = await import(
  join(import.meta.dir, "../lib/providers/claude-cli.ts")
);
const { getAdapter } = await import(
  join(import.meta.dir, "../lib/providers/index.ts")
);

// Helper: create a mock fetch that returns a JSON response
function mockFetch(status: number, body: unknown): typeof fetch {
  return async (_url: string | URL | Request, _init?: RequestInit) => {
    const isOk = status >= 200 && status < 300;
    return {
      ok: isOk,
      status,
      json: async () => body,
      text: async () =>
        typeof body === "string" ? body : JSON.stringify(body),
    } as Response;
  };
}

const BASE_ANTHROPIC_REQ: AdapterRequest = {
  baseUrl: "https://test-anthropic.example.com/v1",
  apiKey: "sk-test-key",
  model: "claude-3-5-sonnet-20241022",
  prompt: "Hello world",
};

const BASE_OPENAI_REQ: AdapterRequest = {
  baseUrl: "https://test-openai.example.com/v1",
  apiKey: "sk-openai-test",
  model: "gpt-4o",
  prompt: "Hello world",
};

const BASE_OLLAMA_REQ: AdapterRequest = {
  baseUrl: "http://test-ollama.example.com",
  apiKey: null,
  model: "llama3.2",
  prompt: "Hello world",
};

// Save and restore global.fetch around each test
let originalFetch: typeof globalThis.fetch;
beforeEach(() => {
  originalFetch = globalThis.fetch;
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---------------------------------------------------------------------------
// Anthropic adapter
// ---------------------------------------------------------------------------

describe("anthropic adapter", () => {
  it("returns normalized AdapterResponse with verbatim text (INV-001)", async () => {
    const responseText = "  The answer is 42.  "; // spaces intentional — must not be trimmed
    globalThis.fetch = mockFetch(200, {
      content: [{ type: "text", text: responseText }],
      model: "claude-3-5-sonnet-20241022",
      stop_reason: "end_turn",
      usage: { input_tokens: 10, output_tokens: 5 },
    });

    const result = await anthropicComplete(BASE_ANTHROPIC_REQ);

    expect(result.text).toBe(responseText); // INV-001: no stripping
    expect(result.model).toBe("claude-3-5-sonnet-20241022");
    expect(result.tokensInput).toBe(10);
    expect(result.tokensOutput).toBe(5);
    expect(result.finishReason).toBe("stop"); // end_turn → stop
  });

  it("extracts the text block past a leading thinking block (extended-thinking models)", async () => {
    // Extended-thinking models (claude-sonnet-5, claude-opus-4, …) return a `thinking`
    // block FIRST, then the `text` block. content[0] is the thinking block, so verbatim
    // extraction must find the first type:"text" block, not blindly read content[0].
    const responseText = "The answer is 42.";
    globalThis.fetch = mockFetch(200, {
      content: [
        { type: "thinking", thinking: "Let me reason about this…", signature: "abc" },
        { type: "text", text: responseText },
      ],
      model: "claude-sonnet-5",
      stop_reason: "end_turn",
      usage: { input_tokens: 10, output_tokens: 5 },
    });

    const result = await anthropicComplete(BASE_ANTHROPIC_REQ);

    expect(result.text).toBe(responseText); // the text block, NOT the thinking block
    expect(result.model).toBe("claude-sonnet-5");
  });

  it("throws with status code on HTTP error (401)", async () => {
    globalThis.fetch = mockFetch(401, "Unauthorized");

    await expect(anthropicComplete(BASE_ANTHROPIC_REQ)).rejects.toThrow("401");
  });
});

// ---------------------------------------------------------------------------
// OpenAI adapter
// ---------------------------------------------------------------------------

describe("openai adapter", () => {
  it("returns normalized AdapterResponse and maps finish_reason 'length' to 'max_tokens'", async () => {
    const responseText = "Response from OpenAI";
    globalThis.fetch = mockFetch(200, {
      choices: [
        {
          message: { content: responseText },
          finish_reason: "length",
        },
      ],
      model: "gpt-4o",
      usage: { prompt_tokens: 20, completion_tokens: 8 },
    });

    const result = await openaiComplete(BASE_OPENAI_REQ);

    expect(result.text).toBe(responseText); // INV-001: verbatim
    expect(result.model).toBe("gpt-4o");
    expect(result.tokensInput).toBe(20);
    expect(result.tokensOutput).toBe(8);
    expect(result.finishReason).toBe("max_tokens"); // 'length' maps to 'max_tokens'
  });

  it("throws with status code on HTTP error (500)", async () => {
    globalThis.fetch = mockFetch(500, "Internal Server Error");

    await expect(openaiComplete(BASE_OPENAI_REQ)).rejects.toThrow("500");
  });
});

// ---------------------------------------------------------------------------
// Ollama adapter
// ---------------------------------------------------------------------------

describe("ollama adapter", () => {
  it("returns normalized AdapterResponse using data.response as text (INV-001)", async () => {
    const responseText = "Ollama says hello";
    globalThis.fetch = mockFetch(200, {
      response: responseText,
      model: "llama3.2",
      done_reason: "stop",
      prompt_eval_count: 15,
      eval_count: 30,
    });

    const result = await ollamaComplete(BASE_OLLAMA_REQ);

    expect(result.text).toBe(responseText); // INV-001: data.response verbatim
    expect(result.model).toBe("llama3.2");
    expect(result.tokensInput).toBe(15);
    expect(result.tokensOutput).toBe(30);
    expect(result.finishReason).toBe("stop");
  });

  it("falls back to 0 when token counts are missing (cached prompt path)", async () => {
    globalThis.fetch = mockFetch(200, {
      response: "Cached response",
      model: "llama3.2",
      done_reason: "stop",
      // prompt_eval_count and eval_count intentionally omitted
    });

    const result = await ollamaComplete(BASE_OLLAMA_REQ);

    expect(result.tokensInput).toBe(0); // || 0 fallback must not be NaN/undefined
    expect(result.tokensOutput).toBe(0);
  });

  it("throws with status code on HTTP error (404)", async () => {
    globalThis.fetch = mockFetch(404, "model not found");

    await expect(ollamaComplete(BASE_OLLAMA_REQ)).rejects.toThrow("404");
  });
});

// ---------------------------------------------------------------------------
// Claude CLI adapter (parser only — subprocess path is integration-tested manually)
// ---------------------------------------------------------------------------

describe("claude-cli parseClaudeJsonResponse()", () => {
  it("extracts result text verbatim and reports the resolved model from modelUsage (INV-001)", () => {
    const envelope = {
      type: "result",
      subtype: "success",
      is_error: false,
      result: "  YES  ", // spaces intentional — must not be trimmed
      stop_reason: "end_turn",
      usage: { input_tokens: 2024, output_tokens: 274 },
      modelUsage: {
        "claude-haiku-4-5-20251001": { inputTokens: 2024, outputTokens: 274 },
      },
    };

    const result = parseClaudeJsonResponse(JSON.stringify(envelope), "haiku");

    expect(result.text).toBe("  YES  "); // INV-001
    expect(result.model).toBe("claude-haiku-4-5-20251001"); // resolved, not the alias
    expect(result.tokensInput).toBe(2024);
    expect(result.tokensOutput).toBe(274);
    expect(result.finishReason).toBe("stop");
  });

  it("falls back to the caller's model alias when modelUsage is absent", () => {
    const envelope = {
      type: "result",
      subtype: "success",
      is_error: false,
      result: "hi",
      stop_reason: "end_turn",
      usage: { input_tokens: 5, output_tokens: 1 },
    };

    const result = parseClaudeJsonResponse(JSON.stringify(envelope), "haiku");

    expect(result.model).toBe("haiku");
  });

  it("maps stop_reason 'max_tokens' to finishReason 'max_tokens'", () => {
    const envelope = {
      type: "result",
      subtype: "success",
      is_error: false,
      result: "truncated...",
      stop_reason: "max_tokens",
      usage: { input_tokens: 1, output_tokens: 100 },
    };

    const result = parseClaudeJsonResponse(JSON.stringify(envelope), "haiku");

    expect(result.finishReason).toBe("max_tokens");
  });

  it("throws on non-JSON stdout with a clear remediation hint", () => {
    expect(() => parseClaudeJsonResponse("not json at all", "haiku")).toThrow(
      "non-JSON output",
    );
  });

  it("throws when envelope reports an error subtype", () => {
    const envelope = {
      type: "result",
      subtype: "error",
      is_error: true,
      result: "auth failed",
    };

    expect(() =>
      parseClaudeJsonResponse(JSON.stringify(envelope), "haiku"),
    ).toThrow("error envelope");
  });

  it("throws when result field is missing or non-string", () => {
    const envelope = {
      type: "result",
      subtype: "success",
      is_error: false,
      // result intentionally omitted
    };

    expect(() =>
      parseClaudeJsonResponse(JSON.stringify(envelope), "haiku"),
    ).toThrow("missing result field");
  });

  it("falls back to 0 when token counts are missing", () => {
    const envelope = {
      type: "result",
      subtype: "success",
      is_error: false,
      result: "x",
      stop_reason: "end_turn",
      // usage intentionally omitted
    };

    const result = parseClaudeJsonResponse(JSON.stringify(envelope), "haiku");

    expect(result.tokensInput).toBe(0);
    expect(result.tokensOutput).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Adapter registry
// ---------------------------------------------------------------------------

describe("getAdapter()", () => {
  it("throws descriptive error for unknown adapter name instead of returning undefined", () => {
    expect(() => getAdapter("nonexistent")).toThrow('"nonexistent"');
    expect(() => getAdapter("nonexistent")).toThrow("Available:");
  });

  it("registers claude-cli adapter so subscription routing is available", () => {
    const adapter = getAdapter("claude-cli");
    expect(typeof adapter.complete).toBe("function");
  });
});
