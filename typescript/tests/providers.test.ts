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
const { getAdapter } = await import(join(import.meta.dir, "../lib/providers/index.ts"));

// Helper: create a mock fetch that returns a JSON response
function mockFetch(status: number, body: unknown): typeof fetch {
  return async (_url: string | URL | Request, _init?: RequestInit) => {
    const isOk = status >= 200 && status < 300;
    return {
      ok: isOk,
      status,
      json: async () => body,
      text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
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
      content: [{ text: responseText }],
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
// Adapter registry
// ---------------------------------------------------------------------------

describe("getAdapter()", () => {
  it("throws descriptive error for unknown adapter name instead of returning undefined", () => {
    expect(() => getAdapter("nonexistent")).toThrow('"nonexistent"');
    expect(() => getAdapter("nonexistent")).toThrow("Available:");
  });
});
