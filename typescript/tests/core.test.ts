/**
 * tests/core.test.ts
 *
 * Integration tests for core.ts — the complete() orchestration function.
 *
 * Strategy: Mock only at external boundaries to avoid contaminating other
 * test files' module caches. Bun 1.3.8 shares the module registry across
 * test files in the same test run, so mock.module() for local files
 * would break providers.test.ts, retry-helpers-pricing.test.ts, etc.
 *
 * Boundaries mocked:
 *   - "os" module: redirect homedir() to temp dir so services.ts writes
 *     its default config there instead of real ~/.config/llm-core/
 *   - "@voidwire/apiconf": inject a test key (config.test.ts runs before
 *     us alphabetically and has already finished — no conflict)
 *   - global.fetch: control HTTP responses from provider adapters
 *
 * This lets the real service resolution, config loading, retry logic,
 * provider adapter selection, and cost estimation code run — we test the
 * full orchestration path, not just the wiring.
 *
 * Invariants protected:
 *   INV-003: CompleteResult always has the same shape regardless of provider
 *   Field mapping: AdapterResponse.tokensInput/Output → CompleteResult.tokens.input/output
 *   Provider field sourced from service.adapter, not adapter response model
 *   durationMs is measured (≥ 0)
 *   cost is null for unknown/local models
 *   Missing model throws before adapter is called
 *   Unknown service throws with listing of available services
 *   API failure propagates to caller
 */

import { afterAll, afterEach, beforeEach, describe, expect, it, mock } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, unlinkSync } from "node:fs";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Temp home dir for service config isolation
// ---------------------------------------------------------------------------

const tempHome = mkdtempSync("/tmp/llm-core-core-test-");
const configDir = join(tempHome, ".config", "llm-core");

// Mock "os" BEFORE importing core.ts so services.ts (loaded by core.ts)
// redirects CONFIG_DIR to our temp dir.
// NOTE: services.test.ts also mocks "os" with its own temp dir — when it
// runs after us (alphabetically: services > core), it re-mocks "os" with
// its own path, which is correct behavior.
mock.module("os", () => ({
  homedir: () => tempHome,
}));

// Mock "@voidwire/apiconf" BEFORE importing core.ts so config.ts returns
// a test key. config.test.ts runs BEFORE core.test.ts (alphabetically
// "config" < "core") and has already finished, so there is no conflict.
mock.module("@voidwire/apiconf", () => ({
  getKey: (keyName: string) => {
    if (keyName === "anthropic") return "sk-test-anthropic";
    if (keyName === "openai") return "sk-test-openai";
    // Unknown key names throw KeyNotFoundError shape
    const err = new Error(`Key "${keyName}" not found`);
    (err as NodeJS.ErrnoException).code = "KEY_NOT_FOUND";
    throw err;
  },
  KeyNotFoundError: class KeyNotFoundError extends Error {
    keyName: string;
    available: string[];
    constructor(keyName: string, available: string[]) {
      super(`Key not found: ${keyName}`);
      this.keyName = keyName;
      this.available = available;
    }
  },
  ConfigNotFoundError: class ConfigNotFoundError extends Error {
    path: string;
    constructor(path: string) {
      super(`Config not found: ${path}`);
      this.path = path;
    }
  },
}));

// Import core.ts AFTER all module mocks are registered.
// Using dynamic import so mocks are fully registered first.
const { complete } = await import(join(import.meta.dir, "../lib/core.ts"));

// Also import _resetServicesCache to clear module-level cache between tests
const { _resetServicesCache } = await import(join(import.meta.dir, "../lib/services.ts"));

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

afterAll(() => {
  rmSync(tempHome, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// Fetch mock helpers
// ---------------------------------------------------------------------------

let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  // Reset services cache so each test starts fresh
  _resetServicesCache();
  // Ensure config dir exists for default service generation
  mkdirSync(configDir, { recursive: true });
  const servicesPath = join(configDir, "services.toml");
  if (existsSync(servicesPath)) unlinkSync(servicesPath);
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

/** Create a mock fetch that returns a JSON response */
function mockFetch(status: number, body: unknown): typeof fetch {
  return async () => {
    const isOk = status >= 200 && status < 300;
    return {
      ok: isOk,
      status,
      json: async () => body,
      text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
    } as Response;
  };
}

/** Ollama success response for "ollama" service (key_required: false) */
function ollamaSuccessResponse(text: string, tokensIn = 15, tokensOut = 30) {
  return {
    response: text,
    model: "llama3.2",
    done_reason: "stop",
    prompt_eval_count: tokensIn,
    eval_count: tokensOut,
  };
}

// ---------------------------------------------------------------------------
// INV-003: CompleteResult envelope normalization
// ---------------------------------------------------------------------------

describe("complete() envelope normalization (INV-003)", () => {
  it("result has all required CompleteResult fields with correct types", async () => {
    globalThis.fetch = mockFetch(200, ollamaSuccessResponse("The answer is 42"));

    const result = await complete({
      prompt: "hello",
      model: "llama3.2",
      service: "ollama",
    });

    // INV-003: every field must be present and typed correctly
    expect(typeof result.text).toBe("string");
    expect(typeof result.model).toBe("string");
    expect(typeof result.provider).toBe("string");
    expect(typeof result.tokens).toBe("object");
    expect(typeof result.tokens.input).toBe("number");
    expect(typeof result.tokens.output).toBe("number");
    expect(typeof result.finishReason).toBe("string");
    expect(typeof result.durationMs).toBe("number");
    // cost is number | null — must not be undefined
    expect(result.cost === null || typeof result.cost === "number").toBe(true);
  });

  it("tokensInput maps to tokens.input and tokensOutput maps to tokens.output (not swapped)", async () => {
    // Ollama returns distinct values: prompt_eval_count=100, eval_count=200
    globalThis.fetch = mockFetch(200, {
      response: "test response",
      model: "llama3.2",
      done_reason: "stop",
      prompt_eval_count: 100, // → tokens.input
      eval_count: 200, // → tokens.output
    });

    const result = await complete({
      prompt: "hello",
      model: "llama3.2",
      service: "ollama",
    });

    expect(result.tokens.input).toBe(100); // must not be 200
    expect(result.tokens.output).toBe(200); // must not be 100
  });

  it("provider field is service.adapter, not the model name returned by adapter", async () => {
    // Ollama service → adapter = "ollama". Response model could differ.
    globalThis.fetch = mockFetch(200, {
      response: "response text",
      model: "llama3.2:instruct", // model name from adapter
      done_reason: "stop",
      prompt_eval_count: 10,
      eval_count: 20,
    });

    const result = await complete({
      prompt: "hello",
      model: "llama3.2:instruct",
      service: "ollama",
    });

    expect(result.provider).toBe("ollama"); // service.adapter
    expect(result.model).toBe("llama3.2:instruct"); // from adapter response
  });

  it("durationMs is non-negative (timing is recorded)", async () => {
    globalThis.fetch = mockFetch(200, ollamaSuccessResponse("hello"));

    const result = await complete({
      prompt: "hello",
      model: "llama3.2",
      service: "ollama",
    });

    expect(result.durationMs).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(result.durationMs)).toBe(true);
  });

  it("cost is null for ollama (local model — no pricing entry)", async () => {
    // The default pricing.toml won't have ollama models → estimateCost returns null
    globalThis.fetch = mockFetch(200, ollamaSuccessResponse("hello"));

    const result = await complete({
      prompt: "hello",
      model: "llama3.2",
      service: "ollama",
    });

    expect(result.cost).toBeNull(); // not 0, not undefined
  });

  it("text field is verbatim from adapter response — no trimming", async () => {
    const verbatimText = "  The answer is 42.  \n"; // spaces intentional
    globalThis.fetch = mockFetch(200, {
      response: verbatimText, // Ollama response field
      model: "llama3.2",
      done_reason: "stop",
      prompt_eval_count: 5,
      eval_count: 10,
    });

    const result = await complete({
      prompt: "hello",
      model: "llama3.2",
      service: "ollama",
    });

    expect(result.text).toBe(verbatimText); // INV-001 carried through
  });
});

// ---------------------------------------------------------------------------
// Error propagation
// ---------------------------------------------------------------------------

describe("complete() error cases", () => {
  it("throws 'Model name required' when model is empty string — before fetch is called", async () => {
    let fetchCalled = false;
    globalThis.fetch = async () => {
      fetchCalled = true;
      return {} as Response;
    };

    await expect(complete({ prompt: "hello", model: "", service: "ollama" })).rejects.toThrow(
      "Model name required",
    );

    expect(fetchCalled).toBe(false); // fetch must never be reached
  });

  it("throws 'Unknown service' when service name is not in services.toml", async () => {
    // Default services.toml has anthropic, openai, ollama — "bogus" is unknown
    await expect(complete({ prompt: "hello", model: "llama3", service: "bogus" })).rejects.toThrow(
      'Unknown service: "bogus"',
    );
  });

  it("propagates HTTP error from adapter — 503 reaches the caller", async () => {
    // withRetry exhausts retries (503 is transient), final error propagates
    globalThis.fetch = mockFetch(503, "Service Unavailable");

    await expect(
      complete({
        prompt: "hello",
        model: "llama3.2",
        service: "ollama",
        // Use delays: [] override not possible here — retry with defaults
        // This test verifies propagation shape, not retry count
      }),
    ).rejects.toThrow("503");
  }, 15000); // Allow up to 15s for retry delays (3 attempts × 4s max)
});
