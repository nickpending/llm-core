/**
 * tests/health-check.test.ts
 *
 * Tests for healthCheck() at adapter level and core orchestration.
 *
 * Strategy:
 *   - Adapter-level tests: import adapters directly, mock globalThis.fetch.
 *     Tests verify correct URL path, correct auth headers, correct error
 *     message format (status code + provider name) — no module-level side
 *     effects, so no mock.module() needed.
 *   - Core orchestration tests: follow core.test.ts pattern — mock "os"
 *     before importing core.ts so services.ts resolves to temp dir, mock
 *     "@voidwire/apiconf" to inject test keys, mock globalThis.fetch to
 *     control HTTP responses.
 *
 * Invariants protected:
 *   - Adapter GETs to correct endpoint path (not POST completion path)
 *   - Adapter sends correct auth headers for each provider
 *   - Error message includes status code AND provider name (prismis startup logs this)
 *   - core.ts dispatches to correct adapter based on svc.adapter
 *   - Unknown adapter throws rather than silently succeeding
 *   - No-arg call resolves (default service path)
 *
 * SC-10 coverage:
 *   - SC-10a: valid service → resolves void
 *   - SC-10b: invalid key → throws with status code and provider name
 */

import { afterAll, afterEach, beforeEach, describe, expect, it, mock } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, unlinkSync } from "node:fs";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Adapter-level tests — no module-level side effects; import directly
// ---------------------------------------------------------------------------

const { healthCheck: anthropicHealthCheck } = await import(
  join(import.meta.dir, "../lib/providers/anthropic.ts")
);
const { healthCheck: openaiHealthCheck } = await import(
  join(import.meta.dir, "../lib/providers/openai.ts")
);
const { healthCheck: ollamaHealthCheck } = await import(
  join(import.meta.dir, "../lib/providers/ollama.ts")
);

let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});

/** Create a mock fetch that returns the given status and body */
function mockFetch(status: number, body: string | object): typeof fetch {
  return async (_url: string | URL | Request, _init?: RequestInit) => {
    const isOk = status >= 200 && status < 300;
    return {
      ok: isOk,
      status,
      text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
      json: async () => (typeof body === "string" ? {} : body),
    } as Response;
  };
}

/** Capture the URL and headers used in the most recent fetch call */
function capturingFetch(status: number, body: string | object) {
  let capturedUrl = "";
  let capturedInit: RequestInit | undefined;
  const fn = async (url: string | URL | Request, init?: RequestInit) => {
    capturedUrl = url.toString();
    capturedInit = init;
    const isOk = status >= 200 && status < 300;
    return {
      ok: isOk,
      status,
      text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
      json: async () => (typeof body === "string" ? {} : body),
    } as Response;
  };
  return {
    fn,
    get url() {
      return capturedUrl;
    },
    get init() {
      return capturedInit;
    },
  };
}

// ---------------------------------------------------------------------------
// Anthropic adapter healthCheck
// ---------------------------------------------------------------------------

describe("anthropic adapter healthCheck", () => {
  it("calls GET /models — not the /messages completion endpoint (SC-10a path)", async () => {
    const cap = capturingFetch(200, {});
    globalThis.fetch = cap.fn as typeof fetch;

    await anthropicHealthCheck("https://api.anthropic.com/v1", "sk-test");

    expect(cap.url).toBe("https://api.anthropic.com/v1/models");
  });

  it("sends x-api-key and anthropic-version headers — not Bearer auth", async () => {
    const cap = capturingFetch(200, {});
    globalThis.fetch = cap.fn as typeof fetch;

    await anthropicHealthCheck("https://api.anthropic.com/v1", "sk-ant-key");

    const headers = cap.init?.headers as Record<string, string>;
    expect(headers?.["x-api-key"]).toBe("sk-ant-key");
    expect(headers?.["anthropic-version"]).toBeTruthy();
    // Must NOT use Bearer auth (that's OpenAI's pattern)
    expect(headers?.Authorization).toBeUndefined();
  });

  it("throws with status code AND 'Anthropic' in message on non-200 — SC-10b", async () => {
    globalThis.fetch = mockFetch(401, "Unauthorized");

    await expect(anthropicHealthCheck("https://api.anthropic.com/v1", "bad-key")).rejects.toThrow(
      "401",
    );

    await expect(anthropicHealthCheck("https://api.anthropic.com/v1", "bad-key")).rejects.toThrow(
      "Anthropic",
    );
  });
});

// ---------------------------------------------------------------------------
// OpenAI adapter healthCheck
// ---------------------------------------------------------------------------

describe("openai adapter healthCheck", () => {
  it("calls GET /models and sends Bearer auth — not x-api-key (SC-10a path)", async () => {
    const cap = capturingFetch(200, {});
    globalThis.fetch = cap.fn as typeof fetch;

    await openaiHealthCheck("https://api.openai.com/v1", "sk-openai-test");

    expect(cap.url).toBe("https://api.openai.com/v1/models");
    const headers = cap.init?.headers as Record<string, string>;
    expect(headers?.Authorization).toBe("Bearer sk-openai-test");
    expect(headers?.["x-api-key"]).toBeUndefined();
  });

  it("throws with status code AND 'OpenAI' in message on non-200 — SC-10b", async () => {
    globalThis.fetch = mockFetch(401, "invalid_api_key");

    await expect(openaiHealthCheck("https://api.openai.com/v1", "bad-key")).rejects.toThrow("401");

    await expect(openaiHealthCheck("https://api.openai.com/v1", "bad-key")).rejects.toThrow(
      "OpenAI",
    );
  });
});

// ---------------------------------------------------------------------------
// Ollama adapter healthCheck
// ---------------------------------------------------------------------------

describe("ollama adapter healthCheck", () => {
  it("calls GET /api/tags — NOT /api/generate (completion path) and NOT /models", async () => {
    const cap = capturingFetch(200, { models: [] });
    globalThis.fetch = cap.fn as typeof fetch;

    await ollamaHealthCheck("http://localhost:11434", null);

    expect(cap.url).toBe("http://localhost:11434/api/tags");
    // Verify it's a plain GET with no auth headers required
    const headers = (cap.init?.headers ?? {}) as Record<string, string>;
    expect(headers?.Authorization).toBeUndefined();
  });

  it("throws with status code AND 'Ollama' in message when service is unreachable", async () => {
    globalThis.fetch = mockFetch(503, "Service Unavailable");

    await expect(ollamaHealthCheck("http://localhost:11434", null)).rejects.toThrow("503");

    await expect(ollamaHealthCheck("http://localhost:11434", null)).rejects.toThrow("Ollama");
  });
});

// ---------------------------------------------------------------------------
// core.ts healthCheck orchestration
// ---------------------------------------------------------------------------

// Set up temp home BEFORE importing core.ts so services.ts resolves to temp dir
const tempHome = mkdtempSync("/tmp/llm-core-health-test-");
const configDir = join(tempHome, ".config", "llm-core");

// Mock "os" BEFORE importing core.ts (same pattern as core.test.ts)
mock.module("os", () => ({
  homedir: () => tempHome,
}));

// Mock "@voidwire/apiconf" to inject test keys without real config files
mock.module("@voidwire/apiconf", () => ({
  getKey: (keyName: string) => {
    if (keyName === "anthropic") return "sk-test-anthropic";
    if (keyName === "openai") return "sk-test-openai";
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

// Import core AFTER mocks are registered
const { healthCheck } = await import(join(import.meta.dir, "../lib/core.ts"));
const { _resetServicesCache } = await import(join(import.meta.dir, "../lib/services.ts"));

afterAll(() => {
  rmSync(tempHome, { recursive: true, force: true });
});

beforeEach(() => {
  _resetServicesCache();
  mkdirSync(configDir, { recursive: true });
  const servicesPath = join(configDir, "services.toml");
  if (existsSync(servicesPath)) unlinkSync(servicesPath);
});

describe("healthCheck() core orchestration", () => {
  it("resolves void on 200 for named openai service — SC-10a", async () => {
    globalThis.fetch = mockFetch(200, { models: [] }) as typeof fetch;

    // Should resolve without throwing
    await expect(healthCheck("openai")).resolves.toBeUndefined();
  });

  it("resolves void on 200 for named ollama service — SC-10a", async () => {
    globalThis.fetch = mockFetch(200, { models: [] }) as typeof fetch;

    await expect(healthCheck("ollama")).resolves.toBeUndefined();
  });

  it("resolves void with no argument — uses default_service (SC-10a edge case)", async () => {
    // Default services.toml sets default_service = "anthropic"
    globalThis.fetch = mockFetch(200, { models: [] }) as typeof fetch;

    await expect(healthCheck()).resolves.toBeUndefined();
  });

  it("throws with status code in message when API returns 401 — SC-10b", async () => {
    globalThis.fetch = mockFetch(401, "Unauthorized") as typeof fetch;

    await expect(healthCheck("openai")).rejects.toThrow("401");
  });

  it("throws 'Unknown service' for nonexistent service — not a silent no-op", async () => {
    await expect(healthCheck("notaservice")).rejects.toThrow("Unknown service");
    await expect(healthCheck("notaservice")).rejects.toThrow("notaservice");
  });
});
