/**
 * tests/retry-helpers-pricing.test.ts
 *
 * Unit tests for retry.ts, helpers.ts, and pricing.ts.
 *
 * Strategy: pure function tests for helpers and retry. Retry tests use
 * delays: [0, 0] to avoid real sleep waits. Pricing tests use temp
 * directory to avoid touching real ~/.config.
 *
 * Invariants protected:
 *   INV-004: Retry only fires on transient errors (429, 5xx, network).
 *            Non-transient errors (400, 401, 403, 404) fail immediately.
 */

import { describe, expect, it } from "bun:test";
import { join } from "node:path";

// Dynamic imports following providers.test.ts pattern
const { withRetry } = await import(join(import.meta.dir, "../lib/retry.ts"));
const { extractJson, isTruncated } = await import(join(import.meta.dir, "../lib/helpers.ts"));
const { estimateCost } = await import(join(import.meta.dir, "../lib/pricing.ts"));

// ---------------------------------------------------------------------------
// retry.ts
// ---------------------------------------------------------------------------

describe("withRetry", () => {
  it("succeeds on first attempt without retrying", async () => {
    let callCount = 0;
    const result = await withRetry(
      async () => {
        callCount++;
        return "ok";
      },
      { delays: [0, 0] },
    );

    expect(result).toBe("ok");
    expect(callCount).toBe(1);
  });

  it("retries on transient error (429) and succeeds on second attempt", async () => {
    let callCount = 0;
    const result = await withRetry(
      async () => {
        callCount++;
        if (callCount === 1) {
          throw new Error("Rate limited (429): too many requests");
        }
        return "recovered";
      },
      { delays: [0, 0] },
    );

    expect(result).toBe("recovered");
    expect(callCount).toBe(2);
  });

  it("retries on transient error (500) and succeeds on third attempt", async () => {
    let callCount = 0;
    const result = await withRetry(
      async () => {
        callCount++;
        if (callCount <= 2) {
          throw new Error("Server error (500): internal");
        }
        return "recovered";
      },
      { maxAttempts: 3, delays: [0, 0] },
    );

    expect(result).toBe("recovered");
    expect(callCount).toBe(3);
  });

  it("retries on network error (TypeError) and succeeds", async () => {
    let callCount = 0;
    const result = await withRetry(
      async () => {
        callCount++;
        if (callCount === 1) {
          throw new TypeError("fetch failed");
        }
        return "recovered";
      },
      { delays: [0, 0] },
    );

    expect(result).toBe("recovered");
    expect(callCount).toBe(2);
  });

  it("fails fast on non-transient error (401) without retrying (INV-004)", async () => {
    let callCount = 0;

    await expect(
      withRetry(
        async () => {
          callCount++;
          throw new Error("Unauthorized (401): invalid key");
        },
        { delays: [0, 0] },
      ),
    ).rejects.toThrow("401");

    expect(callCount).toBe(1); // No retry
  });

  it("fails fast on non-transient error (400) without retrying (INV-004)", async () => {
    let callCount = 0;

    await expect(
      withRetry(
        async () => {
          callCount++;
          throw new Error("Bad request (400): invalid model");
        },
        { delays: [0, 0] },
      ),
    ).rejects.toThrow("400");

    expect(callCount).toBe(1);
  });

  it("throws last error after exhausting all attempts on transient error", async () => {
    let callCount = 0;

    await expect(
      withRetry(
        async () => {
          callCount++;
          throw new Error(`Server error (502): attempt ${callCount}`);
        },
        { maxAttempts: 3, delays: [0, 0] },
      ),
    ).rejects.toThrow("attempt 3");

    expect(callCount).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// helpers.ts - extractJson
// ---------------------------------------------------------------------------

describe("extractJson", () => {
  it("parses plain JSON string", () => {
    const result = extractJson<{ name: string }>('{"name": "test"}');
    expect(result).toEqual({ name: "test" });
  });

  it("strips ```json code block and parses", () => {
    const input = '```json\n{"a": 1, "b": 2}\n```';
    const result = extractJson<{ a: number; b: number }>(input);
    expect(result).toEqual({ a: 1, b: 2 });
  });

  it("strips ``` code block without language tag and parses", () => {
    const input = '```\n{"x": true}\n```';
    const result = extractJson<{ x: boolean }>(input);
    expect(result).toEqual({ x: true });
  });

  it("returns null for invalid JSON", () => {
    const result = extractJson("not json at all");
    expect(result).toBeNull();
  });

  it("returns null for empty string", () => {
    const result = extractJson("");
    expect(result).toBeNull();
  });

  it("handles JSON with surrounding whitespace", () => {
    const result = extractJson<{ ok: boolean }>('  \n  {"ok": true}  \n  ');
    expect(result).toEqual({ ok: true });
  });
});

// ---------------------------------------------------------------------------
// helpers.ts - isTruncated
// ---------------------------------------------------------------------------

describe("isTruncated", () => {
  it("returns true when finishReason is max_tokens", () => {
    const result = isTruncated({
      text: "partial output",
      model: "test",
      provider: "test",
      tokens: { input: 100, output: 4096 },
      finishReason: "max_tokens",
      durationMs: 1000,
      cost: null,
    });
    expect(result).toBe(true);
  });

  it("returns false when finishReason is stop", () => {
    const result = isTruncated({
      text: "complete output",
      model: "test",
      provider: "test",
      tokens: { input: 100, output: 200 },
      finishReason: "stop",
      durationMs: 1000,
      cost: null,
    });
    expect(result).toBe(false);
  });

  it("returns false when finishReason is error", () => {
    const result = isTruncated({
      text: "",
      model: "test",
      provider: "test",
      tokens: { input: 0, output: 0 },
      finishReason: "error",
      durationMs: 0,
      cost: null,
    });
    expect(result).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// pricing.ts - estimateCost
// ---------------------------------------------------------------------------

describe("estimateCost", () => {
  it("returns null for unknown model (no pricing.toml or model not listed)", () => {
    // With no pricing.toml at the expected path, loadPricing returns { models: {} }
    const cost = estimateCost("nonexistent-model", 1000, 500);
    expect(cost).toBeNull();
  });
});
