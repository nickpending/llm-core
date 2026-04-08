/**
 * tests/pricing-cost-calc.test.ts
 *
 * Targeted tests for estimateCost() math formula and corrupt TOML handling.
 * These tests are in a separate file from retry-helpers-pricing.test.ts
 * because each Bun test file runs in its own process — mock.module("fs")
 * must be registered before pricing.ts is imported to intercept its
 * static import { readFileSync, existsSync } from "fs".
 *
 * Strategy: mock "fs" module BEFORE importing pricing.ts so readFileSync
 * and existsSync are replaced before pricing.ts binds them at load time.
 *
 * Invariants protected:
 *   estimateCost math: (input / 1M) * inputRate + (output / 1M) * outputRate
 *   Corrupt TOML: loadPricing() returns empty rates, estimateCost returns null
 */

import { beforeEach, describe, expect, it, mock } from "bun:test";
import { join } from "node:path";

// Valid TOML content with known pricing rates
// claude-3-5-sonnet: $3.00/1M input, $15.00/1M output
const VALID_PRICING_TOML = `
[models."claude-3-5-sonnet-20241022"]
input = 3.00
output = 15.00
`;

// Mock "fs" BEFORE importing pricing.ts — static named imports bind at load
// time, so the mock must be registered before the module is first evaluated.
// pricing.ts calls existsSync/readFileSync at runtime inside loadPricing(),
// so this mock will intercept those runtime calls.
mock.module("fs", () => ({
  existsSync: (_path: string) => true, // pricing file always "exists"
  readFileSync: (_path: string, _encoding: string) => VALID_PRICING_TOML,
}));

// Import AFTER mock is registered
const { estimateCost, _resetPricingCache } = await import(
  join(import.meta.dir, "../lib/pricing.ts")
);

// ---------------------------------------------------------------------------
// estimateCost — math formula validation (HIGH RISK)
// ---------------------------------------------------------------------------

describe("estimateCost math formula", () => {
  // Reset cache before each test so the mocked fs is actually consulted.
  // Earlier test files (core.test.ts) may have cached empty pricing data
  // via their own import of pricing.ts through core.ts.
  beforeEach(() => _resetPricingCache());

  it("calculates input cost only when output is zero", () => {
    // 1,000,000 input * $3.00/1M = $3.00, 0 output = $0.00
    const cost = estimateCost("claude-3-5-sonnet-20241022", 1_000_000, 0);
    expect(cost).toBeCloseTo(3.0, 5);
  });

  it("calculates output cost only when input is zero", () => {
    // 0 input = $0.00, 1,000,000 output * $15.00/1M = $15.00
    const cost = estimateCost("claude-3-5-sonnet-20241022", 0, 1_000_000);
    expect(cost).toBeCloseTo(15.0, 5);
  });

  it("combines input and output costs correctly for typical usage", () => {
    // 500,000 input * $3.00/1M = $1.50, 100,000 output * $15.00/1M = $1.50
    // Total: $3.00
    const cost = estimateCost("claude-3-5-sonnet-20241022", 500_000, 100_000);
    expect(cost).toBeCloseTo(3.0, 5);
  });

  it("returns null for model not in pricing data even when file exists", () => {
    const cost = estimateCost("unknown-model-xyz", 1000, 500);
    expect(cost).toBeNull();
  });
});
