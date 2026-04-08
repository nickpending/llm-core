/**
 * tests/config.test.ts
 *
 * Tests for loadApiKey() — the credential loading layer.
 *
 * loadApiKey() takes a plain ServiceConfig object and calls @voidwire/apiconf's
 * getKey(). We mock apiconf to test error handling without touching real config.
 *
 * loadApiKey() is pure with respect to filesystem (no direct fs calls) —
 * all fs access goes through apiconf. Mocking apiconf covers all error paths.
 */

import { describe, expect, it, mock } from "bun:test";
import { join } from "node:path";
import { ConfigNotFoundError, KeyNotFoundError } from "@voidwire/apiconf";
import type { ServiceConfig } from "../lib/types.ts";

// Mock apiconf BEFORE importing config.ts
mock.module("@voidwire/apiconf", () => ({
  getKey: (keyName: string) => {
    if (keyName === "valid-key") return "sk-test-1234";
    if (keyName === "missing-key") throw new KeyNotFoundError("missing-key", ["other-key"]);
    if (keyName === "no-config") throw new ConfigNotFoundError("/fake/path/config.toml");
    throw new Error("Unexpected key name in test");
  },
  KeyNotFoundError,
  ConfigNotFoundError,
}));

const { loadApiKey } = await import(join(import.meta.dir, "../lib/config.ts"));

describe("loadApiKey()", () => {
  it("returns null when key_required is false (ollama pattern)", async () => {
    const service: ServiceConfig = {
      adapter: "ollama",
      base_url: "http://localhost:11434",
      key_required: false,
    };
    const result = await loadApiKey(service);
    expect(result).toBeNull();
  });

  it("throws with clear message when key field is missing but required", async () => {
    const service: ServiceConfig = {
      adapter: "anthropic",
      base_url: "https://api.anthropic.com/v1",
      // key field intentionally omitted
    };
    await expect(loadApiKey(service)).rejects.toThrow(
      'Service requires an API key but no "key" field is configured',
    );
  });

  it("returns key string when apiconf resolves successfully", async () => {
    const service: ServiceConfig = {
      adapter: "anthropic",
      key: "valid-key",
      base_url: "https://api.anthropic.com/v1",
    };
    const result = await loadApiKey(service);
    expect(result).toBe("sk-test-1234");
  });

  it("throws with key name and instructions when apiconf key not found", async () => {
    const service: ServiceConfig = {
      adapter: "anthropic",
      key: "missing-key",
      base_url: "https://api.anthropic.com/v1",
    };
    await expect(loadApiKey(service)).rejects.toThrow('"missing-key"');
    await expect(loadApiKey(service)).rejects.toThrow("apiconf");
  });
});
