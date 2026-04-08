/**
 * tests/services.test.ts
 *
 * Integration tests for service resolution:
 *   loadServices(), resolveService(), listServices()
 *
 * Strategy: mock "os" to redirect homedir() to a temp directory before the
 * module is loaded. This ensures CONFIG_DIR and SERVICES_PATH in services.ts
 * point to our temp dir, not the real ~/.config/llm-core/.
 *
 * Each describe block resets the module cache via _resetServicesCache()
 * so tests are order-independent.
 */

import { afterAll, beforeEach, describe, expect, it, mock } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Create temp home directory BEFORE importing services.ts
const tempHome = mkdtempSync("/tmp/llm-core-test-");
const configDir = join(tempHome, ".config", "llm-core");
const servicesPath = join(configDir, "services.toml");

// Mock "os" BEFORE importing services.ts — the module evaluates
// CONFIG_DIR/SERVICES_PATH using homedir() at load time
mock.module("os", () => ({
  homedir: () => tempHome,
}));

// Import AFTER mock is registered
const { loadServices, resolveService, listServices, _resetServicesCache } = await import(
  join(import.meta.dir, "../lib/services.ts")
);

afterAll(() => {
  rmSync(tempHome, { recursive: true, force: true });
});

describe("loadServices() error path", () => {
  beforeEach(() => {
    _resetServicesCache();
    // Ensure config dir exists with invalid TOML
    mkdirSync(configDir, { recursive: true });
    writeFileSync(servicesPath, "this is [broken toml = !!!", "utf-8");
  });

  it("throws with file path in message on invalid TOML", () => {
    expect(() => loadServices()).toThrow(servicesPath);
  });
});

describe("loadServices() first run", () => {
  beforeEach(() => {
    _resetServicesCache();
    // Ensure config dir exists but no services file
    mkdirSync(configDir, { recursive: true });
    if (existsSync(servicesPath)) unlinkSync(servicesPath);
  });

  it("generates default services.toml when no file exists", () => {
    expect(existsSync(servicesPath)).toBe(false);

    const map = loadServices();

    expect(existsSync(servicesPath)).toBe(true);
    expect(map.default_service).toBe("anthropic");
    expect(typeof map.services).toBe("object");
    expect(Object.keys(map.services)).toHaveLength(3);
  });
});

describe("loadServices() cached results", () => {
  beforeEach(() => {
    _resetServicesCache();
    // Ensure default config exists
    mkdirSync(configDir, { recursive: true });
    if (existsSync(servicesPath)) unlinkSync(servicesPath);
    loadServices(); // generates defaults and populates cache
  });

  it("returns all 3 default services with correct endpoints", () => {
    const map = loadServices(); // hits cache

    expect(map.services.anthropic).toMatchObject({
      adapter: "anthropic",
      key: "anthropic",
      base_url: "https://api.anthropic.com/v1",
    });
    expect(map.services.openai).toMatchObject({
      adapter: "openai",
      key: "openai",
      base_url: "https://api.openai.com/v1",
    });
    expect(map.services.ollama).toMatchObject({
      adapter: "ollama",
      base_url: "http://localhost:11434",
      key_required: false,
    });
  });
});

describe("resolveService()", () => {
  beforeEach(() => {
    _resetServicesCache();
    mkdirSync(configDir, { recursive: true });
    if (existsSync(servicesPath)) unlinkSync(servicesPath);
    loadServices(); // populate with defaults
  });

  it("returns correct config for a named service", () => {
    const svc = resolveService("openai");
    expect(svc.adapter).toBe("openai");
    expect(svc.base_url).toBe("https://api.openai.com/v1");
  });

  it("returns default service when name is undefined", () => {
    const svc = resolveService(undefined);
    expect(svc.adapter).toBe("anthropic");
    expect(svc.base_url).toBe("https://api.anthropic.com/v1");
  });

  it("throws with helpful message listing available services for unknown name", () => {
    expect(() => resolveService("nonexistent-service")).toThrow(
      'Unknown service: "nonexistent-service"',
    );
    expect(() => resolveService("nonexistent-service")).toThrow("Available:");
  });
});

describe("listServices()", () => {
  beforeEach(() => {
    _resetServicesCache();
    mkdirSync(configDir, { recursive: true });
    if (existsSync(servicesPath)) unlinkSync(servicesPath);
    loadServices();
  });

  it("returns all configured service names", () => {
    const names = listServices();
    expect(names).toContain("anthropic");
    expect(names).toContain("openai");
    expect(names).toContain("ollama");
    expect(names).toHaveLength(3);
  });
});
