/**
 * lib/services.ts - Service resolution from ~/.config/llm-core/services.toml
 *
 * Loads service configuration, generates defaults on first run,
 * and resolves named services to their config.
 *
 * Usage:
 *   import { loadServices, resolveService, listServices } from "./services";
 *   const map = loadServices();
 *   const svc = resolveService("anthropic");
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { parse as parseToml } from "smol-toml";
import type { ServiceConfig, ServiceMap } from "./types";

const CONFIG_DIR = join(homedir(), ".config", "llm-core");
const SERVICES_PATH = join(CONFIG_DIR, "services.toml");

const DEFAULT_SERVICES_TOML = `\
default_service = "anthropic"

[services.anthropic]
adapter = "anthropic"
key = "anthropic"
base_url = "https://api.anthropic.com/v1"

[services.openai]
adapter = "openai"
key = "openai"
base_url = "https://api.openai.com/v1"

[services.ollama]
adapter = "ollama"
base_url = "http://localhost:11434"
key_required = false
`;

let cachedServices: ServiceMap | null = null;

/** Reset cached services — test use only. */
export function _resetServicesCache(): void {
  cachedServices = null;
}

/**
 * Load and parse services.toml. Generates default config on first run.
 * Caches the result for subsequent calls.
 */
export function loadServices(): ServiceMap {
  if (cachedServices) return cachedServices;

  // Generate default config if missing
  if (!existsSync(SERVICES_PATH)) {
    try {
      mkdirSync(CONFIG_DIR, { recursive: true });
      writeFileSync(SERVICES_PATH, DEFAULT_SERVICES_TOML, "utf-8");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(
        `Failed to create default services.toml at ${SERVICES_PATH}: ${message}`,
      );
    }
  }

  let raw: string;
  try {
    raw = readFileSync(SERVICES_PATH, "utf-8");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to read ${SERVICES_PATH}: ${message}`);
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = parseToml(raw) as Record<string, unknown>;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to parse ${SERVICES_PATH}: ${message}`);
  }

  // Validate required fields
  if (typeof parsed.default_service !== "string") {
    throw new Error(
      `Invalid config: missing or non-string "default_service" in ${SERVICES_PATH}`,
    );
  }

  if (
    !parsed.services ||
    typeof parsed.services !== "object" ||
    Array.isArray(parsed.services)
  ) {
    throw new Error(
      `Invalid config: missing or invalid [services] section in ${SERVICES_PATH}`,
    );
  }

  const services = parsed.services as Record<string, unknown>;

  // Validate each service entry
  for (const [name, entry] of Object.entries(services)) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(
        `Invalid config: service "${name}" must be a table in ${SERVICES_PATH}`,
      );
    }
    const svc = entry as Record<string, unknown>;
    if (typeof svc.adapter !== "string") {
      throw new Error(
        `Invalid config: service "${name}" missing "adapter" field in ${SERVICES_PATH}`,
      );
    }
    if (typeof svc.base_url !== "string") {
      throw new Error(
        `Invalid config: service "${name}" missing "base_url" field in ${SERVICES_PATH}`,
      );
    }
  }

  // Validate default_service references a known service
  if (!(parsed.default_service in services)) {
    const available = Object.keys(services).join(", ");
    throw new Error(
      `Invalid config: default_service "${parsed.default_service}" not found in [services]. Available: [${available}]`,
    );
  }

  cachedServices = {
    default_service: parsed.default_service,
    services: services as Record<string, ServiceConfig>,
  };

  return cachedServices;
}

/**
 * Resolve a service by name. If name is undefined, returns the default service.
 */
export function resolveService(name?: string): ServiceConfig {
  const map = loadServices();
  const serviceName = name ?? map.default_service;

  const service = map.services[serviceName];
  if (!service) {
    const available = Object.keys(map.services).join(", ");
    throw new Error(
      `Unknown service: "${serviceName}". Available: [${available}]`,
    );
  }

  return service;
}

/**
 * List all configured service names.
 */
export function listServices(): string[] {
  const map = loadServices();
  return Object.keys(map.services);
}
