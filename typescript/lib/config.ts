/**
 * lib/config.ts - API key loading via apiconf
 *
 * Wraps @voidwire/apiconf's getKey() with service-aware error handling.
 * Translates apiconf errors into actionable messages for the user.
 *
 * Usage:
 *   import { loadApiKey } from "./config";
 *   const key = await loadApiKey(service);
 */

import {
  getKey,
  KeyNotFoundError,
  ConfigNotFoundError,
} from "@voidwire/apiconf";
import type { ServiceConfig } from "./types";

/**
 * Load the API key for a service using apiconf.
 *
 * Returns null if the service does not require a key (e.g., ollama).
 * Throws with actionable error messages for missing keys or config.
 *
 * Note: getKey() is synchronous internally, but loadApiKey() is async
 * for API consistency with the rest of the library.
 */
export async function loadApiKey(
  service: ServiceConfig,
): Promise<string | null> {
  // Services that don't require keys (e.g., ollama)
  if (service.key_required === false) {
    return null;
  }

  // Missing key field when key is required
  if (!service.key) {
    throw new Error(
      `Service requires an API key but no "key" field is configured. ` +
        `Add a "key" field pointing to an apiconf key name.`,
    );
  }

  try {
    return getKey(service.key);
  } catch (err) {
    if (err instanceof KeyNotFoundError) {
      throw new Error(
        `API key "${err.keyName}" not found in apiconf. ` +
          `Available keys: [${err.available.join(", ")}]. ` +
          `Add it to ~/.config/apiconf/config.toml`,
      );
    }

    if (err instanceof ConfigNotFoundError) {
      throw new Error(
        `apiconf config not found at ${err.path}. ` +
          `Create ~/.config/apiconf/config.toml with your API keys. ` +
          `See: https://github.com/nickpending/apiconf`,
      );
    }

    // Re-throw unexpected errors
    throw err;
  }
}
