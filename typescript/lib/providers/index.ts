/**
 * lib/providers/index.ts - Adapter registry
 *
 * Maps adapter names to their provider modules.
 * Used by the core orchestration layer to route requests.
 */

import type { ProviderAdapter } from "../types";
import * as anthropic from "./anthropic";
import * as ollama from "./ollama";
import * as openai from "./openai";

const ADAPTERS: Record<string, ProviderAdapter> = {
  anthropic,
  openai,
  ollama,
};

export function getAdapter(name: string): ProviderAdapter {
  const adapter = ADAPTERS[name];
  if (!adapter) {
    throw new Error(`Unknown adapter: "${name}". Available: ${Object.keys(ADAPTERS).join(", ")}`);
  }
  return adapter;
}
