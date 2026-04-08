/**
 * lib/pricing.ts - Cost estimation from ~/.config/llm-core/pricing.toml
 *
 * Loads pricing rates ($/1M tokens) and estimates cost from token counts.
 * Pricing is best-effort: missing file or unknown model returns null.
 *
 * pricing.toml format:
 *   [models."claude-3-5-sonnet-20241022"]
 *   input = 3.00
 *   output = 15.00
 *
 * Usage:
 *   import { estimateCost } from "./pricing";
 *   const cost = estimateCost("claude-3-5-sonnet-20241022", 1000, 500);
 */

import { readFileSync, existsSync } from "fs";
import { parse as parseToml } from "smol-toml";
import { join } from "path";
import { homedir } from "os";

const CONFIG_DIR = join(homedir(), ".config", "llm-core");
const PRICING_PATH = join(CONFIG_DIR, "pricing.toml");

interface PricingRates {
  models: Record<string, { input: number; output: number }>;
}

let cachedPricing: PricingRates | null = null;

/** Reset cached pricing — test use only. */
export function _resetPricingCache(): void {
  cachedPricing = null;
}

/**
 * Load pricing.toml if it exists, otherwise return empty rates.
 * Pricing is best-effort — if file missing or corrupt, cost returns null.
 * Caches the result for subsequent calls (same pattern as services.ts).
 */
function loadPricing(): PricingRates {
  if (cachedPricing) return cachedPricing;

  if (!existsSync(PRICING_PATH)) {
    cachedPricing = { models: {} };
    return cachedPricing;
  }

  try {
    const content = readFileSync(PRICING_PATH, "utf-8");
    cachedPricing = parseToml(content) as unknown as PricingRates;
    return cachedPricing;
  } catch {
    cachedPricing = { models: {} };
    return cachedPricing;
  }
}

/**
 * Estimate cost in USD from token counts and model name.
 * Returns null if pricing data unavailable for the model.
 * Rates in pricing.toml are per 1M tokens.
 */
export function estimateCost(
  model: string,
  tokensInput: number,
  tokensOutput: number,
): number | null {
  const pricing = loadPricing();
  const rates = pricing.models[model];

  if (!rates) {
    return null; // Unknown model, can't estimate
  }

  // Rates are per 1M tokens
  const inputCost = (tokensInput / 1_000_000) * rates.input;
  const outputCost = (tokensOutput / 1_000_000) * rates.output;

  return inputCost + outputCost;
}

/**
 * Fetch pricing data from LiteLLM community database and update pricing.toml.
 * Returns count of models updated.
 *
 * NOTE: Implementation deferred -- LiteLLM JSON format needs investigation.
 * For now, pricing.toml must be manually populated.
 */
export async function updatePricing(): Promise<{ updated: number }> {
  throw new Error(
    "updatePricing() not yet implemented - manual pricing.toml population required",
  );
}
