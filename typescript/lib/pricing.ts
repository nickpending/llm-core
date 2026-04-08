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

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const CONFIG_DIR = join(homedir(), ".config", "llm-core");
const PRICING_PATH = join(CONFIG_DIR, "pricing.toml");
const PRICING_URL =
  "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

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
    cachedPricing = Bun.TOML.parse(content) as unknown as PricingRates;
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
 * Filters to models with non-zero input and output rates, converts per-token
 * rates to per-1M-token rates, and writes TOML to ~/.config/llm-core/pricing.toml.
 * Returns count of models written.
 */
export async function updatePricing(): Promise<{ updated: number }> {
  const response = await fetch(PRICING_URL);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch pricing data: ${response.status} ${response.statusText}`,
    );
  }

  const data = (await response.json()) as Record<
    string,
    Record<string, unknown>
  >;

  const lines: string[] = [];
  let count = 0;

  for (const [model, entry] of Object.entries(data)) {
    if (!entry || typeof entry !== "object") continue;

    const inputRate = Number(entry.input_cost_per_token);
    const outputRate = Number(entry.output_cost_per_token);

    if (
      isNaN(inputRate) ||
      isNaN(outputRate) ||
      inputRate <= 0 ||
      outputRate <= 0
    )
      continue;

    const inputPer1M =
      Math.round(inputRate * 1_000_000 * 1_000_000) / 1_000_000;
    const outputPer1M =
      Math.round(outputRate * 1_000_000 * 1_000_000) / 1_000_000;

    const safeName = model.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
    lines.push(`[models."${safeName}"]`);
    lines.push(`input = ${inputPer1M}`);
    lines.push(`output = ${outputPer1M}`);
    lines.push("");
    count++;
  }

  mkdirSync(CONFIG_DIR, { recursive: true });
  writeFileSync(PRICING_PATH, lines.join("\n"), "utf-8");
  cachedPricing = null;

  return { updated: count };
}
