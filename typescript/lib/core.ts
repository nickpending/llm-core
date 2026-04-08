/**
 * lib/core.ts - complete() orchestration
 *
 * Wires together service resolution, credential loading, provider adapters,
 * retry logic, and cost estimation into a single function call.
 *
 * Usage:
 *   import { complete } from "./core";
 *   const result = await complete({ prompt: "hello", service: "ollama", model: "llama3" });
 */

import { loadApiKey } from "./config";
import { estimateCost } from "./pricing";
import * as anthropic from "./providers/anthropic";
import { getAdapter } from "./providers/index";
import * as ollama from "./providers/ollama";
import * as openai from "./providers/openai";
import { withRetry } from "./retry";
import { resolveService } from "./services";
import type { CompleteOptions, CompleteResult } from "./types";

export async function complete(options: CompleteOptions): Promise<CompleteResult> {
  const startTime = Date.now();

  // 1. Resolve service configuration
  const service = resolveService(options.service);

  // 2. Load API key (if required)
  const apiKey = await loadApiKey(service);

  // 3. Get provider adapter
  const adapter = getAdapter(service.adapter);

  // 4. Build adapter request — caller model > service default_model
  const model = options.model || service.default_model || "";
  if (!model) {
    throw new Error(
      "Model name required: pass model in CompleteOptions or set default_model in services.toml",
    );
  }

  const adapterRequest = {
    baseUrl: service.base_url,
    apiKey,
    model,
    prompt: options.prompt,
    systemPrompt: options.systemPrompt,
    temperature: options.temperature,
    maxTokens: options.maxTokens,
    json: options.json,
  };

  // 5. Call provider with retry
  const response = await withRetry(() => adapter.complete(adapterRequest));

  // 6. Estimate cost
  const cost = estimateCost(response.model, response.tokensInput, response.tokensOutput);

  // 7. Return normalized envelope
  const durationMs = Date.now() - startTime;

  return {
    text: response.text,
    model: response.model,
    provider: service.adapter,
    tokens: {
      input: response.tokensInput,
      output: response.tokensOutput,
    },
    finishReason: response.finishReason,
    durationMs,
    cost,
  };
}

export async function healthCheck(service?: string): Promise<void> {
  const svc = resolveService(service);
  const apiKey = await loadApiKey(svc);

  if (svc.adapter === "anthropic") {
    await anthropic.healthCheck(svc.base_url, apiKey);
  } else if (svc.adapter === "openai") {
    await openai.healthCheck(svc.base_url, apiKey);
  } else if (svc.adapter === "ollama") {
    await ollama.healthCheck(svc.base_url, apiKey);
  } else {
    throw new Error(`Unknown adapter: "${svc.adapter}"`);
  }
}
