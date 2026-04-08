/**
 * lib/providers/anthropic.ts - Anthropic Messages API adapter
 *
 * Converts AdapterRequest to Anthropic's /messages endpoint format
 * and normalizes the response into AdapterResponse.
 *
 * Reference: packages/llm-summarize/index.ts callAnthropic()
 */

import type { AdapterRequest, AdapterResponse } from "../types";

/** Anthropic API version — pinned per https://docs.anthropic.com/en/api/versioning */
const ANTHROPIC_API_VERSION = "2023-06-01";

/** Anthropic requires max_tokens (unlike OpenAI/Ollama). Default if caller doesn't specify. */
const DEFAULT_MAX_TOKENS = 8192;

export async function complete(req: AdapterRequest): Promise<AdapterResponse> {
  const body = {
    model: req.model,
    max_tokens: req.maxTokens || DEFAULT_MAX_TOKENS,
    messages: [{ role: "user", content: req.prompt }],
    ...(req.systemPrompt && { system: req.systemPrompt }),
    ...(req.temperature !== undefined && { temperature: req.temperature }),
  };

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "anthropic-version": ANTHROPIC_API_VERSION,
  };
  if (req.apiKey) {
    headers["x-api-key"] = req.apiKey;
  }

  const response = await fetch(`${req.baseUrl}/messages`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Anthropic API error (${response.status}): ${error}`);
  }

  const data = await response.json();

  // Validate response shape
  if (!data.content?.length || typeof data.content[0].text !== "string") {
    throw new Error(`Anthropic API returned unexpected response shape: missing content[0].text`);
  }

  // Map stop_reason to normalized finishReason
  let finishReason: "stop" | "max_tokens" | "error" = "stop";
  if (data.stop_reason === "max_tokens" || data.stop_reason === "end_turn") {
    finishReason = data.stop_reason === "max_tokens" ? "max_tokens" : "stop";
  }

  return {
    text: data.content[0].text, // INV-001: verbatim extraction
    model: data.model,
    tokensInput: data.usage?.input_tokens ?? 0,
    tokensOutput: data.usage?.output_tokens ?? 0,
    finishReason,
  };
}

export async function healthCheck(baseUrl: string, apiKey: string | null): Promise<void> {
  const response = await fetch(`${baseUrl}/models`, {
    headers: {
      "x-api-key": apiKey ?? "",
      "anthropic-version": ANTHROPIC_API_VERSION,
    },
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Anthropic health check failed (${response.status}): ${error}`);
  }
}
