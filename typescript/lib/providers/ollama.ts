/**
 * lib/providers/ollama.ts - Ollama Generate API adapter
 *
 * Converts AdapterRequest to Ollama's /api/generate endpoint format
 * and normalizes the response into AdapterResponse.
 *
 * Note: Uses /api/generate (prompt + system fields), NOT /api/chat
 * (messages array). The generate endpoint maps directly to
 * AdapterRequest's prompt/systemPrompt structure.
 *
 * Reference: packages/llm-summarize/index.ts callOllama()
 */

import type { AdapterRequest, AdapterResponse } from "../types";

export async function complete(req: AdapterRequest): Promise<AdapterResponse> {
  const body = {
    model: req.model,
    prompt: req.prompt,
    stream: false,
    ...(req.systemPrompt && { system: req.systemPrompt }),
    options: {
      ...(req.temperature !== undefined && { temperature: req.temperature }),
      ...(req.maxTokens && { num_predict: req.maxTokens }),
    },
  };

  const response = await fetch(`${req.baseUrl}/api/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Ollama API error (${response.status}): ${error}`);
  }

  const data = await response.json();

  // Validate response shape
  if (typeof data.response !== "string") {
    throw new Error(
      `Ollama API returned unexpected response shape: missing or non-string response field`,
    );
  }

  // Map done_reason to normalized finishReason
  let finishReason: "stop" | "max_tokens" | "error" = "stop";
  if (data.done_reason === "length") {
    finishReason = "max_tokens";
  }

  return {
    text: data.response, // INV-001: verbatim extraction
    model: data.model,
    tokensInput: data.prompt_eval_count || 0,
    tokensOutput: data.eval_count || 0,
    finishReason,
  };
}

export async function healthCheck(baseUrl: string, _apiKey: string | null): Promise<void> {
  const response = await fetch(`${baseUrl}/api/tags`);
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Ollama health check failed (${response.status}): ${error}`);
  }
}
