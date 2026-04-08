/**
 * lib/providers/openai.ts - OpenAI Chat Completions API adapter
 *
 * Converts AdapterRequest to OpenAI's /chat/completions endpoint format
 * and normalizes the response into AdapterResponse.
 *
 * Reference: packages/llm-summarize/index.ts callOpenAI()
 */

import type { AdapterRequest, AdapterResponse } from "../types";

export async function complete(req: AdapterRequest): Promise<AdapterResponse> {
  const body = {
    model: req.model,
    messages: [
      ...(req.systemPrompt
        ? [{ role: "system", content: req.systemPrompt }]
        : []),
      { role: "user", content: req.prompt },
    ],
    ...(req.maxTokens && { max_tokens: req.maxTokens }),
    ...(req.temperature !== undefined && { temperature: req.temperature }),
    ...(req.json && { response_format: { type: "json_object" } }),
  };

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (req.apiKey) {
    headers.Authorization = `Bearer ${req.apiKey}`;
  }
  if (req.appTitle) {
    headers["X-Title"] = req.appTitle;
  }
  if (req.appUrl) {
    headers["HTTP-Referer"] = req.appUrl;
  }

  const response = await fetch(`${req.baseUrl}/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error (${response.status}): ${error}`);
  }

  const data = await response.json();

  // Validate response shape
  if (!data.choices?.length || !data.choices[0].message) {
    throw new Error(
      `OpenAI API returned unexpected response shape: missing choices[0].message`,
    );
  }

  const choice = data.choices[0];

  // Map finish_reason to normalized finishReason
  let finishReason: "stop" | "max_tokens" | "error" = "stop";
  if (choice.finish_reason === "length") {
    finishReason = "max_tokens";
  } else if (choice.finish_reason === "stop") {
    finishReason = "stop";
  }

  return {
    text: choice.message.content, // INV-001: verbatim extraction
    model: data.model,
    tokensInput: data.usage?.prompt_tokens ?? 0,
    tokensOutput: data.usage?.completion_tokens ?? 0,
    finishReason,
  };
}

export async function healthCheck(
  baseUrl: string,
  apiKey: string | null,
): Promise<void> {
  const response = await fetch(`${baseUrl}/models`, {
    headers: { Authorization: `Bearer ${apiKey ?? ""}` },
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(
      `OpenAI health check failed (${response.status}): ${error}`,
    );
  }
}
