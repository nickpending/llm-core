/**
 * lib/embed.ts - embed() client function
 *
 * Calls the embed server via HTTP. No provider adapter dispatch —
 * the embed server is a known HTTP protocol, not an LLM provider.
 * Uses resolveService() to locate the embed endpoint from services.toml.
 *
 * Usage:
 *   import { embed } from "./embed";
 *   const result = await embed({ text: "hello", prefix: "search_query" });
 */

import { resolveService } from "./services";
import type { EmbedOptions, EmbedResult } from "./types";

export async function embed(options: EmbedOptions): Promise<EmbedResult> {
  const startTime = Date.now();

  // 1. Resolve service configuration — bypasses adapter dispatch entirely
  const service = resolveService(options.service ?? "embed");
  const baseUrl = service.base_url;

  // 2. POST to embed endpoint
  let resp: Response;
  try {
    resp = await fetch(`${baseUrl}/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: options.text,
        prefix: options.prefix ?? "search_query",
      }),
      signal: AbortSignal.timeout(5000),
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new Error(`Embed server not running at ${baseUrl}. Start with: llm embed-server start`);
    }
    if (err instanceof TypeError && (err as NodeJS.ErrnoException).code === "ECONNREFUSED") {
      throw new Error(`Embed server not running at ${baseUrl}. Start with: llm embed-server start`);
    }
    throw new Error(`Embed server not running at ${baseUrl}. Start with: llm embed-server start`);
  }

  // 3. Validate response
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`Embed server returned ${resp.status}: ${body}`);
  }

  const data = (await resp.json()) as {
    embedding?: number[];
    dims?: number;
    durationMs?: number;
  };

  if (!Array.isArray(data.embedding)) {
    throw new Error("Embed server returned invalid response: missing embedding array");
  }

  // 4. Return normalized envelope
  const durationMs = Date.now() - startTime;

  return {
    embedding: data.embedding,
    dims: data.dims ?? data.embedding.length,
    durationMs,
  };
}
