/**
 * lib/helpers.ts - Opt-in helper functions for completion results
 *
 * extractJson(): Strip markdown code blocks and parse JSON.
 * isTruncated(): Check if completion hit max_tokens.
 *
 * These are convenience utilities — callers opt in by importing them.
 * They are NOT used internally by complete().
 *
 * Usage:
 *   import { extractJson, isTruncated } from "@voidwire/llm-core";
 */

import type { CompleteResult } from "./types";

/**
 * Extract JSON from text, stripping markdown code blocks if present.
 * Returns null if parsing fails.
 */
export function extractJson<T>(text: string): T | null {
  let clean = text.trim();

  // Strip markdown code blocks: ```json ... ``` or ``` ... ```
  const codeBlockMatch = clean.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (codeBlockMatch) {
    clean = codeBlockMatch[1].trim();
  }

  try {
    return JSON.parse(clean) as T;
  } catch {
    return null;
  }
}

/**
 * Check if completion was truncated due to max_tokens.
 */
export function isTruncated(result: CompleteResult): boolean {
  return result.finishReason === "max_tokens";
}
