/**
 * lib/retry.ts - Exponential backoff retry with transient error classification
 *
 * Wraps async functions with retry logic. Only retries on transient errors
 * (429, 5xx, network failures). Non-transient errors (400, 401, 403, 404)
 * fail immediately.
 *
 * Error format assumption: provider adapters throw errors with HTTP status
 * codes in parentheses, e.g. "Anthropic API error (429): rate limited".
 * isTransientError() checks for `(${code})` in error.message to match
 * this format (see lib/providers/anthropic.ts, openai.ts, ollama.ts).
 *
 * Usage:
 *   import { withRetry } from "./retry";
 *   const result = await withRetry(() => adapter.complete(req));
 */

export interface RetryOptions {
  maxAttempts: number;
  delays: number[]; // Backoff delays in ms
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxAttempts: 3,
  delays: [1000, 2000, 4000],
};

function isTransientError(error: unknown): boolean {
  // Network errors (fetch failures, DNS, connection refused)
  if (error instanceof TypeError) return true;

  // HTTP status code errors — check for (status) pattern in message
  if (error instanceof Error && error.message) {
    const msg = error.message;
    // Transient HTTP status codes: 429 (rate limit), 5xx (server errors)
    const transientCodes = [429, 500, 502, 503, 504];
    for (const code of transientCodes) {
      if (msg.includes(`(${code})`)) return true;
    }
  }

  return false;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  options: Partial<RetryOptions> = {},
): Promise<T> {
  const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastError: unknown;

  for (let attempt = 0; attempt < opts.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;

      // Don't retry non-transient errors
      if (!isTransientError(err)) {
        throw err;
      }

      // Don't delay after last attempt
      if (attempt < opts.maxAttempts - 1) {
        await sleep(
          opts.delays[attempt] || opts.delays[opts.delays.length - 1],
        );
      }
    }
  }

  throw lastError;
}
