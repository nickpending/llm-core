/**
 * lib/providers/claude-cli.ts - Claude Code subscription adapter via `claude --print`
 *
 * Routes inference through the user's Claude Code subscription by spawning the
 * claude CLI as a subprocess. Zero per-call API cost — billed against the
 * subscription's cap.
 *
 * Why this isn't the anthropic adapter with a different base_url:
 *   The Anthropic Messages API requires an API key and bills per-token.
 *   Claude Code's subscription auth is OAuth-bound to the local CLI install
 *   and is only honored by the `claude` binary, not by HTTP requests to
 *   api.anthropic.com.
 *
 * Subscription auth precondition:
 *   ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, and CLAUDECODE are scrubbed from
 *   the child env so the CLI falls through to its OAuth credentials. See
 *   ~/.claude/PAI/TOOLS/Inference.ts for the canonical pattern.
 *
 * Flag lockdown — these aren't optional:
 *   --system-prompt           replace (not append) — the caller owns the system prompt
 *   --tools ''                no tool execution mid-completion (pure inference)
 *   --setting-sources ''      ignore local settings.json (deterministic env)
 *   --output-format json      structured envelope so we can extract real token counts
 *   --exclude-dynamic-system-prompt-sections   cache-friendly prompt prefix
 */

import { spawn } from "node:child_process";
import type { AdapterRequest, AdapterResponse } from "../types";

/**
 * Default subprocess timeout. claude --print can stall on auth issues or large
 * prompts; withRetry covers transient failures but won't bound a hung process.
 */
const DEFAULT_TIMEOUT_MS = 120_000;

/** Shape of the JSON envelope produced by `claude --print --output-format json`. */
interface ClaudeJsonResult {
  type: string;
  subtype?: string;
  is_error?: boolean;
  result?: string;
  stop_reason?: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  modelUsage?: Record<string, unknown>;
}

/**
 * Parse the JSON envelope from `claude --print --output-format json` into an
 * AdapterResponse. Pure function — extracted for unit testing without spawn.
 *
 * INV-001: result text is verbatim — no trim, no rewrite.
 */
export function parseClaudeJsonResponse(
  stdout: string,
  fallbackModel: string,
): AdapterResponse {
  let data: ClaudeJsonResult;
  try {
    data = JSON.parse(stdout) as ClaudeJsonResult;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(
      `claude --print returned non-JSON output (--output-format json expected): ${message}`,
    );
  }

  if (data.is_error || data.subtype !== "success") {
    throw new Error(
      `claude --print returned error envelope: ${JSON.stringify({
        subtype: data.subtype,
        result: data.result,
      })}`,
    );
  }

  if (typeof data.result !== "string") {
    throw new Error(
      `claude --print envelope missing result field (got ${typeof data.result})`,
    );
  }

  // The real resolved model is the first key of modelUsage (e.g.
  // "claude-haiku-4-5-20251001"). Falls back to the alias the caller supplied
  // if the envelope shape ever changes — no fabricated model strings.
  const resolvedModel =
    data.modelUsage && Object.keys(data.modelUsage).length > 0
      ? Object.keys(data.modelUsage)[0]
      : fallbackModel;

  let finishReason: "stop" | "max_tokens" | "error" = "stop";
  if (data.stop_reason === "max_tokens") {
    finishReason = "max_tokens";
  }

  return {
    text: data.result,
    model: resolvedModel,
    tokensInput: data.usage?.input_tokens ?? 0,
    tokensOutput: data.usage?.output_tokens ?? 0,
    finishReason,
  };
}

/**
 * Build the env passed to the child process. Subscription auth wins only when
 * the API-key envs are absent and CLAUDECODE is unset (would otherwise trip
 * the nested-session guard).
 */
function buildChildEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.CLAUDECODE;
  delete env.ANTHROPIC_API_KEY;
  delete env.ANTHROPIC_AUTH_TOKEN;
  return env;
}

export async function complete(req: AdapterRequest): Promise<AdapterResponse> {
  const args = [
    "--print",
    "--model",
    req.model,
    "--output-format",
    "json",
    "--tools",
    "",
    "--setting-sources",
    "",
    "--exclude-dynamic-system-prompt-sections",
  ];

  if (req.systemPrompt) {
    args.push("--system-prompt", req.systemPrompt);
  }

  // req.maxTokens, req.temperature, req.json: not exposed by claude --print.
  // Silently ignored (matches anthropic-adapter parity for unsupported fields).

  return new Promise((resolve, reject) => {
    const proc = spawn("claude", args, {
      env: buildChildEnv(),
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      proc.kill("SIGKILL");
      reject(
        new Error(`claude --print timed out after ${DEFAULT_TIMEOUT_MS}ms`),
      );
    }, DEFAULT_TIMEOUT_MS);

    proc.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    proc.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      // ENOENT: claude binary not on PATH — surface a clear remediation hint.
      const code = (err as NodeJS.ErrnoException).code;
      if (code === "ENOENT") {
        reject(
          new Error(
            `claude CLI not found on PATH. Install Claude Code (https://claude.com/code) or remove the claude-cli service from services.toml.`,
          ),
        );
        return;
      }
      reject(err);
    });

    proc.on("close", (exitCode) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (exitCode !== 0) {
        reject(
          new Error(`claude --print exited ${exitCode}: ${stderr.trim()}`),
        );
        return;
      }
      try {
        resolve(parseClaudeJsonResponse(stdout, req.model));
      } catch (err) {
        reject(err);
      }
    });

    proc.stdin.write(req.prompt);
    proc.stdin.end();
  });
}

/**
 * Health check: invoke `claude --version` and confirm the binary responds.
 * Doesn't validate subscription auth — that surfaces on the first complete() call.
 */
export async function healthCheck(
  _baseUrl: string,
  _apiKey: string | null,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn("claude", ["--version"], {
      env: buildChildEnv(),
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stderr = "";
    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    proc.on("error", (err) => {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === "ENOENT") {
        reject(new Error("claude CLI not found on PATH"));
        return;
      }
      reject(err);
    });
    proc.on("close", (exitCode) => {
      if (exitCode !== 0) {
        reject(
          new Error(`claude --version exited ${exitCode}: ${stderr.trim()}`),
        );
        return;
      }
      resolve();
    });
  });
}
