#!/usr/bin/env bun
/**
 * llm-core CLI - Thin wrapper around complete()
 *
 * Usage:
 *   llm-core "prompt text" [options]
 *   llm-core --list-services
 *
 * Output: JSON to stdout, diagnostics to stderr
 * Exit codes: 0 = success, 1 = API/runtime error, 2 = client error
 */

import { complete, listServices } from "./index";

interface ParsedArgs {
  prompt?: string;
  service?: string;
  model?: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  json?: boolean;
  listServices?: boolean;
}

function printUsage(): void {
  process.stderr.write(`llm-core - Shared LLM transport layer

Usage:
  llm-core "prompt text" [options]
  llm-core --list-services

Options:
  --service NAME       Service name from services.toml (default: from config)
  --model NAME         Model name (uses service default_model if omitted)
  --system PROMPT      System prompt
  --temperature NUM    0-1 (default: provider default)
  --max-tokens NUM     Max output tokens
  --json               Request JSON output mode
  --list-services      List configured services

Examples:
  llm-core "hello" --service ollama --model llama3
  llm-core "summarize this" --service anthropic --model claude-3-5-haiku-20241022
  llm-core --list-services

Output: JSON to stdout, diagnostics to stderr
Exit codes: 0 = success, 1 = API/runtime error, 2 = client error
`);
}

function parseArgs(argv: string[]): ParsedArgs | null {
  const args = argv.slice(2);
  const parsed: ParsedArgs = {};

  if (args.length === 0) {
    return null;
  }

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === "--list-services") {
      parsed.listServices = true;
    } else if (arg === "--service") {
      parsed.service = args[++i];
    } else if (arg === "--model") {
      parsed.model = args[++i];
    } else if (arg === "--system") {
      parsed.systemPrompt = args[++i];
    } else if (arg === "--temperature") {
      parsed.temperature = parseFloat(args[++i]);
    } else if (arg === "--max-tokens") {
      parsed.maxTokens = parseInt(args[++i], 10);
    } else if (arg === "--json") {
      parsed.json = true;
    } else if (!arg.startsWith("--")) {
      parsed.prompt = arg;
    }
  }

  return parsed;
}

async function main(): Promise<void> {
  const parsed = parseArgs(process.argv);

  if (!parsed) {
    printUsage();
    process.exit(0);
  }

  try {
    if (parsed.listServices) {
      const services = listServices();
      console.log(JSON.stringify(services, null, 2));
      process.exit(0);
    }

    if (!parsed.prompt) {
      process.stderr.write("Error: Prompt required\n\n");
      printUsage();
      process.exit(2);
    }

    const result = await complete({
      prompt: parsed.prompt,
      service: parsed.service,
      model: parsed.model,
      systemPrompt: parsed.systemPrompt,
      temperature: parsed.temperature,
      maxTokens: parsed.maxTokens,
      json: parsed.json,
    });

    console.log(JSON.stringify(result, null, 2));
    process.exit(0);
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    console.log(JSON.stringify({ error }, null, 2));
    process.stderr.write(`Error: ${error}\n`);
    process.exit(1);
  }
}

main();
