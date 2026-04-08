/**
 * llm-core - Shared LLM transport layer
 *
 * Service-based routing to provider APIs with normalized envelope.
 * Pure functions, no process.exit, no stderr output in library.
 *
 * Usage:
 *   import { complete, loadServices, listServices } from "@voidwire/llm-core";
 */

export { complete, healthCheck } from "./lib/core";
export { embed } from "./lib/embed";
export { extractJson, isTruncated } from "./lib/helpers";
export { listServices, loadServices, resolveService } from "./lib/services";
export type {
  CompleteOptions,
  CompleteResult,
  EmbedOptions,
  EmbedResult,
  ServiceConfig,
  ServiceMap,
} from "./lib/types";
