// Core API surface
export interface CompleteOptions {
  prompt: string;
  service?: string; // Named service from services.toml
  model?: string; // Model name
  systemPrompt?: string; // System instructions
  temperature?: number; // 0-1
  maxTokens?: number; // Max output tokens
  /** Hint for JSON output mode. Only effective for OpenAI (response_format). Silently ignored by Anthropic and Ollama. */
  json?: boolean;
}

export interface CompleteResult {
  text: string; // Raw model output — no interpretation
  model: string; // Actual model that ran
  provider: string; // Which adapter handled it
  tokens: { input: number; output: number };
  finishReason: "stop" | "max_tokens" | "error";
  durationMs: number;
  cost: number | null; // Estimated USD, null for local/unknown
}

// Service configuration
export interface ServiceConfig {
  adapter: string;
  key?: string; // apiconf key name
  base_url: string;
  key_required?: boolean; // Default true, false for ollama
  default_model?: string; // Used when caller doesn't specify a model
  app_title?: string; // Sent as X-Title header (OpenRouter app attribution)
  app_url?: string; // Sent as HTTP-Referer header (OpenRouter app attribution)
}

export interface ServiceMap {
  default_service: string;
  services: Record<string, ServiceConfig>;
}

// Provider adapter interface
export interface AdapterRequest {
  baseUrl: string;
  apiKey: string | null;
  model: string;
  prompt: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  json?: boolean;
  appTitle?: string; // Sent as X-Title header (OpenRouter app attribution)
  appUrl?: string; // Sent as HTTP-Referer header (OpenRouter app attribution)
}

export interface AdapterResponse {
  text: string;
  model: string;
  tokensInput: number;
  tokensOutput: number;
  finishReason: "stop" | "max_tokens" | "error";
}

export interface ProviderAdapter {
  complete(request: AdapterRequest): Promise<AdapterResponse>;
}

// Embedding API surface
export interface EmbedOptions {
  text: string;
  prefix?: "search_query" | "search_document"; // default: "search_query"
  service?: string; // Named service from services.toml, default: "embed"
}

export interface EmbedResult {
  embedding: number[];
  dims: number;
  durationMs: number;
}
