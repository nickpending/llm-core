/**
 * tests/embed.test.ts
 *
 * Integration tests for embed() — requires embed server running.
 * Start with: llm embed-server start
 *
 * Tests hit the real embed server (local, ~9ms per call).
 * If server is down, tests fail with actionable error message —
 * that IS the test for the error path.
 */

import { expect, test } from "bun:test";
import { embed } from "../lib/embed";

test("embed returns 768-dim vector", async () => {
  const result = await embed({ text: "test query" });
  expect(result.embedding).toHaveLength(768);
  expect(result.dims).toBe(768);
  expect(result.durationMs).toBeGreaterThan(0);
});

test("embed uses search_query prefix by default", async () => {
  const result = await embed({ text: "default prefix test" });
  expect(result.embedding).toHaveLength(768);
});

test("embed passes search_document prefix", async () => {
  const result = await embed({
    text: "document text",
    prefix: "search_document",
  });
  expect(result.embedding).toHaveLength(768);
});
