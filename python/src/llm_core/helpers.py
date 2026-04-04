"""Opt-in helper functions for completion results.

extractJson(): Strip markdown code blocks and parse JSON.
isTruncated(): Check if completion hit max_tokens.

These are convenience utilities -- callers opt in by importing them.
They are NOT used internally by complete().
"""

from __future__ import annotations

import json
import re

from .types import CompleteResult


def extract_json(text: str) -> dict | None:  # type: ignore[type-arg]
    """Extract JSON from text, stripping markdown code blocks if present.

    Returns None if parsing fails.
    """
    clean = text.strip()

    # Strip markdown code blocks: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", clean)
    if match:
        clean = match.group(1).strip()

    try:
        return json.loads(clean)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


def is_truncated(result: CompleteResult) -> bool:
    """Check if completion was truncated due to max_tokens."""
    return result.finish_reason == "max_tokens"
