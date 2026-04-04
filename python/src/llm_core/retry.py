"""Retry logic with transient error classification.

Wraps synchronous functions with retry logic. Only retries on transient errors
(429, 5xx, network failures). Non-transient errors (400, 401, 403, 404)
fail immediately.

Python adaptation: uses time.sleep() (sync), typed ProviderError with status_code
instead of string-matching error messages.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import httpx

from .exceptions import ProviderError

T = TypeVar("T")

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_DELAYS = [1, 2, 4]  # seconds


def is_transient_error(error: Exception) -> bool:
    """Classify whether an error is transient (worth retrying).

    Transient:
    - httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout (network)
    - ProviderError with status_code in {429, 500, 502, 503, 504}

    Non-transient (fail immediately):
    - ProviderError with any other status_code (400, 401, 403, 404, etc.)
    - Any other exception type
    """
    if isinstance(error, httpx.ConnectError | httpx.TimeoutException | httpx.ReadTimeout):
        return True
    return isinstance(error, ProviderError) and error.status_code in TRANSIENT_STATUS_CODES


def with_retry(
    fn: Callable[[], T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delays: list[int] | None = None,
) -> T:
    """Execute fn with retry on transient errors.

    Args:
        fn: Zero-argument callable to execute.
        max_attempts: Maximum number of attempts (default 3).
        delays: Backoff delays in seconds between attempts (default [1, 2, 4]).

    Returns:
        The return value of fn on success.

    Raises:
        The last error if all attempts exhausted, or immediately on non-transient errors.
    """
    if delays is None:
        delays = DEFAULT_DELAYS

    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as err:
            last_error = err

            # Don't retry non-transient errors
            if not is_transient_error(err):
                raise

            # Don't delay after last attempt
            if attempt < max_attempts - 1:
                delay = delays[attempt] if attempt < len(delays) else delays[-1]
                time.sleep(delay)

    # All attempts exhausted
    raise last_error  # type: ignore[misc]
