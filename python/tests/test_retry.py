"""Tests for retry.py — transient error classification and retry behavior.

Covers:
- SC-6: retry on transient 5xx, fail-fast on auth errors
- INV-004: is_transient_error() classification
- with_retry() call count and success on 3rd attempt
- Non-transient errors raised immediately (1 attempt)
"""

import pytest

from llm_core.exceptions import ProviderError
from llm_core.retry import is_transient_error, with_retry


def test_503_is_transient() -> None:
    """ProviderError with status_code=503 is classified as transient."""
    assert is_transient_error(ProviderError("service unavailable", status_code=503))


def test_429_is_transient() -> None:
    """ProviderError with status_code=429 is classified as transient."""
    assert is_transient_error(ProviderError("rate limited", status_code=429))


def test_401_is_not_transient() -> None:
    """ProviderError with status_code=401 is not transient — no retry."""
    assert not is_transient_error(ProviderError("unauthorized", status_code=401))


def test_404_is_not_transient() -> None:
    """ProviderError with status_code=404 is not transient — no retry."""
    assert not is_transient_error(ProviderError("not found", status_code=404))


def test_transient_503_retries_until_success() -> None:
    """SC-6: 503 on first 2 calls, success on 3rd — called 3 times total."""
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ProviderError("unavailable", status_code=503)
        return "ok"

    result = with_retry(fn, max_attempts=3, delays=[0, 0])

    assert result == "ok"
    assert len(calls) == 3


def test_non_transient_401_raises_immediately() -> None:
    """SC-6: 401 raised immediately, function called exactly once."""
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        raise ProviderError("unauthorized", status_code=401)

    with pytest.raises(ProviderError):
        with_retry(fn, max_attempts=3, delays=[0, 0])

    assert len(calls) == 1, "Non-transient error must not be retried"


def test_all_attempts_exhausted_raises_last_error() -> None:
    """If all attempts fail with transient error, last exception is raised."""
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        raise ProviderError(f"fail attempt {len(calls)}", status_code=503)

    with pytest.raises(ProviderError, match="fail attempt 3"):
        with_retry(fn, max_attempts=3, delays=[0, 0])

    assert len(calls) == 3
