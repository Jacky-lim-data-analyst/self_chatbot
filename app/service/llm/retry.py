"""
Exponential-backoff retry wrapper for BaseLLM.stream_chat

1. **RetryLLM wrapper class** — wraps any existing BaseLLM instance:
    llm = build_llm("deepseek", model="deepseek-chat")
    llm = RetryLLM(llm)

2. **retry_stream decorator** — wrap a single stream_chat method ad-hoc:
        class MyLLM(BaseLLM):
            @retry_stream()
            async def stream_chat(self, messages):
                ...
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from email.utils import parsedate_to_datetime
from typing import AsyncIterator, Callable

from app.models import Message
from .base import BaseLLM
from app.util.logging import get_logger

MAX_DELAYS = 60.0  # second

logger = get_logger("retry")


# ---------------------------------------------------------------------------
# Retry-After header parsing
# ---------------------------------------------------------------------------
def _parse_retry_after(exc: BaseException) -> float | None:
    """
    Extract a wait duration (seconds) from the Retry-After header attached
    to *exc*, if present.

    Handles both formats defined in RFC 9110:
    - Delta-seconds:  ``Retry-After: 30``
    - HTTP-date:      ``Retry-After: Fri, 21 Mar 2026 12:00:00 GMT``

    Returns None if the exception carries no parsable Retry-After value.
    """
    # Most HTTP client libraries (httpx, openai, requests) attach the
    # raw response object as exc.response
    response = getattr(exc, "response", None)
    if response is None:
        return None

    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None

    raw = raw.strip()

    # case 1: plain integer
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    # case 2: HTTP-date
    try:
        retry_at = parsedate_to_datetime(raw)
        delta = retry_at.timestamp() - time.time()
        return max(0.0, delta)
    except Exception:
        pass

    logger.info(f"Could not parse Retry-After header value: {raw}")
    return None


# ---------------------------------------------------------------------------
# Backoff calculation
# ---------------------------------------------------------------------------
def _backoff(
    attempt: int,
    *,
    base: float,
    multiplier: float,
    cap: float,
    jitter: bool,
) -> float:
    """
    Compute the next wait duration using full-jitter exponential backoff.

    Formula (before cap):  base * multiplier ** attempt
    Jitter:                uniform sample in [0, computed_delay]
    """
    delay = min(cap, base * (multiplier**attempt))
    if jitter:
        delay = random.uniform(0, delay)
    return delay


# ---------------------------------------------------------------------------
# Retryable exception predicate
# ---------------------------------------------------------------------------

# HTTP status codes that are safe to retry
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset(
    {
        429,  # Too many requests
        500,  # Internal server error
        502,  # bad gateway
        503,  # service unavailable
        504,  # gateway timeout
    }
)


def _is_retryable(exc: BaseException) -> bool:
    """
    Return True if *exc* is worth retrying.

    Checks (in order):
    1. The exception has a .response with a retryable status code.
    2. The exception class name contains a retryable keyword.
    """
    # check HTTP status code
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status code", None)
        if status in _RETRYABLE_STATUS_CODES:
            return True

    # transient network / timeout errors
    exc_type = type(exc).__name__
    retryable_keywords = (
        "Timeout",
        "ConnectionError",
        "NetworkError",
        "RateLimitError",
        "ServiceUnvailable",
        "InternalServerError",
        "APIConnectionError",
        "APIStatusError",
    )
    if any(kw in exc_type for kw in retryable_keywords):
        return True

    return False


# ---------------------------------------------------------------------------
# Core async-generator retry logic
# ---------------------------------------------------------------------------
async def _retry_stream(
    stream_fn: Callable[[], AsyncIterator[str]],
    *,
    max_attempts: int,
    base_delay: float,
    multiplier: float,
    max_delay: float,
    jitter: bool,
) -> AsyncIterator[str]:
    """
    Call *stream_fn()* and yield its chunks. On a retryable error,
    call *stream_fn()* again from the beginning.

    Args:
        stream_fn:    Zero-argument callable that returns a fresh AsyncIterator.
        max_attempts: Total number of attempts (1 = no retries).
        base_delay:   Initial backoff in seconds.
        multiplier:   Exponential growth factor.
        max_delay:    Upper bound on computed backoff (before jitter).
        jitter:       Whether to add full jitter to the delay.
    """
    last_exc: BaseException | None = None

    for attempt in range(max_attempts):
        try:
            async for chunk in stream_fn():
                yield chunk
            return

        except Exception as exc:
            last_exc = exc

            if not _is_retryable(exc):
                logger.warning(f"Non-retryable error on attempt {attempt + 1}: {exc!r}")
                raise

            is_last = attempt == max_attempts - 1
            if is_last:
                logger.error(f"stream_chat failed after {max_attempts}: {exc!r}")
                raise

            wait = _parse_retry_after(exc) or _backoff(
                attempt,
                base=base_delay,
                multiplier=multiplier,
                cap=max_delay,
                jitter=jitter,
            )

            logger.warning(
                f"stream_chat attempt {attempt + 1}/{max_attempts} failed."
                f"{type(exc).__name__}: retrying in {wait:.2f}s"
            )

            await asyncio.sleep(wait)

        # just in case
        if last_exc is not None:
            raise last_exc


# ---------------------------------------------------------------------------
# 1. RetryLLM — wrapper class
# ---------------------------------------------------------------------------
class RetryLLM(BaseLLM):
    """
    Wraps an existing BaseLLM instance and adds retry logic to stream_chat

    All other methods are delegated to the inner LLM

        Example::
        base = build_llm("deepseek", model="deepseek-chat")
        llm  = RetryLLM(base, max_attempts=5)

        async with llm:
            async for chunk in llm.stream_chat(messages):
                print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        max_attempts: int = 4,
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        # Bypass BaseLLM.__init__ - config lives on the inner llm
        self._inner = llm
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._multiplier = multiplier
        self._max_delay = max_delay
        self._jitter = jitter

    @property
    def config(self):
        return self._inner.config

    async def stream_chat(self, messages: list[Message]) -> AsyncIterator[str]:
        async for chunk in _retry_stream(
            lambda: self._inner.stream_chat(messages),
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            multiplier=self._multiplier,
            max_delay=self._max_delay,
            jitter=self._jitter,
        ):
            yield chunk

    async def chat(self, messages: list[Message]) -> str:
        return await self.stream_to_string(messages)

    async def close(self) -> None:
        await self._inner.close()

    def __repr__(self) -> str:
        return f"RetryLLM(inner={self._inner!r})max_attempts={self._max_attempts}"


# ---------------------------------------------------------------------------
# 2. retry_stream — method decorator
# ---------------------------------------------------------------------------
def retry_stream(
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
):
    """
    Decorator that wraps an async-generator stream_chat method with retries.

    Example::
        class MyLLM(BaseLLM):
            @retry_stream(max_attempts=3, base_delay=0.5)
            async def stream_chat(self, messages):
                async for chunk in self._client.stream(...):
                    yield chunk
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(self, messages: list[Message]) -> AsyncIterator[str]:
            async for chunk in _retry_stream(
                lambda: fn(self, messages),
                max_attempts=max_attempts,
                base_delay=base_delay,
                multiplier=multiplier,
                max_delay=max_delay,
                jitter=jitter,
            ):
                yield chunk

        return wrapper

    return decorator
