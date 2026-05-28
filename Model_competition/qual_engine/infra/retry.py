"""Retry policies for HTTP calls."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504),
    label: str = "call",
) -> T:
    """Run `fn`, retry with jitter on transient HTTP errors and timeouts."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status not in retry_on_status or attempt >= max_attempts:
                raise
            last_err = e
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            if attempt >= max_attempts:
                raise
            last_err = e
        except Exception:
            # Non-retryable
            raise
        delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.3)
        logger.warning("%s: attempt %d failed (%s), sleeping %.2fs", label, attempt, last_err, delay)
        await asyncio.sleep(delay)
    raise RuntimeError(f"{label}: exhausted retries: {last_err!r}")
