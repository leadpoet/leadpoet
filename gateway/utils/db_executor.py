"""Bounded executor for synchronous gateway database work.

The gateway process uses a single asyncio event loop. Any synchronous
Supabase call made directly from an async endpoint can stall every route.
This module gives hot paths one shared, bounded off-loop executor and sheds
quickly when the DB side is saturated.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

import httpx
from fastapi import HTTPException

from gateway.utils.circuit_breaker import OPEN_SECONDS, db_breaker


logger = logging.getLogger(__name__)

DB_THREADS = int(os.getenv("GATEWAY_DB_THREADS", "24"))
DB_QUEUE_HIGH_WATER = int(os.getenv("GATEWAY_DB_QUEUE_HIGH_WATER", "100"))

_executor = ThreadPoolExecutor(
    max_workers=DB_THREADS,
    thread_name_prefix="gateway-db",
)
_in_flight = 0
_lock = threading.Lock()

T = TypeVar("T")
TRANSPORT_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.NetworkError,
    OSError,
)


def db_queue_depth() -> int:
    with _lock:
        return _in_flight


async def run_db(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run blocking DB work off-loop with bounded admission."""
    global _in_flight

    if not db_breaker.before_call():
        raise HTTPException(
            503,
            detail="Database temporarily unavailable - retry shortly",
            headers={"Retry-After": str(int(OPEN_SECONDS))},
        )

    with _lock:
        if _in_flight >= DB_QUEUE_HIGH_WATER:
            raise HTTPException(
                503,
                detail="Gateway database queue saturated - retry shortly",
                headers={"Retry-After": "10"},
            )
        _in_flight += 1

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            functools.partial(fn, *args, **kwargs),
        )
    except HTTPException:
        db_breaker.record_success()
        raise
    except TRANSPORT_ERRORS as exc:
        db_breaker.record_failure()
        logger.warning("gateway_db_transport_failure: %s", exc)
        raise HTTPException(
            503,
            detail="Database error - retry shortly",
            headers={"Retry-After": "10"},
        ) from exc
    except Exception:
        db_breaker.record_success()
        raise
    else:
        db_breaker.record_success()
        return result
    finally:
        with _lock:
            _in_flight -= 1
