"""Bounded request-priority middleware for the gateway."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from gateway.utils.ops_registry import set_priority_middleware


logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class RouteClass:
    name: str
    max_concurrent: int
    max_waiting: int
    wait_timeout_s: float


VALIDATOR_CLASS = RouteClass(
    "validator",
    _int_env("VALIDATOR_MAX_CONCURRENT", 75),
    _int_env("VALIDATOR_MAX_WAITING", 40),
    _float_env("VALIDATOR_SLOT_WAIT_TIMEOUT_SECONDS", 15),
)
MINER_CLASS = RouteClass(
    "miner",
    _int_env("MINER_MAX_CONCURRENT", 75),
    _int_env("MINER_MAX_WAITING", 40),
    _float_env("MINER_SLOT_WAIT_TIMEOUT_SECONDS", 8),
)
OTHER_CLASS = RouteClass(
    "other",
    _int_env("OTHER_MAX_CONCURRENT", 150),
    _int_env("OTHER_MAX_WAITING", 80),
    _float_env("OTHER_SLOT_WAIT_TIMEOUT_SECONDS", 8),
)


VALIDATOR_EXACT = {
    "/validate",
    "/weights/submit",
    "/weights/submit/v2",
    "/fulfillment/scoring",
    "/fulfillment/score",
    "/fulfillment/rewards/active",
}
VALIDATOR_PREFIXES = (
    "/epoch/",
    "/qualification/validator/",
    "/fulfillment/ban/",
    "/fulfillment/results/",
)
MINER_EXACT = {
    "/presign",
    "/submit",
    "/submit/",
    "/fulfillment/requests/active",
    "/fulfillment/commit",
    "/fulfillment/reveal",
}
MINER_PREFIXES = (
    "/fulfillment/excluded-now/",
)


def _matches(path: str, exact: set[str], prefixes: Iterable[str]) -> bool:
    return path in exact or any(path.startswith(prefix) for prefix in prefixes)


def classify_path(path: str) -> str:
    if _matches(path, VALIDATOR_EXACT, VALIDATOR_PREFIXES):
        return "validator"
    if _matches(path, MINER_EXACT, MINER_PREFIXES):
        return "miner"
    return "other"


class _Pool:
    def __init__(self, route_class: RouteClass) -> None:
        self.route_class = route_class
        self.semaphore = asyncio.Semaphore(route_class.max_concurrent)
        self.waiting = 0
        self.in_flight = 0
        self.shed = 0
        self.requests = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            self.requests += 1
            if self.waiting >= self.route_class.max_waiting:
                self.shed += 1
                return False
            self.waiting += 1
        try:
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.route_class.wait_timeout_s,
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self.shed += 1
            return False
        finally:
            async with self._lock:
                self.waiting = max(0, self.waiting - 1)
        async with self._lock:
            self.in_flight += 1
        return True

    async def release(self) -> None:
        self.semaphore.release()
        async with self._lock:
            self.in_flight = max(0, self.in_flight - 1)

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "shed": self.shed,
            "waiting": self.waiting,
            "in_flight": self.in_flight,
            "max_concurrent": self.route_class.max_concurrent,
            "max_waiting": self.route_class.max_waiting,
        }


class PriorityMiddleware:
    """Pure-ASGI request-priority middleware.

    Implemented as raw ASGI (not Starlette ``BaseHTTPMiddleware``) on purpose.
    ``BaseHTTPMiddleware`` bridges the downstream app through an anyio task
    group and memory-object streams; under concurrent load, or when the
    endpoint raises / the client disconnects mid-request, that task group's
    ``__aexit__`` iterates its internal task deque while it is mutated,
    surfacing as ``RuntimeError: deque mutated during iteration`` and turning
    a valid response into an intermittent 5xx. On the weight path that
    intermittent 5xx burns the validator's tight submission window and drops
    the epoch. Awaiting ``self.app`` directly keeps the concurrency-pool
    accounting identical while removing the task-group wrapper entirely, so
    the race cannot occur.
    """

    def __init__(self, app: ASGIApp, max_concurrent_miners: int | None = None):
        self.app = app
        if max_concurrent_miners is not None and "MINER_MAX_CONCURRENT" not in os.environ:
            miner_class = RouteClass(
                MINER_CLASS.name,
                int(max_concurrent_miners),
                MINER_CLASS.max_waiting,
                MINER_CLASS.wait_timeout_s,
            )
        else:
            miner_class = MINER_CLASS
        self.pools = {
            "validator": _Pool(VALIDATOR_CLASS),
            "miner": _Pool(miner_class),
            "other": _Pool(OTHER_CLASS),
        }
        self.path_counts = Counter()
        set_priority_middleware(self)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only HTTP requests are pooled; websockets and lifespan pass through
        # untouched, exactly as BaseHTTPMiddleware did.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        route_class = classify_path(path)
        pool = self.pools[route_class]
        self.path_counts[route_class] += 1

        acquired = await pool.acquire()
        if not acquired:
            logger.warning(
                "gateway_request_shed class=%s method=%s path=%s",
                route_class,
                scope.get("method", ""),
                path,
            )
            response = JSONResponse(
                status_code=503,
                content={"detail": "Gateway at capacity - retry shortly"},
                headers={"Retry-After": "15"},
            )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            await pool.release()

    def snapshot(self) -> dict:
        pool_snaps = {name: pool.snapshot() for name, pool in self.pools.items()}
        return {
            "requests_total": dict(self.path_counts),
            "shed_total": {name: snap["shed"] for name, snap in pool_snaps.items()},
            "waiting": {name: snap["waiting"] for name, snap in pool_snaps.items()},
            "in_flight": {name: snap["in_flight"] for name, snap in pool_snaps.items()},
            "max_concurrent": {
                name: snap["max_concurrent"] for name, snap in pool_snaps.items()
            },
        }
