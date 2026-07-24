"""Bounded request-priority middleware for the gateway."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

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
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.waiting = 0
        self.in_flight = 0
        self.shed = 0
        self.requests = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._active_calls = 0
        self._runtime_lock = threading.Lock()
        self._state_lock = threading.Lock()

    def _enter_runtime(self) -> asyncio.Semaphore:
        """Bind loop-owned primitives lazily and retain one request lease."""

        loop = asyncio.get_running_loop()
        with self._runtime_lock:
            if self._loop is not loop:
                if self._active_calls:
                    raise RuntimeError(
                        "priority pool cannot move between active event loops"
                    )
                self.semaphore = asyncio.Semaphore(
                    self.route_class.max_concurrent
                )
                self._loop = loop
            if self.semaphore is None:
                raise RuntimeError("priority pool runtime is unavailable")
            self._active_calls += 1
            return self.semaphore

    def _current_runtime(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._runtime_lock:
            if (
                self._loop is not loop
                or self.semaphore is None
                or self._active_calls <= 0
            ):
                raise RuntimeError("priority pool release has no active lease")
            return self.semaphore

    def _leave_runtime(self) -> None:
        with self._runtime_lock:
            if self._active_calls <= 0:
                raise RuntimeError("priority pool lease accounting underflow")
            self._active_calls -= 1

    async def acquire(self) -> bool:
        semaphore = self._enter_runtime()
        acquired = False
        slot_acquired = False
        try:
            with self._state_lock:
                self.requests += 1
                if self.waiting >= self.route_class.max_waiting:
                    self.shed += 1
                    return False
                self.waiting += 1
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=self.route_class.wait_timeout_s,
                )
                slot_acquired = True
            except asyncio.TimeoutError:
                with self._state_lock:
                    self.shed += 1
                return False
            finally:
                with self._state_lock:
                    self.waiting = max(0, self.waiting - 1)
            with self._state_lock:
                self.in_flight += 1
            acquired = True
            return True
        finally:
            if not acquired:
                if slot_acquired:
                    semaphore.release()
                self._leave_runtime()

    async def release(self) -> None:
        semaphore = self._current_runtime()
        try:
            with self._state_lock:
                self.in_flight = max(0, self.in_flight - 1)
            semaphore.release()
        finally:
            self._leave_runtime()

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
    group and memory-object streams, which can obscure the original exception
    when an endpoint fails or a client disconnects. Awaiting ``self.app``
    directly preserves the original failure and keeps pool accounting local to
    this middleware. The independent shared-HTTP/2 fix in ``gateway.db.client``
    prevents the HPACK ``deque mutated during iteration`` failure observed in
    weight publication.
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
