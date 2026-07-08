"""Per-hotkey flow-control buckets for live fulfillment traffic."""

from __future__ import annotations

import os
import threading
import time
from typing import Tuple

from fastapi import HTTPException


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class HotkeyBuckets:
    def __init__(self, name: str, rate_per_min: float, burst: int) -> None:
        self.name = name
        self.rate = max(rate_per_min / 60.0, 0.000001)
        self.burst = max(float(burst), 1.0)
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self.allowed = 0
        self.denied = 0
        self.observed_denied = 0

    def allow(self, hotkey: str) -> Tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(hotkey, [self.burst, now])
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[hotkey] = [tokens - 1.0, now]
                self.allowed += 1
                return True, 0
            self._buckets[hotkey] = [tokens, now]
            self.denied += 1
            retry_after = max(1, int((1.0 - tokens) / self.rate))
            return False, retry_after

    def observe(self, hotkey: str) -> None:
        allowed, _ = self.allow(hotkey)
        if not allowed:
            with self._lock:
                self.observed_denied += 1

    def prune(self, older_than_s: float = 3600.0) -> None:
        now = time.monotonic()
        with self._lock:
            self._buckets = {
                k: v for k, v in self._buckets.items()
                if not (v[0] >= self.burst - 0.01 and now - v[1] > older_than_s)
            }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "active_keys": len(self._buckets),
                "allowed": self.allowed,
                "denied": self.denied,
                "observed_denied": self.observed_denied,
                "rate_per_min": self.rate * 60.0,
                "burst": self.burst,
            }


POLL_BUCKETS = HotkeyBuckets(
    "poll",
    _float_env("BUCKET_POLL_PER_MIN", 6),
    int(_float_env("BUCKET_POLL_BURST", 4)),
)
COMMIT_BUCKETS = HotkeyBuckets(
    "commit",
    _float_env("BUCKET_COMMIT_PER_MIN", 10),
    int(_float_env("BUCKET_COMMIT_BURST", 10)),
)
REVEAL_BUCKETS = HotkeyBuckets(
    "reveal",
    _float_env("BUCKET_REVEAL_PER_MIN", 10),
    int(_float_env("BUCKET_REVEAL_BURST", 10)),
)
ALL_BUCKETS = (POLL_BUCKETS, COMMIT_BUCKETS, REVEAL_BUCKETS)


def enforce(bucket: HotkeyBuckets, hotkey: str) -> None:
    allowed, retry_after = bucket.allow(hotkey)
    if not allowed:
        raise HTTPException(
            429,
            detail=f"Rate limit exceeded for {bucket.name} - retry shortly",
            headers={"Retry-After": str(retry_after)},
        )


def observe(bucket: HotkeyBuckets, hotkey: str) -> None:
    if hotkey:
        bucket.observe(hotkey)


class RecentNonces:
    def __init__(self, ttl_s: float = 600.0) -> None:
        self.ttl_s = ttl_s
        self._seen: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def first_use(self, hotkey: str, nonce: str) -> bool:
        now = time.monotonic()
        key = (hotkey, nonce)
        with self._lock:
            prior = self._seen.get(key)
            if prior is not None and now - prior < self.ttl_s:
                return False
            self._seen[key] = now
            return True

    def prune(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._seen = {
                k: ts for k, ts in self._seen.items()
                if now - ts < self.ttl_s
            }

    def snapshot(self) -> dict:
        with self._lock:
            return {"active_nonces": len(self._seen), "ttl_s": self.ttl_s}


RECENT_NONCES = RecentNonces()
