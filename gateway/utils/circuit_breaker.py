"""Small circuit breaker for gateway upstream dependencies."""

from __future__ import annotations

import os
import threading
import time


FAILURE_THRESHOLD = int(os.getenv("BREAKER_FAILURE_THRESHOLD", "5"))
OPEN_SECONDS = float(os.getenv("BREAKER_OPEN_SECONDS", "15"))


class CircuitBreaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = "closed"
        self.consecutive_failures = 0
        self.opened_at = 0.0
        self.times_opened = 0

    def before_call(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.monotonic() - self.opened_at >= OPEN_SECONDS:
                    self.state = "half_open"
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            self.state = "closed"
            self.consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1
            if (
                self.state == "half_open"
                or self.consecutive_failures >= FAILURE_THRESHOLD
            ):
                if self.state != "open":
                    self.times_opened += 1
                self.state = "open"
                self.opened_at = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "consecutive_failures": self.consecutive_failures,
                "times_opened": self.times_opened,
                "opened_at": self.opened_at,
            }


db_breaker = CircuitBreaker()
