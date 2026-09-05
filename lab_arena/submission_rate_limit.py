"""Small process-local rate limiter for authenticated submission writes."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Deque


SUBMISSION_REQUESTS_PER_WINDOW = 6
SUBMISSION_REQUEST_WINDOW_SECONDS = 60.0
MAX_TRACKED_SUBMISSION_HOTKEYS = 1024


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class SubmissionRequestLimiter:
    """Limit costly presign and finalize work by verified miner identity."""

    def __init__(
        self,
        *,
        limit: int = SUBMISSION_REQUESTS_PER_WINDOW,
        window_seconds: float = SUBMISSION_REQUEST_WINDOW_SECONDS,
        max_identities: int = MAX_TRACKED_SUBMISSION_HOTKEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1 or window_seconds <= 0 or max_identities < 1:
            raise ValueError("submission rate-limit settings must be positive")
        self._limit = int(limit)
        self._window_seconds = float(window_seconds)
        self._max_identities = int(max_identities)
        self._clock = clock
        self._entries: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def check(self, hotkey: str) -> RateLimitDecision:
        now = float(self._clock())
        cutoff = now - self._window_seconds
        identity = str(hotkey)
        with self._lock:
            timestamps = self._entries.pop(identity, deque())
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self._limit:
                self._entries[identity] = timestamps
                retry_after = max(
                    1,
                    int(math.ceil(timestamps[0] + self._window_seconds - now)),
                )
                return RateLimitDecision(False, retry_after)
            timestamps.append(now)
            self._entries[identity] = timestamps
            while len(self._entries) > self._max_identities:
                self._entries.popitem(last=False)
            return RateLimitDecision(True)
