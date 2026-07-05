"""Cross-process fixed-size slot pool for ICP scoring.

Every candidate is scored by its own OS process, so an in-memory pool cannot
bound total concurrency. This pool coordinates all of them through one small
on-disk slots file guarded by an advisory lock: an ICP job must hold one of
``size`` slots while it runs, so the number of concurrent candidate-model
containers stays pinned at ``size`` across every process (the single knob),
and a slot freed by any candidate is immediately taken by the next waiting job
(no idle slots while work remains).

Safety:
- fail-open: any pool error lets the job proceed unthrottled rather than block,
  so a pool fault can never stall scoring.
- self-healing: a slot whose owner process has died, or whose lease has expired,
  is reclaimed on the next acquire, so a crashed worker cannot leak a slot.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import time

_DEFAULT_LEASE_SECONDS = 1800.0
_DEFAULT_POLL_SECONDS = 0.25


def pool_size_from_env() -> int:
    try:
        return max(0, int(os.getenv("RESEARCH_LAB_GLOBAL_SCORING_POOL_SIZE") or "0"))
    except (TypeError, ValueError):
        return 0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class GlobalScoringSlotPool:
    """Fixed-size slot pool shared across processes via a lock file."""

    def __init__(
        self,
        path: str,
        size: int,
        *,
        lease_seconds: float = _DEFAULT_LEASE_SECONDS,
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
        now=time.monotonic,
        wall=time.time,
    ) -> None:
        self.path = path
        self.size = max(1, int(size))
        self.lease_seconds = float(lease_seconds)
        self.poll_seconds = float(poll_seconds)
        self._now = now
        self._wall = wall
        self._counter = 0
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except OSError:
            pass

    def _token(self) -> str:
        self._counter += 1
        return f"{os.getpid()}:{self._counter}"

    def _live_slots(self, slots: list[dict], wall_now: float) -> list[dict]:
        kept: list[dict] = []
        for slot in slots:
            pid = int(slot.get("pid") or 0)
            ts = float(slot.get("ts") or 0.0)
            if wall_now - ts > self.lease_seconds:
                continue
            if pid and not _pid_alive(pid):
                continue
            kept.append(slot)
        return kept

    def _try_claim(self, token: str) -> bool:
        """One atomic attempt to take a slot. True if acquired."""
        flags = os.O_RDWR | os.O_CREAT
        fd = os.open(self.path, flags, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.read(fd, 1 << 20).decode("utf-8") or ""
            try:
                slots = json.loads(raw) if raw.strip() else []
                if not isinstance(slots, list):
                    slots = []
            except (ValueError, TypeError):
                slots = []
            wall_now = self._wall()
            slots = self._live_slots(slots, wall_now)
            already = any(s.get("token") == token for s in slots)
            if not already and len(slots) >= self.size:
                # No room; persist the reclaimed set so dead slots don't linger.
                self._write(fd, slots)
                return False
            if not already:
                slots.append({"token": token, "pid": os.getpid(), "ts": wall_now})
            self._write(fd, slots)
            return True
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _write(self, fd: int, slots: list[dict]) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(slots).encode("utf-8"))

    def _release_token(self, token: str) -> None:
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.read(fd, 1 << 20).decode("utf-8") or ""
            try:
                slots = json.loads(raw) if raw.strip() else []
                if not isinstance(slots, list):
                    slots = []
            except (ValueError, TypeError):
                slots = []
            slots = [s for s in slots if s.get("token") != token]
            self._write(fd, slots)
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextlib.asynccontextmanager
    async def slot(self):
        """Hold one slot for the duration of the block. Fail-open on error."""
        token = self._token()
        acquired = False
        try:
            deadline = self._now() + self.lease_seconds
            while True:
                try:
                    if self._try_claim(token):
                        acquired = True
                        break
                except OSError:
                    # Pool unusable (fs/lock error): proceed unthrottled.
                    break
                if self._now() >= deadline:
                    # Never block a job indefinitely; degrade to unthrottled.
                    break
                await asyncio.sleep(self.poll_seconds)
            yield acquired
        finally:
            if acquired:
                with contextlib.suppress(Exception):
                    self._release_token(token)


_POOL: GlobalScoringSlotPool | None = None
_POOL_KEY: tuple[str, int] | None = None


def get_global_scoring_pool() -> GlobalScoringSlotPool | None:
    """Process-wide singleton, or None when pooling is disabled (size 0)."""
    global _POOL, _POOL_KEY
    size = pool_size_from_env()
    if size <= 0:
        return None
    path = os.getenv("RESEARCH_LAB_GLOBAL_SCORING_POOL_PATH") or "/tmp/leadpoet_scoring_slots.json"
    key = (path, size)
    if _POOL is None or _POOL_KEY != key:
        _POOL = GlobalScoringSlotPool(path, size)
        _POOL_KEY = key
    return _POOL
