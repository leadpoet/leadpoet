"""Pipeline-stage telemetry for the Arena service — a no-op until installed.

The Arena process does its real work off the request path: a once-a-minute
driver tick that promotes baselines, advances every active round, opens the
daily round, activates rewards and reconciles provider costs, plus a code
review worker on its own loop. None of that touches an HTTP handler, so none
of it reached telemetry at all: a wedged driver and an idle one looked
identical from outside.

This module is the seam. Callers record a stage name and an outcome from two
frozen vocabularies; the recorder (installed at startup by
``gateway.observability.otel_bootstrap.configure_arena_otel``) turns that into
one span. With no recorder installed every call is a cheap no-op, so the
pipeline behaves identically whether or not telemetry is configured.

Nothing here can observe a round id, submission id, hotkey, source bundle,
prompt, score, or model output — a stage records its NAME, whether it worked,
the exception CLASS if it did not, and a magnitude.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

NO_ERROR = "-"

_recorder: Optional[Any] = None


def install_recorder(recorder: Optional[Any]) -> None:
    """Install (or clear, with None) the process-wide stage recorder."""
    global _recorder
    _recorder = recorder


def record(
    stage: str,
    outcome: str,
    *,
    error_type: str = NO_ERROR,
    count: int = 0,
    duration_ms: float = 0.0,
    start_ns: Optional[int] = None,
) -> None:
    """Record one stage outcome. Never raises."""
    recorder = _recorder
    if recorder is None:
        return
    try:
        recorder.record(
            stage,
            outcome,
            error_type=error_type,
            count=count,
            duration_ms=duration_ms,
            start_ns=start_ns,
        )
    except BaseException:
        pass


class StageResult:
    """Mutable handle a stage body uses to report magnitude and idleness."""

    __slots__ = ("count", "idle")

    def __init__(self) -> None:
        self.count = 0
        self.idle = False


@contextmanager
def stage(name: str) -> Iterator[StageResult]:
    """Time one pipeline stage and record its outcome, success or failure.

    The wrapped exception is re-raised unchanged — this observes, it never
    swallows. Only the exception's class name is recorded.
    """
    result = StageResult()
    start_ns = time.time_ns()
    start_mono = time.monotonic()
    try:
        yield result
    except BaseException as exc:
        record(
            name,
            "failed",
            error_type=type(exc).__name__,
            duration_ms=(time.monotonic() - start_mono) * 1000.0,
            start_ns=start_ns,
        )
        raise
    record(
        name,
        "idle" if result.idle else "ok",
        count=int(result.count or 0),
        duration_ms=(time.monotonic() - start_mono) * 1000.0,
        start_ns=start_ns,
    )


__all__ = ["NO_ERROR", "StageResult", "install_recorder", "record", "stage"]
