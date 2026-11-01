"""Telemetry seam for the Arena service — a no-op until a recorder is installed.

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

The same seam carries five other kinds of observation, all equally bounded:

- ``record_provider`` — one provider call: which provider and operation, how
  it ended, the HTTP and upstream status, how many credential attempts it
  took, what it cost in micro-USD, and how long it ran.
- ``record_run`` — one finished evaluation run: execution or scoring, and
  which of the thirteen terminal causes ended it.
- ``record_review`` — one persisted submission review attempt outcome, with
  only bounded decision, rejection categories, diagnostic, funding route, and
  submission identity.
- ``record_runtime_upload`` — one accepted upload of runtime events that
  inserted at least one event. Its flags name lifecycle kinds present in the
  batch; ``has_error`` also covers a known failed ``runtime.finished`` status.
  Mixed new/replayed batches do not prove fresh transitions.
- ``record_gate`` — one wait on the shared provider concurrency gate, emitted
  only when a call actually waited, timed out, was cancelled, or found no
  capacity. An uncontended call emits nothing.
- ``note_denial`` — the refusal code the sidecar produced while serving the
  current request, so a 4xx stops being anonymous. Only the literal prefix
  before the first ":" is ever exported.

Provider, committed-run, and accepted-runtime-upload spans may carry seven
bounded identifiers copied from the gateway's authenticated run row. Review
spans carry four bounded identifiers from the accepted submission and committed
attempt. No ICP content, source bundle, prompt, completion, score, model
output, URL, or credential is accepted.
Stage and gate spans remain aggregate, except Deepline billing lookups carry
the bounded round, lane, cursor and run-scope context needed to locate a slow
candidate query. They never carry a run ID or provider payload.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional

NO_ERROR = "-"
# A provider call that names no Deepline tool: any non-Deepline provider, and
# a Deepline operation whose tool could not be resolved from the frame.
NO_TOOL = "-"
# Which wallet the broker settled this call against. "unknown" covers a call
# that raised before a funding source was chosen.
FUNDING_UNKNOWN = "unknown"

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
    source_sha: Optional[str] = None,
    store_error: Optional[Mapping[str, Any]] = None,
    billing_lookup: Optional[Mapping[str, Any]] = None,
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
            source_sha=source_sha,
            **({"store_error": store_error} if store_error else {}),
            **({"billing_lookup": billing_lookup} if billing_lookup else {}),
        )
    except BaseException:
        pass


def flush(timeout_millis: int = 1000) -> None:
    """Bound the final export attempt during graceful service shutdown."""
    recorder = _recorder
    if recorder is None:
        return
    try:
        recorder.flush(timeout_millis=timeout_millis)
    except BaseException:
        pass


def _store_error_metadata(error: Optional[BaseException]) -> Mapping[str, Any]:
    """Read existing diagnostics from exact built-in store errors only."""
    try:
        from lab_arena.store import (
            ArenaStoreError, ArenaStoreUnavailable, safe_store_error_metadata,
        )

        if type(error) in (ArenaStoreError, ArenaStoreUnavailable):
            return safe_store_error_metadata(error.__dict__.get("store_error"))
    except BaseException:
        pass
    return {}


def note_denial(code: str, *, error: Optional[BaseException] = None) -> None:
    """Name the refusal the sidecar is returning for the current request."""
    recorder = _recorder
    if recorder is None:
        return
    try:
        metadata = _store_error_metadata(error)
        recorder.note_denial(code, **({"store_error": metadata} if metadata else {}))
    except BaseException:
        pass


def record_provider(
    provider: str,
    operation: str,
    outcome: str,
    *,
    error_code: str = NO_ERROR,
    error_type: str = NO_ERROR,
    tool: str = NO_TOOL,
    funding_source: str = FUNDING_UNKNOWN,
    http_status: int = 0,
    provider_status: int = 0,
    attempts: int = 1,
    cost_microusd: int = 0,
    duration_ms: float = 0.0,
    start_ns: Optional[int] = None,
    run_identity: Optional[Mapping[str, Any]] = None,
    store_error: Optional[Mapping[str, Any]] = None,
) -> None:
    """Record one provider call. Never raises."""
    recorder = _recorder
    if recorder is None:
        return
    try:
        recorder.record_provider(
            provider,
            operation,
            outcome,
            error_code=error_code,
            error_type=error_type,
            tool=tool,
            funding_source=funding_source,
            http_status=http_status,
            provider_status=provider_status,
            attempts=attempts,
            cost_microusd=cost_microusd,
            duration_ms=duration_ms,
            start_ns=start_ns,
            run_identity=run_identity,
            **({"store_error": store_error} if store_error else {}),
        )
    except BaseException:
        pass


def record_run(
    run_kind: str, terminal_cause: str,
    *, run_identity: Optional[Mapping[str, Any]] = None,
) -> None:
    """Record one finished evaluation run. Never raises.

    No duration: the run row carries no start time, so the sidecar can only
    honestly report that a run ended and which cause ended it.
    """
    recorder = _recorder
    if recorder is None:
        return
    try:
        recorder.record_run(run_kind, terminal_cause, run_identity=run_identity)
    except BaseException:
        pass


def record_review(
    status: str,
    *,
    error_code: str,
    categories: str,
    provider_http_status: int,
    retryable: bool,
    funding_route: str,
    round_id: str,
    submission_id: str,
    miner_hotkey: str,
    attempt: int,
) -> None:
    """Record one committed review decision. Never raises."""
    recorder = _recorder
    if recorder is None:
        return
    try:
        recorder.record_review(
            status,
            error_code=error_code,
            categories=categories,
            provider_http_status=provider_http_status,
            retryable=retryable,
            funding_route=funding_route,
            round_id=round_id,
            submission_id=submission_id,
            miner_hotkey=miner_hotkey,
            attempt=attempt,
        )
    except BaseException:
        pass


def record_runtime_upload(
    *,
    inserted_count: int,
    replayed_count: int,
    has_started: bool,
    has_finished: bool,
    has_error: bool,
    run_identity: Mapping[str, Any],
) -> None:
    """Record a successful upload batch, never an inferred lifecycle event."""
    recorder = _recorder
    if recorder is None:
        return
    try:
        recorder.record_runtime_upload(
            inserted_count=inserted_count,
            replayed_count=replayed_count,
            has_started=has_started,
            has_finished=has_finished,
            has_error=has_error,
            run_identity=run_identity,
        )
    except BaseException:
        pass


def record_gate(
    gate_outcome: str,
    *,
    duration_ms: float = 0.0,
    start_ns: Optional[int] = None,
) -> None:
    """Record one contended wait on the shared provider gate. Never raises."""
    recorder = _recorder
    if recorder is None:
        return
    try:
        recorder.record_gate(
            gate_outcome, duration_ms=duration_ms, start_ns=start_ns
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
def stage(name: str, *, billing_lookup: Optional[Mapping[str, Any]] = None) -> Iterator[StageResult]:
    """Time one pipeline stage and record its outcome, success or failure.

    The wrapped exception is re-raised unchanged — this observes, it never
    swallows. Temporary store failures also retain their existing allowlisted
    machine diagnostics; no exception text or request data is recorded.
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
            store_error=_store_error_metadata(exc),
            billing_lookup=billing_lookup,
        )
        raise
    record(
        name,
        "idle" if result.idle else "ok",
        count=int(result.count or 0),
        duration_ms=(time.monotonic() - start_mono) * 1000.0,
        start_ns=start_ns,
        billing_lookup=billing_lookup,
    )


__all__ = [
    "NO_ERROR",
    "NO_TOOL",
    "StageResult",
    "install_recorder",
    "note_denial",
    "record",
    "record_gate",
    "record_provider",
    "record_run",
    "record_runtime_upload",
    "stage",
]
