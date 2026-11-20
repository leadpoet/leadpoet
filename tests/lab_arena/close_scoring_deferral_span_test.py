"""A close that does not transition the round must not report `ok`.

`lab_arena_close_scoring` returns `closed` only when the round actually
moves on. `draining` (migration 485's baseline-recovery guard), `existing`
and `stale` all leave the round where it was, and the driver re-enters the
same branch on the next tick. On 2026-11-20 that deferral ran for hours
emitting `arena.outcome = ok` every ~71s, so telemetry could not tell an
endless loop from the one real close of the day.
"""

import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability import otel_bootstrap
from lab_arena import telemetry
from lab_arena.service import ArenaService

ROUND_ID = "round-2026-11-20"
SCHEDULE = {
    "submission_cutoff": "2026-11-20T00:00:00+00:00",
    "stage_1_scoring_close": "2026-11-20T11:30:00+00:00",
    "final_scoring_close": "2026-11-20T20:30:00+00:00",
    "publication_deadline": "2026-11-20T20:30:01+00:00",
}


@pytest.fixture
def exporter():
    spans = InMemorySpanExporter()
    telemetry.install_recorder(otel_bootstrap.configure_arena_otel(None, span_exporter=spans))
    yield spans
    telemetry.install_recorder(None)


def _service(close_status):
    """The smallest service that can reach the close-scoring branch."""
    service = object.__new__(ArenaService)
    service._lock = threading.RLock()
    service._store = SimpleNamespace(expire_leases=lambda _round_id: None)
    service._round = lambda _round_id: {
        "status": "stage2_scoring",
        "configuration_doc": {"schedule": SCHEDULE},
    }
    service.now = lambda: datetime(2026, 11, 20, 20, 0, tzinfo=timezone.utc)
    service._recover_baseline = lambda _row: None
    # Every scoring run terminal, which is what makes the driver call close
    # on every tick regardless of the window.
    service._scoring_runs_for_stage = lambda _round_id, _stage: [{"status": "accepted"}]
    service.close_scoring = lambda _round_id, _stage: {
        "status": close_status, "round_status": "stage2_scoring"}
    return service


def _close_span(exporter):
    spans = [s for s in exporter.get_finished_spans() if s.name == "arena.advance_close_scoring"]
    assert len(spans) == 1, [s.name for s in exporter.get_finished_spans()]
    return spans[0]


@pytest.mark.parametrize("close_status", ["draining", "existing", "stale"])
def test_a_close_that_leaves_the_round_alone_reports_idle(exporter, close_status):
    result = _service(close_status)._advance_round_locked(ROUND_ID)

    assert result["status"] == close_status
    assert _close_span(exporter).attributes["arena.outcome"] == "idle"


def test_a_real_close_still_reports_ok(exporter):
    result = _service("closed")._advance_round_locked(ROUND_ID)

    assert result["status"] == "closed"
    assert _close_span(exporter).attributes["arena.outcome"] == "ok"
