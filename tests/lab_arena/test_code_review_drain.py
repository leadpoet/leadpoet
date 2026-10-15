"""A restart drains paid reviews without dispatching queued reviews."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from lab_arena.service import ArenaService


def test_review_stop_finishes_four_active_calls_and_defers_queued_work():
    started = threading.Event()
    release = threading.Event()
    stop = threading.Event()
    seen: list[str] = []
    lock = threading.Lock()

    def review(row):
        with lock:
            seen.append(row["submission_id"])
            if len(seen) == 4:
                started.set()
        assert release.wait(3)
        return {"status": "passed"}

    rows = [{"submission_id": str(index), "code_review_status": "pending"}
            for index in range(12)]
    fake = SimpleNamespace(
        _config=SimpleNamespace(code_reviewer=SimpleNamespace(review=review)),
        active_rounds=lambda: [{"round_id": "round"}],
        _round=lambda _round_id: {"round_id": "round", "status": "closed"},
        _store=SimpleNamespace(list_submissions=lambda _round_id, *, status: rows if status == "accepted" else []),
        _is_daily_baseline=lambda _row, _round: False,
    )
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            ArenaService.review_pending_submissions(fake, should_continue=lambda: not stop.is_set())
        )
    )
    worker.start()
    try:
        assert started.wait(3)
        stop.set()
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert len(seen) == 4
        assert result == [{"reviewed": 4}]
    finally:
        release.set()
        worker.join(3)
