"""Store-level behavior that does not need a database: batched score recording."""

from __future__ import annotations

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, SCORE_BATCH_SIZE


class RecordingTransport:
    def __init__(self, *, stale_at_batch=None):
        self.calls = []
        self.stale_at_batch = stale_at_batch

    def rpc(self, function, params):
        self.calls.append((function, params))
        if self.stale_at_batch is not None and len(self.calls) == self.stale_at_batch:
            return {"status": "stale", "round_status": "scored"}
        count = len(params["p_scores"])
        return {"status": "ok", "recorded": max(0, count - 1), "existing": 1 if count else 0}

    def select(self, *args, **kwargs):  # pragma: no cover - unused
        raise AssertionError("no selects expected")

    def close(self):
        pass


def scores(count):
    return [{"run_id": "run-%05d" % index, "per_icp_score": 50.0, "score_ref": "arena/r/scores/stage1.json", "score_doc": {"i": index}} for index in range(count)]


def test_scores_are_written_in_bounded_idempotent_batches():
    transport = RecordingTransport()
    store = ArenaStore(transport)
    total = store.record_run_scores("arena-2026-09-02", 1, scores(12_850))
    assert total["batches"] == 26 and total["recorded"] == 12_850 - 26 and total["existing"] == 26
    sizes = [len(params["p_scores"]) for _function, params in transport.calls]
    assert max(sizes) == SCORE_BATCH_SIZE and sum(sizes) == 12_850 and sizes[-1] == 350
    assert all(function == "lab_arena_record_run_scores" and params["p_stage"] == 1 for function, params in transport.calls)
    # Order is preserved across batches so a partial write is resumable.
    assert transport.calls[0][1]["p_scores"][0]["run_id"] == "run-00000" and transport.calls[-1][1]["p_scores"][-1]["run_id"] == "run-12849"


def test_an_empty_stage_still_makes_one_status_checked_call():
    transport = RecordingTransport()
    total = ArenaStore(transport).record_run_scores("arena-2026-09-02", 2, [])
    assert total == {"status": "ok", "recorded": 0, "existing": 0, "batches": 1} and len(transport.calls) == 1


def test_a_stale_round_stops_the_batches_and_surfaces_the_status():
    transport = RecordingTransport(stale_at_batch=2)
    result = ArenaStore(transport).record_run_scores("arena-2026-09-02", 1, scores(1_200))
    assert result["status"] == "stale" and len(transport.calls) == 2


def test_batch_size_must_be_positive():
    with pytest.raises(ArenaStoreError):
        ArenaStore(RecordingTransport()).record_run_scores("arena-2026-09-02", 1, scores(3), batch_size=0)
