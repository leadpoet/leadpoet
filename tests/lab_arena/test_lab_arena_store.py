"""Store-level behavior that does not need a database: batched score recording."""

from __future__ import annotations

import httpx
import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, FUNCTION_SIGNATURES, PostgrestTransport, SCORE_BATCH_SIZE, TABLES, create_http1_client


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
    return [{"run_id": "run-%05d" % index, "per_icp_score": 50.0} for index in range(count)]


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


class ShapeTransport:
    def __init__(self):
        self.calls = []

    def rpc(self, function, params):
        self.calls.append((function, params))
        return {"status": "ok"}

    def select(self, *args, **kwargs):  # pragma: no cover - unused
        raise AssertionError("no selects expected")

    def close(self):
        pass


def test_simple_stage_and_completion_rpc_shapes():
    transport = ShapeTransport()
    store = ArenaStore(transport)
    store.open_stage("arena-2026-09-02", 1, [{"submission_id": "s1", "miner_hotkey": "h"}], [0, 1])
    store.complete_attempt(
        run_id="run-1",
        lease_token_hash="sha256:" + "1" * 64,
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref="arena/round/run-1.json",
    )
    assert transport.calls == [
        (
            "lab_arena_open_stage",
            {
                "p_round_id": "arena-2026-09-02",
                "p_stage": 1,
                "p_participants": [{"submission_id": "s1", "miner_hotkey": "h"}],
                "p_icp_positions": [0, 1],
            },
        ),
        (
            "lab_arena_complete_attempt",
            {
                "p_run_id": "run-1",
                "p_lease_token_hash": "sha256:" + "1" * 64,
                "p_result": {"terminal_status": "accepted"},
                "p_terminal_cause": "accepted",
                "p_output_ref": "arena/round/run-1.json",
            },
        ),
    ]


def test_removed_durable_state_has_no_store_boundary():
    assert TABLES == (
        "lab_arena_rounds",
        "lab_arena_submissions",
        "lab_arena_runs",
        "lab_arena_ledger",
    )
    for removed in (
        "lab_arena_append_generation_attempt",
        "lab_arena_upsert_account_credential",
        "lab_arena_record_preflight",
    ):
        assert removed not in FUNCTION_SIGNATURES


def test_postgrest_rejects_http_loopback_lookalikes():
    for url in (
        "http://localhost.evil.example",
        "http://127.0.0.1.evil.example",
        "http://[::2]",
        "https://user:password@example.com",
    ):
        with pytest.raises(ArenaStoreError, match="base URL"):
            PostgrestTransport(url, anon_key="anon", service_jwt="a.b.c")


def test_production_http_client_ignores_proxy_environment_and_redirects(monkeypatch):
    captured = {}
    marker = object()

    def build(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(httpx, "Client", build)
    assert create_http1_client(8) is marker
    assert captured["follow_redirects"] is False
    assert captured["trust_env"] is False


def test_postgrest_does_not_follow_a_cross_origin_redirect():
    contacted = []

    def handler(request):
        contacted.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://attacker.example/collect"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False)
    transport = PostgrestTransport("https://project.example", anon_key="anon", service_jwt="a.b.c", http_client=client)
    with pytest.raises(ArenaStoreError, match="HTTP 307"):
        transport.rpc("lab_arena_whoami", {})
    assert contacted == ["https://project.example/rest/v1/rpc/lab_arena_whoami"]
    transport.close()
