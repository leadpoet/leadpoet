"""A failed evidence read does not prove an incomplete saved evaluation."""
import base64
import copy
from datetime import datetime, timezone

import httpx
import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, ArenaStoreUnavailable, PostgrestTransport
from tests.lab_arena.empty_execution_publication_test import case


def publication_case():
    service, reads, _, ledger, output, _ = case()
    # The accepted empty execution and its zero score are already durable.
    reads.runs[0]["per_icp_score"] = 0.0
    ledger[-1]["terminal_response"].update(
        status=200, call_succeeded=True,
        body_b64=base64.b64encode(b'{"result":"valid response"}').decode(),
    )
    service.now = lambda: datetime(2026, 10, 21, 21, tzinfo=timezone.utc)
    service._round = lambda _: reads.round_row
    # Isolate the execution-evidence gate, after baseline validation succeeds.
    service._baseline_fully_succeeded = lambda _: True
    service._store.list_runs = lambda *a, **kw: (
        [] if kw.get("kind") == "score" else copy.deepcopy(reads.runs)
    )
    service._store.submission_costs = lambda _: {"providers": []}
    cancellations = []
    service._store.cancel_round = lambda *a: cancellations.append(a) or {"status": "cancelled"}
    return service, reads, ledger, output, cancellations


@pytest.mark.parametrize("phase", ["reservation", "lifecycle"])
@pytest.mark.parametrize("fault", ["timeout", "statement_timeout", 429, 502, 503, 504])
def test_unavailable_authenticated_select_does_not_cancel_saved_evaluation(phase, fault):
    service, reads, ledger, output, cancellations = publication_case()
    original = copy.deepcopy((reads.round_row, reads.runs, ledger, output))
    requests = []
    unavailable = True

    def handle(request):
        # Actual authenticated GET transport, without any network or live key.
        assert request.headers["apikey"] == "sb_secret_test"
        assert request.method == "GET"
        assert request.url.path == "/rest/v1/lab_arena_ledger"
        requests.append(request)
        reservation = request.url.params.get("entry_kind") == "eq.reservation"
        if unavailable and (phase == "reservation") == reservation:
            if fault == "timeout":
                raise httpx.ReadTimeout("synthetic unavailable read", request=request)
            if fault == "statement_timeout":
                return httpx.Response(500, json={"code": "57014", "message": "canceling statement due to statement timeout"})
            return httpx.Response(fault, json={})
        return httpx.Response(200, json=[ledger[0]] if reservation else ledger)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        transport_store = ArenaStore.__new__(ArenaStore)
        transport_store._transport = PostgrestTransport(
            "https://db.example", service_key="sb_secret_test", http_client=client,
        )
        service._store.latest_execution_model_reservation = transport_store.latest_execution_model_reservation
        service._store.list_ledger = transport_store.list_ledger
        with pytest.raises(ArenaStoreUnavailable) as caught:
            try:
                service.publish("round")
            finally:
                assert cancellations == []
        assert caught.value.store_error == {
            "operation": "select", "resource": "lab_arena_ledger",
            **({"transport_error": "ReadTimeout"} if fault == "timeout" else
               {"http_status": 500, "code": "57014"} if fault == "statement_timeout" else
               {"http_status": fault}),
        }
        attempts = 1 if fault == "statement_timeout" else 2
        assert len(requests) == attempts + int(phase == "lifecycle")
        assert (reads.round_row, reads.runs, ledger, output) == original

        # The next ordinary publication attempt can use the same saved result.
        unavailable = False
        service._participant_incomplete_reasons = lambda *a: {}
        class RankingReached(Exception):
            pass
        def ranking(*a, **kw):
            raise RankingReached
        service._score_entries_from_runs = ranking
        with pytest.raises(RankingReached):
            service.publish("round")
        assert len(requests) == attempts + int(phase == "lifecycle") + 2
    assert cancellations == []
    assert (reads.round_row, reads.runs, ledger, output) == original


@pytest.mark.parametrize("fault", ["missing_dispatch", "invalid_reservation", "missing_terminal"])
def test_malformed_authoritative_lifecycle_keeps_existing_deadline_cancellation(fault):
    service, _, ledger, _, cancellations = publication_case()
    if fault == "missing_dispatch":
        ledger.pop(1)  # A successful read proves the dispatch receipt is absent.
    elif fault == "invalid_reservation":
        ledger[0]["entry_doc"]["action_sequence"] = "3"
    else:
        ledger[-1]["terminal_response"] = None
    assert service.publish("round") == {"status": "cancelled"}
    assert cancellations == [("round", "scoring_incomplete")]


def test_incomplete_baseline_keeps_existing_deadline_cancellation():
    service, _, _, _, cancellations = publication_case()
    service._baseline_fully_succeeded = lambda _: False
    service._store.latest_execution_model_reservation = lambda _: pytest.fail("baseline gate must come first")
    assert service.publish("round") == {"status": "cancelled"}
    assert cancellations == [("round", "scoring_incomplete")]


@pytest.mark.parametrize("operation,code,expected", [
    ("select", "57014", ArenaStoreUnavailable),
    ("select", "42501", ArenaStoreError),
    ("select", "XX000", ArenaStoreError),
    ("select", "22023", ArenaStoreError),
    ("rpc", "57014", ArenaStoreError),
])
def test_only_select_query_cancellation_is_unavailable_without_added_retry(operation, code, expected):
    requests = []
    def handle(request):
        assert request.headers["apikey"] == "sb_secret_test"
        requests.append(request)
        return httpx.Response(500, json={"code": code})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        store = PostgrestTransport("https://db.example", service_key="sb_secret_test", http_client=client)
        with pytest.raises(ArenaStoreError) as caught:
            if operation == "select":
                store.select("lab_arena_ledger")
            else:
                store.rpc("lab_arena_reserve_call", {})
    assert type(caught.value) is expected
    assert caught.value.store_error == {
        "operation": operation,
        "resource": "lab_arena_ledger" if operation == "select" else "lab_arena_reserve_call",
        "http_status": 500, "code": code,
    }
    assert len(requests) == 1
    assert requests[0].method == ("GET" if operation == "select" else "POST")
