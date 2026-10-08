"""Same exact-recovery and ledger cursor contract in both service transports."""

import json

import httpx
import pytest

from lab_arena.store import (
    ArenaStore, ArenaStoreError, ArenaStoreUnavailable,
    BULK_ROUND_RPC_READ_TIMEOUT_SECONDS, PostgrestTransport,
)


@pytest.mark.parametrize(
    ("function", "params", "expected_read"),
    [
        ("lab_arena_open_stage", {}, BULK_ROUND_RPC_READ_TIMEOUT_SECONDS),
        ("lab_arena_open_scoring", {}, BULK_ROUND_RPC_READ_TIMEOUT_SECONDS),
        ("lab_arena_open_scoring_v2", {}, BULK_ROUND_RPC_READ_TIMEOUT_SECONDS),
        ("lab_arena_open_scoring_v3", {}, BULK_ROUND_RPC_READ_TIMEOUT_SECONDS),
        ("lab_arena_record_run_scores", {}, BULK_ROUND_RPC_READ_TIMEOUT_SECONDS),
        ("lab_arena_transition_round", {"p_expected_status": "scored", "p_next_status": "published"}, BULK_ROUND_RPC_READ_TIMEOUT_SECONDS),
        ("lab_arena_whoami", {}, 8.0),
        ("lab_arena_transition_round", {"p_expected_status": "stage2", "p_next_status": "scoring"}, 8.0),
    ],
)
def test_round_rpc_read_timeout_preserves_other_timeouts(function, params, expected_read):
    observed = []

    def handle(request):
        observed.append(request.extensions["timeout"])
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.Client(
        transport=httpx.MockTransport(handle),
        timeout=httpx.Timeout(connect=1.5, read=8.0, write=2.5, pool=3.5),
    )
    store = PostgrestTransport(
        "https://db.example", service_key="sb_secret_test", http_client=client,
    )
    assert store.rpc(function, params) == {"status": "ok"}
    assert observed == [{"connect": 1.5, "read": expected_read, "write": 2.5, "pool": 3.5}]
    store.close()


@pytest.mark.parametrize("function", [
    "lab_arena_open_scoring", "lab_arena_open_scoring_v2", "lab_arena_open_scoring_v3",
    "lab_arena_record_run_scores",
])
@pytest.mark.parametrize("failure", ["read_timeout", "http_503"])
def test_scoring_rpc_response_failure_never_replays_post(function, failure):
    requests = []

    def handle(request):
        requests.append(request)
        if failure == "read_timeout":
            raise httpx.ReadTimeout("response lost")
        return httpx.Response(503, json={"message": "unavailable"})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    store = PostgrestTransport(
        "https://db.example", service_key="sb_secret_test", http_client=client,
    )
    error = ArenaStoreUnavailable if failure == "read_timeout" else ArenaStoreError
    with pytest.raises(error):
        store.rpc(function, {})
    assert len(requests) == 1
    store.close()


def test_keyed_reconciliation_uses_v2_and_legacy_uses_v1():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "settled"})

    store = ArenaStore(
        PostgrestTransport(
            "https://db.example",
            service_key="sb_secret_test",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
    )
    args = dict(
        round_id="round",
        run_id="run",
        call_identity="sha256:" + "a" * 64,
        uncertain_entry_id=1,
        request_id="ctx-tool-" + "a" * 32,
        operation="exa_search",
        credential_fingerprint="sha256:" + "b" * 64,
        actual_microusd=2000,
        cost_units="0.02",
    )
    assert store.reconcile_deepline_cost(**args)["status"] == "settled"
    assert requests[-1].url.path.endswith("/lab_arena_reconcile_deepline_cost_v1")
    assert (
        store.reconcile_deepline_cost(
            **args, execution_key="arena:" + "a" * 64, recovered_request_id="native-id"
        )["status"]
        == "settled"
    )
    assert requests[-1].url.path.endswith("/lab_arena_reconcile_deepline_cost_v2")
    assert store.reconcile_deepline_cost(
        **args, recovered_request_id="native-id"
    )["status"] == "settled"
    assert requests[-1].url.path.endswith("/lab_arena_reconcile_deepline_cost_v2")
    body = json.loads(requests[-1].content)
    assert body["p_execution_key"] is None and body["p_recovered_request_id"] == "native-id"
    with pytest.raises(ArenaStoreError, match="requires key and request ID"):
        store.reconcile_deepline_cost(**args, execution_key="key")
    assert len(requests) == 3
    store.close()


def test_ledger_cursor_and_operation_filter_use_parameterized_read():
    requests = []
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, json=[])
        )
    )
    transport = PostgrestTransport(
        "https://db.example", service_key="sb_secret_test", http_client=client
    )
    store = ArenaStore(transport)
    assert (
        store.list_ledger(
            run_id="run",
            provider="deepline",
            operation_id="deepline.execute",
            after_entry_id=500,
            limit=1000,
        )
        == []
    )
    params = requests[0].url.params
    assert params["run_id"] == "eq.run" and params["provider"] == "eq.deepline"
    assert (
        params["operation_id"] == "eq.deepline.execute"
        and params["entry_id"] == "gt.500"
    )
    assert params["order"] == "entry_id.asc" and params["limit"] == "1000"
    for value in (-1, True, "1"):
        with pytest.raises(ArenaStoreError, match="cursor"):
            store.list_ledger(after_entry_id=value)
    for options in (
        {"table": "lab_arena_runs", "order": "entry_id"},
        {"table": "lab_arena_ledger", "order": "run_id"},
        {"table": "lab_arena_ledger", "order": "entry_id", "descending": True},
    ):
        with pytest.raises(ArenaStoreError, match="cursor"):
            transport.select(after_entry_id=10, **options)
    assert len(requests) == 1
    store.close()


def test_priority_list_uses_v2_and_ordinary_keeps_v1():
    requests = []
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(
            200, json={"status": "ok", "items": []}
        )
    ))
    store = ArenaStore(PostgrestTransport(
        "https://db.example", service_key="sb_secret_test", http_client=client
    ))
    for priority in (False, True):
        assert store.list_deepline_cost_reconciliations(
            "round", after_entry_id=100, successful_execute_only=priority
        ) == []
    assert requests[0].url.path.endswith('/lab_arena_list_deepline_cost_reconciliations_v1')
    assert requests[1].url.path.endswith('/lab_arena_list_deepline_cost_reconciliations_v2')
    ordinary = json.loads(requests[0].content)
    priority = json.loads(requests[1].content)
    assert ordinary == dict(p_round_id='round', p_run_id='', p_after_entry_id=100, p_limit=1)
    assert priority == {**ordinary, 'p_successful_execute_only': True}
    store.close()
