"""Same exact-recovery and ledger cursor contract in both service transports."""

import httpx
import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PostgrestTransport


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
    for extra in ({"execution_key": "key"}, {"recovered_request_id": "id"}):
        with pytest.raises(ArenaStoreError, match="requires key and request ID"):
            store.reconcile_deepline_cost(**args, **extra)
    assert len(requests) == 2
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
