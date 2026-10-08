"""Pending request errors through a real worker socket, gateway and ledger."""

import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import runner
from lab_arena import broker as br
from lab_arena.api import create_app
from lab_arena.store import hash_lease_token
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import (
    database as base_database,
)
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import (
    database, frame, setup,
)
from tests.lab_arena.deepline_completed_response_recovery_test import catalog
from tests.lab_arena.deepline_terminal_error_test import ErrorTransport, REQUEST_ID, billing


class ParallelErrorTransport(ErrorTransport):
    def send(self, **request):
        response = super().send(**request)
        if "/executions/by-key/" in request["url"]:
            document = json.loads(response.body)
            document["toolId"] = "parallel_search"
            return br.ProviderResponse(response.status, response.headers, json.dumps(document).encode())
        return response


class RecoverableErrorTransport(ParallelErrorTransport):
    def send(self, **request):
        response = super().send(**request)
        if "/executions/by-key/" in request["url"]:
            document = json.loads(response.body)
            document["executionRecovery"]["state"] = "completed"
            document["responseStatus"] = self.status
            document["response"] = json.loads(self.body)
            return br.ProviderResponse(response.status, response.headers, json.dumps(document).encode())
        return response


@pytest.mark.parametrize("settle_during_replay", [False, True])
@pytest.mark.parametrize("status,credits,amount,padding,body_kind,body", [
    (404, "0.5", 50_000, 0, "json", None),
    (422, "0", 0, 1_048_000, "large-json", None),
    (400, "0", 0, 0, "plain", b"Invalid tool"),
    (404, "0.5", 50_000, 0, "html", b"<html>Not Found</html>"),
    (422, "0", 0, 0, "empty", b""),
    (400, "0.5", 50_000, 0, "scalar", b'"invalid"'),
    (404, "0", 0, 0, "malformed", b'{"error":'),
    (422, "0.5", 50_000, 0, "invalid-utf8", b"\xff"),
])
def test_pending_error_reaches_client_and_late_charge_settles_once(
    database, tmp_path, request, monkeypatch, status, credits, amount, padding,
    body_kind, body, settle_during_replay,
):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    transport = ParallelErrorTransport(status, body=body)
    if padding:
        document = json.loads(transport.body)
        document["detail"] = "x" * padding
        transport.body = json.dumps(document).encode()
    body_index = ["json", "large-json", "plain", "html", "empty", "scalar", "malformed", "invalid-utf8"].index(body_kind)
    label = str(10 + body_index * 2 + int(settle_during_replay))
    h, lease, token, connect, broker = setup(database, tmp_path, label, transport)

    http = TestClient(create_app(h.service))
    request.addfinalizer(http.close)

    class GatewayApi(runner.HttpArenaApiClient):
        def __init__(self):
            super().__init__("http://127.0.0.1", client=http)
            self.frames = []

        def provider(self, run_id, lease_token, document):
            self.frames.append(dict(document))
            return super().provider(run_id, lease_token, document)

    api = GatewayApi()
    state = runner.RunState(
        lease=dict(lease, deepline_catalog=catalog()), lease_token=token,
    )
    # Keep the Unix socket path short on macOS as well as Linux.
    with tempfile.TemporaryDirectory(prefix="dl-error-", dir="/tmp") as socket_dir:
        path = Path(socket_dir) / "worker.sock"
        worker = runner.WorkerSocketServer(path, api, state)
        worker.start()
        try:
            with httpx.HTTPTransport(uds=str(path)) as client:
                provider_request = httpx.Request(
                    "POST",
                    "http://code.deepline.com/api/v2/integrations/parallel_search/execute",
                    json={"payload": frame()["parameters"]["payload"]},
                    extensions={"timeout": {key: 10 for key in ("connect", "read", "write", "pool")}},
                )
                response = client.handle_request(provider_request)
                response.request = provider_request
                response.read()
            assert response.status_code == status
            if body is None:
                assert response.content == transport.body
            else:
                assert isinstance(response.json().get("error"), dict)
                assert response.content != transport.body
                assert response.headers["content-type"] == "application/json"
            with pytest.raises(httpx.HTTPStatusError):
                response.raise_for_status()
            assert runner.SETTLED_MICROUSD_HEADER not in response.headers
            assert len(api.frames) == state.action_sequence == len(state.calls) == 1
            identity = state.calls[0]["call_identity"]
            assert response.headers[runner.CALL_IDENTITY_HEADER] == identity
            assert state.calls[0]["outcome"] == "uncertain"
        finally:
            worker.stop()

    before_replay = len(transport.sent)
    pending = api.provider(lease["run_id"], token, api.frames[0])
    assert pending["status"] == status and base64.b64decode(pending["body_b64"]) == response.content
    assert pending["call"]["idempotent"] is True and pending["call"]["outcome"] == "uncertain"
    assert "actual_microusd" not in pending["call"] and len(transport.sent) == before_replay

    store = h.service.store
    candidates = store.list_deepline_cost_reconciliations(h.round_id, run_id=lease["run_id"])
    assert len(candidates) == 1 and candidates[0]["call_identity"] == identity
    with connect() as connection, connection.cursor() as cur:
        cur.execute(
            "SELECT entry_kind,amount_microusd FROM public.lab_arena_ledger "
            "WHERE call_identity=%s ORDER BY entry_id", (identity,),
        )
        before = cur.fetchall()
        assert ("reservation", 0) in before and not any(row[0] == "settlement" for row in before)
        cur.execute(
            "SELECT content FROM public.lab_arena_trajectory_events "
            "WHERE run_id=%s AND event_kind='provider.response'", (lease["run_id"],),
        )
        (saved_response,) = cur.fetchone()
        assert saved_response["http_status"] == status
        assert saved_response["call"]["outcome"] == "uncertain"

    # A provider bill may charge a failed request. Keep that exact real spend,
    # without treating the error as successful research for cost eligibility.
    transport.bill = billing(credits=credits)
    entry = transport.bill["recent"]["entries"][0]
    entry.update(provider="parallel", operation="parallel_search")

    def reconcile():
        with ThreadPoolExecutor(max_workers=2) as pool:
            reconciled = list(pool.map(lambda _: broker.reconcile_deepline_cost(candidates[0]), range(2)))
        assert all(result["status"] == "settled" and result["actual_microusd"] == amount
                   for result in reconciled)

    if settle_during_replay:
        original_list_ledger = store.list_ledger

        def settle_before_ledger_read(**kwargs):
            monkeypatch.setattr(store, "list_ledger", original_list_ledger)
            reconcile()
            return original_list_ledger(**kwargs)

        # The reservation RPC sees uncertainty; the following history read sees
        # the concurrent settlement. Both reads are genuine PostgreSQL results.
        monkeypatch.setattr(store, "list_ledger", settle_before_ledger_read)
        raced = api.provider(lease["run_id"], token, api.frames[0])
        assert raced["status"] == status and base64.b64decode(raced["body_b64"]) == response.content
        assert raced["call"]["outcome"] == "settled"
        assert raced["call"]["actual_microusd"] == amount
    else:
        reconcile()
    assert store.list_deepline_cost_reconciliations(h.round_id, run_id=lease["run_id"]) == []
    with connect() as connection, connection.cursor() as cur:
        cur.execute(
            "SELECT entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger "
            "WHERE call_identity=%s ORDER BY entry_id", (identity,),
        )
        after = cur.fetchall()
        assert [(row[0], row[1]) for row in after[:len(before)]] == before
        settlements = [row for row in after if row[0] == "settlement"]
        assert len(settlements) == 1 and settlements[0][1] == amount
        assert settlements[0][2]["call_succeeded"] is False
        cur.execute(
            "SELECT public.lab_arena__successful_icp_cost_state(%s,%s,%s)",
            (h.round_id, lease["submission_id"], lease["icp_position"]),
        )
        cost = cur.fetchone()[0]
        assert cost["settled_microusd"] == amount
        assert cost["successful_microusd"] == cost["successful_calls"] == 0
        assert cost["uncertain_calls"] == cost["success_unresolved_calls"] == 0
    snapshot = store.run_quota_snapshot(lease["run_id"], hash_lease_token(token))
    assert snapshot["providers"]["deepline"]["used"] == 1
    assert [request["method"] for request in transport.sent].count("POST") == 1

    before_replay = len(transport.sent)
    replay = api.provider(lease["run_id"], token, api.frames[0])
    assert replay["status"] == status and base64.b64decode(replay["body_b64"]) == response.content
    assert replay["call"]["idempotent"] is True and replay["call"]["outcome"] == "settled"
    assert replay["call"]["actual_microusd"] == amount
    assert len(transport.sent) == before_replay  # Neither billing reads nor another paid POST.

    # Retained private responses remain protected by the live run lease.
    denied = api.provider(lease["run_id"], "0" * 64, api.frames[0])
    assert denied["call"]["error_code"] == "lease_stale"
    assert base64.b64decode(denied["body_b64"]) != response.content
    assert len(transport.sent) == before_replay


@pytest.mark.parametrize("status", [400, 404, 422])
@pytest.mark.parametrize("credits,amount", [("0", 0), ("0.5", 50_000)])
def test_settlement_before_mark_uncertain_returns_original_error_and_final_cost(
    database, tmp_path, monkeypatch, status, credits, amount,
):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    body = json.dumps({"request_id": REQUEST_ID, "error": {"code": "NOT_FOUND"}}, separators=(",", ":")).encode()
    transport = RecoverableErrorTransport(status, body=body)
    label = str(26 + [400, 404, 422].index(status) * 2 + int(amount > 0))
    h, lease, token, connect, broker = setup(database, tmp_path, label, transport)
    store = h.service.store
    original_mark = store.mark_uncertain
    observed = {}

    def settle_before_original_mark(**kwargs):
        monkeypatch.setattr(store, "mark_uncertain", original_mark)
        recovered = h.service.handle_provider(lease["run_id"], token, frame())
        assert recovered["status"] == status
        assert base64.b64decode(recovered["body_b64"]) == body
        assert recovered["call"]["outcome"] == "uncertain"
        candidates = store.list_deepline_cost_reconciliations(h.round_id, run_id=lease["run_id"])
        assert len(candidates) == 1
        transport.bill = billing(credits=credits)
        transport.bill["recent"]["entries"][0].update(provider="parallel", operation="parallel_search")
        settled = broker.reconcile_deepline_cost(candidates[0])
        assert settled["status"] == "settled" and settled["actual_microusd"] == amount
        state = original_mark(**kwargs)
        observed.update(state)
        return state

    # A same-action recovery finishes while the original request is still
    # returning. Each state change below uses the actual PostgreSQL RPC.
    monkeypatch.setattr(store, "mark_uncertain", settle_before_original_mark)
    result = h.service.handle_provider(lease["run_id"], token, frame())
    assert observed["status"] == "settled" and observed["amount_microusd"] == amount
    assert observed["terminal_response"]["status"] == 502  # Billing has no original response body.
    assert result["status"] == status and base64.b64decode(result["body_b64"]) == body
    assert result["call"]["outcome"] == "settled" and result["call"]["actual_microusd"] == amount
    assert "error_code" not in result["call"]
    identity = result["call"]["call_identity"]
    before = len(transport.sent)
    replay = h.service.handle_provider(lease["run_id"], token, frame())
    assert replay["status"] == status and base64.b64decode(replay["body_b64"]) == body
    assert replay["call"]["outcome"] == "settled" and replay["call"]["actual_microusd"] == amount
    assert replay["call"]["idempotent"] is True and len(transport.sent) == before
    assert sum(r["method"] == "POST" for r in transport.sent) == 1
    assert store.list_deepline_cost_reconciliations(h.round_id, run_id=lease["run_id"]) == []
    with connect() as connection, connection.cursor() as cur:
        cur.execute(
            "SELECT entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger "
            "WHERE call_identity=%s ORDER BY entry_id", (identity,),
        )
        entries = cur.fetchall()
        assert [row[0] for row in entries] == ["reservation", "dispatch", "uncertain", "settlement"]
        assert entries[-1][1] == amount and entries[-1][2]["call_succeeded"] is False
        cur.execute("SELECT public.lab_arena__successful_icp_cost_state(%s,%s,%s)",
                    (h.round_id, lease["submission_id"], lease["icp_position"]))
        cost = cur.fetchone()[0]
        assert cost["settled_microusd"] == amount
        assert cost["successful_microusd"] == cost["successful_calls"] == 0
        assert cost["uncertain_calls"] == cost["success_unresolved_calls"] == 0
