"""Host payment refusal survives real gateway, socket and ledger transitions."""

import base64
import json
from pathlib import Path
import tempfile

import httpx
from fastapi.testclient import TestClient

from lab_arena import broker as br, runner
from lab_arena.api import create_app
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import database as base_database
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import database, frame, setup
from tests.lab_arena.deepline_completed_response_recovery_test import catalog
from tests.lab_arena.deepline_host_payment_status_test import REFUSAL
from tests.lab_arena.deepline_terminal_error_postgres_test import ParallelErrorTransport
from tests.lab_arena.deepline_terminal_error_test import billing


def test_host_score_402_uncertainty_replay_exact_settlement_and_credit_retry_gate(
    database, tmp_path, monkeypatch,
):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    transport = ParallelErrorTransport(402, body=json.dumps(REFUSAL).encode())
    h, lease, token, connect, broker = setup(database, tmp_path, "30", transport)
    # Follow the existing disposable score-lease fixture. Keep its baseline
    # payer so the real funding RPC classifies this score as host funded.
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1_scoring' WHERE round_id=%s",
                       (h.round_id,))
        cursor.execute("UPDATE public.lab_arena_runs SET kind='score' WHERE run_id=%s", (lease["run_id"],))
    lease = {**lease, "kind": "score"}
    h.service._hot_rounds.clear()
    store = h.service.store
    assert store.provider_funding(lease["run_id"], "deepline")["funding_source"] == "host"

    with TestClient(create_app(h.service)) as http:
        class GatewayApi(runner.HttpArenaApiClient):
            def __init__(self):
                super().__init__("http://127.0.0.1", client=http)
                self.frames = []

            def provider(self, run_id, lease_token, document):
                self.frames.append(dict(document))
                return super().provider(run_id, lease_token, document)

        api = GatewayApi()
        state = runner.RunState(lease=dict(lease, deepline_catalog=catalog()), lease_token=token)
        with tempfile.TemporaryDirectory(prefix="dl-host-402-", dir="/tmp") as directory:
            socket_path = Path(directory) / "worker.sock"
            worker = runner.WorkerSocketServer(socket_path, api, state)
            worker.start()
            try:
                with httpx.HTTPTransport(uds=str(socket_path)) as client:
                    request = httpx.Request("POST",
                        "http://code.deepline.com/api/v2/integrations/parallel_search/execute",
                        json={"payload": frame()["parameters"]["payload"]},
                        extensions={"timeout": {key: 10 for key in ("connect", "read", "write", "pool")}})
                    response = client.handle_request(request)
                    response.request = request
                    response.read()
                assert response.status_code == 402 and response.json() == {"error": {"code": "provider_unavailable"}}
                assert runner.SETTLED_MICROUSD_HEADER not in response.headers
                assert len(api.frames) == len(state.calls) == state.action_sequence == 1
                call = state.calls[0]
                identity = call["call_identity"]
                assert call["funding_source"] == "host" and call["outcome"] == "uncertain"
                assert call["error_code"] == "provider_unavailable" and state.refusals == 0
                assert "actual_microusd" not in call and "credit_failure_proof" not in call
            finally:
                worker.stop()

        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT entry_kind,amount_microusd,entry_doc,terminal_response "
                           "FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id", (identity,))
            before = cursor.fetchall()
        assert [row[0] for row in before] == ["reservation", "dispatch", "uncertain"]
        assert before[0][1] == 0  # Production confirmed-cost admission preserves this reservation.
        saved = before[-1][2]["call"]
        assert saved["deepline_host_payment_refusal"] is True and saved["call_succeeded"] is False
        assert saved["deepline_terminal_response"]["status"] == 402
        assert "credit_failure_proof" not in saved and "account_failure_evidence" not in saved
        before_requests = len(transport.sent)
        replay = api.provider(lease["run_id"], token, api.frames[0])
        assert replay["status"] == 402 and base64.b64decode(replay["body_b64"]) == response.content
        assert replay["call"]["outcome"] == "uncertain" and replay["call"]["idempotent"]
        assert "actual_microusd" not in replay["call"] and len(transport.sent) == before_requests

        candidates = store.list_deepline_cost_reconciliations(h.round_id, run_id=lease["run_id"])
        assert len(candidates) == 1 and candidates[0]["funding_source"] == "host"
        transport.bill = billing(credits="0.02")
        transport.bill["recent"]["entries"][0].update(provider="parallel", operation="parallel_search")
        settled = broker.reconcile_deepline_cost(candidates[0])
        assert settled["status"] == "settled" and settled["actual_microusd"] == 2000
        assert broker.reconcile_deepline_cost(candidates[0])["actual_microusd"] == 2000
        before_requests = len(transport.sent)
        replay = api.provider(lease["run_id"], token, api.frames[0])
        assert replay["status"] == 402 and base64.b64decode(replay["body_b64"]) == response.content
        assert replay["call"]["outcome"] == "settled" and replay["call"]["actual_microusd"] == 2000
        assert replay["call"]["funding_source"] == "host" and replay["call"]["error_code"] == "provider_unavailable"
        assert "credit_failure_proof" not in replay["call"] and len(transport.sent) == before_requests
        assert sum(request["method"] == "POST" for request in transport.sent) == 1

        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT entry_kind,amount_microusd,entry_doc,terminal_response "
                           "FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id", (identity,))
            after = cursor.fetchall()
            assert after[:-1] == before and after[-1][0:2] == ("settlement", 2000)
            assert after[-1][3]["call_succeeded"] is False and "credit_failure_proof" not in after[-1][3]
            cursor.execute("SELECT public.lab_arena__successful_icp_cost_state(%s,%s,%s)",
                           (h.round_id, lease["submission_id"], lease["icp_position"]))
            cost = cursor.fetchone()[0]
            # This sourcing-cost projection excludes judge calls; the real
            # scoring ledger above still retains the full positive charge.
            assert cost["settled_microusd"] == cost["successful_microusd"] == cost["successful_calls"] == 0
            assert cost["uncertain_calls"] == 0

            # Even a wrong terminal label cannot turn host funding and absent
            # credit proof into authority for the miner's credit-retry RPC.
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_submissions SET is_king=FALSE WHERE submission_id=%s",
                           (lease["submission_id"],))
            cursor.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='credential_error',"
                           "result_doc=%s::jsonb WHERE run_id=%s",
                           (json.dumps({"terminal_status": "credential_error"}), lease["run_id"]))
            cursor.execute("SET LOCAL session_replication_role=origin")
            cursor.execute("SELECT public.lab_arena_retry_credit_failures_v1(%s,%s,%s,%s)",
                           (h.round_id, lease["submission_id"], lease["miner_hotkey"], "sha256:" + "d" * 64))
            retry = cursor.fetchone()[0]
            assert retry["status"] == "no_eligible" and retry["requeued_count"] == 0
            assert retry["reason"] == "no_proved_failures"
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE credit_retry_parent_run_id=%s", (lease["run_id"],))
            assert cursor.fetchone()[0] == 0
