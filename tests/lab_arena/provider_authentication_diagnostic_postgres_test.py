"""Shared Runner sends its fixed diagnostic through HTTP and real SQL guards."""
from datetime import datetime, timezone
import json

from fastapi.testclient import TestClient
import pytest

from lab_arena import broker as br, public_dashboard, runner, scoring
from lab_arena.api import create_app
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import database as base_database
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import database
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _start_parallel_round, _verified_test_pool
from tests.lab_arena.test_lab_arena_broker import FakeTransport, HOST_KEYS, price_table
from tests.lab_arena.test_lab_arena_runner import RefusedJudgeRuntime
from tests.lab_arena.test_lab_arena_service_round import Harness, keypair


@pytest.mark.parametrize("upstream", [False, True])
def test_runner_http_completion_persists_auth_reason_and_public_progress(database, tmp_path, upstream):
    connect = lambda: database[0].connect(**database[1])
    flavor = "AuthenticationRefusal" + ("Upstream" if upstream else "Direct")
    h = Harness(connect, tmp_path, challengers=[flavor], runners=["alpha"])
    miner_hotkey = keypair("svc-miner-" + flavor).ss58_address
    h.chain.runners.append(miner_hotkey)
    h.chain.permits[miner_hotkey] = False
    _start_parallel_round(h, "arena-2099-09-" + ("52" if upstream else "51"), slot_ceiling=2)
    # Keep the baseline positions out of this single miner execution.
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND miner_hotkey=%s", (h.round_id, h.baseline_hotkey))
        baseline_ids = [item[0] for item in cursor.fetchall()]
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE run_id=ANY(%s)", (baseline_ids,))
        cursor.execute("SET LOCAL session_replication_role=origin")
    native = {"error": {"code": 401, "message": "private auth detail"}}
    if upstream:
        native["error"]["metadata"] = {"provider_name": "synthetic_vendor", "provider_code": "invalid_api_key"}
    transport = FakeTransport([(401, native)])
    broker = br.Broker(store=h.service.store, key_for=lambda provider: HOST_KEYS[provider],
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key", price_table=price_table(),
        transport=transport, clock=h.clock)
    h.service._broker_for = lambda _round_id: broker
    h.service._hot_rounds.clear()
    h.clock.now = datetime.now(timezone.utc)
    with TestClient(create_app(h.service)) as http:
        class GatewayApi(runner.HttpArenaApiClient):
            held = None
            def claim(self, envelope):
                if self.held is not None:
                    return {"status": "no_pending"}
                self.held = super().claim(envelope)
                return self.held
        api = GatewayApi("http://127.0.0.1", client=http)
        h.api_factory = lambda: api
        h.sandbox = RefusedJudgeRuntime(output=scoring.build_scoring_failure("unused", "judge_error", detail="private"))
        shared_runner = h.runner(0, parallel=1)
        shared_runner._config.proxy_worker_pool = _verified_test_pool(2)
        assert shared_runner.run_once() == 1
        assert shared_runner.abandoned == 0, shared_runner.completed
        held = api.held
    saved = h.service.store.get_run(held["run_id"])
    assert saved["status"] == "failed" and saved["terminal_cause"] == "credential_error"
    expected = "provider_credentials" if upstream else "provider_authentication_failed"
    if upstream:
        assert "failure_diagnostic" not in saved["result_doc"]
    else:
        assert saved["result_doc"]["failure_diagnostic"]["reason"] == expected
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger "
                       "WHERE run_id=%s ORDER BY entry_id", (held["run_id"],))
        ledger = cursor.fetchall()
        assert [entry[0] for entry in ledger] == ["reservation", "dispatch", "settlement"]
        assert ledger[-1][1] == 0
        assert ledger[0][1] >= 0 and ledger[1][1] >= 0
        assert ledger[-1][2]["call_succeeded"] is False
        evidence = ledger[-1][2]["account_failure_evidence"]
        assert evidence["provider_status"] == 401
        assert evidence.get("submitted_account_authentication") is (None if upstream else True)
        cursor.execute("SELECT run_id,assignment_id,submission_id,kind,status,terminal_cause,"
                       "result_doc#>>'{failure_diagnostic,reason}' AS failure_reason "
                       "FROM public.lab_arena_runs WHERE run_id=%s", (held["run_id"],))
        projected_run = dict(zip([column.name for column in cursor.description], cursor.fetchone()))
    progress = public_dashboard.evaluation_progress([projected_run], h.clock.now)
    assert progress["failure_reasons"] == [expected]
    assert "private" not in json.dumps(progress)
    assert len(transport.sent) == 1
