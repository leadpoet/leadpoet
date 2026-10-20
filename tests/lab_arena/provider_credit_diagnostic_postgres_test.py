"""Miner credit diagnostics through real HTTP and append-only SQL accounting."""

from dataclasses import replace
import json

from fastapi.testclient import TestClient
import pytest

from lab_arena import broker as br, runner
from lab_arena.api import create_app
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import (
    database as base_database, fixture_configuration,
)
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import database, frame
from tests.lab_arena.deepline_completed_response_recovery_test import catalog
from tests.lab_arena.deepline_host_payment_status_test import REFUSAL
from tests.lab_arena.deepline_terminal_error_postgres_test import ParallelErrorTransport
from tests.lab_arena.deepline_terminal_error_test import billing
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _start_parallel_round
from tests.lab_arena.test_lab_arena_broker import HOST_KEYS, price_table
from tests.lab_arena.test_lab_arena_migration_postgres import claim
from tests.lab_arena.test_lab_arena_service_round import Harness, keypair


@pytest.mark.parametrize("credits,actual,label", [("0", 0, "41"), ("0.02", 2000, "42")])
def test_miner_unknown_fee_credit_reason_survives_sql_replay_and_reconciliation(
    database, tmp_path, monkeypatch, credits, actual, label,
):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    connect = lambda: database[0].connect(**database[1])
    h = Harness(connect, tmp_path, challengers=["CreditRefusal" + label], runners=[label])
    miner_hotkey = keypair("svc-miner-CreditRefusal" + label).ss58_address
    h.chain.runners.append(miner_hotkey)
    h.chain.permits[miner_hotkey] = False
    h.service.config.defaults = replace(h.service.config.defaults,
        per_icp_cost_policy=True, integrity_from="2000-01-01T00:00:00Z")
    _start_parallel_round(h, "arena-2099-09-" + label, slot_ceiling=2)
    with fixture_configuration(connect) as cursor:
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                       "jsonb_set(jsonb_set(configuration_doc,'{call_quotas,deepline}','0'),"
                       "'{deepline_catalog}',%s::jsonb) WHERE round_id=%s",
                       (json.dumps(catalog()), h.round_id))
    lease, token = claim(h.service.store, h.round_id, h.runner_keys[0],
                        parallelism=2, ceiling=2, excluded=(h.baseline_hotkey,))[:2]
    assert not h.service.store.get_submission(lease["submission_id"])["is_king"]
    assert h.service.store.provider_funding(lease["run_id"], "deepline")["funding_source"] == "miner_key"
    transport = ParallelErrorTransport(402, body=json.dumps(REFUSAL).encode())
    broker = br.Broker(store=h.service.store, key_for=lambda provider: HOST_KEYS[provider],
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key", price_table=price_table(),
        transport=transport, clock=h.clock)
    h.service._broker_for = lambda _round_id: broker
    h.service._hot_rounds.clear()

    with TestClient(create_app(h.service)) as http:
        api = runner.HttpArenaApiClient("http://127.0.0.1", client=http)
        first = api.provider(lease["run_id"], token, frame())
        assert first["status"] == 402
        assert first["call"]["error_code"] == "miner_credentials_unavailable"
        assert first["call"]["outcome"] == "uncertain"
        assert "actual_microusd" not in first["call"] and "credit_failure_proof" not in first["call"]
        assert br.provider_credit_failure_reason(first["call"]) == "provider_credit_exhausted"
        evidence = first["call"]["account_failure_evidence"]
        identity = first["call"]["call_identity"]
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT entry_kind,amount_microusd,entry_doc,terminal_response "
                           "FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id", (identity,))
            original = cursor.fetchall()
            assert [entry[0] for entry in original] == ["reservation", "dispatch", "uncertain"]
            assert all(entry[1] == 0 for entry in original)
            assert original[-1][2]["call"]["account_failure_evidence"] == evidence
            assert evidence["reason"] == "out_of_credit"
            cursor.execute("SELECT content#>>'{call,account_failure_evidence,reason}' "
                           "FROM public.lab_arena_trajectory_events WHERE run_id=%s "
                           "AND event_kind='provider.response' ORDER BY trajectory_id", (lease["run_id"],))
            assert cursor.fetchall() == [("out_of_credit",)]
        before_requests = len(transport.sent)
        replay = api.provider(lease["run_id"], token, frame())
        assert replay["status"] == 409 and replay["call"]["error_code"] == "call_uncertain"
        assert replay["call"]["account_failure_evidence"] == evidence
        assert br.provider_credit_failure_reason(replay["call"]) == "provider_credit_exhausted"
        assert len(transport.sent) == before_requests

        candidates = h.service.store.list_deepline_cost_reconciliations(h.round_id, run_id=lease["run_id"])
        assert len(candidates) == 1 and candidates[0]["funding_source"] == "miner_key"
        transport.bill = billing(credits=credits)
        transport.bill["recent"]["entries"][0].update(provider="parallel", operation="parallel_search")
        settled = broker.reconcile_deepline_cost(candidates[0])
        assert settled["status"] == "settled" and settled["actual_microusd"] == actual
        assert broker.reconcile_deepline_cost(candidates[0])["actual_microusd"] == actual
        # An unavailable original response keeps the existing billing-only
        # 502 reply. Only its bound diagnostic is restored from SQL history.
        original_send = transport.send
        def without_saved_response(**request):
            if "/executions/by-key/" in request["url"]:
                transport.sent.append(request)
                return br.ProviderResponse(404, {}, b"{}")
            return original_send(**request)
        transport.send = without_saved_response
        replay = api.provider(lease["run_id"], token, frame())
        assert replay["status"] == 502 and replay["call"]["outcome"] == "settled"
        assert replay["call"]["actual_microusd"] == actual
        assert replay["call"]["account_failure_evidence"] == evidence
        assert br.provider_credit_failure_reason(replay["call"]) == "provider_credit_exhausted"
        assert "credit_failure_proof" not in replay["call"]
        assert sum(request["method"] == "POST" for request in transport.sent) == 1

    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT entry_kind,amount_microusd,entry_doc,terminal_response "
                       "FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id", (identity,))
        after = cursor.fetchall()
        assert after[:-1] == original and after[-1][0:2] == ("settlement", actual)
        assert "credit_failure_proof" not in after[-1][3]
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='credential_error',"
                       "result_doc=%s::jsonb WHERE run_id=%s",
                       (json.dumps({"terminal_status": "credential_error"}), lease["run_id"]))
        cursor.execute("SET LOCAL session_replication_role=origin")
        cursor.execute("SELECT public.lab_arena_retry_credit_failures_v1(%s,%s,%s,%s)",
                       (h.round_id, lease["submission_id"], lease["miner_hotkey"], "sha256:" + "e" * 64))
        retry = cursor.fetchone()[0]
        assert retry["status"] == "no_eligible" and retry["requeued_count"] == 0
        assert retry["reason"] == "no_proved_failures"
