"""Exact late miner-key judge bills change only the audit ledger."""

from pathlib import Path

import pytest

from lab_arena import broker as br
from lab_arena.store import hash_lease_token
from tests.lab_arena.accepted_host_score_billing_postgres_test import database
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import (
    database as base_database,
)
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import (
    frame, setup,
)
from tests.lab_arena.deepline_completed_response_recovery_test import NATIVE
from tests.lab_arena.deepline_score_billing_latency_test import CompletedTransport
from tests.lab_arena.test_lab_arena_migration_postgres import complete

SCRIPTS = Path(__file__).parents[2] / "scripts"
UPGRADES = (
    "432-lab-arena-closed-host-score-billing.sql",
    "435-lab-arena-deepline-candidate-query.sql",
    "436-lab-arena-closed-provider-candidate-query.sql",
    "438-lab-arena-accepted-host-score-billing.sql",
    "439-lab-arena-closed-miner-success-score-billing.sql",
)
MINER_HELPER = "public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)"
HOST_HELPER = "public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)"
UNCHANGED = (
    HOST_HELPER,
    "public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)",
    "public.lab_arena__submission_kind_admission_spend_v1(text,text,boolean)",
    "public.lab_arena__submission_kind_has_admission_uncertainty_v1(text,text)",
)
CHANGED = (
    "public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)",
    "public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)",
    "public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)",
    "public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)",
    "public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)",
)


def definitions(cursor, signatures):
    found = {}
    for name in signatures:
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (name,))
        found[name] = cursor.fetchone()[0]
    return found


@pytest.fixture(scope="module")
def upgraded(database):
    connect = lambda: database[0].connect(**database[1])
    with connect() as connection, connection.cursor() as cursor:
        for name in UPGRADES[:-1]:
            cursor.execute((SCRIPTS / name).read_text())
        before = definitions(cursor, (*UNCHANGED, *CHANGED))
        cursor.execute((SCRIPTS / UPGRADES[-1]).read_text())
        after = definitions(cursor, (*UNCHANGED, *CHANGED, MINER_HELPER))
        cursor.execute((SCRIPTS / UPGRADES[-1]).read_text())
        assert definitions(cursor, (*UNCHANGED, *CHANGED, MINER_HELPER)) == after
        assert all(before[name] == after[name] for name in UNCHANGED)
        assert all(before[name] != after[name] for name in CHANGED)
    return database


def _make_call(database, tmp_path, label, terminal):
    transport = CompletedTransport()
    transport.response["job_id"] = NATIVE
    h, lease, token, connect, broker = setup(database, tmp_path, label, transport)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage1_scoring',"
            "configuration_doc=configuration_doc || "
            "'{\"sourcing_cost_eligibility_policy\":\"successful_calls_per_icp_v1\","
            "\"mode\":\"live\",\"network_name\":\"finney\",\"netuid\":71}'::jsonb "
            "WHERE round_id=%s", (h.round_id,),
        )
        cursor.execute("UPDATE public.lab_arena_runs SET kind='score' WHERE run_id=%s",
                       (lease["run_id"],))
        cursor.execute("UPDATE public.lab_arena_submissions SET is_king=FALSE "
                       "WHERE submission_id=%s", (lease["submission_id"],))
    h.service._hot_rounds.clear()
    assert h.service.store.provider_funding(lease["run_id"], "deepline")["funding_source"] == "miner_key"
    result = h.service.handle_provider(lease["run_id"], token, frame())
    assert result["status"] == 200 and result["call"]["outcome"] == "uncertain"
    candidate = h.service.store.list_deepline_cost_reconciliations(h.round_id)[0]
    completion = complete(h.service.store, lease["run_id"], hash_lease_token(token),
                          terminal, output_ref=("arena/test/miner-" + label + ".json")
                          if terminal == "accepted" else "")
    assert completion["status"] == ("accepted" if terminal == "accepted" else "failed")
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET status='published',"
                       "publication_doc='{\"immutable_snapshot\":true}'::jsonb "
                       "WHERE round_id=%s", (h.round_id,))
    return h, lease, connect, broker, transport, candidate


def _eligibility(cursor, round_id, entry_id):
    cursor.execute("SELECT public.lab_arena__closed_miner_score_success_uncertainty_v1(%s)",
                   (entry_id,))
    helper = cursor.fetchone()[0]
    cursor.execute("SELECT public.lab_arena_next_closed_deepline_reconciliation_v1("
                   "'live','finney',71,%s,0)", (round_id,))
    return helper, cursor.fetchone()[0]


def test_exact_miner_success_recovery_after_accepted_or_failed_score(
    upgraded, tmp_path, monkeypatch,
):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    for label, terminal in (("42", "accepted"), ("43", "judge_error")):
        h, lease, connect, broker, transport, candidate = _make_call(
            upgraded, tmp_path, label, terminal)
        store = h.service.store
        entry_id = candidate["uncertain_entry_id"]
        original_ledger = store.list_ledger(call_identity=candidate["call_identity"])
        assert [row["entry_kind"] for row in original_ledger] == [
            "reservation", "dispatch", "uncertain"]
        assert {row["funding_source"] for row in original_ledger} == {"miner_key"}
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r "
                           "WHERE round_id=%s", (h.round_id,))
            round_hash = cursor.fetchone()[0]
            cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_runs r "
                           "WHERE run_id=%s", (lease["run_id"],))
            run_hash = cursor.fetchone()[0]
            cursor.execute("SELECT public.lab_arena__submission_kind_has_admission_uncertainty_v1(%s,'score')",
                           (lease["submission_id"],))
            assert cursor.fetchone() == (True,)
            assert _eligibility(cursor, h.round_id, entry_id)[1]["uncertain_entry_id"] == entry_id
            cursor.execute("SELECT public.lab_arena_list_deepline_cost_reconciliations_v2(%s,'',0,1,TRUE)",
                           (h.round_id,))
            assert cursor.fetchone()[0]["items"] == []  # execute-only lane remains separate
            for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
                cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')",
                               (role, MINER_HELPER))
                assert cursor.fetchone() == (False,)
            cursor.execute("SELECT has_function_privilege('lab_arena_owner',%s,'EXECUTE')",
                           (MINER_HELPER,))
            assert cursor.fetchone() == (True,)
            cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r "
                           "WHERE round_id=%s", (h.round_id,))
            assert cursor.fetchone()[0] == round_hash
            cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_runs r "
                           "WHERE run_id=%s", (lease["run_id"],))
            assert cursor.fetchone()[0] == run_hash

        assert store.list_deepline_cost_reconciliations(h.round_id)[0]["uncertain_entry_id"] == entry_id
        selected = store.next_closed_provider_reconciliation(
            mode="live", network_name="finney", netuid=71, round_id=h.round_id)
        assert selected["provider"] == "deepline"
        assert selected["uncertain_entry_id"] == entry_id
        assert broker.reconcile_deepline_cost(candidate)["status"] == "pending"
        attempted_settlement = dict(
            round_id=candidate["round_id"], run_id=candidate["run_id"],
            call_identity=candidate["call_identity"], uncertain_entry_id=entry_id,
            request_id=candidate["request_id"], operation=candidate["operation"],
            credential_fingerprint="sha256:" + "b" * 64,
            actual_microusd=2000, cost_units="0.02",
            execution_key=candidate["execution_key"], recovered_request_id=NATIVE,
        )
        assert store.reconcile_deepline_cost(**attempted_settlement) == {"status": "stale"}
        assert store.list_ledger(call_identity=candidate["call_identity"]) == original_ledger
        post_count = sum(req["method"] == "POST" for req in transport.requests)
        assert post_count == 1
        transport.bill_final = True
        settled = broker.reconcile_deepline_cost(candidate)
        assert settled["status"] == "settled" and settled["actual_microusd"] == 2000
        assert broker.reconcile_deepline_cost(candidate)["status"] == "settled"
        assert sum(req["method"] == "POST" for req in transport.requests) == post_count
        after = store.list_ledger(call_identity=candidate["call_identity"])
        assert after[:-1] == original_ledger
        assert len(after) == 4 and after[-1]["entry_kind"] == "settlement"
        assert after[-1]["amount_microusd"] == 2000
        assert store.list_deepline_cost_reconciliations(h.round_id) == []
        assert store.next_closed_provider_reconciliation(
            mode="live", network_name="finney", netuid=71,
            round_id=h.round_id) == {"status": "none"}
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r "
                           "WHERE round_id=%s", (h.round_id,))
            assert cursor.fetchone()[0] == round_hash
            cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_runs r "
                           "WHERE run_id=%s", (lease["run_id"],))
            assert cursor.fetchone()[0] == run_hash


def test_miner_success_rejects_wrong_scope_and_damaged_identity(upgraded, tmp_path):
    h, lease, connect, _, _, candidate = _make_call(
        upgraded, tmp_path, "44", "accepted")
    entry_id = candidate["uncertain_entry_id"]
    assert _eligibility_for_connection(connect, h.round_id, entry_id)[0] is True
    changes = (
        ("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s", (lease["run_id"],)),
        ("UPDATE public.lab_arena_runs SET status='leased' WHERE run_id=%s", (lease["run_id"],)),
        ("UPDATE public.lab_arena_runs SET terminal_cause='judge_error' WHERE run_id=%s", (lease["run_id"],)),
        ("UPDATE public.lab_arena_ledger SET funding_source='host' WHERE entry_id=%s",
         (entry_id,)),
        ("UPDATE public.lab_arena_ledger SET stage=2 WHERE call_identity=%s AND entry_kind='dispatch'",
         (candidate["call_identity"],)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc-'deepline_execution_key' "
         "WHERE call_identity=%s AND entry_kind='reservation'", (candidate["call_identity"],)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,call_succeeded}',"
         "'false'::jsonb) WHERE entry_id=%s", (entry_id,)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,provider_status}',"
         "'402'::jsonb) WHERE entry_id=%s", (entry_id,)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,credential_fingerprint}',"
         "'\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"'::jsonb) "
         "WHERE entry_id=%s", (entry_id,)),
    )
    with connect() as connection, connection.cursor() as cursor:
        for statement, params in changes:
            cursor.execute("SAVEPOINT bad_miner_bill")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(statement, params)
            assert _eligibility(cursor, h.round_id, entry_id) == (
                False, {"status": "none"}), statement
            cursor.execute("ROLLBACK TO SAVEPOINT bad_miner_bill")


def _eligibility_for_connection(connect, round_id, entry_id):
    with connect() as connection, connection.cursor() as cursor:
        return _eligibility(cursor, round_id, entry_id)
