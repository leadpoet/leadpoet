"""An accepted host-funded score may recover its exact late bill after publication."""

from pathlib import Path

from lab_arena import broker as br
from lab_arena.store import hash_lease_token
from tests.lab_arena.closed_host_score_billing_postgres_test import (
    closed_call,
    database,
)
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import (
    database as base_database,
)
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import (
    frame,
    setup,
)
from tests.lab_arena.deepline_completed_response_recovery_test import NATIVE
from tests.lab_arena.deepline_score_billing_latency_test import CompletedTransport
from tests.lab_arena.test_lab_arena_migration_postgres import complete

SCRIPTS = Path(__file__).parents[2] / "scripts"
MIGRATION_432 = SCRIPTS / "432-lab-arena-closed-host-score-billing.sql"
MIGRATION_436_QUERY = SCRIPTS / "436-lab-arena-closed-provider-candidate-query.sql"
MIGRATION_438 = SCRIPTS / "438-lab-arena-accepted-host-score-billing.sql"
HELPER = "public.lab_arena__closed_host_score_success_uncertainty_v1"
HELPER_SIGNATURE = HELPER + "(bigint)"
SELECTOR_SIGNATURE = (
    "public.lab_arena_next_closed_deepline_reconciliation_v1"
    "(text,text,integer,text,bigint)"
)
UNCHANGED_FUNCTIONS = (
    "public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)",
    "public.lab_arena__submission_kind_admission_spend_v1(text,text,boolean)",
    "public.lab_arena__submission_kind_has_admission_uncertainty_v1(text,text)",
)


def _function_definitions(cursor):
    signatures = (HELPER_SIGNATURE, SELECTOR_SIGNATURE, *UNCHANGED_FUNCTIONS)
    definitions = {}
    for signature in signatures:
        cursor.execute(
            "SELECT pg_get_functiondef(to_regprocedure(%s))", (signature,)
        )
        definitions[signature] = cursor.fetchone()[0]
    return definitions


def _accepted_call(database, tmp_path):
    transport = CompletedTransport()
    transport.response["job_id"] = NATIVE
    h, lease, token, connect, broker = setup(
        database, tmp_path, "24", transport
    )
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage1_scoring',"
            "configuration_doc=configuration_doc || "
            "'{\"sourcing_cost_eligibility_policy\":\"successful_calls_per_icp_v1\","
            "\"mode\":\"live\",\"network_name\":\"finney\",\"netuid\":71}'::jsonb "
            "WHERE round_id=%s", (h.round_id,),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET kind='score' WHERE run_id=%s",
            (lease["run_id"],),
        )
    h.service._hot_rounds.clear()
    result = h.service.handle_provider(lease["run_id"], token, frame())
    assert result["status"] == 200 and result["call"]["outcome"] == "uncertain"
    candidate = h.service.store.list_deepline_cost_reconciliations(h.round_id)[0]
    accepted = complete(
        h.service.store, lease["run_id"], hash_lease_token(token), "accepted",
        output_ref="arena/test/accepted-host436.json",
    )
    assert accepted["status"] == "accepted"
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='published',"
            "publication_doc='{\"immutable_snapshot\":true}'::jsonb "
            "WHERE round_id=%s", (h.round_id,),
        )
    return h, lease, connect, broker, transport, dict(candidate, run_status="accepted")


def _helper(cursor, entry_id):
    cursor.execute("SELECT " + HELPER + "(%s)", (entry_id,))
    return cursor.fetchone()[0]


def _closed_candidate(cursor, round_id):
    cursor.execute(
        "SELECT public.lab_arena_next_closed_deepline_reconciliation_v1("
        "'live','finney',71,%s,0)", (round_id,),
    )
    return cursor.fetchone()[0]


def test_accepted_host_score_exact_bill_without_publication_or_admission_rewrite(
    database, tmp_path, monkeypatch,
):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    h, lease, connect, broker, transport, candidate = _accepted_call(
        database, tmp_path
    )
    store = h.service.store
    old_ledger = store.list_ledger(call_identity=candidate["call_identity"])
    assert [entry["entry_kind"] for entry in old_ledger] == [
        "reservation", "dispatch", "uncertain"
    ]
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(MIGRATION_432.read_text())
        cursor.execute(MIGRATION_436_QUERY.read_text())
        assert _helper(cursor, candidate["uncertain_entry_id"]) is False
        assert _closed_candidate(cursor, h.round_id) == {"status": "none"}
        before = _function_definitions(cursor)
        cursor.execute(
            "SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r "
            "WHERE round_id=%s", (h.round_id,),
        )
        published_hash = cursor.fetchone()[0]
        cursor.execute(MIGRATION_438.read_text())
        first = _function_definitions(cursor)
        cursor.execute(MIGRATION_438.read_text())
        assert _function_definitions(cursor) == first
        assert before[HELPER_SIGNATURE] != first[HELPER_SIGNATURE]
        assert before[SELECTOR_SIGNATURE] != first[SELECTOR_SIGNATURE]
        assert all(before[name] == first[name] for name in UNCHANGED_FUNCTIONS)
        assert _helper(cursor, candidate["uncertain_entry_id"]) is True
        assert _closed_candidate(cursor, h.round_id)["uncertain_entry_id"] == candidate["uncertain_entry_id"]
        cursor.execute(
            "SELECT public.lab_arena_list_deepline_cost_reconciliations_v1(%s,'',0,1)",
            (h.round_id,),
        )
        assert [row["uncertain_entry_id"] for row in cursor.fetchone()[0]["items"]] == [
            candidate["uncertain_entry_id"]
        ]
        cursor.execute(
            "SELECT public.lab_arena_list_deepline_cost_reconciliations_v2(%s,'',0,1,TRUE)",
            (h.round_id,),
        )
        assert cursor.fetchone()[0]["items"] == []  # execute-only priority stays separate
        for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
            cursor.execute(
                "SELECT has_function_privilege(%s,%s,'EXECUTE')",
                (role, HELPER_SIGNATURE),
            )
            assert cursor.fetchone() == (False,)
        cursor.execute(
            "SELECT has_function_privilege('lab_arena_owner',%s,'EXECUTE')",
            (HELPER_SIGNATURE,),
        )
        assert cursor.fetchone() == (True,)

        # Invalid accepted status/cause, wrong payer, unknown provider result,
        # and identity damage may never turn into a closed billing candidate.
        cursor.execute("BEGIN")
        changes = (
            ("UPDATE public.lab_arena_runs SET terminal_cause='judge_error' WHERE run_id=%s", (lease["run_id"],)),
            ("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s", (lease["run_id"],)),
            ("UPDATE public.lab_arena_ledger SET funding_source='miner_key' WHERE call_identity=%s", (candidate["call_identity"],)),
            ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,call_succeeded}','false'::jsonb) WHERE entry_id=%s", (candidate["uncertain_entry_id"],)),
            ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,provider_status}','402'::jsonb) WHERE entry_id=%s", (candidate["uncertain_entry_id"],)),
            ("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc-'deepline_execution_key' WHERE call_identity=%s AND entry_kind='reservation'", (candidate["call_identity"],)),
            ("UPDATE public.lab_arena_ledger SET stage=2 WHERE call_identity=%s AND entry_kind='dispatch'", (candidate["call_identity"],)),
        )
        for statement, params in changes:
            cursor.execute("SAVEPOINT invalid_accepted_host")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(statement, params)
            assert _helper(cursor, candidate["uncertain_entry_id"]) is False
            assert _closed_candidate(cursor, h.round_id) == {"status": "none"}
            cursor.execute("ROLLBACK TO SAVEPOINT invalid_accepted_host")
        cursor.execute("SAVEPOINT invalid_failed_unknown")
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error' WHERE run_id=%s",
            (lease["run_id"],),
        )
        cursor.execute(
            "UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,call_succeeded}','false'::jsonb) "
            "WHERE entry_id=%s", (candidate["uncertain_entry_id"],),
        )
        assert _helper(cursor, candidate["uncertain_entry_id"]) is False
        assert _closed_candidate(cursor, h.round_id) == {"status": "none"}
        cursor.execute("ROLLBACK TO SAVEPOINT invalid_failed_unknown")
        cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s", (h.round_id,))
        assert cursor.fetchone()[0] == published_hash

    assert store.list_deepline_cost_reconciliations(h.round_id)[0]["uncertain_entry_id"] == candidate["uncertain_entry_id"]
    assert broker.reconcile_deepline_cost(candidate)["status"] == "pending"
    assert store.list_ledger(call_identity=candidate["call_identity"]) == old_ledger
    transport.bill_final = True
    settled = broker.reconcile_deepline_cost(candidate)
    assert settled["status"] == "settled" and settled["actual_microusd"] == 2000
    assert broker.reconcile_deepline_cost(candidate)["status"] == "settled"
    after = store.list_ledger(call_identity=candidate["call_identity"])
    assert after[:-1] == old_ledger
    assert after[-1]["entry_kind"] == "settlement" and after[-1]["amount_microusd"] == 2000
    assert store.list_deepline_cost_reconciliations(h.round_id) == []
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s", (h.round_id,))
        assert cursor.fetchone()[0] == published_hash
    failed_round, _, failed_connect, _, _, failed_candidate = closed_call(
        database, tmp_path, "25", native=NATIVE
    )
    with failed_connect() as connection, connection.cursor() as cursor:
        assert _helper(cursor, failed_candidate["uncertain_entry_id"]) is True
        assert _closed_candidate(cursor, failed_round.round_id)["uncertain_entry_id"] == failed_candidate["uncertain_entry_id"]
