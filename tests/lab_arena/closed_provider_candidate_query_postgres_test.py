"""Closed selectors remain JSON-equivalent to their exact prior definitions."""
from pathlib import Path

import pytest

from lab_arena import broker as br
from tests.lab_arena import closed_host_score_billing_postgres_test as host
from tests.lab_arena import closed_openrouter_billing_postgres_test as router
from tests.lab_arena.closed_scoring_billing_postgres_test import _closed_call
from tests.lab_arena.deepline_candidate_query_postgres_test import _catalog
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_PROVIDER_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)

MIGRATION = "436-lab-arena-closed-provider-candidate-query.sql"
NAMES = ["lab_arena_next_closed_deepline_reconciliation_v1",
         "lab_arena_next_closed_provider_reconciliation_v1"]
SUFFIX = "(text,text,integer,text,bigint)"
CAUSES = ["lease_expired", "worker_lost", "result_rejected", "provider_error",
          "stage_closed", "judge_error", "judge_timeout"]


@pytest.fixture(scope="module")
def database():
    migrations = CURRENT_PROVIDER_SERVICE_MIGRATIONS + (
        "435-lab-arena-deepline-candidate-query.sql",)
    for db in database_with_lab_arena_migration(migrations):
        with db[0].connect(**db[1]) as connection, connection.cursor() as cur:
            # The outer oracle must call the original inner oracle too.
            for name in NAMES:
                cur.execute("SELECT pg_get_functiondef(%s::regprocedure)",
                            ("public." + name + SUFFIX,))
                definition = cur.fetchone()[0]
                definition = definition.replace("FUNCTION public." + name + "(",
                    "FUNCTION public.test_before_" + name + "(", 1)
                if name == NAMES[1]:
                    definition = definition.replace("public." + NAMES[0] + "(",
                        "public.test_before_" + NAMES[0] + "(")
                cur.execute(definition)
            before_functions, before_relations = _catalog(cur)
            migration = (Path(__file__).parents[2] / "scripts" / MIGRATION).read_text()
            cur.execute(migration)
            after_functions, after_relations = _catalog(cur)
            assert after_relations == before_relations
            assert len(before_functions) == len(after_functions)
            for before, after in zip(before_functions, after_functions):
                if before[1] in NAMES:
                    assert before[:2] == after[:2] and before[3:] == after[3:]
                    assert before[2] != after[2]
                else:
                    assert before == after  # Includes active V1/V2 and settlement.
            cur.execute(migration)
            assert _catalog(cur) == (after_functions, after_relations)
        yield db


def _equivalent(cur, rid="", cursor=0, mode="live", network="finney", netuid=71):
    results = []
    args = (mode, network, netuid, rid, cursor)
    for name in NAMES:
        cur.execute("SELECT public." + name + "(%s,%s,%s,%s,%s), "
                    "public.test_before_" + name + "(%s,%s,%s,%s,%s)", args + args)
        actual, original = cur.fetchone()
        assert actual == original, (name, args, actual, original)
        results.append(actual)
    return results


def test_miner_policies_causes_cancelled_liabilities_and_scope(database):
    store, rid, candidate = _closed_call(database, "query436")
    try:
        with database[0].connect(**database[1]) as connection, connection.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            for status in ("published", "cancelled"):
                for policy in ("successful_calls_v1", "successful_calls_per_icp_v1", "legacy", None):
                    cur.execute("UPDATE public.lab_arena_rounds SET status=%s, configuration_doc="
                        "configuration_doc || jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) "
                        "WHERE round_id=%s", (status, policy, rid))
                    for cause in CAUSES + ["credential_error", "accepted"]:
                        cur.execute("UPDATE public.lab_arena_runs SET terminal_cause=%s WHERE run_id=%s",
                                    (cause, candidate["run_id"]))
                        actual = _equivalent(cur, rid)[0]
                        eligible = cause in CAUSES and (status == "cancelled" or policy in (
                            "successful_calls_v1", "successful_calls_per_icp_v1"))
                        assert (actual["status"] == "ok") == eligible
            # Exclude execute success, accepted score and malformed receipts.
            cur.execute("UPDATE public.lab_arena_rounds SET status='cancelled' WHERE round_id=%s", (rid,))
            cur.execute("UPDATE public.lab_arena_runs SET terminal_cause='judge_error' WHERE run_id=%s",
                        (candidate["run_id"],))
            for field, value in (("kind", "execute"), ("status", "accepted")):
                cur.execute("SAVEPOINT invalid_run")
                cur.execute("UPDATE public.lab_arena_runs SET " + field + "=%s WHERE run_id=%s",
                            (value, candidate["run_id"]))
                assert _equivalent(cur, rid)[0] == {"status": "none"}
                cur.execute("ROLLBACK TO SAVEPOINT invalid_run")
            for statement, args in (
                ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,"
                 "'{reserve_remaining_budget}','false'::jsonb) WHERE call_identity=%s "
                 "AND entry_kind='reservation'", (candidate["call_identity"],)),
                ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,"
                 "'{call,call_succeeded}','true'::jsonb) WHERE entry_id=%s",
                 (candidate["uncertain_entry_id"],)),
                ("UPDATE public.lab_arena_ledger SET stage=2 WHERE call_identity=%s "
                 "AND entry_kind='dispatch'", (candidate["call_identity"],)),
                ("UPDATE public.lab_arena_ledger SET run_id='wrong-run' WHERE call_identity=%s "
                 "AND entry_kind='reservation'", (candidate["call_identity"],)),
            ):
                cur.execute("SAVEPOINT invalid_binding")
                cur.execute(statement, args)
                assert _equivalent(cur, rid) == [{"status": "none"}] * 2
                cur.execute("ROLLBACK TO SAVEPOINT invalid_binding")
            for kwargs in ({"mode": "test"}, {"network": "test"}, {"netuid": 1},
                           {"mode": None}, {"network": None}, {"netuid": None}):
                assert _equivalent(cur, rid, **kwargs) == [{"status": "none"}] * 2
            assert _equivalent(cur, "missing-round") == [{"status": "none"}] * 2
    finally:
        store.close()


def test_cursor_wrap_latest_head_and_settlement(database):
    calls = [_closed_call(database, "q436c" + str(i)) for i in range(3)]
    try:
        ids = [call[2]["uncertain_entry_id"] for call in calls]
        with database[0].connect(**database[1]) as connection, connection.cursor() as cur:
            for cursor, expected in ((None, ids[0]), (-1, ids[0]), (0, ids[0]),
                                     (ids[0], ids[1]), (ids[1], ids[2]),
                                     (ids[2], ids[0]), (9223372036854775807, ids[0])):
                # Use a distinct scope to exclude the other tests' fixtures.
                cur.execute("SET LOCAL session_replication_role=replica")
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                    "configuration_doc || '{\"network_name\":\"cursor436\"}'::jsonb "
                    "WHERE round_id=ANY(%s)", ([call[1] for call in calls],))
                for rid in ("", None):
                    actual = _equivalent(cur, rid, cursor, network="cursor436")
                    assert all(doc["uncertain_entry_id"] == expected for doc in actual)
            # Any later entry suppresses a head, not only a settlement.
            for kind in ("dispatch", "settlement"):
                cur.execute("SAVEPOINT later_head")
                if kind == "dispatch":
                    cur.execute("UPDATE public.lab_arena_ledger SET entry_id=-entry_id "
                        "WHERE entry_id=%s", (ids[0],))
                else:
                    cur.execute("""INSERT INTO public.lab_arena_ledger
                      (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
                       call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)
                      SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,
                        call_identity,provider,operation_id,funding_source,0,'{}'::jsonb
                      FROM public.lab_arena_ledger WHERE entry_id=%s""", (ids[0],))
                assert _equivalent(cur, calls[0][1], network="cursor436") == [{"status": "none"}] * 2
                assert all(doc["uncertain_entry_id"] == ids[1] for doc in
                           _equivalent(cur, network="cursor436"))
                cur.execute("ROLLBACK TO SAVEPOINT later_head")
    finally:
        for store, _, _ in calls:
            store.close()


def test_host_success_full_binding_and_failed_score_prerequisite(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    h, _, connect, _, _, candidate = host.closed_call(database, tmp_path, "23")
    try:
        with connect() as connection, connection.cursor() as cur:
            for status in ("published", "cancelled"):
                for policy in ("successful_calls_v1", "successful_calls_per_icp_v1", "legacy", None):
                    cur.execute("SET LOCAL session_replication_role=replica")
                    cur.execute("UPDATE public.lab_arena_rounds SET status=%s, configuration_doc="
                        "configuration_doc || jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) "
                        "WHERE round_id=%s", (status, policy, h.round_id))
                    for cause in CAUSES:
                        cur.execute("UPDATE public.lab_arena_runs SET terminal_cause=%s WHERE run_id=%s",
                                    (cause, candidate["run_id"]))
                        actual = _equivalent(cur, h.round_id)
                        assert all((doc["status"] == "ok") == (status == "cancelled" or policy in (
                            "successful_calls_v1", "successful_calls_per_icp_v1")) for doc in actual)
            cur.execute("UPDATE public.lab_arena_rounds SET status='published', configuration_doc="
                "configuration_doc || '{\"sourcing_cost_eligibility_policy\":\"successful_calls_v1\"}'::jsonb "
                "WHERE round_id=%s", (h.round_id,))
            # Existing exhaustive receipt controls still exercise the real helper.
            host.negatives(cur, candidate)
            assert _equivalent(cur, h.round_id)[0]["status"] == "ok"
    finally:
        h.service.store.close()


@pytest.mark.parametrize("status", ["accepted", "failed"])
@pytest.mark.parametrize("closed", ["published", "cancelled"])
def test_openrouter_lane_remains_independent(database, status, closed):
    store, rid, _, _, candidate = router._call(database, "436" + status[:2] + closed[:2],
        run_status=status, round_status=closed, reason="settle_failure")
    try:
        with database[0].connect(**database[1]) as connection, connection.cursor() as cur:
            for cursor in (0, candidate["uncertain_entry_id"], 9223372036854775807):
                deep, provider = _equivalent(cur, rid, cursor)
                assert deep == {"status": "none"}
                assert provider["provider"] == "openrouter"
    finally:
        store.close()


def test_mixed_provider_cursor_keeps_one_global_order(database):
    deep_store, deep_round, deep = _closed_call(database, "436mixed")
    router_store, router_round, _, _, other = router._call(database, "436mixor")
    try:
        with database[0].connect(**database[1]) as connection, connection.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc || "
                "'{\"network_name\":\"mixed436\"}'::jsonb WHERE round_id=ANY(%s)",
                ([deep_round, router_round],))
            first, second = deep["uncertain_entry_id"], other["uncertain_entry_id"]
            assert first < second
            for cursor, expected, provider in ((0, first, "deepline"),
                (first, second, "openrouter"), (second, first, "deepline")):
                selected = _equivalent(cur, cursor=cursor, network="mixed436")[1]
                assert selected["uncertain_entry_id"] == expected
                assert selected["provider"] == provider
    finally:
        deep_store.close()
        router_store.close()


@pytest.mark.parametrize("mutation", [
    "ALTER FUNCTION public.lab_arena_next_closed_provider_reconciliation_v1"
    "(text,text,integer,text,bigint) SET statement_timeout='1s'",
    "ALTER FUNCTION public.lab_arena_next_closed_deepline_reconciliation_v1"
    "(text,text,integer,text,bigint) SET statement_timeout='1s'",
    "CREATE OR REPLACE FUNCTION public.lab_arena__closed_score_dynamic_uncertainty_v1"
    "(p_uncertain_entry_id bigint) RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER "
    "SET search_path=pg_catalog, public AS 'SELECT true'",
    "CREATE OR REPLACE FUNCTION public.lab_arena__closed_host_score_success_uncertainty_v1"
    "(p_uncertain_entry_id bigint) RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER "
    "SET search_path=pg_catalog, public AS 'SELECT true'",
])
def test_migration_rejects_changed_security_or_eligibility(database, mutation):
    with database[0].connect(**database[1]) as connection:
        with connection.cursor() as cur:
            cur.execute(mutation)
            with pytest.raises(Exception, match="(security shape changed|helper prerequisite changed)"):
                cur.execute((Path(__file__).parents[2] / "scripts" / MIGRATION).read_text())
        connection.rollback()
