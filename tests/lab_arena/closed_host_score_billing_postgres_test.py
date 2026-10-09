"""Closed host judge bills extend audit eligibility only, with exact identity."""
from pathlib import Path
import time

import pytest

from lab_arena import broker as br
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import database as base_database
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import database, frame, setup
from tests.lab_arena.deepline_score_billing_latency_test import CompletedTransport
from tests.lab_arena.deepline_interrupted_cost_reconciliation_postgres_test import _settle
from tests.lab_arena import deepline_completed_response_recovery_test as recovery
from tests.lab_arena.test_lab_arena_broker import HOST_KEYS
from tests.lab_arena.closed_scoring_billing_postgres_test import _closed_call
from tests.lab_arena.test_lab_arena_migration_postgres import complete

MIGRATION = Path(__file__).parents[2] / "scripts/432-lab-arena-closed-host-score-billing.sql"


def closed_call(database, tmp_path, label="28", *, native=None, policy="successful_calls_v1"):
    transport = CompletedTransport()
    if native is not None:
        transport.response["job_id"] = native
    h, lease, token, connect, broker = setup(database, tmp_path, label, transport)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1_scoring',"
                       "configuration_doc=configuration_doc || '{\"sourcing_cost_eligibility_policy\":\"successful_calls_v1\","
                       "\"mode\":\"live\",\"network_name\":\"finney\",\"netuid\":71}'::jsonb WHERE round_id=%s", (h.round_id,))
        cursor.execute("UPDATE public.lab_arena_runs SET kind='score' WHERE run_id=%s", (lease["run_id"],))
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                       "jsonb_set(configuration_doc,'{sourcing_cost_eligibility_policy}',to_jsonb(%s::text)) "
                       "WHERE round_id=%s", (policy, h.round_id))
    h.service._hot_rounds.clear()
    result = h.service.handle_provider(lease["run_id"], token, frame())
    assert result["status"] == 200 and result["call"]["outcome"] == "uncertain"
    store = h.service.store
    candidate = store.list_deepline_cost_reconciliations(h.round_id)[0]
    complete(store, lease["run_id"], hash_lease_token(token), "judge_error")
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET status='published',"
                       "publication_doc='{\"immutable_snapshot\":true}'::jsonb WHERE round_id=%s", (h.round_id,))
    return h, lease, connect, broker, transport, dict(candidate, run_status="failed")


HELPER = "public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)"
CHANGED = {
    "lab_arena_list_deepline_cost_reconciliations_v1",
    "lab_arena_list_deepline_cost_reconciliations_v2",
    "lab_arena_next_closed_deepline_reconciliation_v1",
    "lab_arena_reconcile_deepline_cost_v1",
    "lab_arena_reconcile_deepline_cost_v2",
}


def definitions(cursor):
    cursor.execute("SELECT oid::regprocedure::text,pg_get_functiondef(oid) FROM pg_proc "
                   "WHERE pronamespace='public'::regnamespace AND proname LIKE 'lab_arena_%'")
    return dict(cursor.fetchall())


def security(cursor):
    cursor.execute("SELECT relname,relrowsecurity,relforcerowsecurity,relacl::text "
                   "FROM pg_class WHERE relnamespace='public'::regnamespace "
                   "AND relname LIKE 'lab_arena_%' ORDER BY relname")
    relations = cursor.fetchall()
    cursor.execute("SELECT * FROM pg_policies WHERE schemaname='public' "
                   "AND tablename LIKE 'lab_arena_%' ORDER BY tablename,policyname")
    return relations, cursor.fetchall()


def next_closed(store, round_id):
    return store.next_closed_deepline_reconciliation(
        mode="live", network_name="finney", netuid=71, round_id=round_id)


def raw_list(cursor, round_id, *, successful=False):
    cursor.execute("SELECT public.lab_arena_list_deepline_cost_reconciliations_v2(%s,'',0,20,%s)",
                   (round_id, successful))
    return cursor.fetchone()[0]["items"]


def negatives(cursor, candidate):
    entry = candidate["uncertain_entry_id"]
    identity = candidate["call_identity"]
    # Every mutation is rolled back in this disposable database. These inputs
    # model damaged or mismatched saved receipts, not production write paths.
    changes = [
        ("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s", (candidate["run_id"],)),
        ("UPDATE public.lab_arena_runs SET status='accepted' WHERE run_id=%s", (candidate["run_id"],)),
        ("UPDATE public.lab_arena_runs SET terminal_cause='credential_error' WHERE run_id=%s", (candidate["run_id"],)),
        ("UPDATE public.lab_arena_ledger SET funding_source='miner_key' WHERE call_identity=%s", (identity,)),
        ("UPDATE public.lab_arena_ledger SET operation_id='exa.search' WHERE entry_id=%s", (entry,)),
        ("UPDATE public.lab_arena_ledger SET stage=2 WHERE call_identity=%s AND entry_kind='dispatch'", (identity,)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc-'deepline_execution_key' "
         "WHERE call_identity=%s AND entry_kind='reservation'", (identity,)),
    ]
    for path, value in [
        ("call,call_succeeded", 'false'), ("call,call_succeeded", '\"bogus\"'),
        ("call,provider_status", '402'), ("call,provider_status", '\"200\"'),
        ("call,reason", '\"transport_failure\"'),
        ("call,deepline_request_id", '\"ctx-tool-ffffffffffffffffffffffffffffffff\"'),
        ("call,deepline_execution_key", '\"arena:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\"'),
        ("call,deepline_job_id", 'null'), ("call,deepline_job_id", '123'),
        ("call,deepline_job_id", '\"bad/id\"'),
        ("call,deepline_job_id", '\"ctx-tool-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"'),
        ("call,credential_fingerprint", '\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"'),
    ]:
        changes.append(("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,%s::text[],%s::jsonb) "
                        "WHERE entry_id=%s", ("{" + path + "}", value, entry)))
    for statement, arguments in changes:
        cursor.execute("SAVEPOINT invalid_receipt")
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(statement, arguments)
        cursor.execute("SELECT " + HELPER.split("(")[0] + "(%s)", (entry,))
        assert cursor.fetchone() == (False,), (statement, arguments)
        assert raw_list(cursor, candidate["round_id"]) == []
        cursor.execute("SELECT public.lab_arena_next_closed_deepline_reconciliation_v1('live','finney',71,%s,0)",
                       (candidate["round_id"],))
        assert cursor.fetchone()[0] == {"status": "none"}
        cursor.execute("ROLLBACK TO SAVEPOINT invalid_receipt")
    cursor.execute("SELECT " + HELPER.split("(")[0] + "(%s)", (entry,))
    assert cursor.fetchone() == (True,)


@pytest.mark.parametrize("version,label,policy", [
    (1, "28", "successful_calls_v1"),
    (2, "27", "successful_calls_v1"),
    (2, "26", "successful_calls_per_icp_v1"),
])
def test_published_host_score_exact_bill_remains_reconcilable(database, tmp_path, monkeypatch, version, label, policy):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    native = "iad1::execute-1791330233666-aeb0e7021ee6" if version == 1 else recovery.NATIVE
    monkeypatch.setattr(recovery, "NATIVE", native)
    h, lease, connect, broker, transport, candidate = closed_call(database, tmp_path, label, native=native, policy=policy)
    store = h.service.store
    old_ledger = store.list_ledger(call_identity=candidate["call_identity"])
    # Legacy policy keeps a positive hold; the actual per-ICP policy also works
    # with zero monetary holds, without treating an unknown bill as free.
    assert (old_ledger[0]["amount_microusd"] > 0) == (policy == "successful_calls_v1")
    assert old_ledger[-1]["amount_microusd"] == old_ledger[0]["amount_microusd"]
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s", (h.round_id,))
        round_hash = cursor.fetchone()[0]
        before_defs, before_security = definitions(cursor), security(cursor)
        first_run = HELPER.removeprefix("public.") not in before_defs
        if first_run:
            # Current migration regression: no closed driver/list authority, and
            # even a known exact bill cannot pass the published settlement gate.
            assert store.list_deepline_cost_reconciliations(h.round_id) == []
            assert raw_list(cursor, h.round_id) == []
            assert next_closed(store, h.round_id) == {"status": "none"}
            if version == 1:
                assert _settle(store, candidate) == {"status": "stale"}
            else:
                transport.bill_final = True
                assert broker.reconcile_deepline_cost(candidate) == {"status": "stale"}
                transport.bill_final = False
        cursor.execute(MIGRATION.read_text())
        first_defs = definitions(cursor)
        cursor.execute(MIGRATION.read_text())
        cursor.execute("BEGIN")
        assert definitions(cursor) == first_defs
        assert security(cursor) == before_security
        assert next(row for row in before_security[0] if row[0] == "lab_arena_ledger")[1] is True
        changed = {name.split("(")[0] for name in before_defs if before_defs[name] != first_defs[name]}
        assert changed == (CHANGED if first_run else set())
        cursor.execute("SET LOCAL ROLE lab_arena_service")
        assert raw_list(cursor, h.round_id) == [candidate]
        assert raw_list(cursor, h.round_id, successful=True) == []  # execute-only sourcing cursor
        cursor.execute("RESET ROLE")
        cursor.execute("SELECT public.lab_arena__submission_kind_has_admission_uncertainty_v1(%s,'score'),"
                       "public.lab_arena__submission_kind_admission_spend_v1(%s,'score',FALSE)",
                       (lease["submission_id"], lease["submission_id"]))
        assert cursor.fetchone() == (True, old_ledger[0]["amount_microusd"])
        negatives(cursor, candidate)
    assert store.list_deepline_cost_reconciliations(h.round_id) == [candidate]
    assert next_closed(store, h.round_id)["uncertain_entry_id"] == candidate["uncertain_entry_id"]
    # Pending is not authority to manufacture any charge, including zero.
    assert broker.reconcile_deepline_cost(candidate)["status"] == "pending"
    assert store.list_ledger(call_identity=candidate["call_identity"]) == old_ledger
    # An exact positive receipt for this native ID is not authority to bill
    # another native ID, credential or execution key.
    arguments = dict(
        round_id=candidate["round_id"], run_id=candidate["run_id"],
        call_identity=candidate["call_identity"], uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=candidate["request_id"], operation=candidate["operation"],
        credential_fingerprint=candidate["credential_fingerprint"], actual_microusd=2000,
        cost_units="0.02", execution_key=candidate["execution_key"], recovered_request_id=native,
    )
    assert store.reconcile_deepline_cost(**dict(arguments, recovered_request_id="another-native-call")) == {"status": "stale"}
    assert store.reconcile_deepline_cost(**dict(arguments, credential_fingerprint="sha256:" + "b" * 64)) == {"status": "stale"}
    with pytest.raises(ArenaStoreError):
        store.reconcile_deepline_cost(**dict(arguments, execution_key="arena:" + "b" * 64))
    assert store.list_ledger(call_identity=candidate["call_identity"]) == old_ledger
    transport.bill_final = True
    if version == 1:
        recovered_id, cost = br._deepline_exact_readback(
            transport=transport, secret=HOST_KEYS["deepline"], request_id=native,
            execution_key=candidate["execution_key"], operation=candidate["operation"],
            provider=candidate["billing_provider"], poll=False,
            reconciliation_deadline=time.monotonic() + 5,
        )
        assert recovered_id == native and cost is not None and cost.microusd == 2000
        settle = lambda: _settle(store, candidate, amount=cost.microusd, units=format(cost.units, "f"))
    else:
        settle = lambda: broker.reconcile_deepline_cost(candidate)
    settled = settle()
    assert settled["status"] == "settled" and settled["actual_microusd"] == 2000
    assert settle()["status"] == "settled"
    after = store.list_ledger(call_identity=candidate["call_identity"])
    assert after[:-1] == old_ledger and after[-1]["amount_microusd"] == 2000
    assert sum(request["method"] == "POST" for request in transport.requests) == 1
    assert store.list_deepline_cost_reconciliations(h.round_id) == []
    assert next_closed(store, h.round_id) == {"status": "none"}
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s", (h.round_id,))
        assert cursor.fetchone()[0] == round_hash
        for signature in definitions(cursor):
            if signature.split("(")[0] not in CHANGED:
                continue
            for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
                cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, signature))
                assert cursor.fetchone() == (role == "lab_arena_service",)
        for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
            cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, HELPER))
            assert cursor.fetchone() == (False,)
        cursor.execute("SELECT has_function_privilege('lab_arena_owner',%s,'EXECUTE')", (HELPER,))
        assert cursor.fetchone() == (True,)
        cursor.execute("SAVEPOINT denied_helper")
        cursor.execute("SET LOCAL ROLE lab_arena_service")
        with pytest.raises(database[0].errors.InsufficientPrivilege):
            cursor.execute("SELECT " + HELPER.split("(")[0] + "(%s)", (candidate["uncertain_entry_id"],))
        cursor.execute("ROLLBACK TO SAVEPOINT denied_helper")
        cursor.execute("SAVEPOINT denied_ledger")
        cursor.execute("SET LOCAL ROLE anon")
        with pytest.raises(database[0].errors.InsufficientPrivilege):
            cursor.execute("SELECT entry_doc FROM public.lab_arena_ledger WHERE call_identity=%s",
                           (candidate["call_identity"],))
        cursor.execute("ROLLBACK TO SAVEPOINT denied_ledger")


@pytest.mark.parametrize("dynamic,label", [(True,"dynamic432"),(False,"fixed432")])
def test_existing_miner_dynamic_rule_and_fixed_unknown_stay_unchanged(database, dynamic, label):
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    store, round_id, candidate = _closed_call(database, label, dynamic=dynamic)
    try:
        with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (round_id,))
            cursor.execute("SELECT public.lab_arena__closed_score_dynamic_uncertainty_v1(%s),"
                           "public.lab_arena__closed_host_score_success_uncertainty_v1(%s)",
                           (candidate["uncertain_entry_id"], candidate["uncertain_entry_id"]))
            assert cursor.fetchone() == (dynamic, False)
        assert (next_closed(store, round_id)["status"] == "ok") == dynamic
        assert store.list_deepline_cost_reconciliations(round_id) == ([dict(candidate, run_status="failed")] if dynamic else [])
        assert _settle(store, candidate)["status"] == ("settled" if dynamic else "stale")
    finally:
        store.close()
