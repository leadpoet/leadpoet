"""Query-only Deepline optimization on disposable PostgreSQL."""

from pathlib import Path

import pytest

from lab_arena import broker as br
from lab_arena.store import hash_lease_token
from tests.lab_arena import deepline_cost_priority_postgres_test as priority
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import run
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_PROVIDER_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, sha

MIGRATION = "435-lab-arena-deepline-candidate-query.sql"
NAMES = ["lab_arena_list_deepline_cost_reconciliations_v1",
         "lab_arena_list_deepline_cost_reconciliations_v2"]
SIGNATURES = [priority.V1, priority.V2]


def _catalog(cur):
    # Include every public function and relation, so a query-only migration
    # cannot silently change a settlement function, ACL, timeout or index.
    cur.execute("""
        SELECT p.oid, p.proname, p.prosrc, p.proowner, p.proacl::text,
               p.proconfig, p.prosecdef, p.provolatile
        FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
        WHERE n.nspname='public' ORDER BY p.oid
    """)
    functions = cur.fetchall()
    cur.execute("""
        SELECT c.oid, c.relname, c.relkind, c.relowner, c.relacl::text,
               c.relrowsecurity, i.indisvalid, i.indisready,
               pg_get_indexdef(i.indexrelid)
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        LEFT JOIN pg_index i ON i.indexrelid=c.oid
        WHERE n.nspname='public' ORDER BY c.oid
    """)
    return functions, cur.fetchall()


@pytest.fixture(scope="module")
def database():
    for db in database_with_lab_arena_migration(CURRENT_PROVIDER_SERVICE_MIGRATIONS):
        with db[0].connect(**db[1]) as c, c.cursor() as cur:
            # Keep the exact prior definitions as a JSON equivalence oracle.
            for name, signature in zip(NAMES, SIGNATURES):
                cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
                definition = cur.fetchone()[0]
                cur.execute(definition.replace("FUNCTION public." + name + "(",
                    "FUNCTION public.test_before_" + name + "(", 1))
            before_functions, before_relations = _catalog(cur)
            migration = (Path(__file__).parents[2] / "scripts" / MIGRATION).read_text()
            cur.execute(migration)
            after_functions, after_relations = _catalog(cur)
            assert after_relations == before_relations
            assert len(before_functions) == len(after_functions)
            for before, after in zip(before_functions, after_functions):
                if before[1] == NAMES[0]:
                    assert before[:2] == after[:2] and before[3:] == after[3:]
                    assert before[2] != after[2]
                else:
                    assert before == after
            cur.execute(migration)
            assert _catalog(cur) == (after_functions, after_relations)
        yield db


def _list(cur, rid, run_id="", cursor=0, limit=20, success=False):
    results = []
    for version in (1, 2):
        name = NAMES[version - 1]
        args = (rid, run_id, cursor, limit) + ((success,) if version == 2 else ())
        placeholders = ",".join(["%s"] * len(args))
        cur.execute("SELECT public." + name + "(" + placeholders + "), "
            "public.test_before_" + name + "(" + placeholders + ")", args + args)
        actual, before = cur.fetchone()
        assert actual == before
        results.append(actual["items"])
    if success is False:
        assert results[0] == results[1]
    return results


def test_all_guards_and_cursor_json_are_preserved(database, tmp_path):
    h, lease, token, connect = run(database, tmp_path, "25")
    store = h.service.store
    bad_bindings = [priority.uncertain(h, lease, token, "bad-binding-" + str(i))[0]
                    for i in range(24)]
    wrong_identities = []
    for kind in ("reservation", "dispatch"):
        for field, value in (
            ("call_identity", sha("wrong-" + kind)),
            ("run_id", "wrong-run"), ("round_id", "arena-2099-01-01-wrong"),
            ("submission_id", "wrong-submission"), ("miner_hotkey", "wrong-miner"),
            ("stage", 2), ("provider", "openrouter"),
            ("operation_id", "wrong-operation"), ("funding_source", "host"),
        ):
            identity = priority.uncertain(h, lease, token, kind + "-" + field)[0]
            wrong_identities.append((identity, kind, field, value))
    missing_dispatch = priority.uncertain(h, lease, token, "missing-dispatch")[0]
    settled = priority.uncertain(h, lease, token, "already-settled")[0]
    later = priority.uncertain(h, lease, token, "later-entry")[0]
    malformed = priority.uncertain(h, lease, token, "malformed-success", success="true")[0]
    good = [priority.uncertain(h, lease, token, "transport-failure")[0]]
    good += [priority.uncertain(h, lease, token, "good-" + str(i), success=True)[0]
             for i in range(21)]
    saved, _, key, fingerprint = priority.uncertain(h, lease, token, "stored-response")
    assert store.recover_deepline_response(
        run_id=lease["run_id"], lease_token_hash=hash_lease_token(token),
        call_identity=saved, request_hash=sha("stored-response"), execution_key=key,
        credential_fingerprint=fingerprint, request_id="public-execute:stored",
        operation="exa_search", actual_microusd=None,
        terminal_response=br._terminal_response_document(
            200, {}, b'{"result":"saved"}', call_succeeded=True),
        lease_ttl_seconds=420,
    )["status"] == "uncertain"
    good.append(saved)
    other_lease, other_token = claim(store, h.round_id, h.runner_keys[0],
                                   parallelism=2, ceiling=2)[:2]
    assert other_lease["run_id"] != lease["run_id"]
    other_good = priority.uncertain(h, other_lease, other_token, "other-run", success=True)[0]
    for active, active_token in ((lease, token), (other_lease, other_token)):
        assert complete(store, active["run_id"], hash_lease_token(active_token), "accepted",
                        output_ref="arena/" + h.round_id + "/" + active["run_id"] + ".json")["status"] == "accepted"

    # Corrupt only this disposable fixture to test guards independently.
    # Production DDL and data never disable an append-only trigger.
    with connect() as c, c.cursor() as cur:
        cur.execute("ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER lab_arena_ledger_append_only")
        cur.execute("""
            UPDATE public.lab_arena_ledger
            SET entry_doc=jsonb_set(entry_doc,'{call,credential_fingerprint}',
                                   to_jsonb(%s::text))
            WHERE call_identity=ANY(%s) AND entry_kind='uncertain'
        """, ("sha256:" + "b" * 64, bad_bindings))
        for identity, kind, field, value in wrong_identities:
            # Field names are the fixed literals above; values stay bound.
            if field == "funding_source":
                value = "miner_key" if store.provider_funding(lease["run_id"], "deepline")["funding_source"] == "host" else "host"
            cur.execute("UPDATE public.lab_arena_ledger SET " + field +
                        "=%s WHERE call_identity=%s AND entry_kind=%s", (value, identity, kind))
        cur.execute("DELETE FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='dispatch'", (missing_dispatch,))
        cur.execute("""
            INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
               call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)
            SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,
                   call_identity,provider,operation_id,funding_source,0,'{}'::jsonb
            FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'
        """, (settled,))
        # Put a dispatch after the uncertainty to exercise the generic later
        # entry guard separately from settlement exclusion.
        cur.execute("UPDATE public.lab_arena_ledger SET entry_id=-entry_id WHERE call_identity=%s AND entry_kind='uncertain'", (later,))
        cur.execute("ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER lab_arena_ledger_append_only")
    with connect() as c, c.cursor() as cur:
        cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE call_identity=ANY(%s) AND entry_kind='uncertain' ORDER BY entry_id", (good + [other_good],))
        ids = [row[0] for row in cur.fetchall()]
        assert len(ids) == 24
        for run_id in ("", None, lease["run_id"], other_lease["run_id"], "missing-run"):
            for cursor in (None, -1, 0, ids[10], 9223372036854775807):
                for limit in (None, 0, 1, 20, 21):
                    for success in (False, True, None):
                        v1, v2 = _list(cur, h.round_id, run_id, cursor, limit, success)
                        if limit not in (1, 20) or run_id == "missing-run":
                            assert v1 == v2 == []
                        if success is None:
                            assert v2 == []
        ordinary, successful = _list(cur, h.round_id, cursor=ids[19], success=True)
        # Selecting after the cursor wraps before aggregation sorts the final
        # JSON by entry ID, exactly as the prior implementation did.
        assert [item["uncertain_entry_id"] for item in ordinary] == ids[:16] + ids[20:]
        assert len(successful) == 20
        selected = _list(cur, h.round_id, lease["run_id"], limit=20)[0]
        assert len(selected) == 20
        assert [item["call_identity"] for item in selected] == good[:20]
        assert all(item["call_identity"] != malformed for item in selected)
        assert _list(cur, "arena-2099-01-01-missing")[0] == []
    store.close()


def test_exact_success_stored_response_and_settlement(database, tmp_path):
    priority.test_priority_is_exact_success_and_recovered_success_with_fair_wrap(database, tmp_path)


def test_cancelled_transport_failure_stays_recoverable(database, tmp_path):
    priority.test_raw_failure_and_cancelled_liabilities_stay_in_ordinary_lane(database, tmp_path)


def test_scoring_is_excluded_from_execute_lane(database):
    priority.test_successful_and_failed_scoring_liabilities_stay_in_ordinary_lane(database)


def test_owner_settings_and_service_acl(database):
    priority.test_new_list_is_service_only(database)
