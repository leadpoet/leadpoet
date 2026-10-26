"""Preserve exact closed bills while validating in global cursor order."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import closed_provider_eligible_runs573_postgres_test as prior
from tests.lab_arena import closed_openrouter_billing_postgres_test as router
from tests.lab_arena import closed_provider_candidate_query_postgres_test as old
from tests.lab_arena import closed_scoring_billing_postgres_test as deepline
from tests.lab_arena.flat_per_icp_cost500_postgres_test import catalog, rows

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/578-lab-arena-closed-provider-ordered-candidate.sql"
SIG = "public.lab_arena_next_closed_provider_reconciliation_v1" + old.SUFFIX
ORACLE = "public.test578_original" + old.SUFFIX
PRE = "89a779452595cbab4537fec883dfaa0191eb327244a311421d297c73fecf1476"
POST = "91a5e8ebb464aaf2252a34f4b80e00411fdb0decac2f267d25c099c9bf2d9d17"


def definition(cur, signature=SIG):
    cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
    return cur.fetchone()[0]


def old_source(source):
    sql = MIGRATION.read_text()
    before = re.search(r"\$old_branch578\$(.*?)\$old_branch578\$", sql, re.S)[1]
    after = re.search(r"\$new_branch578\$(.*?)\$new_branch578\$", sql, re.S)[1]
    assert source.count(after) == 1
    result = source.replace(after, before, 1)
    assert hashlib.sha256(result.encode()).hexdigest() == PRE
    return result


@pytest.fixture(scope="module")
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        conn = db[0].connect(**db[1])
        conn.autocommit = True
        with conn.cursor() as cur:
            # Both plans get the existing production index, never a new index.
            cur.execute((ROOT / "scripts/532-lab-arena-round-call-head-index.sql").read_text())
            before = definition(cur)
            assert hashlib.sha256(before.encode()).hexdigest() == PRE
            cur.execute(before.replace(
                "FUNCTION public.lab_arena_next_closed_provider_reconciliation_v1(",
                "FUNCTION public.test578_original(", 1,
            ))
            cur.execute("ALTER FUNCTION " + ORACLE + " OWNER TO lab_arena_owner")
            cur.execute("REVOKE ALL ON FUNCTION " + ORACLE + " FROM PUBLIC")
            cur.execute("GRANT EXECUTE ON FUNCTION " + ORACLE + " TO lab_arena_service")
            functions, data, relations = catalog(cur), rows(cur), old._catalog(cur)[1]
            cur.execute(MIGRATION.read_text())
            after = definition(cur)
            assert old_source(after) == before
            assert hashlib.sha256(after.encode()).hexdigest() == POST
            current = catalog(cur)
            changed = []
            for a, b in zip(functions, current):
                assert a[0] == b[0] and a[2:] == b[2:]
                if a[1] != b[1]:
                    changed.append(a[0])
            assert len(functions) == len(current)
            assert changed == [SIG.removeprefix("public.")]
            assert rows(cur) == data and old._catalog(cur)[1] == relations
            cur.execute(MIGRATION.read_text())
            assert catalog(cur) == current and rows(cur) == data
        conn.close()
        yield db
    finally:
        gen.close()


def equivalent(cur, *, rid="", cursor=0, mode="live", network="finney", netuid=71):
    args = (mode, network, netuid, rid, cursor)
    cur.execute("SET LOCAL ROLE lab_arena_service")
    cur.execute("SELECT public.lab_arena_next_closed_provider_reconciliation_v1"
                "(%s,%s,%s,%s,%s),public.test578_original(%s,%s,%s,%s,%s)", args + args)
    actual, before = cur.fetchone()
    cur.execute("RESET ROLE")
    assert actual == before
    return actual


def test_global_provider_minimum_and_every_cursor_boundary(database):
    calls = [router._call(database, "578first"), router._call(database, "578second")]
    deepline = old._closed_call(database, "578mixed")
    try:
        ids = [call[4]["uncertain_entry_id"] for call in calls] + [deepline[2]["uncertain_entry_id"]]
        rids = [call[1] for call in calls] + [deepline[1]]
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||"
                        "'{\"network_name\":\"cursor578\"}'::jsonb WHERE round_id=ANY(%s)", (rids,))
            for scope in (None, ""):
                for cursor in (None, -1, 0, *ids, ids[-1] + 1, 2**63 - 1):
                    result = equivalent(cur, rid=scope, cursor=cursor, network="cursor578")
                    after = [entry for entry in ids if entry > (cursor or 0)]
                    assert result["uncertain_entry_id"] == min(after or ids)
            for rid, expected in zip(rids, ids):
                assert equivalent(cur, rid=rid, network="cursor578")["uncertain_entry_id"] == expected
            for kwargs in ({"rid": "absent578"}, {"mode": None}, {"network": None},
                           {"netuid": None}, {"mode": "test"}, {"netuid": 1}):
                assert equivalent(cur, **kwargs) == {"status": "none"}
    finally:
        for call in calls:
            call[0].close()
        deepline[0].close()


@pytest.mark.parametrize("fault", [
    "operation", "reason", "failure_stage", "generation", "fingerprint",
    "run_kind", "run_status", "run_binding", "null_call", "reservation",
    "reservation_payer", "dispatch", "later_head",
])
def test_invalid_earlier_candidate_never_hides_later_valid_bill(database, fault):
    label = hashlib.sha256(fault.encode()).hexdigest()[:4]
    network = "proof578" + label
    first, second = router._call(database, "578b" + label), router._call(database, "578g" + label)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||"
                        "jsonb_build_object('network_name',%s::text) WHERE round_id=ANY(%s)",
                        (network, [first[1], second[1]]))
            entry, identity, run = first[4]["uncertain_entry_id"], first[3], first[2]["run_id"]
            if fault in {"operation", "reason", "failure_stage", "generation", "fingerprint"}:
                if fault == "operation":
                    cur.execute("UPDATE lab_arena_ledger SET operation_id='other' WHERE entry_id=%s", (entry,))
                else:
                    path, value = {
                        "reason": (["reason"], "other"),
                        "failure_stage": (["call", "reason"], "settle_failure"),
                        "generation": (["call", "openrouter_generation_id"], "invalid id"),
                        "fingerprint": (["call", "credential_fingerprint"], "invalid"),
                    }[fault]
                    cur.execute("UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,%s,to_jsonb(%s::text)) WHERE entry_id=%s", (path, value, entry))
            elif fault in {"run_kind", "run_status"}:
                column, value = ("kind", "execute") if fault == "run_kind" else ("status", "leased")
                cur.execute("UPDATE lab_arena_runs SET " + column + "=%s WHERE run_id=%s", (value, run))
            elif fault == "run_binding":
                cur.execute("UPDATE lab_arena_ledger SET stage=2 WHERE entry_id=%s", (entry,))
            elif fault == "null_call":
                cur.execute("UPDATE lab_arena_ledger SET call_identity=NULL WHERE entry_id=%s", (entry,))
            elif fault == "reservation_payer":
                cur.execute("UPDATE lab_arena_ledger SET funding_source='host' WHERE call_identity=%s AND entry_kind='reservation'", (identity,))
            elif fault in {"reservation", "dispatch"}:
                cur.execute("DELETE FROM lab_arena_ledger WHERE call_identity=%s AND entry_kind=%s", (identity, fault))
            else:
                cur.execute("INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,0,'{}'::jsonb FROM lab_arena_ledger WHERE entry_id=%s", (entry,))
            for cursor in (0, entry, second[4]["uncertain_entry_id"], 2**63 - 1):
                actual = equivalent(cur, cursor=cursor, network=network)
                assert actual["uncertain_entry_id"] == second[4]["uncertain_entry_id"]
            assert equivalent(cur, rid=first[1], network=network) == {"status": "none"}
            cur.execute("UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,"
                        "'{call,openrouter_generation_id}','\"invalid id\"'::jsonb) WHERE entry_id=%s",
                        (second[4]["uncertain_entry_id"],))
            assert equivalent(cur, network=network) == {"status": "none"}
    finally:
        first[0].close()
        second[0].close()


# Native exact-cost recovery still settles once and preserves accepted state.
test_closed_openrouter_settles_once = router.test_exact_closed_judge_bill_settles_once_without_changing_accepted_state
test_closed_deepline_settles_once = deepline.test_published_late_bill_is_exact_idempotent_and_does_not_rewrite_result


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("drift", ["source", "owner", "acl", "security", "volatility", "path", "inner", "target"])
def test_exact_source_security_and_replay_guards(database, installed, drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if not installed:
            cur.execute(old_source(definition(cur)))
        if drift == "source":
            cur.execute(definition(cur).replace("AS $function$\n", "AS $function$\n-- drift\n", 1))
        elif drift == "inner":
            cur.execute(definition(cur, prior.SIG).replace("AS $function$\n", "AS $function$\n-- drift\n", 1))
        elif drift != "target":
            cur.execute({
                "owner": "ALTER FUNCTION " + SIG + " OWNER TO postgres",
                "acl": "GRANT EXECUTE ON FUNCTION " + SIG + " TO anon",
                "security": "ALTER FUNCTION " + SIG + " SECURITY INVOKER",
                "volatility": "ALTER FUNCTION " + SIG + " VOLATILE",
                "path": "ALTER FUNCTION " + SIG + " SET search_path=public,pg_catalog",
            }[drift])
        sql = MIGRATION.read_text().replace(POST, "0" * 64) if drift == "target" else MIGRATION.read_text()
        body = sql.split("DO $closed_provider_order578$")[1].split("$closed_provider_order578$;")[0]
        cur.execute("SAVEPOINT hostile")
        with pytest.raises(database[0].Error, match="Closed provider order"):
            cur.execute("DO $test$" + body + "$test$")
        cur.execute("ROLLBACK TO SAVEPOINT hostile")
        conn.rollback()


def test_owner_member_without_schema_create_can_apply_and_replay(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        functions, data = catalog(cur), rows(cur)
        cur.execute(old_source(definition(cur)))
        cur.execute("CREATE ROLE closed578_migrator NOLOGIN NOSUPERUSER INHERIT")
        cur.execute("GRANT lab_arena_owner TO closed578_migrator")
        cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO closed578_migrator")
        cur.execute("REVOKE CREATE ON SCHEMA public FROM lab_arena_owner")
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        acl = cur.fetchone()[0]
        cur.execute("SET ROLE closed578_migrator")
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())
        cur.execute("RESET ROLE")
        assert catalog(cur) == functions and rows(cur) == data
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        assert cur.fetchone() == (acl,)
        conn.rollback()


def nodes(plan):
    yield plan
    for child in plan.get("Plans", []):
        yield from nodes(child)


def test_generic_plan_proves_one_candidate_before_full_billing_validation(database, tmp_path):
    store, rid, run, identity, candidate = router._call(database, "perf578")
    conn = database[0].connect(**database[1])
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||"
                        "'{\"network_name\":\"perf578\"}'::jsonb WHERE round_id=%s", (rid,))

            def clone(table, alias, replacements, count, where, args):
                cur.execute("SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum", (table,))
                columns = [x[0] for x in cur.fetchall()]
                cur.execute("INSERT INTO " + table + "(" + ",".join(columns) + ") SELECT "
                            + ",".join(replacements.get(k, alias + "." + k) for k in columns)
                            + " FROM " + table + " " + alias + " CROSS JOIN generate_series(1,"
                            + str(count) + ") s WHERE " + where, args)

            clone("lab_arena_rounds", "r", {
                "round_id": "'arena-2026-09-14-perf578'||s",
                "reward_basis_hash": "NULL", "effective_reward_epoch": "NULL",
                "status": "CASE WHEN s%%2=0 THEN 'cancelled' ELSE 'published' END",
            }, 64, "r.round_id=%s", (rid,))
            clone("lab_arena_runs", "r", {
                "run_id": "'perf578-run-'||s", "assignment_id": "'perf578-assignment-'||s",
                "round_id": "'arena-2026-09-14-perf578'||s", "claim_request_id": "NULL",
            }, 64, "r.run_id=%s", (run["run_id"],))
            cur.execute("SELECT max(entry_id) FROM lab_arena_ledger")
            maximum = cur.fetchone()[0]
            number = "(1+(s-1)%%64)"
            count = 6400
            clone("lab_arena_ledger", "l", {
                "entry_id": "%s+s*10+CASE l.entry_kind WHEN 'reservation' THEN 0 WHEN 'dispatch' THEN 1 ELSE 2 END",
                "run_id": "'perf578-run-'||" + number,
                "round_id": "'arena-2026-09-14-perf578'||" + number,
                "call_identity": "'sha256:'||md5('perf578-'||s::text)||md5('perf578-'||s::text)",
                "entry_doc": "CASE WHEN l.entry_kind='uncertain' THEN jsonb_set(l.entry_doc,'{call,openrouter_generation_id}',to_jsonb('gen-perf578-'||s)) ELSE l.entry_doc END",
            }, count, "l.call_identity=%s", (maximum, identity))
            cur.execute("SELECT setval(pg_get_serial_sequence('lab_arena_ledger','entry_id'),(SELECT max(entry_id) FROM lab_arena_ledger)+1)")
        conn.commit()
        conn.autocommit = True
        with conn.cursor() as cur:
            for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger"):
                cur.execute("VACUUM(ANALYZE) " + table)
            source = definition(cur)
            cur.execute("SELECT count(*),max(entry_id),sum(entry_id) FROM lab_arena_ledger")
            before_rows = cur.fetchone()
            plans, summary, selected = {}, {}, {}
            for label, value in (("before", old_source(source)), ("after", source)):
                body = value.split("AS $function$", 1)[1].split("$function$", 1)[0].strip().rstrip(";")
                for i, key in enumerate(("p_mode", "p_network_name", "p_netuid", "p_round_id", "p_after_entry_id"), 1):
                    body = re.sub(r"\b" + key + r"\b", "$" + str(i), body)
                cur.execute("SET plan_cache_mode=force_generic_plan")
                cur.execute("SET statement_timeout='15s'")
                cur.execute("SET jit=off")
                cur.execute("SET ROLE lab_arena_owner")
                cur.execute("PREPARE order578(text,text,integer,text,bigint) AS " + body)
                for name, scope, cursor in (("beginning", "", 0),
                        ("middle", "", maximum + 3200 * 10),
                        ("wrapped", "", 2**63 - 1), ("pinned", rid, 0),
                        ("missing", "missing578", 0)):
                    args = ("live", "perf578", 71, scope, cursor)
                    cur.execute("EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) EXECUTE order578(%s,%s,%s,%s,%s)", args)
                    plan = cur.fetchone()[0][0]
                    cur.execute("EXECUTE order578(%s,%s,%s,%s,%s)", args)
                    selected[label + "_" + name] = cur.fetchone()[0]
                    proofs = [n for n in nodes(plan["Plan"]) if n.get("Alias") in {"reservation", "dispatch"}]
                    work = sum(n["Actual Loops"] * max(1, n["Actual Rows"] + n.get("Rows Removed by Filter", 0)) for n in proofs)
                    summary[label + "_" + name] = {"ms": plan["Execution Time"], "proof_rows_or_probes": work,
                        "shared_hits": plan["Plan"]["Shared Hit Blocks"], "shared_reads": plan["Plan"]["Shared Read Blocks"]}
                    plans[label + "_" + name] = plan
                cur.execute("DEALLOCATE order578")
                cur.execute("RESET ROLE")
            for name in ("beginning", "middle", "wrapped", "pinned", "missing"):
                assert selected["before_" + name] == selected["after_" + name]
            for name in ("beginning", "middle", "wrapped"):
                assert selected["after_" + name]["status"] == "ok"
                # This work invariant fails on exact573, including the simple
                # branch-LIMIT-only prototype whose sort follows validation.
                assert summary["before_" + name]["proof_rows_or_probes"] >= count
                assert summary["after_" + name]["proof_rows_or_probes"] == 2
            assert selected["after_missing"] == {"status": "none"}
            cur.execute("SELECT count(*),max(entry_id),sum(entry_id) FROM lab_arena_ledger")
            assert cur.fetchone() == before_rows
            assert definition(cur) == source
            artifact = tmp_path / "closed-provider578-plans.json"
            artifact.write_text(json.dumps({"summary": summary, "plans": plans, "selected": selected}, indent=2))
            print("CLOSED_PROVIDER578_PLAN_PROOF " + json.dumps({"artifact": str(artifact), "summary": summary}), flush=True)
    finally:
        conn.close()
        store.close()
