"""Keep557 intact; prove scalar uncertainty access, authority, and replay safety."""

import hashlib
import json
import time
from pathlib import Path

import pytest

from tests.lab_arena import closed_provider_scoped_candidates557_postgres_test as prior
from tests.lab_arena import closed_provider_candidate_query_postgres_test as old

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/559-lab-arena-deepline-uncertain-round-entry-index.sql"
DDL = MIGRATION.read_text()
INDEX = "lab_arena_ledger_deepline_uncertain_round_entry_idx"
EXPECTED = (
    "CREATE INDEX "
    + INDEX
    + " ON public.lab_arena_ledger USING btree (round_id, entry_id) "
    "INCLUDE (run_id, call_identity) WHERE ((provider = 'deepline'::text) AND "
    "(entry_kind = 'uncertain'::text) AND (call_identity IS NOT NULL))"
)


def _assert_index(cur):
    cur.execute(
        "SELECT i.indisvalid,i.indisready,i.indisunique,pg_get_userbyid(c.relowner),"
        "pg_get_indexdef(c.oid) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
        "WHERE c.oid=to_regclass(%s)",
        ("public." + INDEX,),
    )
    assert cur.fetchone() == (True, True, False, "lab_arena_owner", EXPECTED)


@pytest.fixture(scope="module")
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        conn = db[0].connect(**db[1])
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(DDL)
                cur.execute(DDL)
                _assert_index(cur)
        finally:
            conn.close()
        yield db
    finally:
        gen.close()


# Full preexisting billing proof fixtures remain the oracle: no eligibility copy.
test_scope_policy_cause_and_proof_controls = (
    prior.test_scope_policy_cause_and_proof_controls
)
test_cursor_latest_head_and_settlement_controls = (
    prior.test_cursor_latest_head_and_settlement_controls
)
test_host_success_bindings = prior.test_host_success_bindings
test_openrouter_independent = prior.test_openrouter_independent
test_mixed_provider_global_cursor = prior.test_mixed_provider_global_cursor
test_flat_policy_and_null_scope_equivalence = (
    prior.test_flat_policy_and_null_scope_equivalence
)
test_active_rounds_malformed_metadata_and_failed_unknowns = (
    prior.test_active_rounds_malformed_metadata_and_failed_unknowns
)
test_closed_bill_full_transition = prior.test_closed_bill_full_transition
test_global_null_and_explicit_scope_select_same_genuine_candidate = (
    prior.test_global_null_and_explicit_scope_select_same_genuine_candidate
)
test_null_call_binding_cannot_qualify_any_full_helper_or_selector = (
    prior.test_null_call_binding_cannot_qualify_any_full_helper_or_selector
)


@pytest.mark.parametrize("drift", ["invalid", "unready", "owner", "definition"])
def test_required_readback_detects_unusable_or_wrong_existing_index(database, drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        _assert_index(cur)
        if drift in {"invalid", "unready"}:
            column = "indisvalid" if drift == "invalid" else "indisready"
            cur.execute(
                "UPDATE pg_index SET "
                + column
                + "=false WHERE indexrelid=%s::regclass",
                ("public." + INDEX,),
            )
        elif drift == "owner":
            cur.execute("ALTER TABLE public.lab_arena_ledger OWNER TO postgres")
        else:
            cur.execute("DROP INDEX public." + INDEX)
            cur.execute(
                "CREATE INDEX " + INDEX + " ON public.lab_arena_ledger(entry_id)"
            )
        with pytest.raises(AssertionError):
            _assert_index(cur)
        conn.rollback()


@pytest.fixture
def performance_database():
    # Keep plan statistics independent of the many mutation/control fixtures.
    gen = prior.database.__wrapped__()
    try:
        yield next(gen)
    finally:
        gen.close()


def test_global64_closed_rounds_sparse_uncertainty_uses_covering_index(
    performance_database, tmp_path
):
    """Before559 fails the access/buffer bound despite all64 rounds matching.

    Resolved uncertainty calls retain real wide TOAST; the dense settlement
    history models the existing round-only index fetching unrelated call kinds.
    No functions, ledger data, eligibility, latest-head or cursor rules change.
    """
    database = performance_database
    pg, dsn = database
    store, rid, candidate = old._closed_call(database, "index559perf")
    c = pg.connect(**dsn)
    q = c.cursor()
    try:
        q.execute("SET LOCAL session_replication_role=replica")
        q.execute("SET LOCAL statement_timeout='90s'")
        q.execute(
            'UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||\'{"network_name":"index559"}\'::jsonb WHERE round_id=%s',
            (rid,),
        )

        def cols(table):
            q.execute(
                "SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum",
                (table,),
            )
            return [x[0] for x in q.fetchall()]

        def clone(table, alias, repl, series, where, args):
            names = cols(table)
            q.execute(
                "INSERT INTO "
                + table
                + "("
                + ",".join(names)
                + ") SELECT "
                + ",".join(repl.get(k, alias + "." + k) for k in names)
                + " FROM "
                + table
                + " "
                + alias
                + " CROSS JOIN generate_series("
                + series
                + ") s WHERE "
                + where,
                args,
            )

        clone(
            "lab_arena_rounds",
            "r",
            {
                "round_id": "'arena-2026-10-20-index559'||s",
                "reward_basis_hash": "NULL",
                "effective_reward_epoch": "NULL",
                "status": "CASE WHEN s%%2=0 THEN 'cancelled' ELSE 'published' END",
            },
            "1,63",
            "r.round_id=%s",
            (rid,),
        )
        clone(
            "lab_arena_runs",
            "r",
            {
                "run_id": "'index559-run-'||s",
                "assignment_id": "'index559-assignment-'||s",
                "round_id": "'arena-2026-10-20-index559'||(1+s%%63)",
                "status": "'accepted'",
                "terminal_cause": "'accepted'",
                "claim_request_id": "NULL",
            },
            "1,2000",
            "r.run_id=%s",
            (candidate["run_id"],),
        )
        q.execute("SELECT max(entry_id)FROM lab_arena_ledger")
        maximum = q.fetchone()[0]
        n = "(((s-1)/4)+1)"
        cid = "('sha256:'||md5((" + n + ")::text)||md5((" + n + ")::text))"
        clone(
            "lab_arena_ledger",
            "l",
            {
                "entry_id": "%s+s",
                "run_id": "'index559-run-'||" + n,
                "round_id": "'arena-2026-10-20-index559'||(1+(" + n + ")%%63)",
                "call_identity": cid,
                "entry_kind": "CASE(s-1)%%4 WHEN 0 THEN 'reservation' WHEN 1 THEN 'dispatch' WHEN 2 THEN 'uncertain' ELSE 'settlement' END",
                "entry_doc": "CASE WHEN(s-1)%%4=2 THEN l.entry_doc||jsonb_build_object('retained_body',(SELECT string_agg(md5(("
                + n
                + ")::text||':'||g::text),'')FROM generate_series(1,256)g))ELSE l.entry_doc END",
            },
            "1,8000",
            "l.call_identity=%s AND l.entry_kind='uncertain'",
            (maximum, candidate["call_identity"]),
        )
        clone(
            "lab_arena_ledger",
            "l",
            {
                "entry_id": "%s+8000+s",
                "run_id": "'index559-run-'||(1+(s-1)%%2000)",
                "round_id": "'arena-2026-10-20-index559'||(1+(1+(s-1)%%2000)%%63)",
                "call_identity": "'sha256:'||md5('dense559-'||s::text)||md5('dense559-'||s::text)",
                "entry_kind": "'settlement'",
            },
            "1,250000",
            "l.call_identity=%s AND l.entry_kind='uncertain'",
            (maximum, candidate["call_identity"]),
        )
        c.commit()
        c.autocommit = True
        for table in ("lab_arena_ledger", "lab_arena_runs", "lab_arena_rounds"):
            q.execute("VACUUM(ANALYZE) " + table)
        q.execute("SELECT pg_get_functiondef(%s::regprocedure)", (prior.SIG,))
        definition = q.fetchone()[0]
        assert hashlib.sha256(definition.encode()).hexdigest() == prior.POST
        q.execute("SELECT count(*),max(entry_id),sum(entry_id)FROM lab_arena_ledger")
        rows_before = q.fetchone()
        q.execute(
            "SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid"
        )
        functions = q.fetchall()
        body = (
            definition.split("AS $function$", 1)[1]
            .split("$function$", 1)[0]
            .strip()
            .rstrip(";")
        )
        for number, key in enumerate(
            ("p_mode", "p_network_name", "p_netuid", "p_round_id", "p_after_entry_id"),
            1,
        ):
            body = body.replace(key, "$" + str(number))

        def measure(label):
            q.execute("SET statement_timeout='20s'")
            q.execute("SET plan_cache_mode=force_generic_plan")
            q.execute("SET jit=off")
            q.execute("SET ROLE lab_arena_owner")
            q.execute("PREPARE index559plan(text,text,integer,text,bigint) AS " + body)
            q.execute(
                "EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) EXECUTE index559plan('live','index559',71,'',0)"
            )
            plan = q.fetchone()[0][0]
            q.execute("DEALLOCATE index559plan")
            q.execute(
                "SELECT public.lab_arena_next_closed_deepline_reconciliation_v1('live','index559',71,'',0)"
            )
            result = q.fetchone()[0]
            q.execute("RESET ROLE")
            return {"result": result, "plan": plan}

        q.execute("DROP INDEX IF EXISTS public." + INDEX)
        before = measure("before")
        (tmp_path / "before.json").write_text(json.dumps(before, indent=2))
        t = time.monotonic()
        q.execute(DDL)
        build = time.monotonic() - t
        q.execute(DDL)
        _assert_index(q)
        after = measure("after")
        (tmp_path / "after.json").write_text(json.dumps(after, indent=2))
        q.execute("SELECT count(*),max(entry_id),sum(entry_id)FROM lab_arena_ledger")
        assert q.fetchone() == rows_before
        q.execute(
            "SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid"
        )
        assert q.fetchall() == functions
        assert before["result"] == after["result"] and after["result"]["status"] == "ok"

        def nodes(p):
            yield p
            for x in p.get("Plans", []):
                yield from nodes(x)

        used = [
            x
            for x in nodes(after["plan"]["Plan"])
            if x.get("Index Name")
            == "lab_arena_ledger_deepline_uncertain_round_entry_idx"
        ]
        assert used
        assert all(n["Node Type"] == "Index Only Scan" for n in used)
        assert sum(n["Actual Loops"] for n in used) == 64
        assert all(n["Heap Fetches"] == 0 for n in used)
        assert not any(
            n.get("Index Name") == INDEX for n in nodes(before["plan"]["Plan"])
        )
        assert (
            after["plan"]["Plan"]["Shared Hit Blocks"]
            < before["plan"]["Plan"]["Shared Hit Blocks"] / 5
        )
        q.execute(
            "SELECT pg_relation_size('lab_arena_ledger_deepline_uncertain_round_entry_idx'),pg_relation_size('lab_arena_ledger'),pg_relation_size(reltoastrelid)FROM pg_class WHERE oid='lab_arena_ledger'::regclass"
        )
        sizes = q.fetchone()
        out = {
            "closed_rounds": 64,
            "sparse_dense_history_rows": 250000,
            "uncertain_history_calls": 2000,
            "wide_toast_bytes": sizes[2],
            "index_bytes": sizes[0],
            "heap_bytes": sizes[1],
            "build_seconds": build,
            "function_sha256_unchanged": prior.POST,
            "all_function_metadata_unchanged": True,
            "ledger_rows_unchanged": True,
            "before": before,
            "after": after,
        }
        (tmp_path / "closed-index559-local-proof.json").write_text(
            json.dumps(out, indent=2)
        )
        print(
            json.dumps(
                {
                    **{k: v for k, v in out.items() if k not in ("before", "after")},
                    "before_ms": before["plan"]["Execution Time"],
                    "after_ms": after["plan"]["Execution Time"],
                    "before_buffers": before["plan"]["Plan"]["Shared Hit Blocks"],
                    "after_buffers": after["plan"]["Plan"]["Shared Hit Blocks"],
                    "new_index_nodes": [
                        {
                            "type": n["Node Type"],
                            "rows": n["Actual Rows"],
                            "loops": n["Actual Loops"],
                            "heap_fetches": n.get("Heap Fetches"),
                        }
                        for n in used
                    ],
                }
            )
        )
    finally:
        q.close()
        c.close()
        store.close()
