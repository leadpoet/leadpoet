"""Push only the existing eligible-run predicate ahead of global head checks."""

from pathlib import Path
import hashlib
import json
import re
import pytest
from tests.lab_arena import closed_provider_scoped_candidates557_postgres_test as prior
from tests.lab_arena import closed_provider_candidate_query_postgres_test as old
from tests.lab_arena.flat_per_icp_cost500_postgres_test import catalog, rows

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/560-lab-arena-closed-provider-score-run-scope.sql"
SIG = prior.SIG
PRE = prior.POST
POST = "9762e92771dfa3333811caabeaec56d01b5c02c888869193c258660796d29bdd"


def definition(cur, signature=SIG):
    return prior.definition(cur, signature)


def old_source(source):
    text = MIGRATION.read_text()
    before = re.search(r"\$old_run_scope560\$(.*?)\$old_run_scope560\$", text, re.S)[1]
    after = re.search(r"\$new_run_scope560\$(.*?)\$new_run_scope560\$", text, re.S)[1]
    assert source.count(after) == 1
    source = source.replace(after, before, 1)
    assert hashlib.sha256(source.encode()).hexdigest() == PRE
    return source


@pytest.fixture(scope="module")
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == PRE
            for name in old.NAMES:
                body = definition(cur, "public." + name + old.SUFFIX)
                body = body.replace(
                    "FUNCTION public." + name + "(",
                    "FUNCTION public.test_before_" + name + "(",
                    1,
                )
                if name == old.NAMES[1]:
                    body = body.replace(
                        "public." + old.NAMES[0] + "(",
                        "public.test_before_" + old.NAMES[0] + "(",
                    )
                cur.execute(body)
                cur.execute(
                    "ALTER FUNCTION public.test_before_"
                    + name
                    + old.SUFFIX
                    + " OWNER TO lab_arena_owner"
                )
            functions, before_rows = catalog(cur), rows(cur)
            relations = old._catalog(cur)[1]
            cur.execute(MIGRATION.read_text())
            after = catalog(cur)
            assert rows(cur) == before_rows and old._catalog(cur)[1] == relations
            changed = []
            for a, b in zip(functions, after):
                assert a[0] == b[0] and a[2:] == b[2:]
                if a[1] != b[1]:
                    changed.append(a[0])
            assert len(functions) == len(after) and changed == [
                SIG.removeprefix("public.")
            ]
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == POST
            assert old_source(definition(cur))
            cur.execute(MIGRATION.read_text())
            assert catalog(cur) == after and rows(cur) == before_rows
        yield db
    finally:
        gen.close()


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


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    "drift",
    [
        "source",
        "owner",
        "acl",
        "security",
        "volatility",
        "path",
        "helper_dynamic",
        "helper_host",
        "helper_miner",
        "helper_adaptation",
        "helper_cost",
        "target",
    ],
)
def test_source_security_and_helper_guards_precede_replay(database, installed, drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        post = definition(cur)
        if not installed:
            cur.execute(old_source(post))
        if drift == "source":
            cur.execute(
                definition(cur).replace(
                    "WITH uncertainties AS", "-- drift\n  WITH uncertainties AS", 1
                )
            )
        elif drift.startswith("helper_"):
            signature = {
                "helper_dynamic": "public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)",
                "helper_host": "public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)",
                "helper_miner": "public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)",
                "helper_adaptation": "public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)",
                "helper_cost": "public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)",
            }[drift]
            cur.execute(
                definition(cur, signature).replace(
                    "AS $function$\n", "AS $function$\n-- drift\n", 1
                )
            )
        elif drift != "target":
            cur.execute(
                {
                    "owner": "ALTER FUNCTION " + SIG + " OWNER TO postgres",
                    "acl": "GRANT EXECUTE ON FUNCTION " + SIG + " TO anon",
                    "security": "ALTER FUNCTION " + SIG + " SECURITY INVOKER",
                    "volatility": "ALTER FUNCTION " + SIG + " VOLATILE",
                    "path": "ALTER FUNCTION "
                    + SIG
                    + " SET search_path=public,pg_catalog",
                }[drift]
            )
        sql = MIGRATION.read_text()
        if drift == "target":
            sql = sql.replace(POST, "0" * 64)
        body = sql.split("DO $closed_provider_score_scope560$")[1].split(
            "$closed_provider_score_scope560$;"
        )[0]
        cur.execute("SAVEPOINT hostile")
        with pytest.raises(database[0].Error, match="Closed provider score scope"):
            cur.execute("DO $test$" + body + "$test$")
        cur.execute("ROLLBACK TO SAVEPOINT hostile")
        conn.rollback()


def test_non_superuser_owner_member_replay_preserves_catalog_and_rows(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        post = definition(cur)
        before_functions, before_rows = catalog(cur), rows(cur)
        cur.execute(old_source(post))
        cur.execute("CREATE ROLE closed557_migrator NOLOGIN NOSUPERUSER INHERIT")
        cur.execute("GRANT lab_arena_owner TO closed557_migrator")
        cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO closed557_migrator")
        cur.execute("SET ROLE closed557_migrator")
        cur.execute("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
        assert cur.fetchone() == (False,)
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())
        cur.execute("RESET ROLE")
        assert catalog(cur) == before_functions and rows(cur) == before_rows


@pytest.fixture
def history_database():
    gen = database.__wrapped__()
    db = next(gen)
    conn = db[0].connect(**db[1])
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                (
                    ROOT
                    / "scripts/559-lab-arena-deepline-uncertain-round-entry-index.sql"
                ).read_text()
            )
        conn.close()
        yield db
    finally:
        conn.close()
        gen.close()


def test_all_closed_history_none_candidate_filters_execution_before_global_heads(
    history_database, tmp_path
):
    """50k latest heads,97.7% execute; evaluate all four full proofs to NONE.

    The old function must inspect every historical call head before rejecting
    execution runs. The new function applies only the same existing eligibility
    predicate first. No unresolved call is deleted or automatically settled.
    """
    pg, dsn = history_database
    store, rid, candidate = old._closed_call(history_database, "none560")
    conn = pg.connect(**dsn)
    cur = conn.cursor()
    try:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("SET LOCAL statement_timeout='90s'")
        cur.execute(
            'UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||\'{"network_name":"none560"}\'::jsonb WHERE round_id=%s',
            (rid,),
        )
        cur.execute(
            "UPDATE lab_arena_ledger SET entry_doc='{}'::jsonb WHERE entry_id=%s",
            (candidate["uncertain_entry_id"],),
        )

        def clone(table, alias, replacements, series, where, params):
            cur.execute(
                "SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum",
                (table,),
            )
            names = [row[0] for row in cur.fetchall()]
            cur.execute(
                "INSERT INTO "
                + table
                + "("
                + ",".join(names)
                + ") SELECT "
                + ",".join(replacements.get(k, alias + "." + k) for k in names)
                + " FROM "
                + table
                + " "
                + alias
                + " CROSS JOIN generate_series("
                + series
                + ") s WHERE "
                + where,
                params,
            )

        clone(
            "lab_arena_rounds",
            "r",
            {
                "round_id": "'arena-2026-10-20-none560'||s",
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
                "run_id": "'none560-run-'||s",
                "assignment_id": "'none560-assignment-'||s",
                "round_id": "'arena-2026-10-20-none560'||(1+s%%63)",
                "kind": "CASE WHEN s<=977 THEN 'execute' ELSE 'score' END",
                "status": "'accepted'",
                "terminal_cause": "'accepted'",
                "claim_request_id": "NULL",
            },
            "1,1000",
            "r.run_id=%s",
            (candidate["run_id"],),
        )
        cur.execute("SELECT max(entry_id)FROM lab_arena_ledger")
        maximum = cur.fetchone()[0]
        n = "(1+(s-1)%%1000)"
        clone(
            "lab_arena_ledger",
            "l",
            {
                "entry_id": "%s+s",
                "run_id": "'none560-run-'||" + n,
                "round_id": "'arena-2026-10-20-none560'||(1+(" + n + ")%%63)",
                "call_identity": "'sha256:'||md5('none560-'||s::text)||md5('none560-'||s::text)",
                "entry_doc": "CASE WHEN s<=1000 THEN jsonb_build_object('retained_body',(SELECT string_agg(md5(s::text||':'||g::text),'')FROM generate_series(1,256)g))ELSE '{}'::jsonb END",
            },
            "1,50000",
            "l.entry_id=%s",
            (maximum, candidate["uncertain_entry_id"]),
        )
        conn.commit()
        conn.autocommit = True
        for table in ("lab_arena_ledger", "lab_arena_runs", "lab_arena_rounds"):
            cur.execute("VACUUM(ANALYZE) " + table)
        cur.execute("SELECT count(*),max(entry_id),sum(entry_id)FROM lab_arena_ledger")
        ledger = cur.fetchone()
        functions = catalog(cur)
        post = definition(cur)
        pre = old_source(post)
        results = {}

        def nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from nodes(child)

        for label, source in (("before", pre), ("after", post)):
            body = (
                source.split("AS $function$", 1)[1]
                .split("$function$", 1)[0]
                .strip()
                .rstrip(";")
            )
            for number, key in enumerate(
                (
                    "p_mode",
                    "p_network_name",
                    "p_netuid",
                    "p_round_id",
                    "p_after_entry_id",
                ),
                1,
            ):
                body = body.replace(key, "$" + str(number))
            cur.execute("SET statement_timeout='30s'")
            cur.execute("SET plan_cache_mode=force_generic_plan")
            cur.execute("SET jit=off")
            cur.execute("SET ROLE lab_arena_owner")
            cur.execute("PREPARE none560(text,text,integer,text,bigint)AS " + body)
            cur.execute(
                "EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE)EXECUTE none560('live','none560',71,'',0)"
            )
            plan = cur.fetchone()[0][0]
            cur.execute("EXECUTE none560('live','none560',71,'',0)")
            assert cur.fetchone() == ({"status": "none"},)
            cur.execute("DEALLOCATE none560")
            cur.execute("RESET ROLE")
            cte = next(
                n
                for n in nodes(plan["Plan"])
                if n.get("Subplan Name") == "CTE uncertainties"
            )
            later = [n for n in nodes(cte) if n.get("Alias") == "later"]
            results[label] = {
                "plan": plan,
                "uncertainty_rows": cte["Actual Rows"],
                "latest_head_probe_loops": sum(n["Actual Loops"] for n in later),
            }
        assert results["before"]["uncertainty_rows"] == 50001
        assert results["after"]["uncertainty_rows"] == 1151
        assert results["before"]["latest_head_probe_loops"] >= 50001
        assert results["after"]["latest_head_probe_loops"] == 1151
        assert old._equivalent(cur, "", network="none560") == [{"status": "none"}] * 2
        cur.execute("SELECT count(*),max(entry_id),sum(entry_id)FROM lab_arena_ledger")
        assert cur.fetchone() == ledger
        assert catalog(cur) == functions
        (tmp_path / "closed-score560-none-history-proof.json").write_text(
            json.dumps(results, indent=2)
        )
        print(
            json.dumps(
                {
                    label: {
                        "ms": value["plan"]["Execution Time"],
                        "uncertainty_rows": value["uncertainty_rows"],
                        "head_probe_loops": value["latest_head_probe_loops"],
                    }
                    for label, value in results.items()
                }
            )
        )
    finally:
        cur.close()
        conn.close()
        store.close()


def test_newer_execution_head_still_suppresses_eligible_score_across_rounds(database):
    """The new score prefilter never narrows the GLOBAL later-CID fence."""
    first = old._closed_call(database, "q560a")
    later = old._closed_call(database, "q560b")
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            assert old._equivalent(cur, first[1])[0]["status"] == "ok"
            cur.execute(
                "UPDATE lab_arena_runs SET kind='execute' WHERE run_id=%s",
                (later[2]["run_id"],),
            )
            cur.execute(
                "UPDATE lab_arena_ledger SET call_identity=%s,entry_kind='settlement' WHERE entry_id=%s",
                (first[2]["call_identity"], later[2]["uncertain_entry_id"]),
            )
            assert later[2]["uncertain_entry_id"] > first[2]["uncertain_entry_id"]
            assert old._equivalent(cur, first[1]) == [{"status": "none"}] * 2
            conn.rollback()
    finally:
        first[0].close()
        later[0].close()
