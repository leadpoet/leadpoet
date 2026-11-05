"""Preserve exact OpenRouter bills while proving only ordered candidates needed."""

import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import closed_openrouter_billing_postgres_test as router
from tests.lab_arena import closed_score_run_covering_index615_postgres_test as covering
from tests.lab_arena.flat_per_icp_cost500_postgres_test import catalog, rows

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/632-lab-arena-openrouter-ordered-cost-candidates.sql"
SQL = MIGRATION.read_text()
TARGET = "lab_arena_list_openrouter_cost_reconciliations_v1"
NAME = "test632_before_openrouter_costs"
CANDIDATE = TARGET
SUFFIX = "(text,text,bigint,integer)"
SIG = "public." + TARGET + SUFFIX
PRE = "e2d75725cb2fa7c5e901f27a0625b86c9a9d5f0727fcbfafb5ebcb235d46ab78"
POST = "4554bbd62c6fe41589f8142cf4b92d6845ea2813f466bd77110e809dd6030e72"
INDEXES = (
    "lab_arena_ledger_openrouter_uncertain_run_entry_idx",
    "lab_arena_ledger_round_call_head_idx",
)
SOURCE = ""


def sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def definition(cur):
    cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIG,))
    return cur.fetchone()[0]


def snapshot(cur):
    cur.execute(
        "SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY indexname"
    )
    return catalog(cur), rows(cur), cur.fetchall()


def guard_body(sql=SQL):
    return sql.split("DO $openrouter_ordered_cost632$", 1)[1].split(
        "$openrouter_ordered_cost632$;", 1
    )[0]


@pytest.fixture(scope="module")
def database():
    global SOURCE
    generator = router.database.__wrapped__()
    db = next(generator)
    try:
        conn = db[0].connect(**db[1])
        conn.autocommit = True
        with conn.cursor() as cur:
            # Historical native lifecycle fixtures predate500. Upgrade only
            # this read seam with500's exact policy substitution, not its claim
            # changes. Pin the complete deployed preimage before testing632.
            source = definition(cur)
            assert (
                sha(source)
                == "380311d654e1e7b5a8b4f20d7a41eed277f9d8a2a5e2c8a2ba24850d195a58c7"
            )
            before = "IN ('successful_calls_v1', 'successful_calls_per_icp_v1')"
            after = "IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')"
            assert source.count(before) == 1
            SOURCE = source.replace(before, after)
            assert sha(SOURCE) == PRE
            cur.execute(SOURCE)
            oracle = SOURCE.replace(
                "FUNCTION public." + TARGET + "(", "FUNCTION public." + NAME + "(", 1
            )
            cur.execute(oracle)
            cur.execute(
                "ALTER FUNCTION public." + NAME + SUFFIX + " OWNER TO lab_arena_owner"
            )
            cur.execute(
                "REVOKE ALL ON FUNCTION public." + NAME + SUFFIX + " FROM PUBLIC"
            )
            cur.execute(
                "GRANT EXECUTE ON FUNCTION public."
                + NAME
                + SUFFIX
                + " TO lab_arena_service"
            )
            for file in (
                "508-lab-arena-openrouter-uncertain-run-index.sql",
                "532-lab-arena-round-call-head-index.sql",
            ):
                cur.execute((ROOT / "scripts" / file).read_text())
        seed = router._call(db, "order632seed")
        seed[0].close()
        with conn.cursor() as cur:
            before = snapshot(cur)
            cur.execute(SQL)
            assert sha(definition(cur)) == POST
            after = snapshot(cur)
            assert before[1:] == after[1:]
            changed = []
            for a, b in zip(before[0], after[0]):
                assert a[0] == b[0] and a[2:] == b[2:]
                if a[1] != b[1]:
                    changed.append(a[0])
            assert len(before[0]) == len(after[0])
            assert changed == [TARGET + SUFFIX]
            cur.execute(SQL)
            assert snapshot(cur) == after
        conn.close()
        yield db
    finally:
        generator.close()


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    "drift", ["source", "owner", "acl", "volatility", "path", "security", "cost"]
)
def test_source_security_guards_precede_replay(database, installed, drift):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        if not installed:
            cur.execute(SOURCE)
        if drift == "source":
            cur.execute(
                definition(cur).replace(
                    "AS $function$\n", "AS $function$\n-- drift\n", 1
                )
            )
        else:
            cur.execute(
                {
                    "owner": "ALTER FUNCTION " + SIG + " OWNER TO postgres",
                    "acl": "GRANT EXECUTE ON FUNCTION " + SIG + " TO anon",
                    "volatility": "ALTER FUNCTION " + SIG + " VOLATILE",
                    "path": "ALTER FUNCTION "
                    + SIG
                    + " SET search_path=public,pg_catalog",
                    "security": "ALTER FUNCTION " + SIG + " SECURITY INVOKER",
                    "cost": "ALTER FUNCTION " + SIG + " COST 999",
                }[drift]
            )
        before = snapshot(cur)
        cur.execute("SAVEPOINT rejected")
        with pytest.raises(pg.Error, match="OpenRouter ordered cost") as failure:
            cur.execute("DO $test$" + guard_body() + "$test$")
        assert failure.value.pgcode == "55000"
        cur.execute("ROLLBACK TO SAVEPOINT rejected")
        assert snapshot(cur) == before
        conn.rollback()


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("index", INDEXES)
@pytest.mark.parametrize(
    "drift", ["missing", "invalid", "unready", "owner", "definition"]
)
def test_index_guards_precede_replay(database, installed, index, drift):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        if not installed:
            cur.execute(SOURCE)
        if drift == "missing":
            cur.execute("DROP INDEX public." + index)
        elif drift in {"invalid", "unready"}:
            column = "indisvalid" if drift == "invalid" else "indisready"
            cur.execute(
                "UPDATE pg_index SET "
                + column
                + "=false WHERE indexrelid=%s::regclass",
                ("public." + index,),
            )
        elif drift == "owner":
            cur.execute("ALTER TABLE public.lab_arena_ledger OWNER TO postgres")
        else:
            cur.execute("DROP INDEX public." + index)
            cur.execute("CREATE INDEX " + index + " ON public.lab_arena_ledger(run_id)")
        before = snapshot(cur)
        cur.execute("SAVEPOINT rejected")
        with pytest.raises(pg.Error, match="OpenRouter ordered cost index"):
            cur.execute("DO $test$" + guard_body() + "$test$")
        cur.execute("ROLLBACK TO SAVEPOINT rejected")
        assert snapshot(cur) == before
        conn.rollback()


@pytest.mark.parametrize("fault", ["postimage", "readback", "post_write"])
def test_full_transaction_is_atomic(database, fault):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute(SOURCE)
        conn.commit()
        before = snapshot(cur)
        text = SQL
        if fault == "postimage":
            text = text.replace(POST, "0" * 64)
        elif fault == "readback":
            anchor = "       IS DISTINCT FROM '" + POST + "'"
            assert text.count(anchor) == 1
            text = text.replace(anchor, "       IS DISTINCT FROM '" + "0" * 64 + "'", 1)
        else:
            text = text.replace(
                "  EXECUTE updated;",
                "  EXECUTE updated;\n  RAISE EXCEPTION 'OpenRouter ordered cost injected post-write failure';",
                1,
            )
        try:
            with pytest.raises(pg.Error, match="OpenRouter ordered cost"):
                cur.execute(text)
            conn.rollback()
            assert snapshot(cur) == before
            assert sha(definition(cur)) == PRE
        finally:
            cur.execute(SQL)
            conn.commit()


def test_owner_member_applies_and_replays_without_owner_schema_create(database):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        before = snapshot(cur)
        cur.execute(SOURCE)
        cur.execute("CREATE ROLE ordered632_migrator NOLOGIN NOSUPERUSER INHERIT")
        cur.execute("GRANT lab_arena_owner TO ordered632_migrator")
        cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO ordered632_migrator")
        cur.execute("REVOKE CREATE ON SCHEMA public FROM lab_arena_owner")
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        acl = cur.fetchone()[0]
        cur.execute("SET ROLE ordered632_migrator")
        cur.execute(SQL)
        cur.execute(SQL)
        cur.execute("RESET ROLE")
        assert snapshot(cur) == before
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        assert cur.fetchone() == (acl,)
        conn.rollback()


def equivalent(cur, rid, run="", cursor=0, limit=1):
    cur.execute("SET LOCAL ROLE lab_arena_service")
    args = (rid, run, cursor, limit)
    cur.execute(
        "SELECT public." + NAME + "(%s,%s,%s,%s),public." + CANDIDATE + "(%s,%s,%s,%s)",
        args + args,
    )
    old, new = cur.fetchone()
    assert old == new, (args, old, new)
    cur.execute("RESET ROLE")
    return old


def body_plan(cur, source, mode, args):
    body = (
        source.split("AS $function$", 1)[1]
        .rsplit("$function$", 1)[0]
        .strip()
        .rstrip(";")
    )
    for pos, key in enumerate(
        ["p_round_id", "p_run_id", "p_after_entry_id", "p_limit"], 1
    ):
        body = re.sub(r"\b" + key + r"\b", "$" + str(pos), body)
    cur.execute("SET plan_cache_mode=" + mode)
    cur.execute("SET ROLE lab_arena_owner")
    cur.execute("PREPARE activeor(text,text,bigint,integer) AS " + body)
    cur.execute(
        "EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) EXECUTE activeor(%s,%s,%s,%s)",
        args,
    )
    result = cur.fetchone()[0][0]
    cur.execute("DEALLOCATE activeor")
    cur.execute("RESET ROLE")
    return result


@pytest.mark.parametrize(
    "fault",
    [
        "operation",
        "generation",
        "fingerprint",
        "top_reason",
        "settle_failure_stage",
        "settle_failure_error",
        "reservation",
        "dispatch",
        "reservation_payer",
        "reservation_run",
        "null_call",
        "later_head",
    ],
)
def test_invalid_earlier_candidate_preserves_later_valid_bill(database, fault):
    tag = hashlib.sha256(fault.encode()).hexdigest()[:6]
    a = router._call(database, "orda" + tag)
    b = router._call(database, "ordb" + tag)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            # Put both native lifecycles in one exact run, round and full binding scope.
            cur.execute(
                "UPDATE lab_arena_ledger SET run_id=%s,round_id=%s,submission_id=l.submission_id,miner_hotkey=l.miner_hotkey,stage=l.stage FROM lab_arena_ledger l WHERE l.entry_id=%s AND lab_arena_ledger.call_identity=%s",
                (a[2]["run_id"], a[1], a[4]["uncertain_entry_id"], b[3]),
            )
            entry = a[4]["uncertain_entry_id"]
            cid = a[3]
            if fault == "operation":
                cur.execute(
                    "UPDATE lab_arena_ledger SET operation_id='other' WHERE entry_id=%s",
                    (entry,),
                )
            elif fault in {
                "generation",
                "fingerprint",
                "top_reason",
                "settle_failure_stage",
                "settle_failure_error",
            }:
                paths = {
                    "generation": (["call", "openrouter_generation_id"], "bad id"),
                    "fingerprint": (["call", "credential_fingerprint"], "bad"),
                    "top_reason": (["reason"], "other"),
                    "settle_failure_stage": (["call", "reason"], "settle_failure"),
                    "settle_failure_error": (["call", "reason"], "settle_failure"),
                }
                path, value = paths[fault]
                cur.execute(
                    "UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,%s,to_jsonb(%s::text)) WHERE entry_id=%s",
                    (path, value, entry),
                )
            elif fault in {"reservation", "dispatch"}:
                cur.execute(
                    "DELETE FROM lab_arena_ledger WHERE call_identity=%s AND entry_kind=%s",
                    (cid, fault),
                )
            elif fault == "reservation_payer":
                cur.execute(
                    "UPDATE lab_arena_ledger SET funding_source='host' WHERE call_identity=%s AND entry_kind='reservation'",
                    (cid,),
                )
            elif fault == "reservation_run":
                cur.execute(
                    "UPDATE lab_arena_ledger SET run_id='missing' WHERE call_identity=%s AND entry_kind='reservation'",
                    (cid,),
                )
            elif fault == "null_call":
                cur.execute(
                    "UPDATE lab_arena_ledger SET call_identity=NULL WHERE entry_id=%s",
                    (entry,),
                )
            else:
                cur.execute(
                    "INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,0,'{}'::jsonb FROM lab_arena_ledger WHERE entry_id=%s",
                    (entry,),
                )
            result = equivalent(cur, a[1])
            assert (
                len(result["items"]) == 1
                and result["items"][0]["call_identity"] == b[3]
            )
    finally:
        a[0].close()
        b[0].close()


def test_scope_policy_reason_cursor_null_and_error_parity(database):
    call = router._call(database, "ordscope")
    store, rid, run, cid, candidate = call
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            for state in ["stage2_judged", "open", "published", "cancelled"]:
                for kind, status in [
                    ("execute", "accepted"),
                    ("score", "accepted"),
                    ("score", "failed"),
                    ("score", "leased"),
                ]:
                    for policy in [
                        "successful_calls_v1",
                        "successful_calls_per_icp_v1",
                        "successful_calls_flat_per_icp_v1",
                        "legacy",
                        None,
                    ]:
                        cur.execute(
                            "UPDATE lab_arena_rounds SET status=%s,configuration_doc=configuration_doc||jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) WHERE round_id=%s",
                            (state, policy, rid),
                        )
                        cur.execute(
                            "UPDATE lab_arena_runs SET kind=%s,status=%s WHERE run_id=%s",
                            (kind, status, run["run_id"]),
                        )
                        equivalent(cur, rid)
            cur.execute(
                "UPDATE lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s",
                (rid,),
            )
            for operation in ["openrouter.chat", "openrouter.responses"]:
                for funding in ["host", "miner_key"]:
                    cur.execute(
                        "UPDATE lab_arena_ledger SET operation_id=%s,funding_source=%s WHERE call_identity=%s",
                        (operation, funding, cid),
                    )
                    assert len(equivalent(cur, rid)["items"]) == 1
            for reason, error in [
                ("missing_provider_cost", None),
                ("settle_failure", "ArenaStoreError"),
                ("settle_failure", "Exception"),
                ("settle_failure", "OperationError"),
            ]:
                cur.execute(
                    "UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call}',(entry_doc->'call')||jsonb_build_object('reason',%s::text,'failure_stage','settlement','error_class',%s::text)) WHERE entry_id=%s",
                    (reason, error, candidate["uncertain_entry_id"]),
                )
                equivalent(cur, rid)
            for scope in [rid, None, "missing"]:
                for run_filter in ["", None, run["run_id"], "missing"]:
                    for cursor in [
                        None,
                        -1,
                        0,
                        candidate["uncertain_entry_id"],
                        2**63 - 1,
                    ]:
                        for limit in [None, -1, 0, 1, 20, 21]:
                            equivalent(cur, scope, run_filter, cursor, limit)
            documents = [None, [], True, "bad", {}, {"call": None}] + [
                {
                    "reason": "worker_reported",
                    "call": {
                        "reason": "missing_provider_cost",
                        "openrouter_generation_id": v,
                        "credential_fingerprint": v,
                    },
                }
                for v in [None, False, True, "bad id", 0, {}, []]
            ]
            for mode in ["force_generic_plan", "force_custom_plan"]:
                cur.execute("SET LOCAL plan_cache_mode=" + mode)
                for doc in documents:
                    cur.execute("SAVEPOINT malformed")
                    cur.execute(
                        "UPDATE lab_arena_ledger SET entry_doc=%s::jsonb WHERE entry_id=%s",
                        (json.dumps(doc), candidate["uncertain_entry_id"]),
                    )
                    outcomes = []
                    for name in [NAME, CANDIDATE]:
                        cur.execute("SAVEPOINT native_outcome")
                        try:
                            cur.execute(
                                "SELECT public." + name + "(%s,%s,%s,%s)",
                                (rid, "", 0, 1),
                            )
                            outcomes.append(("ok", cur.fetchone()[0]))
                        except database[0].Error as exc:
                            outcomes.append(("error", exc.pgcode))
                        cur.execute("ROLLBACK TO SAVEPOINT native_outcome")
                    assert outcomes[0] == outcomes[1], (doc, outcomes)
                    cur.execute("ROLLBACK TO SAVEPOINT malformed")
    finally:
        store.close()


@pytest.mark.parametrize("reason", ["missing_provider_cost", "settle_failure"])
@pytest.mark.parametrize("run_status", ["accepted", "failed"])
@pytest.mark.parametrize("round_status", ["published", "cancelled"])
def test_existing_closed_settlement_lifecycle_stays_exact(
    database, reason, run_status, round_status
):
    # Existing native controls use a synthetic billing transport. They verify
    # exact append-once settlement without altering accepted round/run state.
    router.test_exact_closed_judge_bill_settles_once_without_changing_accepted_state(
        database, reason, run_status, round_status
    )


def test_5001_valid_heads_bound_proofs_and_preserve_cursor_wrap(database, tmp_path):
    store, rid, run, cid, candidate = router._call(database, "ordperf")
    conn = database[0].connect(**database[1])
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='20s'")
            cur.execute("SET jit=off")
            cur.execute("SET session_replication_role=replica")
            cur.execute(
                "UPDATE lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s",
                (rid,),
            )
            cur.execute("SELECT max(entry_id) FROM lab_arena_ledger")
            maximum = cur.fetchone()[0]
            cur.execute(
                "SELECT entry_kind,entry_id FROM lab_arena_ledger WHERE call_identity=%s AND entry_kind IN('reservation','dispatch','uncertain')",
                (cid,),
            )
            templates = dict(cur.fetchall())
            for offset, kind in enumerate(["reservation", "dispatch", "uncertain"]):
                covering.clone(
                    cur,
                    "lab_arena_ledger",
                    "l",
                    {
                        "entry_id": str(maximum) + "+3*s+" + str(offset),
                        "call_identity": "'sha256:'||md5('active-or-'||s::text)||md5('active-or-'||s::text)",
                    },
                    5000,
                    "l.entry_id=%s",
                    (templates[kind],),
                )
            cur.execute("RESET session_replication_role")
            for table in ["lab_arena_ledger", "lab_arena_runs", "lab_arena_rounds"]:
                cur.execute("VACUUM(ANALYZE) " + table)
            reports = {}
            for mode in ["force_generic_plan", "force_custom_plan"]:
                cur.execute("SET plan_cache_mode=" + mode)
                # Autocommit: SET LOCAL service role is unnecessary here; direct SQL has
                # the same definer owner and service ACL, already exercised above.
                for cursor, limit in [
                    (0, 1),
                    (0, 20),
                    (maximum + 14999, 20),
                    (2**63 - 1, 1),
                    (None, 0),
                ]:
                    args = (rid, "", cursor, limit)
                    cur.execute(
                        "SELECT public."
                        + NAME
                        + "(%s,%s,%s,%s),public."
                        + CANDIDATE
                        + "(%s,%s,%s,%s)",
                        args + args,
                    )
                    old, new = cur.fetchone()
                    assert old == new
                    plans = [
                        body_plan(cur, s, mode, args) for s in [SOURCE, definition(cur)]
                    ]
                    old_nodes = list(covering.nodes(plans[0]["Plan"]))
                    new_nodes = list(covering.nodes(plans[1]["Plan"]))
                    metadata = [n for n in new_nodes if n.get("Alias") == "metadata"]
                    if limit:
                        assert (
                            metadata
                            and sum(n.get("Actual Loops", 0) for n in metadata) == limit
                        ), (mode, args, metadata)
                        old_proofs = {
                            alias: sum(
                                n.get("Actual Rows", 0) * n.get("Actual Loops", 0)
                                for n in old_nodes
                                if n.get("Alias") == alias
                            )
                            for alias in ["later", "reservation", "dispatch"]
                        }
                        assert max(old_proofs.values()) >= 5001, (
                            mode,
                            args,
                            old_proofs,
                        )
                    reports[mode + "|" + str(cursor) + "|" + str(limit)] = {
                        "before": plans[0],
                        "after": plans[1],
                    }
            (tmp_path / "active-or-plans.json").write_text(json.dumps(reports))
    finally:
        conn.close()
        store.close()
