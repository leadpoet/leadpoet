"""Keyed cost reads preserve native evidence and avoid unrelated history scans."""
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import nov05_infrastructure_recovery633_postgres_test as source
from tests.lab_arena.test_lab_arena_migration_postgres import sha

SQL = (Path(__file__).resolve().parents[2] / "scripts/662-lab-arena-keyed-cost-evidence-lookups.sql").read_text()
SIGNATURES = (
    "public.lab_arena__successful_call_cost_state(text,text,text)",
    "public.lab_arena__successful_icp_cost_state(text,text,integer)",
    "public.lab_arena_submission_costs(text)",
)


@pytest.fixture(scope="module")
def database():
    yield from source.database.__wrapped__()


@pytest.fixture
def prepared(database):
    p = source.seed(database)
    with p.conn.cursor() as q:
        q.execute("SELECT pg_get_functiondef(oid) FROM pg_proc WHERE oid=ANY(%s::regprocedure[])", (list(SIGNATURES),))
        originals = [row[0] for row in q.fetchall()]
    p.conn.commit()
    try:
        yield p
    finally:
        p.conn.rollback()
        with p.conn.cursor() as q:
            for definition in originals:
                q.execute(definition)
        p.conn.commit()
        p.store.close()
        p.conn.close()


def apply(p):
    with p.conn.cursor() as q:
        q.execute(SQL)
    p.conn.commit()


def metadata(q):
    q.execute("SELECT oid,to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=ANY(%s::regprocedure[]) ORDER BY oid", (list(SIGNATURES),))
    return q.fetchall()


def snapshot(q):
    values = {}
    for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger", "lab_arena_deepline_call_responses", "lab_arena_restart_claim_control", "lab_arena_company_judgments", "lab_arena_signal_judgments", "lab_arena_scoring_supersessions"):
        q.execute("SELECT coalesce(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text),'[]'::jsonb) FROM public." + table + " t")
        values[table] = q.fetchone()[0]
    return values


def run(p, index=0, kind="execute"):
    with p.conn.cursor() as q:
        q.execute("SELECT run_id,round_id,submission_id,miner_hotkey,stage FROM lab_arena_runs WHERE submission_id=%s AND kind=%s ORDER BY run_id LIMIT 1", (p.participants[index]["submission_id"], kind))
        return dict(zip([d[0] for d in q.description], q.fetchone()))


def entry(q, r, label, kind="settlement", *, provider="openrouter", amount=100, terminal=None, doc=None, round_id=None):
    q.execute("INSERT INTO lab_arena_ledger(entry_kind,round_id,submission_id,run_id,miner_hotkey,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response)VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'miner_key',%s,%s::jsonb,%s::jsonb)RETURNING entry_id", (kind, round_id or r["round_id"], r["submission_id"], r["run_id"], r["miner_hotkey"], r["stage"], sha("662-" + label), provider, "deepline.test" if provider == "deepline" else "openrouter.chat", amount, json.dumps(doc or {}), json.dumps(terminal) if terminal is not None else None))
    return q.fetchone()[0]


def response(q, r, label, *, succeeded=True, response_run=None):
    reservation = entry(q, r, label, "reservation", provider="deepline")
    q.execute("INSERT INTO lab_arena_deepline_call_responses(call_identity,reservation_entry_id,run_id,request_id,terminal_response)VALUES(%s,%s,%s,%s,%s::jsonb)", (sha("662-" + label), reservation, response_run or r["run_id"], "req-662-" + label, json.dumps({"call_succeeded": succeeded})))


def states(q, r):
    out = []
    for kind in ("execute", "score", None, "invalid"):
        for provider in (None, "deepline", "openrouter", "scrapingdog", "invalid"):
            q.execute("SELECT lab_arena__successful_call_cost_state(%s,%s,%s)", (r["submission_id"], kind, provider))
            out.append(q.fetchone()[0])
    for round_id, pos in ((r["round_id"], 0), (r["round_id"], 1), (None, None), ("arena-other-round", 0)):
        q.execute("SELECT lab_arena__successful_icp_cost_state(%s,%s,%s)", (round_id, r["submission_id"], pos))
        out.append(q.fetchone()[0])
    q.execute("SELECT lab_arena_submission_costs(%s)", (r["submission_id"],))
    out.append(q.fetchone()[0])
    return out


def test_exact_cost_matrix_data_metadata_and_replay(prepared):
    p = prepared
    r, foreign, judge = run(p), run(p, 1), run(p, kind="score")
    with p.conn.cursor() as q:
        q.execute("SET LOCAL session_replication_role=replica")
        for label, value in (("true", True), ("false", False), ("null", None), ("string", "true"), ("object", {})):
            entry(q, r, label, terminal={"call_succeeded": value})
        for label, value in (("true", True), ("false", False), ("null", None)):
            entry(q, r, "uncertain-" + label, "uncertain", doc={"call": {"call_succeeded": value}})
        entry(q, r, "cancelled-uncertain", "uncertain", doc={"reason": "round_cancelled"})
        for kind in ("reservation", "dispatch", "refusal"):
            entry(q, r, kind, kind)
        response(q, r, "recovered")
        entry(q, r, "recovered", "uncertain", provider="deepline", amount=17, doc={"call": {"call_succeeded": False}})
        response(q, r, "recovered-settlement")
        entry(q, r, "recovered-settlement", provider="deepline", terminal={"call_succeeded": False})
        response(q, r, "wrong-run", response_run=foreign["run_id"])
        entry(q, r, "wrong-run", provider="deepline", terminal={"call_succeeded": False})
        response(q, r, "cancelled")
        entry(q, r, "cancelled", provider="deepline", terminal={"call_succeeded": True}, doc={"late_reconciliation": "true", "reconciled_uncertainty_reason": "round_cancelled"})
        prior = entry(q, r, "delayed", "uncertain", doc={"call": {"call_succeeded": True}})
        entry(q, r, "delayed", terminal={"call_succeeded": "wrong"}, doc={"openrouter_delayed_reconciliation": "true", "reconciled_uncertainty_entry_id": str(prior)})
        entry(q, judge, "judge", provider="deepline", terminal={"call_succeeded": True})
        entry(q, foreign, "foreign", amount=999999, terminal={"call_succeeded": True})
        entry(q, r, "foreign-round", amount=777, terminal={"call_succeeded": True}, round_id="arena-other-round")
        entry(q, r, "short-circuit", terminal={"call_succeeded": False}, doc={"openrouter_delayed_reconciliation": "true", "reconciled_uncertainty_entry_id": "not-integer"})
        q.execute("SET LOCAL session_replication_role=origin")
        before = states(q, r)
        assert {k: before[1][k] for k in (
            "settled_microusd", "successful_microusd", "success_unresolved_microusd",
            "inflight_calls", "uncertain_calls", "call_count")
        } == {"settled_microusd": 300, "successful_microusd": 100,
              "success_unresolved_microusd": 17, "inflight_calls": 0,
              "uncertain_calls": 1, "call_count": 4}
        assert before[6]["successful_microusd"] == 100
        identity, data = metadata(q), snapshot(q)
    p.conn.commit()
    for _ in range(2):
        apply(p)
        with p.conn.cursor() as q:
            assert states(q, r) == before
            assert metadata(q) == identity
            assert snapshot(q) == data
        p.conn.commit()


def _plan(q, body, name, args, analyze=False, parameters=("p_submission_id", "p_kind", "p_provider"), types="text,text,text"):
    for number, parameter in enumerate(parameters, 1):
        place = "$" + str(number)
        body = re.sub(r"\b" + parameter + r"\b", lambda _: place, body)
    q.execute("PREPARE " + name + "(" + types + ")AS " + body)
    q.execute("EXPLAIN(FORMAT JSON" + (",ANALYZE,BUFFERS" if analyze else "") + ")EXECUTE " + name + "(" + ",".join(["%s"] * len(args)) + ")", args)
    return q.fetchone()[0]


def _scans(value):
    if isinstance(value, dict):
        if "Node Type" in value and value.get("Relation Name") in ("lab_arena_runs", "lab_arena_deepline_call_responses"):
            yield value
        for item in value.values():
            yield from _scans(item)
    elif isinstance(value, list):
        for item in value:
            yield from _scans(item)


def test_generic_plan_uses_only_keyed_runs_and_response_reads(prepared):
    p = prepared
    r, foreign = run(p), run(p, 1)
    with p.conn.cursor() as q:
        q.execute("SET LOCAL session_replication_role=replica")
        # Unrelated history is deliberately large enough to reproduce the old
        # parameterized hash/sequence plans, without wall-clock assertions.
        q.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,stage_generation,kind,scoring_generation_floor)SELECT 'noise662:'||g,'noise662:'||g,%s,%s,%s,2,0,1,'pending',7,'execute',0 FROM generate_series(1,10000)g", (r["round_id"], foreign["submission_id"], foreign["miner_hotkey"]))
        for label in ("one", "two"):
            response(q, r, label)
            entry(q, r, label, provider="deepline", terminal={"call_succeeded": True})
        q.execute("INSERT INTO lab_arena_ledger(entry_kind,round_id,submission_id,run_id,miner_hotkey,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)SELECT 'reservation',%s,%s,'noise662:'||g,%s,2,'sha256:'||encode(extensions.digest('noise662-'||g,'sha256'),'hex'),'deepline','deepline.noise','miner_key',0,'{}'FROM generate_series(1,4000)g", (r["round_id"], foreign["submission_id"], foreign["miner_hotkey"]))
        q.execute("INSERT INTO lab_arena_deepline_call_responses(call_identity,reservation_entry_id,run_id,request_id,terminal_response)SELECT call_identity,entry_id,run_id,'noise-'||entry_id,jsonb_build_object('call_succeeded',true,'body',repeat(md5(entry_id::text),128))FROM lab_arena_ledger WHERE operation_id='deepline.noise'")
        q.execute("SET LOCAL session_replication_role=origin")
        q.execute("ANALYZE lab_arena_runs;ANALYZE lab_arena_ledger;ANALYZE lab_arena_deepline_call_responses;SET plan_cache_mode=force_generic_plan")
        q.execute("SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURES[0],))
        old_body = q.fetchone()[0]
        old_plan = _plan(q, old_body, "old_cost662", (r["submission_id"], "execute", "deepline"))
        assert any(x["Node Type"] == "Seq Scan" for x in _scans(old_plan))
        q.execute("EXECUTE old_cost662(%s,'execute','deepline')", (r["submission_id"],))
        before = q.fetchone()[0]
        q.execute("PREPARE retained_rpc662(text)AS SELECT public.lab_arena_submission_costs($1)")
        q.execute("EXECUTE retained_rpc662(%s)", (r["submission_id"],))
        retained_before = q.fetchone()[0]
        q.execute("PREPARE retained_icp662(text,text,integer)AS SELECT public.lab_arena__successful_icp_cost_state($1,$2,$3)")
        q.execute("EXECUTE retained_icp662(%s,%s,0)", (r["round_id"], r["submission_id"]))
        icp_before = q.fetchone()[0]
    p.conn.commit()
    apply(p)
    with p.conn.cursor() as q:
        q.execute("SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURES[0],))
        new_plan = _plan(q, q.fetchone()[0], "new_cost662", (r["submission_id"], "execute", "deepline"), analyze=True)
        assert list(_scans(new_plan))
        assert not any(x["Node Type"] == "Seq Scan" for x in _scans(new_plan))
        q.execute("EXECUTE new_cost662(%s,'execute','deepline')", (r["submission_id"],))
        assert q.fetchone()[0] == before
        q.execute("EXECUTE retained_rpc662(%s)", (r["submission_id"],))
        assert q.fetchone()[0] == retained_before
        q.execute("EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON)EXECUTE retained_rpc662(%s)", (r["submission_id"],))
        rpc_plan = q.fetchone()[0]
        assert rpc_plan[0]["Plan"]["Actual Rows"] == 1
        q.execute("EXECUTE retained_icp662(%s,%s,0)", (r["round_id"], r["submission_id"]))
        assert q.fetchone()[0] == icp_before
        q.execute("SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURES[1],))
        icp_plan = _plan(q, q.fetchone()[0], "icp_body662", (r["round_id"], r["submission_id"], 0),
                         analyze=True, parameters=("p_round_id", "p_submission_id", "p_icp_position"), types="text,text,integer")
        assert list(_scans(icp_plan))
        assert not any(x["Node Type"] == "Seq Scan" for x in _scans(icp_plan))
        q.execute("SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURES[2],))
        outer_body = q.fetchone()[0].split("    WITH run_kinds AS MATERIALIZED (", 1)[1].split("  ) AS grouped", 1)[0]
        outer_plan = _plan(q, "WITH run_kinds AS MATERIALIZED (" + outer_body,
                           "outer_body662", (r["submission_id"],), analyze=True,
                           parameters=("p_submission_id",), types="text")
        assert list(_scans(outer_plan))
        assert not any(x["Node Type"] == "Seq Scan" for x in _scans(outer_plan))
    p.conn.commit()


@pytest.mark.parametrize("fault", ["source", "acl", "index"])
def test_wrong_prerequisites_roll_back_all_replacements(prepared, fault):
    p = prepared
    with p.conn.cursor() as q:
        if fault == "source":
            q.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURES[-1],))
            original = q.fetchone()[0]
            changed = original.replace("DECLARE", "DECLARE /* unrelated662 */", 1)
            assert changed != original
            q.execute(changed)
        elif fault == "acl":
            q.execute("GRANT EXECUTE ON FUNCTION " + SIGNATURES[-1] + " TO authenticated")
        else:
            q.execute("ALTER INDEX lab_arena_ledger_cost_run_idx RENAME TO other_cost662")
        before, data = metadata(q), snapshot(q)
    p.conn.commit()
    try:
        with pytest.raises(p.pg.Error, match="Cost keyed lookup662"):
            with p.conn.cursor() as q:
                q.execute(SQL)
        p.conn.rollback()
        with p.conn.cursor() as q:
            assert metadata(q) == before
            assert snapshot(q) == data
    finally:
        p.conn.rollback()
        with p.conn.cursor() as q:
            if fault == "acl":
                q.execute("REVOKE EXECUTE ON FUNCTION " + SIGNATURES[-1] + " FROM authenticated")
            elif fault == "index":
                q.execute("ALTER INDEX other_cost662 RENAME TO lab_arena_ledger_cost_run_idx")
        p.conn.commit()


def test_malformed_delayed_cost_keeps_original_error(prepared):
    p = prepared
    r = run(p)
    with p.conn.cursor() as q:
        # Ensure the delayed-reference lane has a candidate under either a
        # primary-key or sequence plan; an empty lane may never evaluate a cast.
        entry(q, r, "prior-for-invalid", "uncertain", amount=0, doc={"call": {"call_succeeded": False}})
        entry(q, r, "invalid-reference", terminal={}, doc={"openrouter_delayed_reconciliation": "true", "reconciled_uncertainty_entry_id": "not-integer"})
    p.conn.commit()
    for patched in (False, True):
        if patched:
            apply(p)
        with p.conn.cursor() as q:
            q.execute("SELECT entry_doc FROM lab_arena_ledger WHERE call_identity=%s", (sha("662-invalid-reference"),))
            assert q.fetchone()[0]["reconciled_uncertainty_entry_id"] == "not-integer"
        try:
            with p.conn.cursor() as q:
                q.execute("SELECT lab_arena__successful_call_cost_state(%s,'execute','openrouter')", (r["submission_id"],))
                got = q.fetchone()[0]
        except p.pg.Error as failure:
            assert failure.pgcode == "22P02"
        else:
            pytest.fail("malformed reference unexpectedly accepted; patched=%s result=%r" % (patched, got))
        p.conn.rollback()
