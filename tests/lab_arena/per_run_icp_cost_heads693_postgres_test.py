"""Existing per-run index bounds cost-head reads without changing cost authority."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.lab_arena import keyed_cost_evidence662_postgres_test as prior

SQL = (Path(__file__).resolve().parents[2] / "scripts/693-lab-arena-per-run-icp-cost-heads.sql").read_text()
SIGNATURE = prior.SIGNATURES[1]


@pytest.fixture(scope="module")
def database():
    yield from prior.database.__wrapped__()


@pytest.fixture
def prepared(database):
    fixture = prior.prepared.__wrapped__(database)
    p = next(fixture)
    try:
        prior.apply(p)
        yield p
    finally:
        fixture.close()


def apply(p):
    with p.conn.cursor() as q:
        q.execute(SQL)
    p.conn.commit()


def state(q, r, pos=0):
    q.execute("SELECT public.lab_arena__successful_icp_cost_state(%s,%s,%s)",
              (r["round_id"], r["submission_id"], pos))
    return q.fetchone()[0]


def outcome(q, sql, args):
    q.execute("SAVEPOINT cost693_case")
    try:
        q.execute(sql, args)
        value = ("value", q.fetchone()[0])
    except Exception as error:
        if not getattr(error, "pgcode", None):
            raise
        value = ("error", error.pgcode, error.diag.message_primary)
    finally:
        q.execute("ROLLBACK TO SAVEPOINT cost693_case")
    return value


def test_native_matrix_preserves_costs_data_security_and_replay(prepared, monkeypatch):
    # Reuse the native settled/uncertain/inflight, delayed settlement, recovered
    # Deepline, wrong-run, cross-round/provider/kind, and null matrix from662.
    # It checks every result, original pg_proc metadata, data, and two replays.
    monkeypatch.setattr(prior, "SQL", SQL)
    prior.test_exact_cost_matrix_data_metadata_and_replay(prepared)


@pytest.mark.parametrize("mode", ["force_generic_plan", "force_custom_plan"])
def test_heads_read_only_exact_matching_run_with_existing_index(prepared, mode):
    p = prepared
    r, foreign = prior.run(p), prior.run(p, 1)
    with p.conn.cursor() as q:
        q.execute("SET LOCAL session_replication_role=replica")
        # Ten ICPs each have100 paid calls and three records per call. Unrelated
        # rows reproduce plans that previously scanned a submission or round.
        for row, label in ((r, "own"), (foreign, "foreign")):
            q.execute("""INSERT INTO lab_arena_ledger(entry_kind,round_id,submission_id,
                run_id,miner_hotkey,stage,call_identity,provider,operation_id,
                funding_source,amount_microusd,entry_doc,terminal_response)
              SELECT kind,%s,%s,runs.run_id,%s,runs.stage,
                'sha256:'||encode(extensions.digest(%s||g,'sha256'),'hex'),
                'openrouter','openrouter.chat','miner_key',10,'{}',
                CASE WHEN kind='settlement' THEN '{"call_succeeded":true}'::jsonb END
              FROM generate_series(1,1000)g
              CROSS JOIN(VALUES('reservation'),('dispatch'),('settlement'))k(kind)
              CROSS JOIN LATERAL(SELECT run_id,stage FROM lab_arena_runs
                WHERE submission_id=%s AND kind='execute' AND icp_position=g%%10
                ORDER BY attempt DESC LIMIT 1)runs""",
                      (row["round_id"], row["submission_id"], row["miner_hotkey"],
                       "693-" + label, row["submission_id"]))
        q.execute("SET LOCAL session_replication_role=origin;ANALYZE lab_arena_ledger;ANALYZE lab_arena_runs")
        q.execute("SET plan_cache_mode=" + mode)
        q.execute("SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURE,))
        prior._plan(q, q.fetchone()[0], "before_cost693_" + mode,
                    (r["round_id"], r["submission_id"], 9),
                    parameters=("p_round_id", "p_submission_id", "p_icp_position"),
                    types="text,text,integer")
        before = []
        for pos in range(10):
            q.execute("EXECUTE before_cost693_" + mode + "(%s,%s,%s)",
                      (r["round_id"], r["submission_id"], pos))
            before.append(q.fetchone()[0])
    p.conn.commit()
    apply(p)
    with p.conn.cursor() as q:
        q.execute("SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURE,))
        body = q.fetchone()[0]
        plan = prior._plan(q, body, "cost693_" + mode,
                           (r["round_id"], r["submission_id"], 9), analyze=True,
                           parameters=("p_round_id", "p_submission_id", "p_icp_position"),
                           types="text,text,integer")
        after = []
        for pos in range(10):
            q.execute("EXECUTE cost693_" + mode + "(%s,%s,%s)",
                      (r["round_id"], r["submission_id"], pos))
            after.append(q.fetchone()[0])
        assert after == before
        def heads_scans(value):
            if isinstance(value, dict):
                # The discovery CTE deliberately remains unchanged. Inspect
                # only the second ledger read that computes the cost heads.
                if value.get("Subplan Name") == "CTE matching_runs":
                    return
                if value.get("Relation Name") == "lab_arena_ledger" and value.get("Alias") in ("ledger", "found"):
                    yield value
                for item in value.values():
                    yield from heads_scans(item)
            elif isinstance(value, list):
                for item in value:
                    yield from heads_scans(item)
        scans = list(heads_scans(plan))
        assert scans
        assert all(scan.get("Index Name") == "lab_arena_ledger_cost_run_idx"
                   and "submission_id" in scan["Index Cond"]
                   and "run_id = runs.run_id" in scan["Index Cond"] for scan in scans)
        assert all(scan["Actual Rows"] < 1000 for scan in scans)
    p.conn.commit()


def test_exact_errors_null_empty_scopes_and_irrelevant_malformed_evidence(prepared):
    p = prepared
    r = prior.run(p)
    with p.conn.cursor() as q:
        q.execute("SET LOCAL session_replication_role=replica")
        prior.entry(q, r, "uncertain-before-bad", "uncertain", doc={"call": {"call_succeeded": False}})
        prior.entry(q, r, "bad-delayed", doc={"openrouter_delayed_reconciliation": "true",
                                            "reconciled_uncertainty_entry_id": "not-integer"})
        q.execute("SET LOCAL session_replication_role=origin")
        cases = [("SELECT public.lab_arena__successful_icp_cost_state(%s,%s,%s)", args)
                 for args in ((r["round_id"], r["submission_id"], 0),
                              (r["round_id"], r["submission_id"], 9),
                              (None, None, None), ("", "", 0),
                              ("missing", r["submission_id"], 0))]
        cases += [("SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,%s)", args)
                  for args in ((r["round_id"], r["submission_id"], 0, 0),
                               (r["round_id"], r["submission_id"], 9, 5),
                               (None, None, None, None),
                               (r["round_id"], r["submission_id"], -1, 0),
                               (r["round_id"], r["submission_id"], 10, 0),
                               (r["round_id"], "missing", 0, 0))]
        cases += [("SELECT public.lab_arena_submission_costs(%s)", (v,))
                  for v in (None, "", "missing", "bad value")]
        before = [outcome(q, sql, args) for sql, args in cases]
        assert before[0][0:2] == ("error", "22P02")
        assert before[1][0] == "value"
    p.conn.commit()
    apply(p)
    with p.conn.cursor() as q:
        assert [outcome(q, sql, args) for sql, args in cases] == before
    p.conn.commit()


def test_historical_cross_submission_round_and_run_links_are_preserved(prepared):
    p = prepared
    r, foreign, judge = prior.run(p), prior.run(p, 1), prior.run(p, kind="score")
    with p.conn.cursor() as q:
        q.execute("SET LOCAL session_replication_role=replica")
        prior.entry(q, r, "same-call-different-round", "uncertain", amount=111, doc={"call": {"call_succeeded": True}})
        prior.entry(q, r, "same-call-different-round", amount=999,
                    terminal={"call_succeeded": True}, round_id="other693-round")
        # Historical behavior follows the ledger's submission plus exact run-ID
        # join. Additional run-owner/round filters would silently change it.
        prior.entry(q, {**r, "run_id": foreign["run_id"]}, "cross-submission-run",
                    provider="scrapingdog", amount=222, terminal={"call_succeeded": True})
        prior.entry(q, {**foreign, "run_id": r["run_id"]}, "foreign-submission-ledger",
                    amount=9999, terminal={"call_succeeded": True})
        prior.entry(q, {**r, "run_id": "missing693-run"}, "orphan-run",
                    amount=99999, terminal={"call_succeeded": True})
        prior.entry(q, judge, "judge-only", amount=99999, terminal={"call_succeeded": True})
        q.execute("SET LOCAL session_replication_role=origin")
        before = state(q, r)
        assert before["settled_microusd"] == 223
        assert before["reserved_or_uncertain_microusd"] == 111
        assert before["call_count"] == 3
        complete_before = prior.states(q, r)
    p.conn.commit()
    apply(p)
    with p.conn.cursor() as q:
        assert state(q, r) == before
        assert prior.states(q, r) == complete_before
    p.conn.commit()


def publication_row(q, r):
    rows = []
    for pos in range(10):
        q.execute("SELECT public.lab_arena__integrity_submission_summary(%s,%s,ARRAY[%s])",
                  (r["round_id"], r["submission_id"], pos))
        summary = q.fetchone()[0]
        assert summary["valid"]
        q.execute("SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,%s)",
                  (r["round_id"], r["submission_id"], pos, summary["qualified_company_count"]))
        rows.append({**q.fetchone()[0], "qualified_company_count": summary["qualified_company_count"],
                     "returned_company_count": 0})
    costs = {}
    for kind in ("execute", "score"):
        q.execute("SELECT public.lab_arena__cost_kind_summary_v1(%s,%s)", (r["submission_id"], kind))
        costs[kind] = q.fetchone()[0]
    return {"submission_id": r["submission_id"], "final_score": 0, "eligible": True,
            "eligibility_reason": "eligible",
            "training_data_eligibility": {"status": "not_assessed", "eligible": None, "reasons": []},
            "cost_summary": {"sourcing_cost_eligibility_policy": "successful_calls_flat_per_icp_v1",
                "execution_icp_cap_microusd": 6000000, "cost_per_company_cap_microusd": 6000000,
                "returned_company_count": 0, "qualified_company_count": 0, "eligible_icp_count": 10,
                "competition_sourcing_microusd": sum(row["competition_sourcing_microusd"] for row in rows),
                "per_icp": rows, "execution": costs["execute"], "judge": costs["score"]}}


def test_complete_baseline_and_miner_publication_reject_same_forged_reports(prepared):
    p = prepared
    with p.conn.cursor() as q:
        q.execute("SET LOCAL session_replication_role=replica")
        q.execute("UPDATE lab_arena_runs SET per_icp_score=0,qualification_doc='{\"companies\":[]}' WHERE kind='execute' AND status='accepted'")
        q.execute("SET LOCAL session_replication_role=origin")
        sql = "SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)"
        cases = []
        for r in (prior.run(p), prior.run(p, 1)):
            genuine = publication_row(q, r)
            cases.append((r["round_id"], json.dumps(genuine)))
            for key in ("competition_sourcing_microusd", "eligible_icp_count", "qualified_company_count"):
                forged = deepcopy(genuine)
                forged["cost_summary"][key] += 1
                cases.append((r["round_id"], json.dumps(forged)))
            for key in ("settled_microusd", "success_unresolved_calls"):
                forged = deepcopy(genuine)
                forged["cost_summary"]["per_icp"][0]["execution"][key] += 1
                cases.append((r["round_id"], json.dumps(forged)))
            forged = deepcopy(genuine)
            forged["training_data_eligibility"] = {"eligible": True, "reasons": []}
            cases.append((r["round_id"], json.dumps(forged)))
        before = [outcome(q, sql, args) for args in cases]
        assert before[0] == before[7] == ("value", True)
        assert all(row[0:2] == ("error", "22023") for i, row in enumerate(before) if i not in (0, 7))
    p.conn.commit()
    apply(p)
    with p.conn.cursor() as q:
        assert [outcome(q, sql, args) for args in cases] == before
    p.conn.commit()


def test_native_infrastructure_exempt_and_retired_source_costs_are_unchanged(prepared):
    p = prepared
    r = prior.run(p)
    with p.conn.cursor() as q:
        q.execute("SET LOCAL session_replication_role=replica")
        # The original proof helper, not a mock, recognizes independently bound
        # failed-provider dispatch and terminal evidence on this historical run.
        q.execute("""UPDATE lab_arena_runs SET status='failed',output_ref=NULL,
            terminal_cause='provider_error',result_doc='{"terminal_status":"provider_error",
              "resource_summary":{"provider_unresolved_failure_count":1}}',
            runner_hotkey=miner_hotkey,lease_token_hash=%s,claim_request_id='cost693-host',
            lease_generation=1,claim_response='{"status":"leased","lease_generation":1}'
            WHERE run_id=%s""", (prior.sha("693-host"), r["run_id"]))
        prior.entry(q, r, "host-failure", "dispatch")
        prior.entry(q, r, "host-failure", terminal={"call_succeeded": False, "status": 503}, amount=900)
        q.execute("SELECT public.lab_arena__execute_infrastructure_cost_exempt_v1(r) FROM lab_arena_runs r WHERE run_id=%s", (r["run_id"],))
        assert q.fetchone()[0] is True
        infrastructure_before = state(q, r)
        assert infrastructure_before["excluded_infrastructure_settled_microusd"] == 901
        assert infrastructure_before["competition_settled_microusd"] == 0

        retired_round, retired_sub, retired_run = "arena-2026-11-04", "baseline-2026-11-04", "cost693-retired-run"
        q.execute("""INSERT INTO lab_arena_rounds(round_id,status,configuration_doc,participants,evaluation_date)
            SELECT %s,status,configuration_doc,participants,'2026-11-04' FROM lab_arena_rounds WHERE round_id=%s""",
                  (retired_round, r["round_id"]))
        q.execute("""INSERT INTO lab_arena_submissions SELECT (jsonb_populate_record(NULL::lab_arena_submissions,
            to_jsonb(s)||jsonb_build_object('round_id',%s,'submission_id',%s))).*
            FROM lab_arena_submissions s WHERE submission_id=%s""", (retired_round, retired_sub, r["submission_id"]))
        q.execute("""INSERT INTO lab_arena_runs SELECT (jsonb_populate_record(NULL::lab_arena_runs,
            to_jsonb(r)||jsonb_build_object('round_id',%s,'submission_id',%s,'run_id',%s,
                'assignment_id',%s,'stage',1,'status','accepted','output_ref','retained693.json'))).*
            FROM lab_arena_runs r WHERE run_id=%s""", (retired_round, retired_sub, retired_run, retired_run, r["run_id"]))
        q.execute("SELECT configuration_doc FROM lab_arena_rounds WHERE round_id=%s", (retired_round,))
        current_config = q.fetchone()[0]
        previous_config = {**current_config, "reasoning_baseline_source_sha256": prior.sha("old693-source")}
        q.execute("""INSERT INTO lab_arena_scoring_supersessions(round_id,generation_floor,
            request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,
            new_round_doc,old_score_projections,recovery_doc)
            VALUES(%s,1,'nov04-baseline-source621-v1',%s,'test693',1,1,%s::jsonb,%s::jsonb,'{}',%s::jsonb)""",
                  (retired_round, prior.sha("693-history"), json.dumps({"configuration_doc": previous_config}),
                   json.dumps({"configuration_doc": current_config}), json.dumps({"operation": "baseline_source_supersession_v1",
                       "retired_source_run_hashes": {retired_run: prior.sha("693-retired")}})))
        retired = {**r, "round_id": retired_round, "submission_id": retired_sub, "run_id": retired_run, "stage": 1}
        prior.entry(q, retired, "retired", terminal={"call_succeeded": True}, amount=700)
        q.execute("SELECT public.lab_arena__retired_source621_v1(r) FROM lab_arena_runs r WHERE run_id=%s", (retired_run,))
        assert q.fetchone()[0] is True
        q.execute("SET LOCAL session_replication_role=origin")
        retired_before = state(q, retired)
        assert retired_before["settled_microusd"] == 700
        assert retired_before["competition_settled_microusd"] == 0
        assert retired_before["excluded_superseded_source_successful_microusd"] == 700
        data = prior.snapshot(q)
    p.conn.commit()
    apply(p)
    with p.conn.cursor() as q:
        assert state(q, r) == infrastructure_before
        assert state(q, retired) == retired_before
        assert prior.snapshot(q) == data
    p.conn.commit()


@pytest.mark.parametrize("fault", ["source", "acl", "index"])
def test_changed_prerequisites_fail_closed_without_data_change(prepared, fault):
    p = prepared
    with p.conn.cursor() as q:
        q.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        original = q.fetchone()[0]
        if fault == "source":
            q.execute(original.replace("WITH matching_runs", "WITH /* unrelated693 */ matching_runs", 1))
        elif fault == "acl":
            q.execute("GRANT EXECUTE ON FUNCTION " + SIGNATURE + " TO authenticated")
        else:
            q.execute("ALTER INDEX lab_arena_ledger_cost_run_idx RENAME TO other_cost693")
        before, data = prior.metadata(q), prior.snapshot(q)
    p.conn.commit()
    try:
        with pytest.raises(p.pg.Error, match="Cost heads693"):
            with p.conn.cursor() as q:
                q.execute(SQL)
        p.conn.rollback()
        with p.conn.cursor() as q:
            assert prior.metadata(q) == before
            assert prior.snapshot(q) == data
    finally:
        p.conn.rollback()
        with p.conn.cursor() as q:
            q.execute(original)
            if fault == "acl":
                q.execute("REVOKE EXECUTE ON FUNCTION " + SIGNATURE + " FROM authenticated")
            if fault == "index":
                q.execute("ALTER INDEX other_cost693 RENAME TO lab_arena_ledger_cost_run_idx")
        p.conn.commit()


def test_signed_pending_billing_lifecycle_with_current_cost_heads(prepared, tmp_path, monkeypatch):
    from tests.lab_arena import pending_success_lifecycle_e2e_test as lifecycle

    # Its original fixture intentionally has a historical schema tail. Upgrade
    # only the two current cost-proof definitions required by693 from the native
    # 621/630/662 fixture above, retaining that fixture's HTTP/billing authorities.
    signatures = ("public.lab_arena__retired_source621_v1(public.lab_arena_runs)", SIGNATURE)
    with prepared.conn.cursor() as q:
        definitions = []
        for signature in signatures:
            q.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            definitions.append(q.fetchone()[0])
    prepared.conn.commit()
    fixture = lifecycle.database.__wrapped__()
    database = next(fixture)
    try:
        pg, dsn = database
        with pg.connect(**dsn) as connection, connection.cursor() as q:
            for signature, definition in zip(signatures, definitions):
                q.execute(definition)
                q.execute("ALTER FUNCTION " + signature + " OWNER TO lab_arena_owner")
                q.execute("REVOKE ALL ON FUNCTION " + signature +
                          " FROM PUBLIC,anon,authenticated,service_role,lab_arena_service")
            q.execute(SQL)
        lifecycle.test_pending_success_restarts_replays_settles_and_publishes_baseline_and_miner(
            database, tmp_path, monkeypatch)
    finally:
        fixture.close()
