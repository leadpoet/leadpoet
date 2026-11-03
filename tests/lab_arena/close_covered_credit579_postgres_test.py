"""Completed sources bypass obsolete credit proofs without weakening close."""
import copy
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import pytest

from scripts.verify_arena_full_thinking_e2e import full_thinking_database
from tests.lab_arena import company_judgment_validation_projection548_postgres_test as completed
from tests.lab_arena import proven_funding_recovery524_postgres_test as funding
from tests.lab_arena.flat_per_icp_cost500_postgres_test import catalog, rows

ROOT = Path(__file__).parents[2]
SQL = ROOT / "scripts/579-lab-arena-close-covered-credit-candidates.sql"
SIG = "public.lab_arena_close_scoring(text,smallint)"
HELPER = "public.lab_arena_miner_credit_failure_v1(text)"
BEFORE = "public.test579_before(text,smallint)"
PRE = "a07b809f89df827f3e0034e71f01c587b49bf1718715c7b60320e1521f322a6c"
POST = "10093875bac014642baa5056bc1f578611c46a327b9b7e870a9a2ca454c5d636"
HELPER_SHA = "23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6"


def scalar(cur, sql, args=()):
    cur.execute("SELECT " + sql, args)
    return cur.fetchone()[0]


def definition(cur, signature=SIG):
    return scalar(cur, "pg_get_functiondef(%s::regprocedure)", (signature,))


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def old_source(source):
    text = SQL.read_text()
    before = re.search(r"\$old_credit579\$(.*?)\$old_credit579\$", text, re.S)[1]
    after = re.search(r"\$new_credit579\$(.*?)\$new_credit579\$", text, re.S)[1]
    assert source.count(after) == 1
    result = source.replace(after, before, 1)
    assert digest(result) == PRE
    return result


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    generator = full_thinking_database()
    db = next(generator)
    fixture = completed.completed.__wrapped__(db, tmp_path_factory)
    try:
        h, row = next(fixture)
        conn = db[0].connect(**db[1])
        conn.autocommit = True
        with conn.cursor() as cur:
            for name in ("575-lab-arena-score-host-continuation.sql",
                         "576-lab-arena-champion-score-funding.sql",
                         "577-lab-arena-judge-continuation-runner-handoff.sql",
                         "578-lab-arena-closed-provider-ordered-candidate.sql"):
                cur.execute((ROOT / "scripts" / name).read_text())
            before = definition(cur)
            assert digest(before) == PRE and digest(definition(cur, HELPER)) == HELPER_SHA
            cur.execute(before.replace("FUNCTION public.lab_arena_close_scoring(",
                                       "FUNCTION public.test579_before(", 1))
            cur.execute("ALTER FUNCTION " + BEFORE + " OWNER TO lab_arena_owner")
            cur.execute("REVOKE ALL ON FUNCTION " + BEFORE + " FROM PUBLIC")
            cur.execute("GRANT EXECUTE ON FUNCTION " + BEFORE + " TO lab_arena_service")
            functions, data = catalog(cur), rows(cur)
            cur.execute(SQL.read_text())
            assert digest(definition(cur)) == POST and old_source(definition(cur)) == before
            after = catalog(cur)
            changed = []
            for a, b in zip(functions, after):
                assert a[0] == b[0] and a[2:] == b[2:]
                if a[1] != b[1]:
                    changed.append(a[0])
            assert len(functions) == len(after) and changed == [SIG.removeprefix("public.")]
            assert rows(cur) == data and digest(definition(cur, HELPER)) == HELPER_SHA
            cur.execute(SQL.read_text())
            assert catalog(cur) == after and rows(cur) == data
        conn.close()
        yield db, h, row
    finally:
        fixture.close()
        generator.close()


def seed(cur, scenario):
    rid = scenario[2]["round_id"]
    cur.execute("SET LOCAL session_replication_role=replica")
    cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',"
                "configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',"
                "'\"2099-01-01T00:00:00Z\"'::jsonb) WHERE round_id=%s", (rid,))
    cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,"
                "pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',"
                "guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")
    cur.execute("SET LOCAL session_replication_role=origin")
    assert scalar(cur, "public.lab_arena_judge_continuation_enabled_v1(%s)", (rid,)) is True
    run = scalar(cur, "to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s "
                 "AND kind='score' AND stage=2 AND status='accepted' ORDER BY run_id LIMIT 1", (rid,))
    return rid, run


def immutable(cur, rid):
    return {
        table: scalar(cur, "coalesce(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text),'[]') "
                      "FROM public." + table + " t WHERE round_id=%s" + predicate, (rid,))
        for table, predicate in (("lab_arena_runs", " AND status='accepted'"),
                                 ("lab_arena_ledger", ""))
    }


def clone(cur, run, attempt, **patch):
    value = copy.deepcopy(run)
    value.update(run_id=run["assignment_id"] + ":" + str(attempt), attempt=attempt,
                 claim_request_id=None, claim_request_hash=None, claim_response=None)
    value.update(patch)
    cur.execute("INSERT INTO public.lab_arena_runs SELECT "
                "(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*",
                (json.dumps(value),))
    return value


def calls(cur):
    return scalar(cur, "coalesce((SELECT calls FROM pg_stat_xact_user_functions "
                  "WHERE funcid=%s::regprocedure),0)", (HELPER,))


def close(cur, rid, *, before=False):
    cur.execute("SET LOCAL ROLE lab_arena_service")
    result = scalar(cur, ("public.test579_before" if before else "public.lab_arena_close_scoring")
                    + "(%s,2::smallint)", (rid,))
    cur.execute("RESET ROLE")
    return result


def compare(cur, rid, after_close=None):
    result = []
    for before in (True, False):
        cur.execute("SAVEPOINT same_state")
        start = calls(cur)
        output = close(cur, rid, before=before)
        count = calls(cur) - start
        result.append((output, count))
        if after_close:
            after_close()
        cur.execute("ROLLBACK TO SAVEPOINT same_state")
    assert result[0][0] == result[1][0]
    return result


def test_complete_real_cohort_skips_unchanged_classifier(scenario, tmp_path):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        rid, run = seed(cur, scenario)
        cur.execute("SET LOCAL track_functions='all'")
        cur.execute("SET LOCAL session_replication_role=replica")
        for attempt in range(2, 130):
            old = copy.deepcopy(run)
            old.update(run_id=run["assignment_id"] + ":" + str(attempt), attempt=attempt,
                       status="failed", terminal_cause="judge_error", output_ref=None,
                       result_doc={"terminal_status": "judge_error"}, claim_request_id=None,
                       claim_request_hash=None, claim_response=None)
            cur.execute("INSERT INTO public.lab_arena_runs SELECT "
                        "(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*",
                        (json.dumps(old),))
        cur.execute("SET LOCAL session_replication_role=origin")
        preserved = immutable(cur, rid)
        historical = scalar(cur, "jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM "
                            "public.lab_arena_runs r WHERE round_id=%s", (rid,))
        frozen = scalar(cur, "to_jsonb(r)-'status'-'updated_at'-'status_generation'-'stage_generation' FROM "
                        "public.lab_arena_rounds r WHERE round_id=%s", (rid,))

        def preserved_after_close():
            assert immutable(cur, rid) == preserved
            assert scalar(cur, "jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM "
                          "public.lab_arena_runs r WHERE round_id=%s", (rid,)) == historical
            assert scalar(cur, "to_jsonb(r)-'status'-'updated_at'-'status_generation'-'stage_generation' FROM "
                          "public.lab_arena_rounds r WHERE round_id=%s", (rid,)) == frozen

        results = compare(cur, rid, preserved_after_close)
        assert results[0][0]["status"] == "closed"
        assert results[0][0]["incomplete_assignments"] == 0
        assert results[0][1] >= 128 and results[1][1] == 0
        assert digest(definition(cur, HELPER)) == HELPER_SHA
        artifact = tmp_path / "close-credit579-proof.json"
        artifact.write_text(json.dumps({"empty_company_signed_lifecycle_fixture": True,
            "before": {"result": results[0][0], "real_classifier_calls": results[0][1]},
            "after": {"result": results[1][0], "real_classifier_calls": results[1][1]},
            "all_historical_runs_and_accepted_ledger_preserved": True}, indent=2))
        print("CLOSE_CREDIT579_PROOF " + str(artifact), flush=True)
        conn.rollback()


@pytest.mark.parametrize("coverage", [None, "same_floor", "other_floor", "other_round", "after_deadline"])
def test_real_uncovered_credit_drains_until_deadline_and_global_coverage_still_counts(scenario, coverage):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        rid, run = seed(cur, scenario)
        cur.execute("SET LOCAL track_functions='all'")
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='credential_error',"
                    "result_doc='{\"terminal_status\":\"credential_error\"}',output_ref=NULL WHERE run_id=%s", (run["run_id"],))
        funding.call(conn, run["run_id"], "credit579", outcome="credit")
        cur.execute("SET LOCAL session_replication_role=origin")
        assert scalar(cur, "public.lab_arena_miner_credit_failure_v1(%s)", (run["run_id"],)) is True
        if coverage == "after_deadline":
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,"
                        "'{schedule,final_scoring_close}','\"2000-01-01T00:00:00Z\"'::jsonb) WHERE round_id=%s", (rid,))
            cur.execute("SET LOCAL session_replication_role=origin")
        elif coverage:
            accepted = copy.deepcopy(run)
            accepted.update(run_id=run["assignment_id"] + ":2", attempt=2,
                            claim_request_id=None, claim_request_hash=None, claim_response=None)
            if coverage == "other_floor":
                accepted["scoring_generation_floor"] += 1
            if coverage == "other_round":
                accepted["round_id"] = "arena-2099-01-01"
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("INSERT INTO public.lab_arena_runs SELECT "
                        "(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*", (json.dumps(accepted),))
            cur.execute("SET LOCAL session_replication_role=origin")
        preserved = immutable(cur, rid)

        def preserved_after_close():
            assert immutable(cur, rid) == preserved

        results = compare(cur, rid, preserved_after_close)
        assert results[0][0]["status"] == ("closed" if coverage else "draining")
        assert (results[1][1] == 0) if coverage else (results[1][1] > 0)
        conn.rollback()


@pytest.mark.parametrize("baseline", [False, True])
def test_active_score_lease_still_drains(scenario, baseline):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        rid, run = seed(cur, scenario)
        if baseline:
            run = scalar(cur, "to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s "
                         "AND kind='score' AND stage=1 ORDER BY run_id LIMIT 1", (rid,))
        cur.execute("SET LOCAL track_functions='all'")
        cur.execute("SET LOCAL session_replication_role=replica")
        if baseline:
            # A cross-stage baseline recovery is attempt3 after two genuine
            # infrastructure failures, not an ordinary stage1 first attempt.
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',"
                        "result_doc='{\"terminal_status\":\"judge_error\"}',output_ref=NULL WHERE run_id=%s", (run["run_id"],))
            clone(cur, run, 2, status="failed", terminal_cause="judge_error",
                  result_doc={"terminal_status": "judge_error"}, output_ref=None)
            run = clone(cur, run, 3, status="leased", terminal_cause=None,
                        result_doc=None, output_ref=None)
        else:
            cur.execute("UPDATE public.lab_arena_runs SET status='leased' WHERE run_id=%s", (run["run_id"],))
        cur.execute("SET LOCAL session_replication_role=origin")
        if baseline:
            assert scalar(cur, "public.lab_arena__baseline_recovery_run_v1(r) FROM public.lab_arena_runs r WHERE run_id=%s", (run["run_id"],)) is True
        result = compare(cur, rid)
        assert result[0][0]["status"] == "draining"
        conn.rollback()


def test_uncovered_null_source_keeps_exact_old_negative_behavior(scenario):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        rid, run = seed(cur, scenario)
        cur.execute("SET LOCAL track_functions='all'")
        cur.execute("SET LOCAL session_replication_role=replica")
        clone(cur, run, 2, status="failed", terminal_cause="judge_error",
              result_doc={"terminal_status": "judge_error"}, output_ref=None, scored_run_id=None)
        cur.execute("SET LOCAL session_replication_role=origin")
        results = compare(cur, rid)
        assert results[0][0]["status"] == "closed" and results[1][1] == 1
        conn.rollback()


@pytest.mark.parametrize("old_floor", [False, True])
def test_retirement_respects_current_floor_and_accepted_first_precedence(scenario, old_floor):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        rid, run = seed(cur, scenario)
        cur.execute("SET LOCAL track_functions='all'")
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id=%s", (rid,))
        cur.execute("UPDATE public.lab_arena_runs SET scoring_generation_floor=1 WHERE round_id=%s AND kind='score'", (rid,))
        pending = clone(cur, run, 2, status="pending", terminal_cause=None, result_doc=None,
                        output_ref=None, scoring_generation_floor=0 if old_floor else 1)
        cur.execute("SET LOCAL session_replication_role=origin")
        statuses = []
        for before in (True, False):
            cur.execute("SAVEPOINT same_floor")
            result = close(cur, rid, before=before)
            assert result["status"] == "closed" and result["incomplete_assignments"] == 0
            statuses.append(scalar(cur, "status FROM public.lab_arena_runs WHERE run_id=%s", (pending["run_id"],)))
            cur.execute("ROLLBACK TO SAVEPOINT same_floor")
        assert statuses == (["pending"] * 2 if old_floor else ["failed"] * 2)
        conn.rollback()


@pytest.mark.parametrize("status,expected", [("stage2", "stale"), ("stage2_judged", "existing"),
                                            ("scored", "existing"), ("published", "existing")])
def test_existing_and_stale_statuses_are_unchanged(scenario, status, expected):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        rid, _ = seed(cur, scenario)
        cur.execute("SET LOCAL track_functions='all'")
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET status=%s WHERE round_id=%s", (status, rid))
        cur.execute("SET LOCAL session_replication_role=origin")
        results = compare(cur, rid)
        assert results[0][0]["status"] == expected and results[1][1] == 0
        conn.rollback()


def test_operator_pause_remains_fail_closed(scenario):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        rid, _ = seed(cur, scenario)
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton")
        for before in (True, False):
            cur.execute("SAVEPOINT paused")
            with pytest.raises(db[0].Error) as error:
                close(cur, rid, before=before)
            assert error.value.pgcode == "55000"
            cur.execute("ROLLBACK TO SAVEPOINT paused")
            assert scalar(cur, "status FROM public.lab_arena_rounds WHERE round_id=%s", (rid,)) == "stage2_scoring"
        conn.rollback()


def test_claim_control_still_precedes_round_lock(scenario):
    db, _, row = scenario
    for before in (True, False):
        holder = db[0].connect(**db[1])
        worker = db[0].connect(**db[1])
        outcomes = []
        started = threading.Event()
        try:
            with holder.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0))")
            def run():
                try:
                    with worker.cursor() as cur:
                        cur.execute("SET LOCAL statement_timeout='5s'")
                        started.set()
                        outcomes.append(close(cur, row["round_id"], before=before))
                    worker.commit()
                except Exception as error:
                    outcomes.append(type(error).__name__)
            thread = threading.Thread(target=run)
            thread.start()
            assert started.wait(2)
            with db[0].connect(**db[1]) as observer, observer.cursor() as cur:
                until = time.monotonic() + 2
                while time.monotonic() < until:
                    cur.execute("SELECT wait_event,pg_blocking_pids(%s) FROM pg_stat_activity WHERE pid=%s",
                                (worker.get_backend_pid(), worker.get_backend_pid()))
                    waiting = cur.fetchone()
                    if waiting and waiting[0] == "advisory" and holder.get_backend_pid() in waiting[1]:
                        break
                    time.sleep(0.02)
                else:
                    pytest.fail("close did not wait on claim control")
                # A close blocked on claim control must not own the round lock.
                cur.execute("SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE NOWAIT", (row["round_id"],))
                assert cur.fetchone() == (1,)
                observer.rollback()
            holder.rollback()
            thread.join(5)
            assert not thread.is_alive() and outcomes[0]["status"] == "existing", outcomes
        finally:
            holder.rollback()
            holder.close()
            worker.close()


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("drift", ["source", "owner", "acl", "security", "volatility", "path", "helper", "target"])
def test_source_security_and_replay_guards(scenario, installed, drift):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        if not installed:
            cur.execute(old_source(definition(cur)))
        if drift == "source":
            cur.execute(definition(cur).replace("BEGIN\n", "BEGIN\n  -- drift\n", 1))
        elif drift == "helper":
            cur.execute(definition(cur, HELPER).replace("BEGIN\n", "BEGIN\n  -- drift\n", 1))
        elif drift != "target":
            cur.execute({"owner": "ALTER FUNCTION " + SIG + " OWNER TO postgres",
                         "acl": "GRANT EXECUTE ON FUNCTION " + SIG + " TO anon",
                         "security": "ALTER FUNCTION " + SIG + " SECURITY INVOKER",
                         "volatility": "ALTER FUNCTION " + SIG + " STABLE",
                         "path": "ALTER FUNCTION " + SIG + " SET search_path=public,pg_catalog"}[drift])
        text = SQL.read_text().replace(POST, "0" * 64) if drift == "target" else SQL.read_text()
        body = text.split("DO $close_covered_credit579$")[1].split("$close_covered_credit579$;")[0]
        cur.execute("SAVEPOINT hostile")
        with pytest.raises(db[0].Error, match="Close covered credit"):
            cur.execute("DO $test$" + body + "$test$")
        cur.execute("ROLLBACK TO SAVEPOINT hostile")
        conn.rollback()
