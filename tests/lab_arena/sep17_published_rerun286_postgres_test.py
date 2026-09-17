"""Disposable-Postgres proof for the one-time published Sep17 rerun286."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache, scoring
from lab_arena.store import hash_lease_token, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.sep16_native_baseline_rerun_postgres_test import (
    _proof_breakdown,
    _proof_company,
    _proof_execution,
)
from tests.lab_arena.sep17_baseline_recovery285_postgres_test import (
    BANK_SNAPSHOT,
    BASELINE,
    TEMPLATE as TEMPLATE_285,
    TEST_SOURCE_COMMIT as SOURCE_285_COMMIT,
    TEST_SOURCE_SHA as SOURCE_285_SHA,
    TEST_SOURCE_SIZE as SOURCE_285_SIZE,
    captured as captured_284_terminal,
    compact,
    replacements as replacements_285,
    restore_terminal as restore_284_terminal,
)
from tests.lab_arena.test_lab_arena_service_round import Harness


ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "scripts/286-arena-2026-09-17-published-baseline-rerun.sql.template"
ROUND = "arena-2026-09-17"
ARCHIVE = "arena-2026-09-17-rerun285archive"
SOURCE_286_SIZE = 610_286
SOURCE_286_SHA = "d" * 64
SOURCE_286_COMMIT = "e" * 40
pytestmark = pytest.mark.skipif(
    not BANK_SNAPSHOT.is_file(), reason="rerun286 PostgreSQL proof requires the sealed ICP bank",
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _future_schedule(old):
    return {
        **old,
        "benchmark_deadline": "2099-01-01T00:00:00Z",
        "stage_1_start": "2099-01-01T00:00:01Z",
        "stage_1_close": "2099-01-01T01:00:00Z",
        "stage_1_scoring_close": "2099-01-01T02:00:00Z",
        "stage_2_start": "2099-01-01T02:00:01Z",
        "stage_2_close": "2099-01-01T03:00:00Z",
        "final_scoring_close": "2099-01-01T04:00:00Z",
        "publication_deadline": "2099-01-01T04:30:00Z",
    }


def _render_285(snapshot, schedule):
    sql = TEMPLATE_285.read_text()
    for token, value in replacements_285(snapshot).items():
        sql = sql.replace(token, value)
    terminal_schedule = compact(
        json.loads(re.search(r"\$forward_schedule\$(.*?)\$forward_schedule\$", sql).group(1))
    )
    sql = sql.replace(terminal_schedule, compact(schedule))
    assert re.search(r"__[A-Z0-9_]+__", sql) is None
    return sql


def _hash(cursor, table, where, order, strip_identity=False):
    expression = "to_jsonb(x)"
    if strip_identity:
        expression += "-'round_id'-'submission_id'"
    cursor.execute(
        "SELECT 'sha256:'||encode(extensions.digest(coalesce(string_agg("
        "encode(extensions.digest((%s)::text,'sha256'),'hex'),'' ORDER BY %s),''),"
        "'sha256'),'hex') FROM public.%s x WHERE %s"
        % (expression, order, table, where)
    )
    return cursor.fetchone()[0]


def _terminal_seals(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_jsonb(x) FROM public.lab_arena_rounds x WHERE round_id=%s", (ROUND,))
        round_doc = cursor.fetchone()[0]
        cursor.execute(
            "SELECT to_jsonb(x) FROM public.lab_arena_submissions x "
            "WHERE round_id=%s AND submission_id=%s", (ROUND, BASELINE),
        )
        baseline = cursor.fetchone()[0]
        counts = {}
        for name, table, predicate in (
            ("BASELINE_RUN", "lab_arena_runs", "round_id='%s' AND submission_id='%s'" % (ROUND, BASELINE)),
            ("BASELINE_LEDGER", "lab_arena_ledger", "round_id='%s' AND submission_id='%s'" % (ROUND, BASELINE)),
            ("NONBASELINE_SUBMISSION", "lab_arena_submissions", "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE)),
            ("NONBASELINE_RUN", "lab_arena_runs", "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE)),
            ("NONBASELINE_LEDGER", "lab_arena_ledger", "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE)),
        ):
            cursor.execute("SELECT count(*) FROM public.%s WHERE %s" % (table, predicate))
            counts[name] = cursor.fetchone()[0]
        hashes = {
            "BASELINE_RUNS": _hash(cursor, "lab_arena_runs", "round_id='%s' AND submission_id='%s'" % (ROUND, BASELINE), "run_id", True),
            "BASELINE_LEDGER": _hash(cursor, "lab_arena_ledger", "round_id='%s' AND submission_id='%s'" % (ROUND, BASELINE), "entry_id", True),
            "NONBASELINE_SUBMISSIONS": _hash(cursor, "lab_arena_submissions", "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE), "submission_id"),
            "NONBASELINE_RUNS": _hash(cursor, "lab_arena_runs", "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE), "run_id"),
            "NONBASELINE_LEDGER": _hash(cursor, "lab_arena_ledger", "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE), "entry_id"),
        }
    return round_doc, baseline, counts, hashes


def _render_286(round_doc, baseline, counts, hashes, schedule, definition_hash):
    values = {
        "__FORWARD_SCHEDULE_JSON__": compact(schedule),
        "__SCORING_DEFINITION_SHA256__": definition_hash,
        "__NEW_SOURCE_SIZE_BYTES__": str(SOURCE_286_SIZE),
        "__NEW_SOURCE_SHA256__": SOURCE_286_SHA,
        "__NEW_SOURCE_COMMIT__": SOURCE_286_COMMIT,
        "__TERMINAL_ROUND_JSON__": compact(round_doc),
        "__TERMINAL_BASELINE_JSON__": compact(baseline),
        "__TERMINAL_SOURCE_SIZE_BYTES__": str(SOURCE_285_SIZE),
        "__TERMINAL_SOURCE_SHA256__": SOURCE_285_SHA,
        "__TERMINAL_SOURCE_COMMIT__": SOURCE_285_COMMIT,
        "__TERMINAL_BASELINE_RUN_COUNT__": str(counts["BASELINE_RUN"]),
        "__TERMINAL_BASELINE_RUNS_HASH__": hashes["BASELINE_RUNS"],
        "__TERMINAL_BASELINE_LEDGER_COUNT__": str(counts["BASELINE_LEDGER"]),
        "__TERMINAL_BASELINE_LEDGER_HASH__": hashes["BASELINE_LEDGER"],
        "__TERMINAL_NONBASELINE_SUBMISSION_COUNT__": str(counts["NONBASELINE_SUBMISSION"]),
        "__TERMINAL_NONBASELINE_SUBMISSIONS_HASH__": hashes["NONBASELINE_SUBMISSIONS"],
        "__TERMINAL_NONBASELINE_RUN_COUNT__": str(counts["NONBASELINE_RUN"]),
        "__TERMINAL_NONBASELINE_RUNS_HASH__": hashes["NONBASELINE_RUNS"],
        "__TERMINAL_NONBASELINE_LEDGER_COUNT__": str(counts["NONBASELINE_LEDGER"]),
        "__TERMINAL_NONBASELINE_LEDGER_HASH__": hashes["NONBASELINE_LEDGER"],
    }
    sql = TEMPLATE.read_text()
    for token, value in values.items():
        assert token in sql
        sql = sql.replace(token, value)
    assert re.search(r"__[A-Z0-9_]+__", sql) is None
    return sql


def _prepare(cursor, schedule):
    cursor.execute(
        "SELECT public.lab_arena_prepare_sep17_published_rerun286_v1(%s,%s,%s,%s,%s::jsonb)",
        (SOURCE_286_SIZE, SOURCE_286_SHA, SOURCE_286_COMMIT,
         "7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871",
         json.dumps(schedule)),
    )
    return cursor.fetchone()[0]


def _drive_cycle(
    service, objects, icps, *, runner_hotkey, score_suffix, retry_execute=False,
    publish=True, model_version=0,
):
    saw_retry = False
    if retry_execute:
        token = new_lease_token()
        request_id = contracts.new_request_id()
        first = service.store.claim_assignment(
            round_id=ROUND, runner_hotkey=runner_hotkey,
            declared_parallelism=10, slot_ceiling=20, excluded_miner_hotkeys=[],
            request_id=request_id,
            request_hash=contracts.document_hash({"request_id": request_id}),
            lease_token_hash=hash_lease_token(token),
        )
        failed = service.store.complete_attempt(
            run_id=first["run_id"], lease_token_hash=hash_lease_token(token),
            result={"terminal_status": "provider_error"},
            terminal_cause="provider_error", output_ref="",
        )
        assert failed["confirmation_attempt"] == 2
    accepted_positions = set()
    while len(accepted_positions) < 20:
        token = new_lease_token()
        request_id = contracts.new_request_id()
        claim = service.store.claim_assignment(
            round_id=ROUND, runner_hotkey=runner_hotkey,
            declared_parallelism=10, slot_ceiling=20, excluded_miner_hotkeys=[],
            request_id=request_id,
            request_hash=contracts.document_hash({"request_id": request_id}),
            lease_token_hash=hash_lease_token(token),
        )
        assert claim["status"] == "leased" and claim["kind"] == "execute"
        saw_retry |= int(claim["attempt"]) == 2
        position = int(claim["icp_position"])
        version = model_version if position else 0
        _proof_execution(objects, icps[position], version, position, claim["run_id"])
        assert service.store.complete_attempt(
            run_id=claim["run_id"], lease_token_hash=hash_lease_token(token),
            result={"terminal_status": "accepted"}, terminal_cause="accepted",
            output_ref="arena/output/%s.json" % claim["run_id"],
        )["status"] == "accepted"
        accepted_positions.add(position)
    assert saw_retry is retry_execute
    assert service.close_stage(ROUND, 1)["status"] == "ok"
    score_assignments = []
    for stage in (1, 2):
        assert service.open_scoring(ROUND, stage)["assignments"] == 10
        stage_runs = [row for row in service.store.list_runs(ROUND, submission_id=BASELINE)
                      if row["kind"] == "score" and row["stage"] == stage
                      and row["assignment_id"].endswith(score_suffix)]
        assert len(stage_runs) == 10
        score_assignments.extend(row["assignment_id"] for row in stage_runs)
        for _ in range(sum(row["status"] != "accepted" for row in stage_runs)):
            token = new_lease_token()
            request_id = contracts.new_request_id()
            run = service.store.claim_assignment(
                round_id=ROUND, runner_hotkey=runner_hotkey,
                declared_parallelism=1, slot_ceiling=20, excluded_miner_hotkeys=[runner_hotkey],
                request_id=request_id,
                request_hash=contracts.document_hash({"request_id": request_id}),
                lease_token_hash=hash_lease_token(token),
            )
            assert run["status"] == "leased" and run["kind"] == "score"
            assert run["assignment_id"].endswith(score_suffix)
            position = int(run["icp_position"])
            version = model_version if position else 0
            document = scoring.build_scoring_output(
                run["scored_run_id"], [_proof_breakdown(_proof_company(icps[position], version, position), 40)]
            )
            ref = "arena/score/%s.json" % run["run_id"]
            objects.put(ref, json.dumps(document).encode())
            stored = service.store.get_run(run["run_id"])
            evidence = judgment_cache.build_evidence_snapshot(
                output=document, cache_scope=stored["judgment_scope_doc"],
                source_score_run_id=run["run_id"],
                source_scored_run_id=run["scored_run_id"],
                source_output_ref=ref, source_runner_hotkey=runner_hotkey,
                runner_authority_exclusions=run["runner_authority_exclusions"],
            )
            assert service.store.complete_attempt(
                run_id=run["run_id"], lease_token_hash=hash_lease_token(token),
                result={"terminal_status": "accepted"}, terminal_cause="accepted", output_ref=ref,
                judgment_evidence=evidence,
                judgment_evidence_hash=contracts.document_hash(evidence),
            )["status"] == "accepted"
        assert service.close_scoring(ROUND, stage)["status"] == "closed"
        assert service.score_stage(ROUND, stage)["status"] == "ok"
        if stage == 1:
            assert service.open_stage(ROUND, 2)["status"] == "ok"
            assert service.close_stage(ROUND, 2)["status"] == "ok"
    assert len(set(score_assignments)) == 20
    if publish:
        assert service.publish(ROUND)["status"] == "ok"


def test_rerun286_atomic_prepare_normal_driver_and_full_transition(
    database, tmp_path, monkeypatch,
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        snapshot = restore_284_terminal(connection)
        old_schedule = snapshot["proof"]["round"]["schedule"]
        schedule = _future_schedule(old_schedule)
        migration_285 = _render_285(captured_284_terminal(), schedule)
        with connection.cursor() as cursor:
            cursor.execute(migration_285)
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep17_baseline_recovery285_v1(%s,%s,%s,%s,%s::jsonb)",
                (SOURCE_285_SIZE, SOURCE_285_SHA, SOURCE_285_COMMIT,
                 "7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871",
                 json.dumps(schedule)),
            )
            assert cursor.fetchone()[0]["status"] == "prepared"
        connection.commit()

        harness = Harness(lambda: psycopg2.connect(**dsn), tmp_path,
                          challengers=[], runners=["rerun286-proof"])
        service, objects = harness.service, harness.objects
        harness.clock.now = datetime(2099, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        bank = json.loads(BANK_SNAPSHOT.read_text())
        assert hashlib.sha256(contracts.canonical_json(bank["icps"]).encode()).hexdigest() == (
            "7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871"
        )
        objects.put("arena/arena-2026-09-17/benchmark.json", json.dumps(bank).encode())
        _drive_cycle(
            service, objects, bank["icps"], runner_hotkey=harness.runner_keys[0],
            score_suffix=":score",
        )
        # The historical disposable fixture predates reward activation. Model
        # the official recovery285 terminal's already-activated authority.
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET reward_basis_hash=%s,"
                "reward_basis_doc=%s::jsonb,signing_key_doc=%s::jsonb,"
                "effective_reward_epoch=8247552,reward_activated_at=%s "
                "WHERE round_id=%s",
                ("sha256:" + "a" * 64, json.dumps({"sealed": True}),
                 json.dumps({"public_key_hash": "sha256:" + "b" * 64}),
                 datetime(2099, 1, 1, tzinfo=timezone.utc), ROUND),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
        connection.commit()

        round_doc, baseline, counts, hashes = _terminal_seals(connection)
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)")
            definition_hash = hashlib.sha256(cursor.fetchone()[0].encode()).hexdigest()
        sql = _render_286(round_doc, baseline, counts, hashes, schedule, definition_hash)
        # A scorer body without the current cache-source integrity check cannot
        # install the namespace patch, and the failed transaction restores it.
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
            )
            definition = cursor.fetchone()[0]
            assert definition.count("lab_arena_judgment_cache_source_invalid") == 1
            cursor.execute(definition.replace(
                "lab_arena_judgment_cache_source_invalid",
                "lab_arena_judgment_cache_source_untrusted",
            ))
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
            )
            changed = cursor.fetchone()[0]
            assert "lab_arena_judgment_cache_source_untrusted" in changed
            assert "lab_arena_judgment_cache_source_invalid" not in changed
            cursor.execute("SAVEPOINT expected_migration_failure")
            with pytest.raises(Exception, match="current integrity scoring definition differs"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK TO SAVEPOINT expected_migration_failure")
        connection.rollback()
        # The marker and assignment snippet alone cannot authorize a changed
        # function body. Preserve both and change an unrelated body comment.
        with connection.cursor() as cursor:
            cursor.execute(definition.replace('BEGIN', 'BEGIN\n  -- unrelated definition drift', 1))
            cursor.execute("SAVEPOINT expected_unrelated_drift_failure")
            with pytest.raises(Exception, match="current integrity scoring definition differs"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK TO SAVEPOINT expected_unrelated_drift_failure")
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(sql)
            cursor.execute(sql)
        connection.commit()

        # Malformed facts and a partial namespace collision both roll back.
        with pytest.raises(Exception):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep17_published_rerun286_v1(0,%s,%s,%s,%s::jsonb)",
                    (SOURCE_286_SHA, SOURCE_286_COMMIT,
                     "7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871",
                     json.dumps(schedule)),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET assignment_id=assignment_id||':rerun286' "
                "WHERE run_id=(SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s LIMIT 1)", (ROUND, BASELINE),
            )
            assert cursor.rowcount == 1
            cursor.execute("SET LOCAL session_replication_role=origin")
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND assignment_id LIKE '%%:rerun286'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 1
            cursor.execute("SAVEPOINT expected_prepare_failure")
            with pytest.raises(Exception, match="rerun286 replay differs"):
                _prepare(cursor, schedule)
            cursor.execute("ROLLBACK TO SAVEPOINT expected_prepare_failure")
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_jsonb(r)=%s::jsonb,to_jsonb(s)=%s::jsonb,"
                "r.status,r.reward_basis_hash IS NOT NULL,r.reward_basis_doc IS NOT NULL,"
                "r.signing_key_doc IS NOT NULL,r.effective_reward_epoch IS NOT NULL,"
                "r.reward_activated_at IS NOT NULL,s.source_ref,s.source_size_bytes,"
                "(SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND status IN ('pending','leased','submitted')),"
                "public.lab_arena__successful_call_cost_state(%s,'execute',NULL),"
                "public.lab_arena__successful_call_cost_state(%s,'score',NULL),"
                "public.lab_arena_sep17_recovery284_archive_valid_v1(),"
                "public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1(),"
                "public.lab_arena_sep17_rerun286_nonbaseline_valid_v1() "
                "FROM public.lab_arena_rounds r,public.lab_arena_submissions s "
                "WHERE r.round_id=%s AND s.round_id=r.round_id AND s.submission_id=%s",
                (json.dumps(round_doc), json.dumps(baseline), ROUND, BASELINE,
                 BASELINE, ROUND, BASELINE),
            )
            terminal_checks = cursor.fetchone()
            assert terminal_checks[:8] == (True, True, "published", True, True, True, True, True), terminal_checks
            assert terminal_checks[10] == 0, terminal_checks
            assert terminal_checks[13:] == (True, True, True), terminal_checks
            result = _prepare(cursor, schedule)
            assert result["status"] == "prepared"
            assert _prepare(cursor, schedule)["status"] == "existing"
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                "AND assignment_id LIKE '%%:rerun286'", (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s "
                "AND status='cancelled' AND rewards_enabled=false", (ARCHIVE,),
            )
            assert cursor.fetchone()[0] == 1
        connection.commit()

        # The insert guard rejects only a malformed target-baseline score.
        # Challenger and other-round score rows keep their normal namespace.
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE score_guard_probe(round_id text,submission_id text,"
                "kind text,assignment_id text,stage smallint,icp_position integer)"
            )
            cursor.execute(
                "CREATE TRIGGER score_guard_probe BEFORE INSERT ON score_guard_probe "
                "FOR EACH ROW EXECUTE FUNCTION "
                "public.lab_arena_sep17_rerun286_score_namespace_guard_v1()"
            )
            for other_round, submission in (
                (ROUND, "challenger-proof"),
                ("arena-2026-09-18", BASELINE),
            ):
                assignment = "%s:%s:1:0:score" % (other_round, submission)
                cursor.execute(
                    "INSERT INTO score_guard_probe VALUES(%s,%s,'score',%s,1,0)",
                    (other_round, submission, assignment),
                )
            cursor.execute("SELECT count(*) FROM score_guard_probe")
            assert cursor.fetchone()[0] == 2
        connection.rollback()
        with pytest.raises(Exception, match="exact baseline namespace"):
            with connection.cursor() as cursor:
                cursor.execute(
                    "CREATE TEMP TABLE score_guard_probe(round_id text,submission_id text,"
                    "kind text,assignment_id text,stage smallint,icp_position integer)"
                )
                cursor.execute(
                    "CREATE TRIGGER score_guard_probe BEFORE INSERT ON score_guard_probe "
                    "FOR EACH ROW EXECUTE FUNCTION "
                    "public.lab_arena_sep17_rerun286_score_namespace_guard_v1()"
                )
                cursor.execute(
                    "INSERT INTO score_guard_probe VALUES(%s,%s,'score',%s,1,0)",
                    (ROUND, BASELINE, "%s:%s:1:0:score" % (ROUND, BASELINE)),
                )
        connection.rollback()

        _drive_cycle(
            service, objects, bank["icps"],
            runner_hotkey=harness.runner_keys[0],
            score_suffix=":score:rerun286", retry_execute=True, publish=False, model_version=1,
        )

        # Zero is a valid normal outcome. The rerun guard does not turn the
        # operator's positive-score goal into a publication rule.
        for eligible in (True, False):
            with connection.cursor() as cursor:
                cursor.execute(
                    "CREATE TEMP TABLE round_guard_probe AS TABLE "
                    "public.lab_arena_rounds WITH NO DATA"
                )
                cursor.execute(
                    "INSERT INTO round_guard_probe SELECT * FROM public.lab_arena_rounds "
                    "WHERE round_id=%s", (ROUND,),
                )
                cursor.execute(
                    "UPDATE round_guard_probe SET "
                    "publication_doc=jsonb_set(jsonb_set(publication_doc,"
                    "'{final_ranking,0,final_score}','0'::jsonb,true),"
                    "'{final_ranking,0,eligible}',%s::jsonb,true) WHERE round_id=%s",
                    (json.dumps(eligible), ROUND),
                )
                cursor.execute(
                    "CREATE TRIGGER round_guard_probe BEFORE UPDATE ON round_guard_probe "
                    "FOR EACH ROW EXECUTE FUNCTION "
                    "public.lab_arena_sep17_rerun286_publication_guard_v1()"
                )
                cursor.execute(
                    "UPDATE round_guard_probe SET status='published' WHERE round_id=%s",
                    (ROUND,),
                )
            connection.rollback()
        assert service.publish(ROUND)["status"] == "ok"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,publication_doc#>>'{final_ranking,0,final_score}' "
                "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,),
            )
            status, score = cursor.fetchone()
            assert status == "published" and float(score) > 0
            cursor.execute(
                "SELECT count(*),count(output_ref),count(judgment_cache_key) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND kind='score' AND assignment_id LIKE '%%:score:rerun286'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, 20)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_judgment_cache cache "
                "JOIN public.lab_arena_runs source ON source.run_id=cache.source_score_run_id "
                "WHERE source.round_id=%s AND source.submission_id=%s "
                "AND source.kind='score' AND source.status='accepted' "
                "AND source.assignment_id LIKE '%%:score:rerun286' "
                "AND source.runner_hotkey=cache.source_runner_hotkey "
                "AND source.scored_run_id=cache.source_scored_run_id",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 19
            # Identical position zero legitimately reuses immutable evidence from
            # the archived baseline; changed outputs get new source judgments.
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs reused "
                "JOIN public.lab_arena_runs source "
                "ON source.run_id=reused.judgment_cache_source_run_id "
                "WHERE reused.round_id=%s AND reused.submission_id=%s "
                "AND reused.kind='score' AND reused.status='accepted' "
                "AND reused.assignment_id LIKE '%%:score:rerun286' "
                "AND source.round_id=%s AND source.kind='score' "
                "AND source.status='accepted'",
                (ROUND, BASELINE, ARCHIVE),
            )
            assert cursor.fetchone()[0] == 1
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute' AND attempt=2 "
                "AND assignment_id LIKE '%%:rerun286'", (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 1
            assert _prepare(cursor, schedule)["status"] == "existing"
        connection.commit()
    finally:
        connection.close()
