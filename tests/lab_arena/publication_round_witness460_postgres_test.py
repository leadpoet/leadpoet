"""Publication permits only proof-backed unfinished Arena assignments."""

from pathlib import Path
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import copy
import json

import pytest

from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge,
    _enable_verified_proxy_runtime,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_PROVIDER_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/460-lab-arena-publication-round-witness.sql"
HELPER = "public.lab_arena__publication_execution_incomplete_v1(text,text)"
OUTER = "public.lab_arena_integrity_publication_guard_v1()"
BASELINE = "public.lab_arena_publication_baseline_guard_v1()"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_PROVIDER_SERVICE_MIGRATIONS + (
        "410-lab-arena-publication-transition-timeout.sql",
        "423-lab-arena-partial-publication.sql",
        "424-lab-arena-partial-baseline-stage-transition.sql",
        "430-lab-arena-unscored-judge-infrastructure.sql",
    ))


@pytest.mark.parametrize("failed_execution", [False, True])
def test_archived_accepted_judge_blocks_exact_current_incomplete_transition(
    database, tmp_path, monkeypatch, failed_execution,
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(
        connect, tmp_path, challengers=["PartialMiner"], runners=["alpha"]
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True,
    )
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    assert fixtures._start_round(harness, day=27 + int(failed_execution), epoch=62_427) == 2
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("stage1_scored", runners=1)
    harness.service.advance_round(harness.round_id)
    harness.advance_until("scored", runners=1)
    round_row = harness.service.store.get_round(harness.round_id)
    ids = {
        bool(item["is_king"]): item["submission_id"]
        for item in round_row["participants"]
    }
    assert set(ids) == {False, True}

    captured = {}
    transition = harness.service.store.transition_round
    def capture(round_id, expected, target, patch=None):
        assert (round_id, expected, target) == (harness.round_id, "scored", "published")
        captured.update(patch or {})
        return {"status": "captured"}
    harness.service.store.transition_round = capture
    try:
        assert harness.service.publish(harness.round_id)["status"] == "captured"
    finally:
        harness.service.store.transition_round = transition
    full = captured["publication_doc"]
    assert full["king_decision"]["outcome"] == "no_king"

    publication=copy.deepcopy(full)
    submission_id=ids[False]
    ranking=next(item for item in publication['final_ranking'] if item['submission_id']==submission_id)
    ranking.update(final_score=None,cost_summary=None,eligible=False,eligibility_reason='execution_incomplete')
    with connect() as connection,connection.cursor() as cursor:

        cursor.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER')
        cursor.execute('ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER')
        cursor.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='execute' AND status='accepted' ORDER BY icp_position LIMIT 1",(harness.round_id,submission_id))
        execution_id=cursor.fetchone()[0]
        archive_id=harness.round_id+'-r452archive'
        cursor.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,participants) SELECT %s,'cancelled',configuration_doc,participants FROM public.lab_arena_rounds WHERE round_id=%s",(archive_id,harness.round_id))
        cursor.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(j)||jsonb_build_object('run_id',j.run_id||':archived448','round_id',%s,'assignment_id',j.assignment_id||':archived448'))).* FROM public.lab_arena_runs j WHERE scored_run_id=%s AND kind='score' AND status='accepted'",(archive_id,execution_id))
        cursor.execute("UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s",(execution_id,))
        cursor.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc='{}'::jsonb WHERE scored_run_id=%s AND kind='score' AND round_id=%s",(execution_id,harness.round_id))
        if failed_execution:
            cursor.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc='{\"infrastructure_incomplete\":true}'::jsonb WHERE run_id=%s",(execution_id,))
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((now()-interval '1 minute')::text)),'{schedule,stage_2_close}',to_jsonb((now()-interval '1 minute')::text)) WHERE round_id=%s",(harness.round_id,))
        cursor.execute('ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER')
        cursor.execute('ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER')
        cursor.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)',(harness.round_id,submission_id))
        assert cursor.fetchone()[0] is False
        cursor.execute('SAVEPOINT actual_transition')
        with pytest.raises(psycopg2.Error) as error:
            cursor.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",(harness.round_id,json.dumps({'publication_doc':publication,'published_at':publication['published_at']})))
        print('EXACT_TRANSITION_FAILURE',error.value.pgcode,error.value.diag.message_primary)
        cursor.execute('ROLLBACK TO SAVEPOINT actual_transition')
        cursor.execute(MIGRATION.read_text().replace("BEGIN;", "").replace("COMMIT;", ""))
        cursor.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)',(harness.round_id,submission_id))
        assert cursor.fetchone()[0] is True
        if not failed_execution:
            cursor.execute('SAVEPOINT missing_current_judge')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute("DELETE FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND scored_run_id=%s", (harness.round_id,execution_id))
            cursor.execute('SET LOCAL session_replication_role=origin')
            cursor.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)', (harness.round_id,submission_id))
            assert cursor.fetchone()[0] is False
            cursor.execute('ROLLBACK TO SAVEPOINT missing_current_judge')
            cursor.execute('SAVEPOINT malformed_deadline')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}','null'::jsonb) WHERE round_id=%s",(harness.round_id,))
            cursor.execute('SET LOCAL session_replication_role=origin')
            cursor.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)', (harness.round_id,submission_id))
            assert cursor.fetchone()[0] is False
            cursor.execute('ROLLBACK TO SAVEPOINT malformed_deadline')
        cursor.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",(harness.round_id,json.dumps({'publication_doc':publication,'published_at':publication['published_at']})))
        assert cursor.fetchone()[0]['status']=='ok'
        cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND status='accepted'",(archive_id,))
        assert cursor.fetchone()[0]>=1
        connection.rollback()


def test_legacy_current_authority_and_deadline_controls(database, tmp_path, monkeypatch):
    from tests.lab_arena import unscored_judge_infrastructure430_postgres_test as legacy
    monkeypatch.setattr(legacy, "MIGRATION", MIGRATION)
    legacy.test_exhausted_judge_proof_preserves_null_ranking_and_blocks_false_proof(
        database, tmp_path, monkeypatch
    )


def test_migration_replay_and_security(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT proowner,proacl,prosecdef,provolatile,proconfig FROM pg_proc WHERE oid=%s::regprocedure", (HELPER,))
        identity=cur.fetchone()
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())
        cur.execute("SELECT proowner,proacl,prosecdef,provolatile,proconfig,encode(extensions.digest(pg_get_functiondef(oid),'sha256'),'hex') FROM pg_proc WHERE oid=%s::regprocedure", (HELPER,))
        after=cur.fetchone()
        assert after[:-1]==identity
        assert after[-1]=='5fb513506ad7ecd78de3009bce3d004cbc8e9883af163fafb1934b7cebe904d7'


def test_migration_refuses_unknown_definition(database):
    psycopg, dsn = database
    conn=psycopg.connect(**dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (HELPER,))
            definition=cur.fetchone()[0]
            altered=definition.replace("RETURN v_missing > 0;", "RETURN v_missing > 0 OR FALSE;")
            assert altered!=definition
            cur.execute(altered)
            cur.execute('SAVEPOINT unknown_preimage')
            with pytest.raises(psycopg.Error,match='preimage differs'):
                cur.execute(MIGRATION.read_text().replace('BEGIN;','').replace('COMMIT;',''))
            cur.execute('ROLLBACK TO SAVEPOINT unknown_preimage')
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)",(HELPER,))
            assert cur.fetchone()[0]==altered
    finally:
        conn.rollback()
        conn.close()
