"""Disposable PostgreSQL proof for the full October 1 source rejudge."""

import json
from pathlib import Path

import pytest

from lab_arena import contact_policy, provider_observations
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena import oct01_hq_source_hold386_postgres_test as held
from tests.lab_arena import oct01_corrected_scorer_hold378_postgres_test as hashes


database = held.database
ROUND = held.ROUND
ARCHIVE = ROUND + '-r387archive'
MIGRATION = Path(__file__).parents[2] / 'scripts/387-arena-2026-10-01-hq-source-rejudge.sql'
TEST_DIGEST = 'sha256:' + 'a' * 64


def _prepare(cursor):
    cursor.execute(held._prepare(cursor))
    cursor.execute("UPDATE public.lab_arena_restart_claim_control SET "
                   "actor_ref='canonical-active-release:1caf74c9ee0bcd6a466976df4e00e0a16e1a6e42',"
                   "guard_generation=313 WHERE singleton")
    assert cursor.rowcount == 1
    cursor.execute('SET session_replication_role=replica')
    cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',"
                   "output_ref='arena/test/score387-old.json',lease_expires_at=NULL "
                   "WHERE round_id=%s AND kind='score' AND status='leased'", (ROUND,))
    assert cursor.rowcount == 13
    cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',"
                   "output_ref='arena/test/score387-old.json' WHERE run_id IN ("
                   "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
                   "AND kind='score' AND status='pending' ORDER BY run_id LIMIT 17)", (ROUND,))
    assert cursor.rowcount == 17
    cursor.execute("UPDATE public.lab_arena_runs SET qualification_doc='{"
                   "\"companies\":[]}'::jsonb,per_icp_score=54 "
                   "WHERE round_id=%s AND kind='execute' AND stage=1 "
                   "AND status='accepted'", (ROUND,))
    assert cursor.rowcount == 10
    cursor.execute("SELECT count(*) FILTER (WHERE stage=1 AND qualification_doc IS NOT NULL "
                   "AND per_icp_score IS NOT NULL),count(*) FILTER (WHERE stage=2 "
                   "AND qualification_doc IS NULL AND per_icp_score IS NULL) "
                   "FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' "
                   "AND status='accepted'", (ROUND,))
    assert cursor.fetchone() == (10, 130)
    cursor.execute("SELECT count(*) FILTER (WHERE stage=1 AND status='accepted'),"
                   "count(*) FILTER (WHERE stage=2 AND status='accepted'),"
                   "count(*) FILTER (WHERE stage=2 AND status='pending') "
                   "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
    assert cursor.fetchone() == (10, 121, 9)
    cursor.execute("UPDATE public.lab_arena_rounds SET "
                   "stage1_scoring_plan_doc='{}'::jsonb,stage2_scoring_plan_doc='{}'::jsonb,"
                   "finalists='[]'::jsonb WHERE round_id=%s", (ROUND,))
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
        "icp_position,attempt,kind,status,stage_generation,terminal_cause,result_doc) "
        "SELECT 'failed387:audit',assignment_id,round_id,submission_id,miner_hotkey,"
        "stage,icp_position,2,'execute','failed',stage_generation,'provider_error',"
        "'{\"terminal_status\":\"failed\"}'::jsonb FROM public.lab_arena_runs "
        "WHERE round_id=%s AND stage=2 AND kind='execute' AND attempt=1 "
        "ORDER BY run_id LIMIT 1", (ROUND,))
    assert cursor.rowcount == 1
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd) "
        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
        "'sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),"
        "CASE WHEN mod(icp_position,3)=0 THEN 'openrouter' WHEN mod(icp_position,3)=1 "
        "THEN 'deepline' ELSE 'scrapingdog' END,'test_score387','host',100 "
        "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
    assert cursor.rowcount == 140
    cursor.execute(
        "INSERT INTO public.lab_arena_trajectory_events "
        "(run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,"
        "assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,"
        "model_role,event_kind,occurred_at,content) "
        "SELECT run_id,md5(run_id)::uuid,round_id,submission_id,miner_hotkey,"
        "miner_hotkey,assignment_id,'oct01-0',stage,icp_position,attempt,'score',"
        "'baseline','provider.response','2026-10-02T18:00:00Z',"
        "'{\"operation_id\":\"test_score387\"}'::jsonb "
        "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
    assert cursor.rowcount == 140
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd) "
        "SELECT 'uncertain',miner_hotkey,round_id,submission_id,run_id,stage,"
        "'sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),"
        "'openrouter','failed_execute387','host',17 "
        "FROM public.lab_arena_runs WHERE run_id='failed387:audit'")
    assert cursor.rowcount == 1
    cursor.execute(
        "INSERT INTO public.lab_arena_trajectory_events "
        "(run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,"
        "assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,"
        "model_role,event_kind,occurred_at,content) "
        "SELECT run_id,md5(run_id)::uuid,round_id,submission_id,miner_hotkey,"
        "miner_hotkey,assignment_id,'oct01-0',stage,icp_position,attempt,'execute',"
        "'miner','provider.response','2026-10-02T18:00:00Z',"
        "'{\"operation_id\":\"failed_execute387\"}'::jsonb "
        "FROM public.lab_arena_runs WHERE run_id='failed387:audit'")
    assert cursor.rowcount == 1
    cursor.execute('SET session_replication_role=origin')
    sql = MIGRATION.read_text()
    for expected, expression in (
        ('872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19',
         "(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        ('8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61',
         '(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930)'),
        ('1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3',
         "(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-01')"),
        ('24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa',
         "(SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'submission_id',submission_id,'stage',stage,'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,'output_ref',output_ref,'stage_generation',stage_generation) ORDER BY icp_position) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='execute')"),
        ('cfce5676b6c9f1ec45a3b63d54070626f1dcfd1f9782d50cad714c9f114c5b0d',
         "pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"),
        ('52ce5ab8a99bbdad55d7d8bd13227d2e2f5bc6fffc69b2b4d5b8fbf40da0d58c',
         "pg_get_functiondef('public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::regprocedure)"),
    ):
        assert expected in sql
        sql = sql.replace(expected, hashes._hash(cursor, expression))
    return sql.replace('sha256:4256d5790540ace6f739ab31135b855f3ce76eece8809943b5eca68deff7e787', TEST_DIGEST)


def _rows(cursor, kind):
    cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                   "WHERE round_id=%s AND kind=%s ORDER BY run_id", (ROUND, kind))
    return [row[0] for row in cursor.fetchall()]


def test_full_archive_preserves_executions_costs_and_original_credential_ids(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            original = _rows(cursor, 'execute')
            assert len(original) == 141
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l "
                           "WHERE round_id=%s ORDER BY entry_id", (ROUND,))
            original_ledger = [row[0] for row in cursor.fetchall()]
            cursor.execute("SELECT to_jsonb(e) FROM public.lab_arena_trajectory_events e "
                           "WHERE round_id=%s ORDER BY trajectory_id", (ROUND,))
            original_events = [row[0] for row in cursor.fetchall()]
            cursor.execute(sql)
            cursor.execute("SELECT status,status_generation,stage_generation,"
                           "stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,"
                           "configuration_doc->>'scorer_image_digest' "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone() == ('stage1_closed', 35, 29, None, None, None, TEST_DIGEST)
            cursor.execute("SELECT configuration_doc->'schedule',evaluation_date,icp_set_date "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            schedule, evaluation, icp_set = cursor.fetchone()
            assert schedule['stage_1_scoring_close'] == '2026-10-02T19:00:01Z'
            assert schedule['stage_2_start'] == '2026-10-02T19:00:02Z'
            assert schedule['stage_2_close'] == '2026-10-03T02:00:02Z'
            assert schedule['final_scoring_close'] == '2026-10-03T08:00:02Z'
            assert (str(evaluation), str(icp_set)) == ('2026-10-01', '2026-09-30')
            current = _rows(cursor, 'execute')
            assert len(current) == 141
            for before, after in zip(original, current):
                assert {k: v for k, v in before.items() if k not in ('qualification_doc', 'per_icp_score')} == {
                    k: v for k, v in after.items() if k not in ('qualification_doc', 'per_icp_score')}
                if before['status'] == 'accepted':
                    assert after['qualification_doc'] is None and after['per_icp_score'] is None
                else:
                    assert after == before
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT count(*),count(*) FILTER(WHERE status='pending') "
                           "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ARCHIVE,))
            assert cursor.fetchone() == (140, 9)
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s", (ARCHIVE,))
            assert cursor.fetchone()[0] == 140
            cursor.execute("SELECT count(*) FROM public.lab_arena_trajectory_events WHERE round_id=%s", (ARCHIVE,))
            assert cursor.fetchone()[0] == 140
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l "
                           "WHERE round_id IN (%s,%s) ORDER BY entry_id", (ROUND, ARCHIVE))
            after_ledger = [row[0] for row in cursor.fetchall()]
            assert [({**row, 'round_id': ROUND,
                      'submission_id': 'baseline-2026-10-01' if row['submission_id'] == 'baseline-2026-10-01-r387archive' else row['submission_id']})
                    for row in after_ledger] == original_ledger
            cursor.execute("SELECT to_jsonb(e) FROM public.lab_arena_trajectory_events e "
                           "WHERE round_id IN (%s,%s) ORDER BY trajectory_id", (ROUND, ARCHIVE))
            after_events = [row[0] for row in cursor.fetchall()]
            assert [({**row, 'round_id': ROUND,
                      'submission_id': 'baseline-2026-10-01' if row['submission_id'] == 'baseline-2026-10-01-r387archive' else row['submission_id']})
                    for row in after_events] == original_events
            cursor.execute("SELECT count(*) FROM public.lab_arena_submissions WHERE round_id=%s", (ARCHIVE,))
            assert cursor.fetchone()[0] == 1
            cursor.execute("SELECT run_id,submission_id,miner_hotkey FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND kind='score' AND stage=2 LIMIT 1", (ARCHIVE,))
            miner_run_id, miner_submission_id, miner_hotkey = cursor.fetchone()
            for provider in ('openrouter', 'deepline', 'scrapingdog'):
                cursor.execute("SELECT public.lab_arena_provider_funding(%s,%s)", (miner_run_id, provider))
                funding = cursor.fetchone()[0]
                assert funding['funding_source'] == 'miner_key'
                assert funding['credential_submission_id'] == miner_submission_id
                assert funding['credential_miner_hotkey'] == miner_hotkey
            cursor.execute("SELECT run_id FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND kind='score' AND stage=1 LIMIT 1", (ARCHIVE,))
            baseline_run_id = cursor.fetchone()[0]
            for provider in ('openrouter', 'deepline', 'scrapingdog'):
                cursor.execute("SELECT public.lab_arena_provider_funding(%s,%s)", (baseline_run_id, provider))
                funding = cursor.fetchone()[0]
                assert funding['funding_source'] == 'miner_key'
                cursor.execute("SELECT champion_submission_id,champion_hotkey "
                               "FROM public.lab_arena_rounds WHERE round_id=%s", (ARCHIVE,))
                assert (funding['credential_submission_id'],
                        funding['credential_miner_hotkey']) == cursor.fetchone()
            cursor.execute(sql)  # replay is read-only after release


def test_rerun_namespace_rejudges_and_resumes_130_accepted_miner_outputs(database, monkeypatch):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute(_prepare(cursor))
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    service = object.__new__(ArenaService)
    service._store = store
    service._round = lambda rid: store.get_round(rid)
    service._load_scoring_plan = lambda row, stage: row['stage1_scoring_plan_doc']
    service._require_code_review = lambda *args: None
    service.benchmark_icps = lambda rid: daily_icps()[:10]
    service.evaluation_icps = lambda rid: daily_icps()[:10]

    class Objects:
        @staticmethod
        def get_bounded(ref, max_bytes):
            config = store.get_round(ROUND)['configuration_doc']
            return json.dumps({'schema_version': contact_policy.output_schema(config),
                               'companies': []}).encode()

    service._objects = Objects()
    monkeypatch.setattr(provider_observations, 'resolve_observations', lambda *a, **kw: [])
    assert service.commit_scoring_plan(ROUND, 1)['status'] == 'ok'
    opened = service.open_scoring(ROUND, 1)
    assert opened['status'] == 'ok' and opened['assignments'] == 10
    new_scores = store.list_runs(ROUND, stage=1, kind='score')
    assert len(new_scores) == 10
    assert all(r['assignment_id'].endswith(':score:rerun387') for r in new_scores)
    assert all(r['judgment_scope_doc']['scorer_image_digest'] == TEST_DIGEST
               for r in new_scores)
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/newscore387.json',"
                           "result_doc='{\"terminal_status\":\"accepted\"}'::jsonb "
                           "WHERE round_id=%s AND stage=1 AND kind='score' AND status='pending'", (ROUND,))
            assert cursor.rowcount == 10
            cursor.execute('SET session_replication_role=origin')
    assert store.close_scoring(ROUND, 1)['status'] == 'closed'
    plan = store.get_round(ROUND)['stage1_scoring_plan_doc']
    service._outputs_by_run = lambda rid, stage: {
        item['scored_run_id']: [] for item in plan['work_items']}
    service._verified_breakdowns = lambda run, **kwargs: []
    assert service.score_stage(ROUND, 1)['status'] == 'ok'
    participants = [p for p in store.get_round(ROUND)['participants'] if not p['is_king']]
    resumed = store.open_stage(ROUND, 2, participants, list(range(10)))
    assert resumed['status'] == 'ok' and resumed['resumed'] is True
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*),count(*) FILTER(WHERE status='accepted') "
                           "FROM public.lab_arena_runs WHERE round_id=%s AND stage=2 "
                           "AND kind='execute'", (ROUND,))
            assert cursor.fetchone() == (131, 130)


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='foreign' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='oct01-hq-source-hold386' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_generation=312 WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),"
    "owner_commitment='sha256:'||repeat('b',64),guard_generation=1,"
    "guard_expires_at=now()+interval '1 hour',candidate_commit=repeat('c',40),"
    "restart_scope='all',restart_phase='draining' WHERE singleton",
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-01'",
    "UPDATE public.lab_arena_runs SET status='leased' WHERE round_id='arena-2026-10-01' AND kind='score' AND status='pending' AND icp_position=9",
    "UPDATE public.lab_arena_runs SET output_ref=NULL WHERE round_id='arena-2026-10-01' AND kind='execute' AND stage=2 AND icp_position=0",
    "UPDATE public.lab_arena_ledger SET entry_kind='dispatch' WHERE entry_id="
    "(SELECT MIN(entry_id) FROM public.lab_arena_ledger WHERE round_id='arena-2026-10-01' "
    "AND stage=2 AND entry_kind='settlement')",
    "UPDATE public.lab_arena_rounds SET published_at=now() WHERE round_id='arena-2026-10-01'",
])
def test_wrong_state_fails_without_archive_or_hold_release(database, mutation):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute('BEGIN')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute(mutation)
            cursor.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error):
                cursor.execute(sql)
            cursor.execute('ROLLBACK')
            cursor.execute("SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s", (ARCHIVE,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT actor_ref FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone()[0] == 'canonical-active-release:1caf74c9ee0bcd6a466976df4e00e0a16e1a6e42'


def test_replay_rejects_archived_and_retained_payload_drift(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute(sql)
            mutations = (
                "UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
                "WHERE run_id=(SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' ORDER BY run_id LIMIT 1)",
                "UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd+1 "
                "WHERE entry_id=(SELECT MIN(entry_id) FROM public.lab_arena_ledger WHERE round_id=%s)",
                "UPDATE public.lab_arena_trajectory_events SET content='{}'::jsonb "
                "WHERE trajectory_id=(SELECT MIN(trajectory_id) FROM public.lab_arena_trajectory_events WHERE round_id=%s)",
                "UPDATE public.lab_arena_runs SET output_ref='arena/tampered.json' "
                "WHERE run_id=(SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' AND status='accepted' ORDER BY run_id LIMIT 1)",
                "UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
                "WHERE run_id='failed387:audit' AND round_id=%s",
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,"
                "'{recovery_source_receipts_sha256}','\"wrong\"'::jsonb) WHERE round_id=%s",
            )
            for index, mutation in enumerate(mutations):
                cursor.execute('BEGIN')
                cursor.execute('SET LOCAL session_replication_role=replica')
                cursor.execute(mutation, (ARCHIVE if index in (0, 1, 2, 5) else ROUND,))
                assert cursor.rowcount == 1
                cursor.execute('SET LOCAL session_replication_role=origin')
                with pytest.raises(psycopg.Error):
                    cursor.execute(sql)
                cursor.execute('ROLLBACK')
                cursor.execute(sql)
