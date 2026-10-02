"""Disposable PostgreSQL proof of the corrected October 1 score rejudge."""

import json
from pathlib import Path

import pytest

from lab_arena import contact_policy, provider_observations
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena import oct01_corrected_scorer_hold378_postgres_test as held


MIGRATION = Path(__file__).parents[2] / "scripts/379-arena-2026-10-01-corrected-scorer-rejudge.sql"
database = held.database
ROUND = held.ROUND


def _prepare(cursor):
    hold_sql = held._prepare(cursor)
    cursor.execute('SET session_replication_role=replica')
    cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2',"
                   "status_generation=16,stage_generation=13 WHERE round_id=%s", (ROUND,))
    cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                   "terminal_cause='accepted',output_ref='arena/test/score379.json' "
                   "WHERE round_id=%s AND kind='score' AND status<>'accepted'", (ROUND,))
    cursor.execute("UPDATE public.lab_arena_runs SET per_icp_score=0,"
                   "qualification_doc='{\"companies\":[]}'::jsonb "
                   "WHERE round_id=%s AND stage=1 AND kind='execute'", (ROUND,))
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
        "icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref) "
        "SELECT 'miner379:'||s.submission_id||':'||p.n,"
        "s.round_id||':'||s.submission_id||':2:'||p.n,s.round_id,"
        "s.submission_id,s.miner_hotkey,2,p.n,1,'execute',"
        "CASE WHEN p.n=0 AND s.submission_id=(SELECT min(submission_id) "
        "FROM public.lab_arena_submissions WHERE round_id=%s AND NOT is_king) "
        "THEN 'accepted' ELSE 'pending' END,13,"
        "CASE WHEN p.n=0 AND s.submission_id=(SELECT min(submission_id) "
        "FROM public.lab_arena_submissions WHERE round_id=%s AND NOT is_king) "
        "THEN 'accepted' ELSE NULL END,"
        "CASE WHEN p.n=0 AND s.submission_id=(SELECT min(submission_id) "
        "FROM public.lab_arena_submissions WHERE round_id=%s AND NOT is_king) "
        "THEN 'arena/test/miner379.json' ELSE NULL END "
        "FROM public.lab_arena_submissions s CROSS JOIN generate_series(0,9) p(n) "
        "WHERE s.round_id=%s AND NOT s.is_king",
        (ROUND, ROUND, ROUND, ROUND),
    )
    assert cursor.rowcount == 130
    cursor.execute("SET session_replication_role=origin")
    cursor.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
                   "AND stage=2 AND status='accepted'", (ROUND,))
    accepted_miner = cursor.fetchone()[0]
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd) "
        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,2,"
        "'sha256:'||repeat('d',64),'deepline','test_source','miner_key',100 "
        "FROM public.lab_arena_runs WHERE run_id=%s", (accepted_miner,)
    )
    cursor.execute(hold_sql)
    cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                   "SET actor_ref='canonical-active-release:d9b5b669a849b071d19f6002514273c1925e91d3' "
                   "WHERE singleton AND operator_paused AND "
                   "pause_reason='oct01_corrected_scorer_rejudge'")
    assert cursor.rowcount == 1
    sql = MIGRATION.read_text()
    hashes = {
        '6b7b62b7f2f4c8a332398f12941305acec1de5cb311c6c7eeb0b574133c2243f':
            held._hash(cursor, "(SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        '872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19':
            held._hash(cursor, "(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        '8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61':
            held._hash(cursor, "(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930)"),
        '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3':
            held._hash(cursor, "(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-01')"),
        '24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa':
            held._hash(cursor, "(SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'submission_id',submission_id,'stage',stage,'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,'output_ref',output_ref,'stage_generation',stage_generation) ORDER BY icp_position) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='execute')"),
    }
    for expected, fixture in hashes.items():
        assert expected in sql
        sql = sql.replace(expected, fixture)
    for function, expected in (
        ('public.lab_arena_open_scoring_v2(text,smallint,jsonb)',
         'de99ad96d87254e59aa4b431d66d3d944e430be15f1562ab6f732dc73672d6ba'),
        ('public.lab_arena_open_stage(text,smallint,jsonb,integer[])',
         '52fd6b196e561c6dabb27a3b551d98538cfe5f631b1ec1a01a838edd25074eb2'),
    ):
        cursor.execute("SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),"
                       "'sha256'),'hex')", (function,))
        sql = sql.replace(expected, cursor.fetchone()[0])
    return sql, accepted_miner


def test_rejudge_preserves_miner_execution_and_resumes_without_duplicate(database, monkeypatch):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, accepted_miner = _prepare(cursor)
            cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                           "WHERE round_id=%s AND stage=2 ORDER BY run_id", (ROUND,))
            miner_before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l "
                           "WHERE run_id=%s", (accepted_miner,))
            source_cost_before = cursor.fetchall()
            cursor.execute(sql)
            cursor.execute("SELECT status,status_generation,stage_generation,"
                           "configuration_doc->>'scorer_image_digest' "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone() == (
                'stage1_closed', 17, 14,
                'sha256:de2b69fba175c716d2e69b8cadec71cd42ab301ce34425b6dfd5a7df9f2427ea')
            cursor.execute("SELECT operator_paused FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone()[0] is False
            cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                           "WHERE round_id=%s AND stage=2 ORDER BY run_id", (ROUND,))
            assert cursor.fetchall() == miner_before
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l "
                           "WHERE run_id=%s", (accepted_miner,))
            assert cursor.fetchall() == source_cost_before
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                           (ROUND+'-r379archive',))
            assert cursor.fetchone()[0] == 10
            cursor.execute("SELECT jsonb_array_length(configuration_doc->'recovery_baseline_receipts') "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND+'-r379archive',))
            assert cursor.fetchone()[0] == 10
            cursor.execute(sql)  # Replay is read-only even after hold release.
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    service = object.__new__(ArenaService)
    service._store = store
    service._round = lambda rid: store.get_round(rid)
    service._load_scoring_plan = lambda row, stage: row['stage1_scoring_plan_doc']
    service._require_code_review = lambda *args: None
    service.evaluation_icps = lambda rid: daily_icps()[:10]
    class Objects:
        @staticmethod
        def get_bounded(ref, max_bytes):
            config = store.get_round(ROUND)['configuration_doc']
            return json.dumps({'schema_version': contact_policy.output_schema(config),
                               'companies': []}).encode()
    service._objects = Objects()
    monkeypatch.setattr(provider_observations, 'resolve_observations', lambda *a, **kw: [])
    opened_score = service.open_scoring(ROUND, 1)
    assert opened_score['status'] == 'ok' and opened_score['assignments'] == 10
    new_scores = store.list_runs(ROUND, stage=1, kind='score')
    assert len(new_scores) == 10
    assert all(row['assignment_id'].endswith(':score:rerun379') for row in new_scores)
    assert all(row['judgment_scope_doc']['scorer_image_digest'] ==
               'sha256:de2b69fba175c716d2e69b8cadec71cd42ab301ce34425b6dfd5a7df9f2427ea'
               for row in new_scores)
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/newscore379.json',"
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
    assert all(row['per_icp_score'] == 0 for row in
               store.list_runs(ROUND, stage=1, kind='execute'))
    participants = [p for p in store.get_round(ROUND)['participants'] if not p['is_king']]
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            with pytest.raises(psycopg.Error, match='lab_arena_stage_position_invalid'):
                cursor.execute("SELECT public.lab_arena_open_stage(%s,2::smallint,%s::jsonb,"
                               "ARRAY[1,2,3,4,5,6,7,8,9,10]::integer[])",
                               (ROUND, json.dumps(participants)))
            cursor.execute('ROLLBACK')
            wrong = [dict(p) for p in participants]
            wrong[0]['miner_hotkey'] = wrong[1]['miner_hotkey']
            with pytest.raises(psycopg.Error, match='preserved miner execution differs'):
                cursor.execute("SELECT public.lab_arena_open_stage(%s,2::smallint,%s::jsonb,"
                               "ARRAY[0,1,2,3,4,5,6,7,8,9]::integer[])",
                               (ROUND, json.dumps(wrong)))
            cursor.execute('ROLLBACK')
    result = store.open_stage(ROUND, 2, participants, list(range(10)))
    assert result['status'] == 'ok' and result['resumed'] is True
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(DISTINCT assignment_id),count(*) "
                           "FROM public.lab_arena_runs WHERE round_id=%s AND stage=2", (ROUND,))
            assert cursor.fetchone() == (130, 130)
            cursor.execute("SELECT status,stage_generation FROM public.lab_arena_runs "
                           "WHERE run_id=%s", (accepted_miner,))
            assert cursor.fetchone() == ('accepted', 13)
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                           "AND stage=2 AND status='pending' AND stage_generation="
                           "(SELECT stage_generation FROM public.lab_arena_rounds WHERE round_id=%s)",
                           (ROUND, ROUND))
            assert cursor.fetchone()[0] == 129


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_runs SET status='leased' WHERE round_id='arena-2026-10-01' AND stage=2 AND status='pending' AND icp_position=1",
    "UPDATE public.lab_arena_runs SET assignment_id='wrong' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND stage=2 AND status='pending' AND icp_position=1)",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"wrong\"'::jsonb) WHERE round_id='arena-2026-10-01'",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='foreign' WHERE singleton",
])
def test_rejudge_rejects_lease_assignment_or_image_mismatch(database, mutation):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, _ = _prepare(cursor)
            cursor.execute('BEGIN')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute(mutation)
            cursor.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error):
                cursor.execute(sql)
            cursor.execute('ROLLBACK')
            cursor.execute("SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s",
                           (ROUND+'-r379archive',))
            assert cursor.fetchone()[0] == 0
