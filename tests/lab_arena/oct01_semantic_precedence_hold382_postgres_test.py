"""Disposable PostgreSQL proof for the October 1 semantic precedence hold."""

from pathlib import Path

import pytest

from tests.lab_arena import oct01_attribute_evidence_rejudge381_postgres_test as prior
from tests.lab_arena import oct01_corrected_scorer_hold378_postgres_test as held


HOLD = Path(__file__).parents[2] / 'scripts/382-arena-2026-10-01-semantic-precedence-hold.sql'
database = prior.database
ROUND = prior.ROUND


def _prepare(cursor):
    rejudge, _ = prior._prepare(cursor)
    cursor.execute(rejudge)
    cursor.execute('SET session_replication_role=replica')
    cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2',"
                   "status_generation=23,stage_generation=19 WHERE round_id=%s", (ROUND,))
    cursor.execute("UPDATE public.lab_arena_runs SET per_icp_score=0,"
                   "qualification_doc='{\"companies\":[]}'::jsonb "
                   "WHERE round_id=%s AND stage=1 AND kind='execute'", (ROUND,))
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
        "icp_position,attempt,kind,status,stage_generation,scored_run_id,"
        "terminal_cause,output_ref) "
        "SELECT 'score382:'||e.icp_position||':1',"
        "e.round_id||':'||e.submission_id||':1:'||e.icp_position||':score:rerun381',"
        "e.round_id,e.submission_id,e.miner_hotkey,1,e.icp_position,1,'score',"
        "'accepted',19,e.run_id,'accepted','arena/test/score382.json' "
        "FROM public.lab_arena_runs e WHERE e.round_id=%s AND e.stage=1 AND e.kind='execute'",
        (ROUND,),
    )
    assert cursor.rowcount == 10
    cursor.execute("UPDATE public.lab_arena_runs SET status='leased',"
                   "lease_expires_at=now()+interval '1 hour' "
                   "WHERE round_id=%s AND stage=2 AND kind='execute' "
                   "AND status='pending' AND icp_position=1", (ROUND,))
    assert cursor.rowcount == 13
    cursor.execute('SET session_replication_role=origin')
    sql = HOLD.read_text()
    for expected, expr in (
        ('872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19',
         "(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        ('8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61',
         '(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930)'),
        ('1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3',
         "(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-01')"),
        ('24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa',
         "(SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'submission_id',submission_id,'stage',stage,'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,'output_ref',output_ref,'stage_generation',stage_generation) ORDER BY icp_position) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='execute')"),
    ):
        assert expected in sql
        sql = sql.replace(expected, held._hash(cursor, expr))
    return sql


def test_hold_preserves_runs_and_costs_allows_active_completion_and_replay(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute("SELECT encode(extensions.digest(jsonb_agg(to_jsonb(r) ORDER BY run_id)::text,"
                           "'sha256'),'hex') FROM public.lab_arena_runs r WHERE round_id=%s", (ROUND,))
            runs_before = cursor.fetchone()[0]
            cursor.execute("SELECT encode(extensions.digest(jsonb_agg(to_jsonb(l) ORDER BY entry_id)::text,"
                           "'sha256'),'hex') FROM public.lab_arena_ledger l WHERE round_id=%s", (ROUND,))
            ledger_before = cursor.fetchone()[0]
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            held_state = cursor.fetchone()
            assert held_state[:3] == (True, 'oct01_semantic_precedence_review',
                                      'oct01-semantic-precedence-hold382')
            cursor.execute("SELECT encode(extensions.digest(jsonb_agg(to_jsonb(r) ORDER BY run_id)::text,"
                           "'sha256'),'hex') FROM public.lab_arena_runs r WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone()[0] == runs_before
            cursor.execute("SELECT encode(extensions.digest(jsonb_agg(to_jsonb(l) ORDER BY entry_id)::text,"
                           "'sha256'),'hex') FROM public.lab_arena_ledger l WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone()[0] == ledger_before
            with pytest.raises(psycopg.Error, match='lab_arena_claims_paused'):
                cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                               "WHERE round_id=%s AND stage=2 AND icp_position=2 AND status='pending'", (ROUND,))
            cursor.execute('ROLLBACK')
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/new382.json' "
                           "WHERE round_id=%s AND stage=2 AND icp_position=1 AND status='leased'", (ROUND,))
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone() == held_state


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='foreign',actor_ref='foreign' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),owner_commitment='sha256:'||repeat('b',64),guard_generation=1,guard_expires_at=now()+interval '1 hour',candidate_commit=repeat('c',40),restart_scope='all',restart_phase='draining' WHERE singleton",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"wrong\"'::jsonb) WHERE round_id='arena-2026-10-01'",
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-02'",
    "UPDATE public.lab_arena_runs SET assignment_id='wrong' WHERE run_id='score382:0:1'",
])
def test_hold_rejects_foreign_authority_or_frozen_change(database, mutation):
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
            cursor.execute("SELECT operator_paused FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone()[0] is False
