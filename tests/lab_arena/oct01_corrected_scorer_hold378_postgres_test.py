"""Disposable PostgreSQL proof of the October 1 corrected-scorer hold."""

from pathlib import Path

import pytest

from tests.lab_arena import oct01_new_scorer_rejudge376_postgres_test as prior


HOLD = Path(__file__).parents[2] / "scripts/378-arena-2026-10-01-corrected-scorer-hold.sql"
database = prior.database
ROUND = "arena-2026-10-01"


def _hash(cursor, expression):
    cursor.execute("SELECT encode(extensions.digest((" + expression + ")::text,'sha256'),'hex')")
    return cursor.fetchone()[0]


def _prepare(cursor):
    rejudge, _ = prior._prepare(cursor)
    cursor.execute(rejudge)
    cursor.execute("SET session_replication_role=replica")
    cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1_scoring',"
                   "status_generation=13,stage_generation=11 WHERE round_id=%s", (ROUND,))
    cursor.execute("UPDATE public.lab_arena_restart_claim_control SET "
                   "operator_paused=false,pause_reason='',actor_ref='' WHERE singleton")
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
        "icp_position,attempt,kind,status,stage_generation,scored_run_id,"
        "terminal_cause,output_ref,result_doc) "
        "SELECT 'score378:'||e.icp_position||':1',"
        "e.round_id||':'||e.submission_id||':1:'||e.icp_position||':score:rerun376',"
        "e.round_id,e.submission_id,e.miner_hotkey,1,e.icp_position,1,'score',"
        "CASE WHEN e.icp_position<7 THEN 'accepted' "
        "WHEN e.icp_position=7 THEN 'leased' ELSE 'pending' END,11,e.run_id,"
        "CASE WHEN e.icp_position<7 THEN 'accepted' ELSE NULL END,"
        "CASE WHEN e.icp_position<7 THEN 'arena/test/score378.json' ELSE NULL END,"
        "CASE WHEN e.icp_position<7 THEN "
        "'{\"terminal_status\":\"accepted\"}'::jsonb ELSE NULL END "
        "FROM public.lab_arena_runs e WHERE e.round_id=%s AND e.kind='execute'",
        (ROUND,),
    )
    assert cursor.rowcount == 10
    cursor.execute("SET session_replication_role=origin")
    sql = HOLD.read_text()
    hashes = {
        "6b7b62b7f2f4c8a332398f12941305acec1de5cb311c6c7eeb0b574133c2243f":
            _hash(cursor, "(SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        "872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19":
            _hash(cursor, "(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        "8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61":
            _hash(cursor, "(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930)"),
        "1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3":
            _hash(cursor, "(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-01')"),
        "24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa":
            _hash(cursor, "(SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'submission_id',submission_id,'stage',stage,'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,'output_ref',output_ref,'stage_generation',stage_generation) ORDER BY icp_position) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND kind='execute')"),
    }
    for expected, fixture in hashes.items():
        assert expected in sql
        sql = sql.replace(expected, fixture)
    return sql


def test_hold_stops_claim_and_progression_but_allows_completion_and_replay(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            held = cursor.fetchone()
            assert held[:3] == (True, 'oct01_corrected_scorer_rejudge',
                                'oct01-corrected-scorer-hold378')
            with pytest.raises(psycopg.Error, match='lab_arena_claims_paused'):
                cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                               "WHERE run_id='score378:8:1'")
            cursor.execute('ROLLBACK')
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/score378:7.json' "
                           "WHERE run_id='score378:7:1'")
            with pytest.raises(psycopg.Error, match='lab_arena_round_progression_paused'):
                cursor.execute("SELECT public.lab_arena_close_scoring(%s,1::smallint)", (ROUND,))
            cursor.execute('ROLLBACK')
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone() == held
            cursor.execute("SELECT status FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone()[0] == 'stage1_scoring'


@pytest.mark.parametrize('state,generation,stage_generation,derived', [
    ('stage1_judged', 14, 11, False),
    ('stage1_scored', 15, 12, True),
    ('stage2', 16, 13, True),
])
def test_hold_accepts_progression_before_miner_scoring(
        database, state, generation, stage_generation, derived):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_rounds SET status=%s,"
                           "status_generation=%s,stage_generation=%s WHERE round_id=%s",
                           (state, generation, stage_generation, ROUND))
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/score378.json' "
                           "WHERE round_id=%s AND kind='score' AND status<>'accepted'", (ROUND,))
            if derived:
                cursor.execute("UPDATE public.lab_arena_runs SET per_icp_score=0,"
                               "qualification_doc='{\"companies\":[]}'::jsonb "
                               "WHERE round_id=%s AND stage=1 AND kind='execute'", (ROUND,))
            if state == 'stage2':
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs "
                    "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                    "stage,icp_position,attempt,kind,status,stage_generation) "
                    "SELECT 'miner378:'||s.submission_id||':'||p.n,"
                    "'miner378:'||s.submission_id||':'||p.n,s.round_id,"
                    "s.submission_id,s.miner_hotkey,2,p.n,1,'execute','pending',13 "
                    "FROM public.lab_arena_submissions s CROSS JOIN generate_series(0,9) p(n) "
                    "WHERE s.round_id=%s AND s.submission_id<>'baseline-2026-10-01'",
                    (ROUND,),
                )
                assert cursor.rowcount == 130
            cursor.execute('SET session_replication_role=origin')
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone()[0] is True
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND stage=2", (ROUND,))
            assert cursor.fetchone()[0] == (130 if state == 'stage2' else 0)


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_rounds SET status='stage1_judged' WHERE round_id='arena-2026-10-01'",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"changed\"'::jsonb) WHERE round_id='arena-2026-10-01'",
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='foreign',actor_ref='foreign' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),owner_commitment='sha256:'||repeat('b',64),guard_generation=1,guard_expires_at=now()+interval '1 hour',candidate_commit=repeat('c',40),restart_scope='all',restart_phase='draining' WHERE singleton",
    "UPDATE public.lab_arena_runs SET per_icp_score=0 WHERE round_id='arena-2026-10-01' AND kind='execute' AND icp_position=0",
    "UPDATE public.lab_arena_rounds SET status='stage1' WHERE round_id='arena-2026-10-02'",
    "UPDATE public.lab_arena_runs SET status='leased' WHERE run_id='foreign368:1'",
])
def test_hold_rejects_changed_source_or_foreign_owner(database, mutation):
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
