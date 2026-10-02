"""Hold October 1 optional-signal review without losing results or leases."""
from pathlib import Path

import pytest

from tests.lab_arena import oct01_hq_source_rejudge387_postgres_test as prior
from tests.lab_arena import oct01_intent_state_rejudge385_postgres_test as hashes

HOLD = Path(__file__).parents[2] / 'scripts/388-arena-2026-10-01-optional-signal-hold.sql'
database = prior.database
ROUND = prior.ROUND


def _prepare(cursor):
    cursor.execute(prior._prepare(cursor))
    cursor.execute('SET session_replication_role=replica')
    cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',"
                   "status_generation=41,stage_generation=34,"
                   "stage1_scoring_plan_doc='{}'::jsonb,"
                   "stage2_scoring_plan_doc='{}'::jsonb WHERE round_id=%s", (ROUND,))
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
        "icp_position,attempt,kind,status,stage_generation,scored_run_id,"
        "terminal_cause,output_ref,lease_expires_at) "
        "SELECT 'score388:'||e.run_id,"
        "e.round_id||':'||e.submission_id||':'||e.stage||':'||e.icp_position||':score:rerun387',"
        "e.round_id,e.submission_id,e.miner_hotkey,e.stage,e.icp_position,1,'score',"
        "CASE WHEN e.stage=1 OR e.stage2_rank<=104 THEN 'accepted' "
        "WHEN e.stage2_rank<=109 THEN 'leased' ELSE 'pending' END,34,e.run_id,"
        "CASE WHEN e.stage=1 OR e.stage2_rank<=104 THEN 'accepted' END,"
        "CASE WHEN e.stage=1 OR e.stage2_rank<=104 THEN 'arena/test/score388.json' END,"
        "CASE WHEN e.stage=2 AND e.stage2_rank BETWEEN 105 AND 109 "
        "THEN now()+interval '1 hour' END "
        "FROM (SELECT r.*,CASE WHEN stage=2 THEN "
        "row_number() OVER (PARTITION BY stage ORDER BY run_id) END AS stage2_rank "
        "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='execute' "
        "AND status='accepted') e", (ROUND,))
    assert cursor.rowcount == 140
    cursor.execute('SET session_replication_role=origin')
    return hashes._render(cursor, HOLD.read_text().replace(
        'sha256:4256d5790540ace6f739ab31135b855f3ce76eece8809943b5eca68deff7e787',
        prior.TEST_DIGEST,
    ))


def test_hold_preserves_all_work_blocks_new_claims_and_allows_drain(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                           "WHERE round_id=%s ORDER BY run_id", (ROUND,))
            before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id")
            costs = cursor.fetchall()
            cursor.execute(sql)
            cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                           "WHERE round_id=%s ORDER BY run_id", (ROUND,))
            assert cursor.fetchall() == before
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id")
            assert cursor.fetchall() == costs
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            control = cursor.fetchone()
            assert control[:3] == (True, 'oct01_optional_signal_boundary_recovery',
                                   'oct01-optional-signal-hold388')
            with pytest.raises(psycopg.Error, match='lab_arena_claims_paused'):
                cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                               "WHERE round_id=%s AND stage=2 AND kind='score' "
                               "AND status='pending'", (ROUND,))
            cursor.execute('ROLLBACK')
            with pytest.raises(psycopg.Error, match='lab_arena_round_progression_paused'):
                cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' "
                               "WHERE round_id=%s", (ROUND,))
            cursor.execute('ROLLBACK')
            with pytest.raises(psycopg.Error, match='lab_arena_round_progression_paused'):
                cursor.execute("UPDATE public.lab_arena_runs SET per_icp_score=50,"
                               "qualification_doc='{\"companies\":[]}'::jsonb "
                               "WHERE run_id=(SELECT run_id FROM public.lab_arena_runs "
                               "WHERE round_id=%s AND stage=2 AND kind='execute' "
                               "AND status='accepted' ORDER BY run_id LIMIT 1)", (ROUND,))
            cursor.execute('ROLLBACK')
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/drained388.json' "
                           "WHERE round_id=%s AND stage=2 AND kind='score' "
                           "AND status='leased'", (ROUND,))
            assert cursor.rowcount == 5
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone() == control
            for status, generation, stage_generation in (
                ('stage2_judged', 42, 35), ('scored', 43, 35)
            ):
                cursor.execute('SET session_replication_role=replica')
                cursor.execute("UPDATE public.lab_arena_rounds SET status=%s,"
                               "status_generation=%s,stage_generation=%s "
                               "WHERE round_id=%s",
                               (status, generation, stage_generation, ROUND))
                cursor.execute('SET session_replication_role=origin')
                cursor.execute(sql)
                cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                               "FROM public.lab_arena_restart_claim_control WHERE singleton")
                assert cursor.fetchone() == control


def test_hold_accepts_failed_attempt_with_pending_retry(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET status='failed',"
                           "terminal_cause='judge_error' WHERE run_id=("
                           "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
                           "AND stage=2 AND kind='score' AND status='pending' "
                           "ORDER BY run_id LIMIT 1) RETURNING run_id,assignment_id,"
                           "round_id,submission_id,miner_hotkey,stage,icp_position,"
                           "stage_generation,scored_run_id", (ROUND,))
            failed = cursor.fetchone()
            assert failed is not None
            cursor.execute("INSERT INTO public.lab_arena_runs "
                           "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                           "stage,icp_position,attempt,kind,status,stage_generation,"
                           "scored_run_id) VALUES (%s,%s,%s,%s,%s,%s,%s,2,'score',"
                           "'pending',%s,%s)",
                           (failed[0]+':retry2', *failed[1:]))
            cursor.execute('SET session_replication_role=origin')
            cursor.execute(sql)
            cursor.execute("SELECT count(*),count(DISTINCT assignment_id) "
                           "FROM public.lab_arena_runs WHERE round_id=%s "
                           "AND stage=2 AND kind='score'", (ROUND,))
            assert cursor.fetchone() == (131, 130)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone() == (True, 'oct01_optional_signal_boundary_recovery',
                                         'oct01-optional-signal-hold388')


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='foreign',actor_ref='foreign' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),owner_commitment='sha256:'||repeat('b',64),guard_generation=1,guard_expires_at=now()+interval '1 hour',candidate_commit=repeat('c',40),restart_scope='all',restart_phase='draining' WHERE singleton",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"wrong\"'::jsonb) WHERE round_id='arena-2026-10-01'",
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-02'",
    "UPDATE public.lab_arena_runs SET assignment_id='wrong:'||run_id WHERE round_id='arena-2026-10-01' AND kind='score' AND stage=2 AND icp_position=0",
    "UPDATE public.lab_arena_runs SET output_ref=NULL WHERE round_id='arena-2026-10-01' AND kind='execute' AND stage=2 AND icp_position=0",
    "UPDATE public.lab_arena_rounds SET published_at=now() WHERE round_id='arena-2026-10-01'",
], ids=['foreign_owner', 'restart_guard', 'wrong_image', 'other_live_round',
        'bad_assignment', 'missing_source', 'published'])
def test_hold_fails_closed_without_mutation(database, mutation):
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
