"""Only the exact released November 3 hold can resume the retained work."""
import pytest

from tests.lab_arena import nov03_corrected_judge607_postgres_test as prior

SQL=prior.prior.SCRIPTS/'608-arena-2026-11-03-resume-remaining-work.sql'
ROUND=prior.ROUND
ACTOR='nov03-forward-recovery608'
database=prior.database
scalar=prior.scalar


def setup(conn,dsn,pg):
    with conn.cursor() as cur:
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton")
    conn.commit()
    store,_,_,migration,_,_=prior.setup(conn,dsn,pg)
    with conn.cursor() as cur:
        cur.execute(migration)
        # Use the existing ordinary judge cooldown in this disposable fixture.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",(prior.RUN,))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_generation=756,guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]',actor_ref='canonical-release-test608' WHERE singleton")
        cur.execute("INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc) SELECT round_id,99,'nov03-admission-review605-v1','sha256:'||repeat('a',64),'test608',txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(r),'{}','{\"operation\":\"nov03_admission_review605_v1\"}' FROM public.lab_arena_rounds r WHERE round_id=%s",(ROUND,))
    conn.commit()
    return store


def snapshot(conn):
    with conn.cursor() as cur:
        result={}
        for table,key in (('lab_arena_restart_claim_control','singleton'),('lab_arena_rounds','round_id'),
                          ('lab_arena_submissions','submission_id'),('lab_arena_runs','run_id'),
                          ('lab_arena_ledger','entry_id'),('lab_arena_scoring_supersessions','generation_floor')):
            result[table]=scalar(cur,f"SELECT coalesce(jsonb_agg(to_jsonb(t) ORDER BY {key}),'[]'::jsonb) FROM public.{table} t")
        return result


def test_resume_changes_only_exact_hold_and_replays_without_other_state_change(database):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        store=setup(conn,dsn,pg)
        try:
            before=snapshot(conn)
            with conn.cursor() as cur:cur.execute(SQL.read_text())
            after=snapshot(conn)
            control=before['lab_arena_restart_claim_control'][0]
            expected=control|{'operator_paused':False,'pause_reason':'','actor_ref':ACTOR,
                'updated_at':after['lab_arena_restart_claim_control'][0]['updated_at']}
            assert after==before|{'lab_arena_restart_claim_control':[expected]}
            with conn.cursor() as cur:cur.execute(SQL.read_text())
            assert snapshot(conn)==after
            assert prior.claim(conn)['status']=='leased'
            running=snapshot(conn)
            with conn.cursor() as cur:cur.execute(SQL.read_text())
            assert snapshot(conn)==running
        finally:store._transport.close()


@pytest.mark.parametrize('damage',['wrong_hold','active_guard','old_generation','missing604','missing605','missing607',
    'wrong604_config','past_baseline_deadline','past_execution_deadline','active_review','active_lease',
    'damaged_preserved_judge','damaged_corrected_judge','other_unpaused_actor'])
def test_resume_denies_without_changing_any_state(database,damage):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        store=setup(conn,dsn,pg)
        try:
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                if damage=='wrong_hold':cur.execute("UPDATE public.lab_arena_restart_claim_control SET pause_reason='unrelated-hold' WHERE singleton")
                elif damage=='active_guard':cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),owner_commitment='sha256:'||repeat('b',64),guard_expires_at=clock_timestamp()+interval '1 hour',candidate_commit=repeat('c',40),restart_scope='all',restart_phase='draining' WHERE singleton")
                elif damage=='old_generation':cur.execute('UPDATE public.lab_arena_restart_claim_control SET guard_generation=754 WHERE singleton')
                elif damage in ('missing604','missing605','missing607'):
                    request={'missing604':'nov03-forward-only-verifier604-v1','missing605':'nov03-admission-review605-v1','missing607':'nov03-baseline8-corrected607-v1'}[damage]
                    cur.execute('DELETE FROM public.lab_arena_scoring_supersessions WHERE request_id=%s',(request,))
                elif damage=='wrong604_config':cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc||'{\"wrong608\":true}' WHERE round_id=%s",(ROUND,))
                elif damage.startswith('past_'):
                    field='stage_1_scoring_close' if damage=='past_baseline_deadline' else 'stage_2_close'
                    cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,ARRAY['schedule',%s],to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s",(field,ROUND))
                elif damage=='active_review':cur.execute("UPDATE public.lab_arena_submissions SET code_review_status='reviewing',code_review_expires_at=clock_timestamp()+interval '10 minutes',code_review_doc=NULL WHERE submission_id='nov03-miner'")
                elif damage=='active_lease':cur.execute("UPDATE public.lab_arena_runs SET status='leased',lease_token_hash='sha256:'||repeat('a',64),lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=%s",(prior.RUN,))
                elif damage=='damaged_preserved_judge':cur.execute("UPDATE public.lab_arena_runs SET output_ref='arena/wrong608.json' WHERE kind='score' AND status='accepted' AND icp_position=0")
                elif damage=='damaged_corrected_judge':cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs='[]' WHERE run_id=%s",(prior.RUN,))
                else:cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason='',actor_ref='unrelated-resume' WHERE singleton")
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit();before=snapshot(conn)
            with pytest.raises(Exception):
                with conn.cursor() as cur:cur.execute(SQL.read_text())
            conn.rollback()
            assert snapshot(conn)==before
        finally:store._transport.close()
