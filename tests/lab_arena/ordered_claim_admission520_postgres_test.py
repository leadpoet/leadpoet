"""Ordered admission keeps complete proof and row locking on each candidate."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,time
import pytest
from tests.lab_arena import credit_negative_fast_path519_postgres_test as queue
from tests.lab_arena import model_policy_supersession517_postgres_test as revision
from tests.lab_arena import miner_credit_terminal484_postgres_test as credit

SQL=Path(__file__).parents[2]/'scripts/520-lab-arena-ordered-claim-admission.sql'
SIG='public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'

@pytest.fixture(scope='module')
def database():
    gen=queue.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute(SQL.read_text());cur.execute(SQL.read_text())
        yield db
    finally:gen.close()


def prepared(db, *, native_credit=False):
    store,row,old,new=revision.seed(db)
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}','\"2000-01-01T00:00:00Z\"') WHERE round_id=%s",(row['round_id'],))
        previous=old[1]['scored_run_id'][:-2]+':score:1'
        if native_credit:
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='credential_error',result_doc='{\"terminal_status\":\"credential_error\"}'::jsonb WHERE run_id=%s",(previous,))
        cur.execute('SET LOCAL session_replication_role=origin')
        if native_credit:credit.credit(conn,previous,proof=True,funding='miner_key')
        cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=%s',(row['round_id'],));fresh=cur.fetchone()[0]
        revision.invoke(cur,fresh,new)
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
        cur.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND status='pending' ORDER BY icp_position,created_at,assignment_id",(row['round_id'],));ids=[r[0] for r in cur.fetchall()]
    store.close();return row['round_id'],ids


def claim(cur,rid, *, runner=None):
    runner=runner or revision.company.original.hotkey('ordered520')
    cur.execute('SET LOCAL ROLE lab_arena_service')
    cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,7,8,%s,%s,%s,%s,4500)',(rid,runner,[],'a'*32,'sha256:'+'a'*64,'sha256:'+'b'*64))
    return cur.fetchone()[0]


def test_large_post_deadline_queue_concurrent_claims(database):
    queue.test_three_concurrent_real_claims_from_large_post_deadline_queue(database)


@pytest.mark.parametrize('rejection',['review','excluded','native_credit'])
def test_first_rejected_candidate_remains_unlocked_and_later_candidate_is_leased(database,rejection):
    rid,ids=prepared(database,native_credit=rejection=='native_credit')
    bad=ids[1]  # Nonbaseline: preserve the canonical organizer review exemption.
    good=next(r for r in ids if r!=bad)
    with database[0].connect(**database[1]) as setup,setup.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '30 minutes' WHERE run_id=%s",(bad,))
        if rejection=='review':
            cur.execute("UPDATE public.lab_arena_submissions SET code_review_status='pending',code_review_claim=NULL,code_review_started_at=NULL,code_review_doc=NULL,code_review_attempts=0,code_review_expires_at=NULL WHERE submission_id=(SELECT submission_id FROM public.lab_arena_runs WHERE run_id=%s)",(bad,))
        if rejection=='excluded':
            cur.execute("UPDATE public.lab_arena_runs SET judgment_group_miner_hotkeys=ARRAY[%s] WHERE run_id=%s",(revision.company.original.hotkey('ordered520'),bad))
        cur.execute('SET LOCAL session_replication_role=origin')
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        # Excluded control supplies the real caller exclusion list. Keep every
        # SQL admission guard in force rather than faking a helper result.
        if rejection=='excluded':
            runner=revision.company.original.hotkey('ordered520')
            cur.execute('SET LOCAL ROLE lab_arena_service')
            cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,7,8,%s,%s,%s,%s,4500)',(rid,runner,[runner],'a'*32,'sha256:'+'a'*64,'sha256:'+'b'*64));result=cur.fetchone()[0]
        else:result=claim(cur,rid)
        assert result['status']=='leased' and result['run_id']==good,result
        # The successful claim transaction is still open and holds its chosen
        # row. A separate transaction can nevertheless lock the rejected row.
        with database[0].connect(**database[1]) as other,other.cursor() as check:
            check.execute('SELECT run_id FROM public.lab_arena_runs WHERE run_id=%s FOR UPDATE NOWAIT',(bad,));assert check.fetchone()==(bad,)


def test_locked_eligible_candidate_is_skipped_without_changing_order(database):
    rid,ids=prepared(database)
    with database[0].connect(**database[1]) as blocker,blocker.cursor() as lock:
        lock.execute('SELECT run_id FROM public.lab_arena_runs WHERE run_id=%s FOR UPDATE',(ids[0],))
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            result=claim(cur,rid)
            assert result['status']=='leased' and result['run_id']==ids[1],result


@pytest.mark.parametrize('change',['status','floor'])
def test_concurrent_row_change_is_rechecked_before_lease(database,change):
    rid,ids=prepared(database);target=ids[0];gate=7520520
    with database[0].connect(**database[1]) as setup,setup.cursor() as cur:
        cur.execute("SELECT pg_get_functiondef('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'::regprocedure)");definition=cur.fetchone()[0]
        assert definition.count('BEGIN\n')==1
        instrumented=definition.replace('BEGIN\n',"BEGIN\n  IF p_run.run_id="+"'"+target+"' THEN PERFORM pg_advisory_xact_lock("+str(gate)+"); END IF;\n",1)
        cur.execute(instrumented)
    blocker=database[0].connect(**database[1]);blocker.autocommit=False
    pool=ThreadPoolExecutor(max_workers=1)
    def run_claim():
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:return claim(cur,rid)
    future=None
    try:
        with blocker.cursor() as cur:cur.execute('SELECT pg_advisory_xact_lock(%s)',(gate,))
        future=pool.submit(run_claim)
        deadline=time.monotonic()+10
        with database[0].connect(**database[1]) as observer,observer.cursor() as cur:
            while True:
                cur.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' AND objid=%s AND NOT granted)",(gate,))
                if cur.fetchone()[0]:break
                assert time.monotonic()<deadline,'claim did not reach the pre-lock proof gate'
                time.sleep(0.01)
            cur.execute("SET LOCAL lock_timeout='500ms'")
            cur.execute('SET LOCAL session_replication_role=replica')
            if change=='status':cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc='{\"previous_status\":\"pending\"}'::jsonb WHERE run_id=%s",(target,))
            else:cur.execute('UPDATE public.lab_arena_runs SET scoring_generation_floor=0 WHERE run_id=%s',(target,))
            cur.execute('SET LOCAL session_replication_role=origin')
        blocker.commit()
        result=future.result(timeout=10)
        assert result['status']=='leased' and result['run_id']==ids[1],result
    finally:
        blocker.rollback();blocker.close();pool.shutdown(wait=True)
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:cur.execute(definition)


@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('drift',['owner','acl','secdef','volatility','path','source'])
def test_security_and_source_drift_rejected_before_replay(database,installed,drift):
    sql=SQL.read_text()
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));saved=cur.fetchone()[0]
        current=saved
        if not installed:
            for field in ('header','tail'):
                old=sql.split('$old_'+field+'$')[1];new=sql.split('$new_'+field+'$')[1]
                assert current.count(new)==1;current=current.replace(new,old)
            cur.execute(current)
        cur.execute('SAVEPOINT hostile')
        if drift=='source':cur.execute(current.replace('BEGIN\n','BEGIN\n  -- unrecognized source drift\n',1))
        else:
            statements={
                'owner':'ALTER FUNCTION '+SIG+' OWNER TO postgres',
                'acl':'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon',
                'secdef':'ALTER FUNCTION '+SIG+' SECURITY INVOKER',
                'volatility':'ALTER FUNCTION '+SIG+' STABLE',
                'path':'ALTER FUNCTION '+SIG+' SET search_path=public,pg_catalog',
            }
            cur.execute(statements[drift])
        body=sql.split('DO $ordered_claim_admission$')[1].split('$ordered_claim_admission$;')[0]
        with pytest.raises(database[0].Error):cur.execute('DO $ordered_claim_admission$'+body+'$ordered_claim_admission$;')
        cur.execute('ROLLBACK TO SAVEPOINT hostile');cur.execute(saved)
