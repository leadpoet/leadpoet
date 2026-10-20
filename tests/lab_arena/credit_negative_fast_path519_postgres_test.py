"""Credit-proof absence is cheap; positive native proof remains unchanged."""
from pathlib import Path
import pytest
from tests.lab_arena import model_policy_supersession517_postgres_test as revision

SQL=Path(__file__).parents[2]/'scripts/519-lab-arena-credit-negative-fast-path.sql'

@pytest.fixture(scope='module')
def database():
    gen=revision.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute(SQL.read_text());cur.execute(SQL.read_text())
        yield db
    finally:gen.close()

@pytest.mark.parametrize('proof,funding,expected',[(True,'miner_key',True),(False,'miner_key',False),(True,'host',False)])
def test_native_credit_lineage_unchanged(database,proof,funding,expected):
    revision.test_only_native_same_payer_credit_proof_survives_model_revision(database,proof,funding,expected)

@pytest.mark.parametrize('closed',[False,True])
def test_no_credential_predecessor_skips_expensive_company_evidence(database,closed):
    store,row,old,new=revision.seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            revision.invoke(cur,row,new)
            cur.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND status='pending' AND submission_id=%s",(row['round_id'],new[1]['submission_id']))
            run_id=cur.fetchone()[0]
            if closed:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc='{\"previous_status\":\"pending\"}'::jsonb WHERE run_id=%s",(run_id,))
                cur.execute('SET LOCAL session_replication_role=origin')
            # A forbidden validator proves the cheap negative precedes all
            # company-scope work, without a flaky wall-clock assertion.
            cur.execute("SELECT pg_get_functiondef('public.lab_arena__company_judgment_refs_v1(public.lab_arena_rounds,jsonb)'::regprocedure)")
            original=cur.fetchone()[0]
            cur.execute("CREATE OR REPLACE FUNCTION public.lab_arena__company_judgment_refs_v1(p_round public.lab_arena_rounds,p_refs jsonb) RETURNS boolean LANGUAGE plpgsql AS $$BEGIN RAISE EXCEPTION 'expensive company validation invoked';END$$")
            try:
                cur.execute('SELECT count(*) FROM generate_series(1,160) n WHERE public.lab_arena_miner_credit_failure_v1(%s||substring(n::text,1,0))',(run_id,))
                assert cur.fetchone()==(0,)
            finally:cur.execute(original)
    finally:store.close()

@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('drift',['owner','acl'])
def test_security_drift_rejected_before_idempotent_return(database,installed,drift):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SELECT pg_get_functiondef('public.lab_arena_miner_credit_failure_v1(text)'::regprocedure)");saved=cur.fetchone()[0]
        if not installed:
            old=SQL.read_text().split('$old$')[1];new=SQL.read_text().split('$new$')[1]
            assert saved.count(new)==1
            cur.execute(saved.replace(new,old))
        cur.execute('SAVEPOINT hostile')
        cur.execute('ALTER FUNCTION public.lab_arena_miner_credit_failure_v1(text) OWNER TO postgres' if drift=='owner' else 'GRANT EXECUTE ON FUNCTION public.lab_arena_miner_credit_failure_v1(text) TO anon')
        # The migration has its own transaction delimiters; execute its DO body
        # under this savepoint so hostile metadata is never retained by fixture.
        body=SQL.read_text().split('DO $credit_negative_fastpath$')[1].split('$credit_negative_fastpath$;')[0]
        with pytest.raises(Exception):cur.execute('DO $credit_negative_fastpath$'+body+'$credit_negative_fastpath$;')
        cur.execute('ROLLBACK TO SAVEPOINT hostile');cur.execute(saved)


def test_three_concurrent_real_claims_from_large_post_deadline_queue(database):
    """Full claim SQL serializes safely with many negative credit candidates."""
    from concurrent.futures import ThreadPoolExecutor
    import hashlib,json,time
    from lab_arena.store import ArenaStore,PsycopgTransport
    company=revision.company
    store,row,old,new=revision.seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cfg=row['configuration_doc']
            cfg['schedule']['final_scoring_close']='2000-01-01T00:00:00Z'
            # Model a large frozen catalogue without a production fixture or
            # uniform compressible padding. No operation consumes this field.
            cfg['test_large_context']=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(26000)]
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(cfg),row['round_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=%s',(row['round_id'],));fresh=cur.fetchone()[0]
            revision.invoke(cur,fresh,new)
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s AND submission_id=%s ORDER BY kind,attempt",(row['round_id'],new[1]['submission_id']))
            templates=[r[0] for r in cur.fetchall()]
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('SELECT to_jsonb(s) FROM public.lab_arena_submissions s WHERE submission_id=%s',(new[1]['submission_id'],));submission=cur.fetchone()[0]
            participants=list(row['participants']);plan=[]
            for index in range(158):
                group=index//10;position=index%10
                sub=row['round_id']+'-queue-'+str(group)
                miner=company.original.hotkey(sub)
                if position==0:
                    clone_sub=dict(submission);clone_sub['submission_id']=sub;clone_sub['miner_hotkey']=miner
                    cur.execute('INSERT INTO public.lab_arena_submissions SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,%s::jsonb)).*',(json.dumps(clone_sub),))
                    participants.append(dict(submission_id=sub,miner_hotkey=miner,is_king=False))
                refs=revision.deepcopy(new[1]['company_judgment_refs'])
                for ref in refs:
                    ref['company_input_hash']=ref['scope_doc']['company_input_hash']=revision.contracts.document_hash(['queue',index])
                    ref['scope_doc'].pop('cache_key')
                    ref['cache_key']=ref['scope_doc']['cache_key']=revision.contracts.document_hash(ref['scope_doc'])
                execution_id=f"{row['round_id']}:{sub}:2:{position}:1"
                for template in templates:
                    clone=dict(template);clone['submission_id']=sub;clone['miner_hotkey']=miner;clone['icp_position']=position
                    assignment=f"{row['round_id']}:{sub}:2:{position}"+(':score' if template['kind']=='score' else '')
                    clone['assignment_id']=assignment;clone['run_id']=assignment+':'+str(template['attempt'])
                    if template['kind']=='score':
                        clone['scored_run_id']=execution_id;clone['company_judgment_refs']=refs
                    cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*',(json.dumps(clone),))
                item=dict(new[1]);item['submission_id']=sub;item['icp_position']=position;item['scored_run_id']=execution_id;item['company_judgment_refs']=refs
                plan.append(item)
            cur.execute('UPDATE public.lab_arena_rounds SET participants=%s::jsonb,stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s',(json.dumps(participants),json.dumps({'work_items':[new[1]]+plan}),row['round_id']))
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND status='pending'",(row['round_id'],));assert cur.fetchone()==(160,)
            cur.execute("SELECT count(*) FILTER(WHERE public.lab_arena__judge_continuation_v1(a,r)) FROM public.lab_arena_runs r JOIN public.lab_arena_rounds a USING(round_id) WHERE r.round_id=%s AND r.status='pending'",(row['round_id'],));print('LARGE_QUEUE_ELIGIBLE',cur.fetchone(),flush=True)
        runner=company.original.hotkey('credit519-concurrent')
        def claim(_):
            local=ArenaStore(PsycopgTransport(lambda:database[0].connect(**database[1])))
            try:return company.original.claim(local,row['round_id'],runner,parallelism=7,ceiling=8,excluded=[runner])[0]
            finally:local.close()
        started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:leases=list(pool.map(claim,range(3)))
        print('LARGE_QUEUE_THREE_CLAIMS_SECONDS',time.perf_counter()-started,flush=True)
        print('LARGE_QUEUE_CLAIM_STATUSES',[r['status'] for r in leases],flush=True)
        assert all(r['status']=='leased' for r in leases),[r['status'] for r in leases]
        assert len({r['run_id'] for r in leases})==3
        assert all(r['kind']=='score' for r in leases)
    finally:store.close()


def test_non_superuser_caller_replaces_owner_function_without_owner_schema_create(database):
    admin=None
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('CREATE ROLE credit519_fixture_admin LOGIN SUPERUSER PASSWORD %s',(database[1]['password'],))
            cur.execute('GRANT lab_arena_owner TO postgres')
            cur.execute('ALTER SCHEMA public OWNER TO pg_database_owner')
            cur.execute('REVOKE CREATE ON SCHEMA public FROM PUBLIC,lab_arena_owner')
            cur.execute("SELECT pg_get_functiondef('public.lab_arena_miner_credit_failure_v1(text)'::regprocedure)");saved=cur.fetchone()[0]
            old=SQL.read_text().split('$old$')[1];new=SQL.read_text().split('$new$')[1]
            assert saved.count(new)==1;cur.execute(saved.replace(new,old))
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'");acl=cur.fetchone()[0]
        admin=database[0].connect(**dict(database[1],user='credit519_fixture_admin'));admin.autocommit=True
        with admin.cursor() as cur:cur.execute('ALTER ROLE postgres NOSUPERUSER INHERIT CREATEROLE BYPASSRLS')
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute("SELECT (SELECT rolsuper FROM pg_roles WHERE rolname=current_user),has_schema_privilege(current_user,'public','CREATE'),has_schema_privilege('lab_arena_owner','public','USAGE'),has_schema_privilege('lab_arena_owner','public','CREATE')")
            assert cur.fetchone()==(False,True,True,False)
            cur.execute(SQL.read_text());cur.execute(SQL.read_text())
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'");assert cur.fetchone()==(acl,)
            cur.execute("SELECT pg_get_functiondef('public.lab_arena_miner_credit_failure_v1(text)'::regprocedure)");assert cur.fetchone()==(saved,)
    finally:
        if admin is not None:
            with admin.cursor() as cur:cur.execute('ALTER ROLE postgres SUPERUSER')
            admin.close()
