"""Stage quota counts retain exact heads while looking up each run once."""
import hashlib
from pathlib import Path
import re

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/584-lab-arena-stage-quota-run-lookup.sql'
SIG = 'public.lab_arena__submission_stage_consumed(text,smallint,text,text)'
PRE = 'b00d1e9f60bbeaa3dad776811b33fd4a81df9f9278d1091d47a8b9b02768039e'
POST = '488e418fcd9c479b6898ac661bb64392a0fcb7cd271039bc28c062c1923e78e9'
ROUND = 'arena-2099-10-28-quota584'
SUB = 'quota584'


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def definition(q):
    q.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
    return q.fetchone()[0]


def catalog(q):
    q.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p "
              "WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
    return {oid: (source, metadata) for oid, source, metadata in q.fetchall()}


def body(which):
    return re.search(r'\$'+which+r'584\$(.*?)\$'+which+r'584\$', SQL.read_text(), re.S)[1]


@pytest.fixture(scope='module')
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


@pytest.fixture(autouse=True)
def restore(database):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        original = definition(q)
        assert digest(original) == PRE
    yield
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        q.execute(original)
        q.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger RESTART IDENTITY CASCADE')


def seed(q):
    # Only disposable fixture data bypasses write-once triggers.
    q.execute('SET LOCAL session_replication_role=replica')
    q.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,participants,benchmark_ref) "
              "VALUES (%s,'stage2','{}','[]',%s)", (ROUND, 'arena/'+ROUND+'/benchmark.json'))
    for sub in (SUB, SUB+'-other'):
        q.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status) "
                  "VALUES (%s,%s,%s,'frozen')", (sub, ROUND, hotkey(sub)))
        for stage in (1, 2):
            for kind in ('execute', 'score'):
                for position in range(8):
                    for attempt in (1, 2):
                        run = f'{sub}:{stage}:{kind}:{position}:{attempt}'
                        assignment = f'{sub}:{stage}:{kind}:{position}'
                        q.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                                  "miner_hotkey,stage,icp_position,attempt,kind,status) "
                                  "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'failed')",
                                  (run, assignment, ROUND, sub, hotkey(sub), stage, position, attempt, kind))
    q.execute('SET LOCAL session_replication_role=origin')


def ledger(q, *, label, states, sub=SUB, stage=2, provider='openrouter', kind='execute', position=0, attempt=1, identity=None):
    run = f'{sub}:{stage}:{kind}:{position}:{attempt}'
    identity = identity or 'sha256:'+hashlib.sha256(label.encode()).hexdigest()
    for state in states:
        q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,"
                  "stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                  "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'openrouter.chat','miner_key',0,'{}')",
                  (state, hotkey(sub), ROUND, sub, run, stage, identity, provider))


def count(q, args):
    q.execute('SELECT public.lab_arena__submission_stage_consumed(%s,%s::smallint,%s,%s)', args)
    return q.fetchone()[0]


def test_exact_upgrade_replay_preserves_security_other_functions_and_rows(database):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        seed(q)
        ledger(q, label='existing', states=['reservation','dispatch','settlement'])
        before = catalog(q)
        q.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
        rows = q.fetchone()[0]
        q.execute('CREATE ROLE quota584_migrator NOLOGIN NOSUPERUSER INHERIT')
        q.execute('GRANT lab_arena_owner TO quota584_migrator')
        q.execute('GRANT USAGE,CREATE ON SCHEMA public TO quota584_migrator')
        q.execute('SET ROLE quota584_migrator')
        q.execute(SQL.read_text())
        q.execute(SQL.read_text())
        q.execute('RESET ROLE')
        after = catalog(q)
        changed = [oid for oid in before if before[oid] != after[oid]]
        assert len(changed) == 1
        original, metadata = before[changed[0]]
        updated, updated_metadata = after[changed[0]]
        assert metadata == updated_metadata
        assert digest(original) == PRE and digest(updated) == POST
        assert updated == original.replace(body('old'), body('new'), 1)
        q.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
        assert q.fetchone()[0] == rows
        for role in ('anon','authenticated','service_role','lab_arena_service'):
            q.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role,SIG))
            assert q.fetchone()[0] is False


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source','owner','acl','path','security','volatility'])
def test_exact_source_and_security_drift_rejected(database, installed, drift):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        if installed:
            q.execute(SQL.read_text())
        q.execute('BEGIN')
        q.execute('SAVEPOINT hostile')
        if drift == 'source':
            q.execute(definition(q).replace('SELECT COUNT(*)::BIGINT', 'SELECT COUNT(*)::BIGINT + 1', 1))
        else:
            q.execute({
                'owner': 'ALTER FUNCTION '+SIG+' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon',
                'path': 'ALTER FUNCTION '+SIG+' SET search_path=public,pg_catalog',
                'security': 'ALTER FUNCTION '+SIG+' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION '+SIG+' VOLATILE',
            }[drift])
        migration_body = SQL.read_text().split('DO $stage_quota_run_lookup584$')[1].split('$stage_quota_run_lookup584$;')[0]
        with pytest.raises(database[0].Error):
            q.execute('DO $test$'+migration_body+'$test$')
        q.execute('ROLLBACK TO SAVEPOINT hostile')


def test_all_filters_and_latest_state_sequences_match_original(database):
    sequences = [
        ['reservation'], ['reservation','dispatch'],
        ['reservation','dispatch','settlement'],
        ['reservation','dispatch','uncertain'], ['reservation','recovery'], ['refusal'],
        ['reservation','dispatch','uncertain','settlement'],
    ]
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        seed(q)
        for sub in (SUB,SUB+'-other'):
            for stage in (1,2):
                for provider in ('openrouter','deepline','scrapingdog'):
                    for kind in ('execute','score'):
                        for number, states in enumerate(sequences):
                            for attempt in (1,2):
                                ledger(q,label=f'{sub}-{stage}-{provider}-{kind}-{number}-{attempt}',
                                       states=states,sub=sub,stage=stage,provider=provider,kind=kind,
                                       position=number,attempt=attempt)
        # Head order is global within the filtered eligible run set, not per
        # run. A later score entry must not hide an execute reservation.
        shared = 'sha256:'+hashlib.sha256(b'shared-runs').hexdigest()
        ledger(q,label='shared',states=['reservation'],identity=shared)
        ledger(q,label='shared',states=['dispatch'],position=1,identity=shared)
        ledger(q,label='shared',states=['recovery'],kind='score',identity=shared)
        # A null call identity and a missing run do not enter the count.
        q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,"
                  "run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) "
                  "VALUES ('settlement',%s,%s,%s,NULL,2,NULL,'openrouter','openrouter.chat','miner_key',0)",
                  (hotkey(SUB),ROUND,SUB))
        args = [(sub,stage,provider,kind)
                for sub in (SUB,SUB+'-other','missing',None)
                for stage in (1,2,3,None)
                for provider in ('openrouter','deepline','scrapingdog','missing',None)
                for kind in ('execute','score','missing',None)]
        before = [count(q, a) for a in args]
        assert count(q,(SUB,2,'openrouter','execute')) == 11
        q.execute(SQL.read_text())
        assert [count(q,a) for a in args] == before
        assert count(q,(SUB,2,'openrouter','execute')) == 11


def nodes(plan):
    yield plan
    for child in plan.get('Plans',[]):
        yield from nodes(child)


def prepared_plan(q, name, source):
    source = source.replace('p_submission_id','$1').replace('p_stage','$2').replace('p_provider','$3').replace('p_kind','$4')
    q.execute('PREPARE '+name+'(text,smallint,text,text) AS '+source)
    q.execute('EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON) EXECUTE '+name+'(%s,2::smallint,\'openrouter\',\'execute\')',(SUB,))
    plan = q.fetchone()[0][0]['Plan']
    q.execute('DEALLOCATE '+name)
    return plan


def test_generic_nested_loop_run_lookup_work_is_bounded_by_distinct_runs(database):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        seed(q)
        # Realistic repeated calls per ICP; enforce the nested-loop shape measured
        # in production without depending on machine speed or planner statistics.
        q.execute("SET LOCAL plan_cache_mode='force_generic_plan'")
        q.execute('SET LOCAL enable_hashjoin=off')
        q.execute('SET LOCAL enable_mergejoin=off')
        q.execute('SET LOCAL enable_memoize=off')
        for number in range(2000):
            ledger(q,label=f'load-{number}',states=['reservation','dispatch','settlement'],position=number%8)
        # A production-sized unrelated run population makes indexed lookups
        # appropriate; a tiny table can correctly use one bounded table scan.
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                  "miner_hotkey,stage,icp_position,attempt,kind,status) "
                  "SELECT 'distractor:'||n,'distractor:'||n,%s,%s,%s,1,n%%20,1,'execute','failed' "
                  "FROM generate_series(1,10000) n", (ROUND,SUB+'-other',hotkey(SUB+'-other')))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ANALYZE public.lab_arena_ledger')
        q.execute('ANALYZE public.lab_arena_runs')
        old = prepared_plan(q,'quota584_old',body('old'))
        new = prepared_plan(q,'quota584_new',body('new'))
        old_runs = [n for n in nodes(old) if n.get('Relation Name')=='lab_arena_runs']
        new_runs = [n for n in nodes(new) if n.get('Relation Name')=='lab_arena_runs']
        old_work = sum(n['Actual Rows']*n['Actual Loops'] for n in old_runs)
        new_work = sum(n['Actual Rows']*n['Actual Loops'] for n in new_runs)
        assert new_work <= 8
        assert old_work >= 6000
        assert new_work*100 < old_work
        assert sum(n.get('Shared Hit Blocks',0) for n in new_runs)*20 < sum(n.get('Shared Hit Blocks',0) for n in old_runs)
        assert count(q,(SUB,2,'openrouter','execute')) == 2000
        q.execute(SQL.read_text())
        assert count(q,(SUB,2,'openrouter','execute')) == 2000


def assert_real_stage_boundary(database, caller):
    with database[0].connect(**database[1]) as c:
        config = caller.seed(c,quota=1)
        lease = caller.claim(c)
        limit = config['stage_2_icp_count'] * config['max_attempts_per_assignment']
        with c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                      "miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation) "
                      "SELECT run_id||'-sibling',assignment_id||'-sibling',round_id,submission_id,"
                      "miner_hotkey,stage,9,1,kind,'failed',stage_generation "
                      "FROM public.lab_arena_runs WHERE run_id=%s",(lease['run_id'],))
            q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,"
                      "run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) "
                      "SELECT 'reservation',r.miner_hotkey,r.round_id,r.submission_id,r.run_id,r.stage,"
                      "'sha256:'||lpad(to_hex(n+58400000),64,'0'),'deepline','deepline.execute','miner_key',0 "
                      "FROM public.lab_arena_runs r CROSS JOIN generate_series(1,%s) n "
                      "WHERE r.run_id=%s",(limit,lease['run_id']+'-sibling'))
            q.execute('SET LOCAL session_replication_role=origin')
        c.commit()
        denied = caller.reserve(c,lease,call='d')
        assert denied['status']=='refused' and denied['reason']=='stage_quota',denied
        with c.cursor() as q:
            q.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s "
                      "AND entry_kind='dispatch'",(lease['run_id'],))
            assert q.fetchone()[0]==0


def test_real_reservation_and_host_retry_quotas_preserved():
    from tests.lab_arena import host_interruption_retry474_postgres_test as caller
    generator = caller.database.__wrapped__()
    database = next(generator)
    try:
        caller.test_extra_attempt_does_not_reset_cumulative_provider_quota(database)
        assert_real_stage_boundary(database,caller)
        with database[0].connect(**database[1]) as c, c.cursor() as q:
            assert digest(definition(q)) == PRE
            q.execute(SQL.read_text())
        caller.test_extra_attempt_does_not_reset_cumulative_provider_quota(database)
        assert_real_stage_boundary(database,caller)
    finally:
        generator.close()
