"""Closed billing query parity; authority and immutable data stay unchanged."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import execute_reconciliation_scope505_postgres_test as prior

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/506-lab-arena-closed-reconciliation-eligible-run-scope.sql'
FUNCTION = 'public.lab_arena_next_closed_deepline_reconciliation_v1'
SIGNATURE = FUNCTION + '(text,text,integer,text,bigint)'
OLD = 'public.test_before_closed_scope506'
HELPERS = ['dynamic', 'host_success', 'miner_success', 'adaptation']
HELPER_FUNCTIONS = ['lab_arena__closed_score_dynamic_uncertainty_v1',
                    'lab_arena__closed_host_score_success_uncertainty_v1',
                    'lab_arena__closed_miner_score_success_uncertainty_v1',
                    'lab_arena__closed_score_adaptation_uncertainty_v1']


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
            cur.execute(cur.fetchone()[0].replace('FUNCTION ' + FUNCTION + '(', 'FUNCTION ' + OLD + '(', 1))
            conn.commit()
            before = prior.snapshot(cur)
            text = SQL.read_text()
            for name, message in (('v_preimage','preimage differs'), ('v_postimage','postimage differs')):
                digest = text.split(name + " CONSTANT TEXT := '")[1].split("'")[0]
                with pytest.raises(db[0].Error, match=message):
                    cur.execute(text.replace(digest,'0' * 64,1))
                conn.rollback()
                assert prior.snapshot(cur) == before
            cur.execute(f'GRANT EXECUTE ON FUNCTION {SIGNATURE} TO anon')
            with pytest.raises(db[0].Error, match='security shape differs'):
                cur.execute(text)
            conn.rollback()
            assert prior.snapshot(cur) == before
            cur.execute('CREATE ROLE closed506_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO closed506_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO closed506_migrator')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE closed506_migrator IN SCHEMA public '
                        'GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cur.execute('SET ROLE closed506_migrator')
            cur.execute(text)
            after = prior.snapshot(cur)
            assert before[1:] == after[1:]
            changed = []
            for a,b in zip(before[0],after[0]):
                assert a[0] == b[0] and a[2:] == b[2:]
                if a[1] != b[1]:
                    changed.append(a[0])
            assert changed == ['lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)']
            cur.execute(text)
            assert prior.snapshot(cur) == after
            cur.execute('RESET ROLE')
        yield db
    finally:
        gen.close()


def pair(cur, rid='', cursor=0, mode='live', network='finney', netuid=71):
    cur.execute('SET LOCAL jit=off')
    args = (mode,network,netuid,rid,cursor)
    cur.execute('SELECT ' + OLD + '(%s,%s,%s,%s,%s),' + FUNCTION + '(%s,%s,%s,%s,%s)',args+args)
    old,new = cur.fetchone()
    assert old == new
    return new


def seed(database, helper='miner_success'):
    rid,candidate = prior.seed(database,'closed506')
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='published',configuration_doc="
                    "configuration_doc || '{\"mode\":\"live\",\"network_name\":\"finney\",\"netuid\":71,"
                    "\"sourcing_cost_eligibility_policy\":\"successful_calls_flat_per_icp_v1\"}'::jsonb WHERE round_id=%s",(rid,))
        cur.execute("UPDATE public.lab_arena_runs SET kind='score',status=%s,terminal_cause=%s WHERE run_id=%s",
                    ('failed' if helper=='dynamic' else 'accepted','judge_error' if helper=='dynamic' else 'accepted',candidate['run_id']))
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc || "
                    "jsonb_build_object('deepline_execution_key','arena:'||substr(call_identity,8)) "
                    "WHERE entry_kind='reservation' AND round_id=%s",(rid,))
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call}',"
                    "(entry_doc->'call') || jsonb_build_object('provider_status',200,'deepline_execution_key',"
                    "'arena:'||substr(call_identity,8))) WHERE entry_kind='uncertain' AND round_id=%s",(rid,))
        if helper=='dynamic':
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,"
                        "'{call,call_succeeded}','false') WHERE entry_kind='uncertain' AND round_id=%s",(rid,))
        if helper=='host_success':
            cur.execute("UPDATE public.lab_arena_ledger SET funding_source='host' WHERE round_id=%s",(rid,))
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,deepline_job_id}',"
                        "'\"public-execute:host506\"'::jsonb) WHERE entry_kind='uncertain' AND round_id=%s",(rid,))
        if helper=='adaptation':
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call}',"
                        "(entry_doc->'call') || '{\"reason\":\"settle_failure\",\"call_succeeded\":false,"
                        "\"failure_stage\":\"response_adaptation\",\"error_class\":\"CompatibilityResponseError\"}'::jsonb) "
                        "WHERE entry_kind='uncertain' AND round_id=%s",(rid,))
        cur.execute('SELECT ' + ','.join('public.'+f+'(%s)' for f in HELPER_FUNCTIONS),
                    (candidate['uncertain_entry_id'],)*4)
        flags=cur.fetchone()
        assert flags[HELPERS.index(helper)] is True, flags
        assert pair(cur,rid)['uncertain_entry_id'] == candidate['uncertain_entry_id']
    return rid,candidate


@pytest.mark.parametrize('helper',HELPERS)
def test_all_four_unchanged_authority_helpers_positive_and_settled_head(database,helper):
    rid,candidate = seed(database,helper)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        assert pair(cur)['uncertain_entry_id'] == candidate['uncertain_entry_id']
        for cursor in (None,-1,0,candidate['uncertain_entry_id'],9223372036854775807):
            assert pair(cur,rid,cursor)['uncertain_entry_id'] == candidate['uncertain_entry_id']
        cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,"
                    "run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                    "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
                    "provider,operation_id,funding_source,0,'{}'::jsonb FROM public.lab_arena_ledger WHERE entry_id=%s",
                    (candidate['uncertain_entry_id'],))
        assert pair(cur,rid) == {'status':'none'}


def test_scope_policy_terminal_kind_and_null_boundary_parity(database):
    rid,candidate = seed(database)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        for status in ('published','cancelled','open','stage1_scoring'):
            for policy in ('successful_calls_v1','successful_calls_per_icp_v1','successful_calls_flat_per_icp_v1','legacy',None):
                cur.execute("UPDATE public.lab_arena_rounds SET status=%s,configuration_doc="
                            "configuration_doc || jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) "
                            "WHERE round_id=%s",(status,policy,rid))
                result=pair(cur,rid)
                allowed=status=='cancelled' or status=='published' and policy not in ('legacy',None)
                assert result['status'] == ('ok' if allowed else 'none')
        cur.execute("UPDATE public.lab_arena_rounds SET status='cancelled' WHERE round_id=%s",(rid,))
        for run_status,cause in [('accepted','accepted'),('failed','judge_error'),('failed','provider_error'),
                                 ('failed','invalid_output'),('leased',None),('pending',None),('accepted','judge_error')]:
            for kind in ('score','execute'):
                cur.execute('UPDATE public.lab_arena_runs SET status=%s,terminal_cause=%s,kind=%s WHERE run_id=%s',
                            (run_status,cause,kind,candidate['run_id']))
                result=pair(cur,rid)
                allowed=kind=='score' and ((run_status=='accepted' and cause=='accepted') or
                        (run_status=='failed' and cause in ('judge_error','provider_error')))
                assert result['status'] == ('ok' if allowed else 'none')
        cur.execute("UPDATE public.lab_arena_runs SET kind='score',status='accepted',terminal_cause='accepted' WHERE run_id=%s",(candidate['run_id'],))
        assert pair(cur,rid)['status'] == pair(cur,None)['status'] == 'ok'
        for args in [dict(rid='missing-round'),dict(mode='shadow'),dict(network='test'),dict(netuid=1),
                     dict(mode=None),dict(network=None),dict(netuid=None)]:
            assert pair(cur,**args)=={'status':'none'}


@pytest.mark.parametrize('row_kind', ['reservation','dispatch'])
@pytest.mark.parametrize('column,value', [('run_id','foreign-run'),('round_id','foreign-round'),
    ('submission_id','foreign-submission'),('miner_hotkey','foreign-miner'),('stage',1),
    ('provider','openrouter'),('operation_id','foreign-operation'),('funding_source','host')])
def test_final_authority_identity_guards_remain_exact(database,row_kind,column,value):
    rid,candidate = seed(database)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute(f'UPDATE public.lab_arena_ledger SET {column}=%s WHERE call_identity=%s AND entry_kind=%s',
                    (value,candidate['call_identity'],row_kind))
        assert pair(cur,rid) == {'status':'none'}


def test_cursor_wrap_latest_heads_and_ineligible_early_rows(database):
    rid,candidate = seed(database)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        prior.heads._duplicate_calls(cur,candidate,24,start=506000)
        # Each copied call needs its own unchanged execution-key binding.
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc || "
                    "jsonb_build_object('deepline_execution_key','arena:'||substr(call_identity,8)) "
                    "WHERE round_id=%s AND entry_kind='reservation'",(rid,))
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call}',"
                    "(entry_doc->'call') || jsonb_build_object('deepline_execution_key','arena:'||substr(call_identity,8))) "
                    "WHERE round_id=%s AND entry_kind='uncertain'",(rid,))
        cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s AND entry_kind='uncertain' ORDER BY entry_id",(rid,))
        ids=[r[0] for r in cur.fetchall()]
        assert len(ids)==25
        assert pair(cur,rid)['uncertain_entry_id']==ids[0]
        assert pair(cur,rid,ids[12])['uncertain_entry_id']==ids[13]
        assert pair(cur,rid,ids[-1])['uncertain_entry_id']==ids[0]
        # A later nonsettlement head suppresses the first call, even though the
        # old uncertainty and every helper proof still exist.
        cur.execute('UPDATE public.lab_arena_ledger SET entry_id=-entry_id WHERE entry_id=%s',(ids[0],))
        assert pair(cur,rid)['uncertain_entry_id']==ids[1]
        # A helper-invalid earlier row must not block the next valid candidate.
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,provider_status}','500') WHERE entry_id=%s",(ids[1],))
        assert pair(cur,rid)['uncertain_entry_id']==ids[2]
        cur.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s",(candidate['run_id'],))
        assert pair(cur,rid)=={'status':'none'}
