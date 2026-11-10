"""Owner-only transport correction preserves active work and grading authority."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
from threading import Barrier
import uuid

import pytest

from lab_arena import reasoning_capture, scoring
from tests.lab_arena import operational_round_history663_postgres_test as history
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL = Path(__file__).parents[2] / 'scripts/683-lab-arena-judge-provider-routing.sql'
RPC = 'public.lab_arena_set_judge_provider_routing_v1(text,jsonb,jsonb,text,text,boolean)'
MODEL = 'z-ai/glm-5.3-flash'
PLAN = {MODEL: ['fireworks']}


@pytest.fixture(scope='module')
def database():
    source = history.database.__wrapped__()
    pg, dsn = next(source)
    try:
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute(history.SQL)
            cur.execute('SELECT to_regprocedure(%s)', (RPC,))
            assert cur.fetchone()[0] is None  # The pre-change path has no authority.
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
        yield pg, dsn
    finally:
        source.close()


def seed(database, *, change=None, status='stage2'):
    pg, dsn = database
    rid = 'arena-2099-11-10-' + uuid.uuid4().hex[:12]
    config = base_round_configuration()
    config.update(round_id=rid, mode='live', rewards_enabled=True,
                  network_name='finney', netuid=71,
                  scorer_policy=scoring.build_scorer_policy(),
                  judge_reasoning_route_registry=reasoning_capture.load_registry())
    config['schedule'] = dict(config['schedule'],
        final_scoring_close='2099-11-10T20:30:02Z',
        publication_deadline='2099-11-10T20:30:03Z')
    if change:
        change(config)
    conn = pg.connect(**dsn)
    with conn.cursor() as cur:
        # Synthetic rows model an existing live lease. The RPC itself uses all
        # production triggers, grants, locks and append-only history checks.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("INSERT INTO lab_arena_rounds(round_id,status,configuration_doc,rewards_enabled,"
                    "stage_generation,status_generation) VALUES(%s,%s,%s::jsonb,TRUE,4,7)",
                    (rid,status,json.dumps(config)))
        sid='routing683-'+uuid.uuid4().hex[:16]
        hk=hotkey(sid)
        cur.execute("INSERT INTO lab_arena_submissions(submission_id,round_id,miner_hotkey,status)"
                    "VALUES(%s,%s,%s,'frozen')",(sid,rid,hk))
        for kind, state, suffix in [('execute','accepted','e'),('score','leased','l'),('score','pending','p')]:
            run=rid+':'+suffix
            cur.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                        "stage,icp_position,attempt,status,kind,lease_generation,scoring_generation_floor,"
                        "runner_hotkey,lease_token_hash,lease_expires_at,result_doc,per_icp_score) "
                        "VALUES(%s,%s,%s,%s,%s,1,0,1,%s,%s,1,0,%s,%s,'2099-11-10T18:00:00Z',"
                        "'{\"preserve\":true}',%s)",
                        (run,run,rid,sid,hk,state,kind,hk,'sha256:'+'b'*64,37 if kind=='execute'else None))
        cur.execute("INSERT INTO lab_arena_ledger(entry_kind,run_id,round_id,submission_id,miner_hotkey,stage,"
                    "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                    "VALUES('reservation',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','host',1234,"
                    "'{\"preserve\":true}')",(rid+':l',rid,sid,hk,'sha256:'+hashlib.sha256(rid.encode()).hexdigest()))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return conn, rid, config


def invoke(cur,rid,config,plan=PLAN,*,request='routing683-request-0001',dry=False,actor='fixture683'):
    cur.execute('SELECT public.lab_arena_set_judge_provider_routing_v1(%s,%s::jsonb,%s::jsonb,%s,%s,%s)',
                (rid,json.dumps(config),json.dumps(plan),request,actor,dry))
    return cur.fetchone()[0]


def snapshot(cur,rid):
    result={}
    for name in ['lab_arena_rounds','lab_arena_runs','lab_arena_submissions','lab_arena_ledger',
                 'lab_arena_company_judgments','lab_arena_signal_judgments','lab_arena_scoring_supersessions']:
        if name in ['lab_arena_company_judgments','lab_arena_signal_judgments']:
            cur.execute('SELECT to_jsonb(t) FROM public.'+name+' t ORDER BY to_jsonb(t)::text')
        else:
            cur.execute('SELECT to_jsonb(t) FROM public.'+name+' t WHERE round_id=%s ORDER BY to_jsonb(t)::text',(rid,))
        result[name]=cur.fetchall()
    cur.execute('SELECT to_jsonb(t)FROM lab_arena_restart_claim_control t')
    result['guard']=cur.fetchall()
    return result


def test_dry_run_cas_active_work_preserved_history_and_idempotence(database):
    conn,rid,config=seed(database)
    try:
        with conn.cursor() as cur:
            before=snapshot(cur,rid)
            preview=invoke(cur,rid,config,dry=True)
            assert preview['status']=='dry_run' and snapshot(cur,rid)==before
            cur.execute('SELECT public.lab_arena_scoring_policy_history_v1(%s)',(rid,))
            old_history=cur.fetchone()[0]
            applied=invoke(cur,rid,config)
            assert applied['status']=='applied'
            assert applied['configuration_sha256']==preview['configuration_sha256']
            after=snapshot(cur,rid)
            old=before['lab_arena_rounds'][0][0];new=after['lab_arena_rounds'][0][0]
            assert new['configuration_doc']==dict(old['configuration_doc'],judge_provider_routing=PLAN)
            assert {k:v for k,v in new.items()if k not in ['configuration_doc','updated_at']}=={
                k:v for k,v in old.items()if k not in ['configuration_doc','updated_at']}
            assert new['configuration_doc']['scorer_policy']==config['scorer_policy']
            for table in set(before)-{'lab_arena_rounds','lab_arena_scoring_supersessions'}:
                assert after[table]==before[table]
            assert len(after['lab_arena_scoring_supersessions'])==1
            receipt=after['lab_arena_scoring_supersessions'][0][0]
            assert receipt['recovery_doc']['operation']=='round_operational_update_v1'
            assert receipt['recovery_doc']['scoring_generation_floor']==old['scoring_generation_floor']
            cur.execute('SELECT public.lab_arena_scoring_policy_history_v1(%s)',(rid,))
            new_history=cur.fetchone()[0]
            assert new_history['revisions'][:-1]==old_history['revisions']
            assert new_history['revisions'][-1]['generation_floor']==old['scoring_generation_floor']
            assert new_history['revisions'][-1]['old_scorer_policy']==new_history['revisions'][-1]['new_scorer_policy']
            assert invoke(cur,rid,config)['status']=='existing'
            assert snapshot(cur,rid)==after
        conn.commit()
    finally:conn.close()


@pytest.mark.parametrize('plan',[
    {},{MODEL:[]},{MODEL:['fireworks','fireworks']},{MODEL:['fireworks','near-ai/fp8','other']},
    {MODEL:[None]},{MODEL:['unreviewed/provider']},{'unknown/model':['fireworks']},
    {MODEL:{'provider':'fireworks'}},{MODEL:['Fireworks']},
])
def test_invalid_plans_change_nothing(database,plan):
    conn,rid,config=seed(database)
    try:
        with conn.cursor()as cur:before=snapshot(cur,rid)
        conn.commit()
        with pytest.raises(database[0].Error):
            with conn.cursor()as cur:invoke(cur,rid,config,plan)
        conn.rollback()
        with conn.cursor()as cur:assert snapshot(cur,rid)==before
    finally:conn.close()


@pytest.mark.parametrize('fault',['version','parser','semantic','fields','mode','capacity','api','duplicate','unapproved'])
def test_incompatible_frozen_candidates_fail_closed(database,fault):
    def change(config):
        routes=config['judge_reasoning_route_registry']['approved_routes']
        candidate=next(x for x in routes if x['model_id']==MODEL and x['api']=='openrouter.chat'and x['provider']=='fireworks')
        if fault=='version':candidate['model_version']='z-ai/glm-5.3-flash-other'
        elif fault=='parser':candidate['parser_version']='other'
        elif fault=='semantic':candidate['semantic_class']='summary'
        elif fault=='fields':candidate['full_text_fields']=['reasoning']
        elif fault=='mode':candidate['mode']={'enabled':True}
        elif fault=='capacity':candidate['max_output_tokens']=4096
        elif fault=='api':candidate['api']='openrouter.responses'
        elif fault=='duplicate':routes.append(deepcopy(candidate))
        else:candidate['status']='unapproved'
    conn,rid,config=seed(database,change=change)
    try:
        with conn.cursor()as cur:before=snapshot(cur,rid)
        conn.commit()
        with pytest.raises(database[0].Error):
            with conn.cursor()as cur:invoke(cur,rid,config)
        conn.rollback()
        with conn.cursor()as cur:assert snapshot(cur,rid)==before
    finally:conn.close()


def test_two_candidates_preserve_frozen_options_and_cas_rejects_other_changes(database):
    conn,rid,config=seed(database)
    try:
        bad=deepcopy(config);bad['promotion_margin']=99
        with pytest.raises(database[0].Error,match='expected configuration'):
            with conn.cursor()as cur:invoke(cur,rid,bad)
        conn.rollback()
        with conn.cursor()as cur:
            result=invoke(cur,rid,config,{MODEL:['near-ai/fp8','fireworks']})
            assert result['status']=='applied'
        conn.commit()
        with pytest.raises(database[0].Error,match='expected configuration'):
            with conn.cursor()as cur:invoke(cur,rid,config,request='routing683-request-stale')
        conn.rollback()
        with pytest.raises(database[0].Error,match='request conflict'):
            with conn.cursor()as cur:invoke(cur,rid,config,PLAN)
        conn.rollback()
    finally:conn.close()


@pytest.mark.parametrize('role',['anon','authenticated','service_role','lab_arena_service','leadpoet_parity_reader'])
def test_only_owner_can_invoke(database,role):
    conn,rid,config=seed(database)
    try:
        with conn.cursor()as cur:
            cur.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')',(role,RPC))
            assert cur.fetchone()[0] is False
            cur.execute('SET LOCAL ROLE '+role)
            with pytest.raises(database[0].Error,match='permission denied'):
                invoke(cur,rid,config)
        conn.rollback()
    finally:conn.close()


@pytest.mark.parametrize('status',['published','cancelled'])
def test_terminal_round_cannot_be_corrected(database,status):
    conn,rid,config=seed(database,status=status)
    try:
        with pytest.raises(database[0].Error,match='round or expected configuration'):
            with conn.cursor()as cur:invoke(cur,rid,config)
        conn.rollback()
    finally:conn.close()


def test_concurrent_cas_has_one_winner_without_touching_active_jobs(database):
    conn,rid,config=seed(database)
    try:
        with conn.cursor()as cur:before=snapshot(cur,rid)
        conn.commit()
        barrier=Barrier(2)
        def apply(index):
            local=database[0].connect(**database[1])
            try:
                barrier.wait(timeout=5)
                try:
                    with local.cursor()as cur:
                        result=invoke(cur,rid,config,
                            {MODEL:['fireworks']}if index==0 else {MODEL:['near-ai/fp8','fireworks']},
                            request='routing683-concurrent-'+str(index))
                    local.commit();return result['status']
                except database[0].Error as error:
                    local.rollback()
                    assert 'expected configuration' in str(error)
                    return 'stale'
            finally:local.close()
        with ThreadPoolExecutor(max_workers=2)as pool:
            assert sorted(pool.map(apply,[0,1]))==['applied','stale']
        with conn.cursor()as cur:
            after=snapshot(cur,rid)
            assert len(after['lab_arena_scoring_supersessions'])==1
            for table in set(before)-{'lab_arena_rounds','lab_arena_scoring_supersessions'}:
                assert before[table]==after[table]
    finally:conn.close()


def test_old_request_cannot_claim_a_later_correction_is_still_active(database):
    conn,rid,config=seed(database)
    try:
        with conn.cursor()as cur:
            invoke(cur,rid,config)
            current=dict(config,judge_provider_routing=PLAN)
            invoke(cur,rid,current,{MODEL:['near-ai/fp8','fireworks']},request='routing683-new-correction')
        conn.commit()
        with pytest.raises(database[0].Error,match='replay superseded'):
            with conn.cursor()as cur:invoke(cur,rid,config)
        conn.rollback()
    finally:conn.close()


def test_two_candidates_need_distinct_observed_provider_names(database):
    def change(config):
        routes=config['judge_reasoning_route_registry']['approved_routes']
        primary=next(x for x in routes if x['model_id']==MODEL and x['api']=='openrouter.chat'and x['provider']=='near-ai/fp8')
        alternative=next(x for x in routes if x['model_id']==MODEL and x['api']=='openrouter.chat'and x['provider']=='fireworks')
        alternative['provider_name']=primary['provider_name']
    conn,rid,config=seed(database,change=change)
    try:
        with conn.cursor()as cur:before=snapshot(cur,rid)
        conn.commit()
        with pytest.raises(database[0].Error,match='provider observation ambiguous'):
            with conn.cursor()as cur:invoke(cur,rid,config,{MODEL:['near-ai/fp8','fireworks']})
        conn.rollback()
        with conn.cursor()as cur:assert snapshot(cur,rid)==before
    finally:conn.close()


@pytest.mark.parametrize('fault',['source','acl'])
def test_migration_refuses_unknown_function_or_acl_drift(database,fault):
    conn=database[0].connect(**database[1])
    try:
        with conn.cursor()as cur:
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(RPC,))
            original=cur.fetchone()[0]
            if fault=='source':
                cur.execute(original.replace('BEGIN','BEGIN\n -- unreviewed routing change',1))
            else:
                cur.execute('GRANT EXECUTE ON FUNCTION '+RPC+' TO authenticated')
        conn.commit()
        with pytest.raises(database[0].Error,match='existing function differs'):
            with conn.cursor()as cur:cur.execute(SQL.read_text())
        conn.rollback()
    finally:
        conn.rollback()
        with conn.cursor()as cur:
            cur.execute(original)
            cur.execute('REVOKE ALL ON FUNCTION '+RPC+' FROM authenticated')
        conn.commit();conn.close()
