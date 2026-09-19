"""Verbatim migration 306 template against disposable pre-dispatch rounds."""
from __future__ import annotations
import copy, json
from pathlib import Path
import pytest
from lab_arena import contracts
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT=Path(__file__).parents[2]
MIGRATION=ROOT/'scripts/306-arena-2026-09-19-preexecution-source-quota-activation.sql'
ROUND19='arena-2026-09-19'; ROUND20='arena-2026-09-20'; BASELINE='baseline-2026-09-19'
OLD_REF=f'arena/{ROUND19}/sources/{BASELINE}.tar.gz'
NEW_REF=f'arena/{ROUND19}/sources/{BASELINE}-preexecution306.tar.gz'
NEW_SIZE=673_106; NEW_SHA='9ad94f534ffcecd6c43a554d79ec6f30c6f40d8d3892277dcc04a415f9bb5f38'; NEW_COMMIT='66459d982ad43106a9186339cd42558594abb99a'
OLD_QUOTAS=dict(contracts.OPENROUTER_200_CALL_QUOTAS_PER_ICP)
NEW_QUOTAS=dict(contracts.CALL_QUOTAS_PER_ICP)

@pytest.fixture(scope='module')
def database(): yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)

def config(round_id):
 return {'schema_version':contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,'round_id':round_id,'mode':'live',
  'call_quotas':copy.deepcopy(OLD_QUOTAS),'scoring_call_quotas':dict(contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
  'sourcing_cost_eligibility_policy':contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
  'execution_icp_cap_microusd':contracts.PER_ICP_EXECUTION_CAP_MICROUSD,
  'cost_per_company_microusd':contracts.PER_ICP_QUALIFIED_PAIR_CAP_MICROUSD}

def rendered_sql():
 return MIGRATION.read_text()

def rows(cur,table,where,order):
 cur.execute(f"SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY {order}),'[]'::jsonb) FROM public.{table} r WHERE {where}")
 return cur.fetchone()[0]

def seed(conn):
 base_hk=hotkey('sep19-baseline'); miner_hks=[hotkey(f'sep19-miner-{i}') for i in range(4)]
 ids=[BASELINE]+[f'sep19-miner-{i}' for i in range(4)]; hks=[base_hk]+miner_hks
 participants=[]
 with conn.cursor() as cur:
  cur.execute('SET session_replication_role=replica')
  cur.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,public.lab_arena_submissions,public.lab_arena_rounds RESTART IDENTITY CASCADE')
  cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,participants,benchmark_ref,evaluation_date,icp_set_date) VALUES (%s,'stage1',1,1,%s::jsonb,'[]'::jsonb,%s,DATE '2026-09-19',DATE '2026-09-18')",(ROUND19,json.dumps(config(ROUND19)),f'arena/{ROUND19}/benchmark.json'))
  cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc) VALUES (%s,'open',0,0,%s::jsonb)",(ROUND20,json.dumps(config(ROUND20))))
  for i,(sid,hk) in enumerate(zip(ids,hks)):
   ref=OLD_REF if i==0 else f'arena/{ROUND19}/sources/{sid}.tar.gz'; size=673162 if i==0 else 1000+i
   doc={'source_ref':ref,'source_size_bytes':size,'consent':{'public_rerun':True},'is_king':i==0}
   cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref,source_size_bytes,submission_doc,frozen_at) VALUES (%s,%s,%s,'frozen',%s,%s,%s,%s::jsonb,clock_timestamp())",(sid,ROUND19,hk,i==0,ref,size,json.dumps(doc)))
   participants.append({'submission_id':sid,'miner_hotkey':hk,'source_ref':ref,'source_size_bytes':size,'is_king':i==0})
   for pos in range(20):
    stage=1 if pos<10 else 2; assignment=f'{ROUND19}:{sid}:{stage}:{pos}'
    cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation) VALUES (%s,%s,%s,%s,%s,%s,%s,1,'execute','pending',1)",(assignment+':1',assignment,ROUND19,sid,hk,stage,pos))
  cur.execute('UPDATE public.lab_arena_rounds SET participants=%s::jsonb WHERE round_id=%s',(json.dumps(participants),ROUND19))
  for i,(sid,hk) in enumerate(zip(ids[1:],hks[1:])):
   identity=contracts.document_hash(['review',i])
   for kind in ('reservation','dispatch','settlement'):
    cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,call_identity,provider,operation_id,funding_source,amount_microusd) VALUES (%s,%s,%s,%s,%s,'openrouter','openrouter.code_review','miner_key',1)",(kind,hk,ROUND19,sid,identity))
  for i in range(3):
   sid=f'sep20-queued-{i}'; ref=f'arena/{ROUND20}/sources/{sid}.tar.gz'
   cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref,source_size_bytes,submission_doc) VALUES (%s,%s,%s,'uploading',false,%s,1000,%s::jsonb)",(sid,ROUND20,hotkey(sid),ref,json.dumps({'source_ref':ref,'source_size_bytes':1000,'consent':{'public_rerun':True}})))
  cur.execute('SET session_replication_role=origin')
 conn.commit()

def execute(conn):
 with conn.cursor() as cur: cur.execute(rendered_sql())
 conn.commit()

def snapshot(cur):
 return {k:rows(cur,*v) for k,v in {
  'runs':('lab_arena_runs',"round_id='arena-2026-09-19'",'run_id'),
  'ledger':('lab_arena_ledger',"round_id='arena-2026-09-19'",'entry_id'),
  'miners19':('lab_arena_submissions',"round_id='arena-2026-09-19' AND submission_id<>'baseline-2026-09-19'",'submission_id'),
  'submissions20':('lab_arena_submissions',"round_id='arena-2026-09-20'",'submission_id')}.items()}

def test_pending_round_source_and_both_quota_profiles_activate_atomically(database):
 psycopg2,dsn=database; conn=psycopg2.connect(**dsn)
 try:
  seed(conn)
  with conn.cursor() as cur:
   protected_before=snapshot(cur)
   cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(ROUND19,)); round_before=cur.fetchone()[0]
  execute(conn)
  with conn.cursor() as cur:
   assert snapshot(cur)==protected_before
   cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(ROUND19,)); round_after=cur.fetchone()[0]
   cur.execute('SELECT source_ref,source_size_bytes,submission_doc FROM public.lab_arena_submissions WHERE submission_id=%s',(BASELINE,)); ref,size,doc=cur.fetchone()
   cur.execute("SELECT configuration_doc->'call_quotas' FROM public.lab_arena_rounds ORDER BY round_id"); assert [r[0] for r in cur.fetchall()]==[NEW_QUOTAS,NEW_QUOTAS]
   cur.execute("SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgname IN ('lab_arena_rounds_write_once','lab_arena_submissions_frozen') ORDER BY tgname"); assert [r[0] for r in cur.fetchall()]==['O','O']
  assert ref==NEW_REF and size==NEW_SIZE and doc['source_sha256']==NEW_SHA and doc['source_commit']==NEW_COMMIT
  assert doc['preexecution_source_override']['previous_source_sha256']=='780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0'
  assert {k:v for k,v in round_after.items() if k not in {'configuration_doc','participants','updated_at'}}=={k:v for k,v in round_before.items() if k not in {'configuration_doc','participants','updated_at'}}
  execute(conn)
  with conn.cursor() as cur:
   assert snapshot(cur)==protected_before
 finally: conn.close()

@pytest.mark.parametrize('mutation,error',[
 ("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=miner_hotkey,lease_generation=1,lease_token_hash='sha256:'||repeat('a',64),lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-19')",'execution was dispatched'),
 ("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,'sha256:'||repeat('b',64),'deepline','deepline.execute','host',1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-19' LIMIT 1",'ledger'),
 ("UPDATE public.lab_arena_rounds SET benchmark_ref='arena/wrong/benchmark.json' WHERE round_id='arena-2026-09-19'",'source state differs'),
 ("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(jsonb_set(configuration_doc,'{sourcing_cost_eligibility_policy}','null'::jsonb,false),'{execution_icp_cap_microusd}','null'::jsonb,false) WHERE round_id='arena-2026-09-19'",'source state differs'),
 ("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status) SELECT 'sep20-started','sep20-started',round_id,submission_id,miner_hotkey,1,0,1,'execute','pending' FROM public.lab_arena_submissions WHERE round_id='arena-2026-09-20' LIMIT 1",'Sep20 open quota state differs')])
def test_activation_rejects_started_or_changed_state(database,mutation,error):
 psycopg2,dsn=database; conn=psycopg2.connect(**dsn)
 try:
  seed(conn)
  with conn.cursor() as cur:
   cur.execute('SET session_replication_role=replica'); cur.execute(mutation); cur.execute('SET session_replication_role=origin')
  conn.commit()
  with pytest.raises(Exception,match=error): execute(conn)
  conn.rollback()
  with conn.cursor() as cur:
   cur.execute("SELECT configuration_doc->'call_quotas' FROM public.lab_arena_rounds ORDER BY round_id"); assert [r[0] for r in cur.fetchall()]==[OLD_QUOTAS,OLD_QUOTAS]
 finally: conn.close()
