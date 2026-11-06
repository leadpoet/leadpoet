"""Exact manual sixth keeps company0, five paid failures and the frozen grader."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import re
import uuid

import pytest

from lab_arena import company_judgments, contracts, scoring
from lab_arena.service import ArenaService, ServiceConfig
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import nov03_forward_verifier604_postgres_test as prior
from tests.lab_arena.company_judgments_test import _input, _raw
from tests.lab_arena.lab_arena_pg_harness import current_service_database, SCRIPTS
from tests.lab_arena.legacy_partial_postgres_test import _completion
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey
from tests.lab_arena.test_lab_arena_service_round import FakeChain, keypair, wallet_verify

SQL = SCRIPTS/'612-arena-2026-11-03-mychn-checkpoint-continuation.sql'
ROUND = 'arena-2026-11-03'
SUB = 'sub-0df2d0295c950d907e094e258bf98905'
ASSIGNMENT = f'{ROUND}:{SUB}:2:7:score'
SOURCE = f'{ROUND}:{SUB}:2:7:2'
RUN = ASSIGNMENT+':6'
GRADING = 'sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b'
RUNTIME = 'sha256:1b871e1c80e6d3c4095b14d73166dc5d37dde5f5002aa622583d4350b198fa4b'
REPOSITORY = '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model'
HELPER = 'public.lab_arena__nov03_mychn_judge612_v1(public.lab_arena_rounds,public.lab_arena_runs)'
CLAIM = 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
HISTORY = 'public.lab_arena_scoring_policy_history_v1(text)'
scalar = prior.scalar


@pytest.fixture(scope='module')
def database():
    generator = current_service_database(); pg, dsn = next(generator)
    try:
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            for name in ('495-lab-arena-reviewer-runtime-selection.sql', '596-lab-arena-bounded-judge-attempts.sql', '597-lab-arena-zero-call-judge-attempt-credit.sql'):
                cur.execute((SCRIPTS/name).read_text()); conn.commit()
            for number, marker in ((604, '-- Exact one-time live preimage604'), (607, '-- Exact one-time corrected attempt607'),
                                   (609, 'DO $append609$'), (610, '-- Exact one-time append.'), (611, 'DO $append611$')):
                cur.execute(next(SCRIPTS.glob(str(number)+'-*.sql')).read_text().split(marker)[0]+'\nCOMMIT;')
                conn.commit()
            definitions = {sig: scalar(cur, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,)) for sig in (CLAIM, HISTORY)}
            cur.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
        yield pg, dsn, definitions
    finally: generator.close()


def setup(database):
    pg, dsn, definitions = database
    conn = pg.connect(**dsn)
    with conn.cursor() as cur:
        for definition in definitions.values(): cur.execute(definition)
        # Reset this disposable database only; migration612 never alters ledger data.
        cur.execute('TRUNCATE public.lab_arena_ledger CASCADE')
    conn.commit()
    _old, config, _sources, _judges, objects = prior.seed(conn)
    future = (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
    config.update(scorer_image_digest=GRADING, scorer_image_reference=REPOSITORY+'@'+GRADING,
                  scoring_cap_microusd=50000000)
    config['schedule']['final_scoring_close'] = future
    config['schedule']['publication_deadline'] = future
    icps = [_input()['icp']]*10
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('DELETE FROM public.lab_arena_reviewer_runtimes')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='' WHERE singleton")
        cur.execute("UPDATE public.lab_arena_submissions SET submission_id=%s WHERE submission_id='nov03-miner'", (SUB,))
        cur.execute("UPDATE public.lab_arena_runs SET run_id=%s,assignment_id=%s,submission_id=%s,stage=2,icp_position=7,attempt=2,stage_generation=6,output_ref='mychn-output.json' WHERE submission_id='nov03-miner'", (SOURCE, SOURCE.rsplit(':',1)[0], SUB))
        plan = {'work_items': [{'scored_run_id': SOURCE, 'submission_id': SUB, 'icp_position': 7, 'output_ref': 'mychn-output.json'}]}
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',stage_generation=6,configuration_doc=%s::jsonb,participants=replace(participants::text,'nov03-miner',%s)::jsonb,stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s", (json.dumps(config), SUB, json.dumps(plan), ROUND))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    service = ArenaService.__new__(ArenaService); service._store = store
    objects['mychn-output.json'] = contracts.canonical_json({'schema_version': contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, 'companies': _input()['companies']}).encode()
    service._objects = type('Objects', (), {'get': lambda _self,key: objects[key], 'get_bounded': lambda _self,key,_limit: objects[key]})()
    service._round = lambda rid: store.get_round(rid)
    service.evaluation_icps = lambda _rid: icps
    refs = service._scoring_work_items(store.get_round(ROUND), 2, plan)[0]['company_judgment_refs']
    miss = lambda ref: {**{k: ref[k] for k in ('company_index', 'cache_key', 'company_input_hash')}, 'authority_slot': 0}
    first = {**miss(refs[0]), 'raw_judgment': {**_raw(0), 'company_judgment_policy': 'company_judgment_v1', 'failure_reason': 'required_intent_unverified'}}
    partial = scoring.build_scoring_failure(SOURCE, 'judge_error', company_judgments=[first])
    objects['partial1.json'] = contracts.canonical_json(partial).encode()
    old_runner = hotkey('old-mychn-runner612')
    exclusions = [old_runner]
    first_lease = {'schema_version': company_judgments.LEASE_SCHEMA_VERSION, 'hits': [], 'misses': [miss(ref) for ref in refs]}
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        for attempt in range(1,6):
            result = {'terminal_status': 'judge_error', 'failure_diagnostic': {'error_class': 'judge_error', 'stage': 'scorer', 'reason': 'unknown'}}
            if attempt == 1: result['company_judgment_completion_request_hash'] = 'sha256:'+'f'*64
            claim_doc = {'company_judgment_cache': first_lease, 'runner_authority_exclusions': exclusions}
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,runner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,company_judgment_refs,result_doc,scored_run_id,scoring_generation_floor,lease_generation,claim_request_id,claim_request_hash,claim_response,output_ref) VALUES (%s,%s,%s,%s,%s,%s,2,7,%s,'score','failed',6,'judge_error',%s::jsonb,%s::jsonb,%s,0,1,%s,%s,%s::jsonb,%s)", (ASSIGNMENT+':'+str(attempt), ASSIGNMENT, ROUND, SUB, hotkey('nov03-miner'), old_runner, attempt, json.dumps(refs), json.dumps(result), SOURCE, 'c'*31+str(attempt), 'sha256:'+'d'*64, json.dumps(claim_doc), 'partial1.json' if attempt == 1 else None))
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,run_id,amount_microusd,entry_doc) VALUES ('settlement',%s,%s,%s,100,'{}')", (hotkey('nov03-miner'), ROUND, ASSIGNMENT+':'+str(attempt)))
        evidence = company_judgments.build_evidence_snapshot(new_judgment=first, company_ref=refs[0], source_score_run_id=ASSIGNMENT+':1', source_scored_run_id=SOURCE, source_output_ref='partial1.json', source_output_hash=contracts.document_hash(partial), source_runner_hotkey=old_runner, source_claim_request_id='c'*31+'1', source_claim_request_hash='sha256:'+'d'*64, source_lease_generation=1, source_completion_request_hash='sha256:'+'f'*64, runner_authority_exclusions=exclusions)
        cur.execute('INSERT INTO public.lab_arena_company_judgments(cache_key,authority_slot,scope_doc,company_input_hash,evidence_hash,evidence_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey) VALUES (%s,0,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s)', (refs[0]['cache_key'], json.dumps(refs[0]['scope_doc']), refs[0]['company_input_hash'], contracts.document_hash(evidence), json.dumps(evidence), ASSIGNMENT+':1', SOURCE, old_runner))
        cur.execute("INSERT INTO public.lab_arena_signal_judgments(cache_key,authority_slot,signal_input_hash,scope_doc,judgment_hash,judgment_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey,source_claim_request_hash,source_lease_generation,runner_authority_exclusions) VALUES (%s,0,%s,%s::jsonb,%s,'{}',%s,%s,%s,%s,1,%s::jsonb)", ('sha256:'+'9'*64, 'sha256:'+'8'*64, json.dumps({'cache_key': 'sha256:'+'9'*64, 'signal_input_hash': 'sha256:'+'8'*64}), 'sha256:'+'7'*64, ASSIGNMENT+':1', SOURCE, old_runner, 'sha256:'+'d'*64, json.dumps(exclusions)))
        cur.execute("INSERT INTO public.lab_arena_reviewer_runtimes VALUES (%s,%s,%s,%s,now())", (GRADING, REPOSITORY+'@'+GRADING, RUNTIME, REPOSITORY+'@'+RUNTIME))
        # An unrelated live execution must survive the exact transaction.
        cur.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,lease_expires_at=now()+interval '1 hour',runner_hotkey=%s WHERE run_id=%s", (hotkey('unrelated-live612'), f'{ROUND}:nov03-baseline:1:8:1'))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    with conn.cursor() as cur:
        before = snapshot(cur)
        runs = scalar(cur, 'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE assignment_id=%s', (ASSIGNMENT,))
        source = store.get_run(SOURCE)
        manifest = {'source_sha256': prior.pg_hash(cur, source), 'prior_run_hashes': {rid: prior.pg_hash(cur,row) for rid,row in runs.items()}, 'company_judgment_refs': refs, 'runtime_image_digest': RUNTIME, 'runtime_image_reference': REPOSITORY+'@'+RUNTIME}
        for table in ('lab_arena_company_judgments', 'lab_arena_signal_judgments'):
            manifest[table+'_row_hashes'] = scalar(cur, f"SELECT jsonb_object_agg(cache_key,encode(extensions.digest(to_jsonb(a)::text,'sha256'),'hex')) FROM public.{table} a")
        sql = re.sub(r'\$manifest612\$.*?\$manifest612\$', lambda _match: '$manifest612$'+json.dumps(manifest, separators=(',', ':'), sort_keys=True)+'$manifest612$', SQL.read_text(), flags=re.S)
        sql = sql.replace('9c396e3565a7f9d84d636cc55537ac95968d97770f5de798b1f45b56fbfc8d57', prior.pg_hash(cur,config)).replace('2026-11-03T20:30:02Z', future).replace('2026-11-03T20:30:03Z', future)
    return conn, store, service, objects, sql, before, refs, icps


def snapshot(cur):
    return scalar(cur, "SELECT jsonb_build_object('runs',(SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a),'ledger',(SELECT jsonb_agg(to_jsonb(a) ORDER BY entry_id) FROM public.lab_arena_ledger a),'company',(SELECT jsonb_agg(to_jsonb(a) ORDER BY cache_key) FROM public.lab_arena_company_judgments a),'signal',(SELECT jsonb_agg(to_jsonb(a) ORDER BY cache_key) FROM public.lab_arena_signal_judgments a),'round',(SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id='arena-2026-11-03'),'control',(SELECT to_jsonb(a) FROM public.lab_arena_restart_claim_control a WHERE singleton),'schema_acl',(SELECT nspacl FROM pg_namespace WHERE nspname='public'))")


def claim(conn):
    runner = keypair('mychn612').ss58_address
    with conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,%s::text[],%s,%s,%s,1500)', (ROUND, runner, [runner], uuid.uuid4().hex, 'sha256:'+'b'*64, hash_lease_token('mychn612')))
        lease = cur.fetchone()[0]; cur.execute('RESET ROLE')
    conn.commit(); return lease


def mature(conn):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s", (RUN,))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()


@pytest.mark.parametrize('outcome', ['accepted', 'judge_error', 'expired'])
def test_exact_sixth_claim_keeps_company0_and_four_misses_without_seventh(database, outcome):
    conn, store, service, objects, sql, before, refs, icps = setup(database)
    try:
        with conn.cursor() as cur:
            cur.execute(sql); conn.commit(); cur.execute(sql); conn.commit()
            after = snapshot(cur)
            assert {k:v for k,v in after['runs'].items() if k != RUN} == before['runs']
            assert {k:v for k,v in after.items() if k != 'runs'} == {k:v for k,v in before.items() if k != 'runs'}
            assert scalar(cur, 'SELECT public.lab_arena__score_attempt_count_v1(a) FROM public.lab_arena_runs a WHERE run_id=%s', (RUN,)) == 6
        mature(conn); lease = claim(conn)
        assert lease['status'] == 'leased', lease
        assert lease['run_id'] == RUN
        assert [row['company_index'] for row in lease['company_judgment_cache']['hits']] == [0]
        assert sorted(row['company_index'] for row in lease['company_judgment_cache']['misses']) == [1,2,3,4]
        pinned = store.get_run(RUN)['reviewer_runtime_doc']
        assert pinned['runtime_image_digest'] == RUNTIME
        assert pinned['grading_image_digest'] == GRADING
        with conn.cursor() as cur:
            assert scalar(cur, 'SELECT public.lab_arena__nov03_mychn_judge612_v1(r,a) FROM public.lab_arena_rounds r JOIN public.lab_arena_runs a ON a.round_id=r.round_id WHERE a.run_id=%s', (RUN,)) is True
        if outcome == 'accepted':
            key = keypair('mychn612')
            objects_api = type('Objects', (), {'get': lambda _self,k: objects[k], 'get_bounded': lambda _self,k,_limit: objects[k], 'put': lambda _self,k,v: objects.__setitem__(k,v)})()
            gateway = ArenaService(ServiceConfig(mode='live', store=store, object_store=objects_api, signer=None, chain=FakeChain([key.ss58_address]), verify_signature=wallet_verify, daily_icp_source=lambda **_: {}, banned_hotkeys_source=lambda: (), broker_factory=lambda *_: None))
            gateway.evaluation_icps = service.evaluation_icps
            current = store.get_run(RUN); lease = current['claim_response']; cache = lease['company_judgment_cache']
            new = [{**miss, 'raw_judgment': {**_raw(0), 'company_judgment_policy': 'company_judgment_v1', 'failure_reason': 'required_intent_unverified'}} for miss in cache['misses']]
            raw = {row['company_index']: row['evidence_doc']['raw_judgment'] for row in cache['hits']}
            raw.update({row['company_index']: row['raw_judgment'] for row in new})
            companies = json.loads(objects['mychn-output.json'])['companies']
            breakdowns = scoring.project_company_judgments(companies, [raw[i] for i in range(5)], company_quality=False, scored_indexes=scoring.verify.bucket_skip(icps[7], companies, max_scored_companies=0)[0])
            output = {'schema_version': scoring.SCORING_OUTPUT_SCHEMA_VERSION, 'scored_run_id': SOURCE, 'breakdowns': breakdowns, 'company_judgments': new}
            request = _completion(ROUND, key, lease, 'mychn612', output, terminal='accepted')
            accepted = gateway.handle_complete(request)
            assert accepted['status'] == 'accepted'
            assert gateway.handle_complete(request)['idempotent'] is True
        elif outcome == 'expired':
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s", (RUN,))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit(); store.expire_leases(ROUND)
            assert store.get_run(RUN)['terminal_cause'] == 'lease_expired'
        else:
            assert store.complete_attempt(run_id=RUN, lease_token_hash=hash_lease_token('mychn612'), result={'terminal_status': 'judge_error'}, terminal_cause='judge_error', output_ref=None)['status'] == 'failed'
        store.expire_leases(ROUND)
        with conn.cursor() as cur:
            assert scalar(cur, 'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>6', (ASSIGNMENT,)) == 0
            assert scalar(cur, 'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE run_id<>%s', (RUN,)) == before['runs']
            assert scalar(cur, 'SELECT jsonb_agg(to_jsonb(a) ORDER BY entry_id) FROM public.lab_arena_ledger a') == before['ledger']
            assert scalar(cur, 'SELECT jsonb_agg(to_jsonb(a) ORDER BY cache_key) FROM public.lab_arena_signal_judgments a') == before['signal']
            old_cache = before['company'][0]
            assert scalar(cur, 'SELECT to_jsonb(a) FROM public.lab_arena_company_judgments a WHERE cache_key=%s AND authority_slot=0', (old_cache['cache_key'],)) == old_cache
            cur.execute(sql); conn.commit()
            assert scalar(cur, 'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s', (ASSIGNMENT,)) == 6
    finally: store._transport.close(); conn.close()


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_runs SET output_ref='tampered' WHERE run_id='"+SOURCE+"'",
    "UPDATE public.lab_arena_runs SET result_doc='{}' WHERE run_id='"+ASSIGNMENT+":5'",
    "UPDATE public.lab_arena_company_judgments SET evidence_hash='sha256:'||repeat('0',64)",
    "UPDATE public.lab_arena_signal_judgments SET judgment_hash='sha256:'||repeat('0',64)",
    "UPDATE public.lab_arena_reviewer_runtimes SET runtime_image_digest='sha256:'||repeat('0',64),runtime_image_reference='"+REPOSITORY+"@sha256:'||repeat('0',64)",
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((now()-interval '1 second')::text))",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb((now()-interval '1 second')::text))",
    "UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,lease_expires_at=now()+interval '1 hour' WHERE run_id='"+ASSIGNMENT+":5'",
    "INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,run_id,call_identity,amount_microusd,entry_doc) VALUES ('reservation','"+hotkey('nov03-miner')+"','"+ROUND+"','"+ASSIGNMENT+":5','sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',0,'{}')",
])
def test_changed_source_cache_runtime_clock_or_inflight_calls_refuse_append(database, mutation):
    conn, store, _service, _objects, sql, _before, _refs, _icps = setup(database)
    try:
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica'); cur.execute(mutation); cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        with pytest.raises(Exception):
            with conn.cursor() as cur: cur.execute(sql)
        conn.rollback()
        with conn.cursor() as cur:
            assert scalar(cur, 'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s', (ASSIGNMENT,)) == 5
    finally: store._transport.close(); conn.close()


def test_count_fence_runtime_pin_security_and_history(database):
    conn, store, _service, _objects, sql, _before, _refs, _icps = setup(database)
    try:
        with conn.cursor() as cur:
            cur.execute(sql); conn.commit()
            post = scalar(cur, 'SELECT pg_get_functiondef(%s::regprocedure)', (CLAIM,))
            pre = post.replace('\n      OR public.lab_arena__nov03_mychn_judge612_v1(v_round,runs)', '')
            assert pre != post
            cur.execute(pre)
        conn.commit(); mature(conn)
        assert claim(conn)['status'] != 'leased'  # Exact before-correction witness.
        with conn.cursor() as cur: cur.execute(post)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_reviewer_runtimes SET runtime_image_digest='sha256:'||repeat('0',64),runtime_image_reference=%s||'@sha256:'||repeat('0',64)", (REPOSITORY,))
        conn.commit()
        assert claim(conn)['status'] != 'leased'
        with conn.cursor() as cur:
            cur.execute('UPDATE public.lab_arena_reviewer_runtimes SET runtime_image_digest=%s,runtime_image_reference=%s', (RUNTIME, REPOSITORY+'@'+RUNTIME))
        conn.commit()
        lease = claim(conn); assert lease['status'] == 'leased'
        pinned = store.get_run(RUN)['reviewer_runtime_doc']
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_reviewer_runtimes SET runtime_image_digest='sha256:'||repeat('0',64),runtime_image_reference=%s||'@sha256:'||repeat('0',64)", (REPOSITORY,))
        conn.commit()
        assert store.get_run(RUN)['reviewer_runtime_doc'] == pinned
        with conn.cursor() as cur:
            assert scalar(cur, 'SELECT public.lab_arena__nov03_mychn_judge612_v1(r,a) FROM public.lab_arena_rounds r JOIN public.lab_arena_runs a ON a.round_id=r.round_id WHERE a.run_id=%s', (RUN,)) is True
            history = scalar(cur, 'SELECT public.lab_arena_scoring_policy_history_v1(%s)', (ROUND,))
            item = next(row for row in history['revisions'] if row['recovery']['operation'] == 'nov03_mychn_checkpoint_continuation612_v1')
            assert item['generation_floor'] == 0 and item['receipt_generation'] > 0
            for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
                assert scalar(cur, 'SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, HELPER)) is False
        with pytest.raises(Exception):
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_runs SET reviewer_runtime_doc=jsonb_set(reviewer_runtime_doc,'{runtime_image_digest}',to_jsonb(('sha256:'||repeat('0',64))::text)) WHERE run_id=%s", (RUN,))
        conn.rollback()
    finally: store._transport.close(); conn.close()
