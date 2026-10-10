"""Rejected compatibility replies retain exact GET-only delayed accounting."""
import json
from pathlib import Path

import pytest

from lab_arena import broker as broker_module
from tests.lab_arena import completed_score_host_recovery450_postgres_test as recovery
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import _store
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import _broker, SECRET, FINGERPRINT

database = recovery.database
ROOT = Path(__file__).parents[2]
SQL = ROOT/'scripts/453-lab-arena-deepline-adaptation-cost-reconciliation.sql'
PREHASH = 'e90a382a37756fa64768c66b399124746e3bd621ac03f6e6e27956581bfab137'
POSTHASH = 'd8ec50ba997a2aab723feac8341864e2f6cdcedc34cd03ba74fb275a5cdfa550'
V2_PREHASH = 'f3d2e7a1810d4a77cec1ae32e883ab76cd7fec506d2420b3039893eed535e115'
SIGNATURE = 'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)'
CALL = 'sha256:'+'7'*64
REQUEST = 'ctx-tool-'+CALL[7:39]
EXECUTION = 'arena:'+CALL[7:]
NATIVE = 'public-execute:adaptation453-fixture'
CALL_DOC = {'reason':'settle_failure','call_succeeded':False,
            'failure_stage':'response_adaptation','error_class':'CompatibilityResponseError'}
RESERVATION = {'request_hash':'sha256:'+'d'*64,'tool':'firecrawl_scrape',
               'deepline_request_id':REQUEST,'deepline_execution_key':EXECUTION,
               'credential_fingerprint':FINGERPRINT,'deepline_billing_provider':'firecrawl',
               'deepline_operation_aliases':['firecrawl_scrape','scrape']}


def _definition(cur):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIGNATURE,))
    return cur.fetchone()[0]


def _security(cur):
    cur.execute("SELECT proowner,proacl::text,prosecdef,provolatile,proconfig FROM pg_proc WHERE oid=%s::regprocedure",(SIGNATURE,))
    return cur.fetchone()


@pytest.fixture(scope='module')
def migrated(database):
    recovery.migrated.__wrapped__(database)
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            # The host-recovery harness intentionally omits historical261;
            # install the unchanged production binding before the453 upgrade.
            cur.execute((ROOT/'scripts/261-lab-arena-settlement-failure-deepline-cost-reconciliation.sql').read_text())
            before=_definition(cur)
            security=_security(cur)
            rpc_before={}
            for signature in (recovery.prior.identity.EXPIRY,
                'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)',
                'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)'):
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(signature,))
                rpc_before[signature]=cur.fetchone()[0]
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error,match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREHASH,'0'*64,1))
            cur.execute('ROLLBACK')
            assert _definition(cur)==before
            # A second-seam failure must roll back the already changed helper.
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error,match='V2 preimage differs'):
                cur.execute(SQL.read_text().replace(V2_PREHASH,'0'*64,1))
            cur.execute('ROLLBACK')
            assert _definition(cur)==before
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
            assert _security(cur)==security
            cur.execute("SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),'sha256'),'hex')",(SIGNATURE,))
            assert cur.fetchone()[0]==POSTHASH
            for signature,definition in rpc_before.items():
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(signature,))
                assert cur.fetchone()[0]==definition
            for role in ('anon','authenticated','service_role','lab_arena_service'):
                cur.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')',(role,SIGNATURE))
                assert cur.fetchone()[0] is False
    return True


def _binding(cur,call=None,reservation=None):
    cur.execute('SELECT public.lab_arena__deepline_cost_binding_v1(%s::jsonb,%s::jsonb,%s)',
        (json.dumps({'reason':'worker_reported','call':CALL_DOC if call is None else call}),
         json.dumps(RESERVATION if reservation is None else reservation),CALL))
    return cur.fetchone()[0]


def test_exact_failed_witness_and_all_negative_bindings(database,migrated):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn,conn.cursor() as cur:
        assert _binding(cur)
        full={**CALL_DOC,'deepline_request_id':REQUEST,'deepline_operation':'firecrawl_scrape',
              'credential_fingerprint':FINGERPRINT}
        assert _binding(cur,full)
        for patch in ({'failure_stage':'settlement'},{'error_class':'ValueError'},
            {'reason':'missing_provider_cost'},{'call_succeeded':True},
            {'call_succeeded':'false'},{'call_succeeded':None},
            {'deepline_request_id':REQUEST},{'deepline_job_id':None}):
            assert not _binding(cur,{**CALL_DOC,**patch}),patch
        missing=dict(CALL_DOC);missing.pop('call_succeeded')
        assert not _binding(cur,missing)
        for key,value in (('deepline_request_id','ctx-tool-'+'9'*32),
            ('deepline_operation','exa_contents'),('credential_fingerprint','sha256:'+'9'*64),
            ('deepline_job_id','unbound-native-id')):
            assert not _binding(cur,{**full,key:value}),(key,value)
        for key,value in (('deepline_execution_key','arena:'+'9'*64),
            ('deepline_execution_key',None),('deepline_request_id','ctx-tool-'+'9'*32),
            ('credential_fingerprint','bad'),('tool','Bad Tool')):
            assert not _binding(cur,reservation={**RESERVATION,key:value}),(key,value)
        # Existing legacy settlement-store errors remain eligible even when
        # their older reservation predates execution-key recovery.
        legacy={**CALL_DOC,'failure_stage':'settlement','error_class':'ArenaStoreError'}
        old_reservation={k:v for k,v in RESERVATION.items() if k!='deepline_execution_key'}
        assert _binding(cur,legacy,old_reservation)
        assert _binding(cur,{**legacy,'call_succeeded':True},old_reservation)
        for reason in ('transport_failure','missing_provider_cost'):
            assert _binding(cur,{**full,'reason':reason},old_reservation)


class ExactBillingGet:
    def __init__(self):
        self.sent=[]
    def send(self,**request):
        self.sent.append(request)
        assert request['method']=='GET'  # No execution/provider POST is permitted.
        if '/executions/by-key/' in request['url']:
            document={'toolId':'firecrawl_scrape','requestId':NATIVE,
                'executionRecovery':{'idempotencyKey':EXECUTION,'state':'completed'}}
            return broker_module.ProviderResponse(200,{'x-deepline-idempotency-supported':'true'},json.dumps(document).encode())
        assert NATIVE.replace(':','%3A') in request['url']
        entry={'id':'charge453','request_id':NATIVE,'provider':'firecrawl',
            'operation':'firecrawl_scrape','credits':0.02,'delta':-0.02,
            'charge_state':'posted','charge_finality':'final','metadata':{}}
        return broker_module.ProviderResponse(200,{'content-type':'application/json'},
            json.dumps({'recent':{'request_id':NATIVE,'entries':[entry]}}).encode())


def test_normal_exact_get_settlement_then_authenticated_recovery_keeps_charge(database,migrated):
    psycopg,dsn=database
    prior=recovery.prior
    round_id=prior.recovery.prior.ROUND
    with psycopg.connect(**dsn) as conn:
        lease,_=recovery._seed(conn,count=0)
        assert prior._rpc(conn,"lab_arena_reserve_call(%s,%s,%s,'scrapingdog.scrape','deepline','host',0,%s::jsonb,4500)",
            (lease['run_id'],prior.TOKEN_HASH,CALL,json.dumps(RESERVATION)))['status']=='reserved'
        assert prior._dispatch(conn,lease,CALL)['status']=='dispatched'
        assert prior._rpc(conn,'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
            (lease['run_id'],prior.TOKEN_HASH,CALL,json.dumps(CALL_DOC)))['status']=='uncertain'
        prior._event(conn,lease,'runtime.error',{'status':'abandoned','failure_stage':'runtime',
            'error_class':'RuntimeHostError','runtime_host_reason':'sandbox_launcher_signaled',
            'launch_exit_code':-2,'launch_timed_out':False})
        assert prior._expire(conn)=={'status':'ok','expired':0,'retried':0}
        cost_args=(round_id,lease['submission_id'],lease['icp_position'],1)
        source_cost=prior._rpc(conn,'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)',cost_args)
        store=_store(database)
        try:
            candidates=store.list_deepline_cost_reconciliations(round_id,run_id=lease['run_id'])
            assert len(candidates)==1 and candidates[0]['call_identity']==CALL
            candidate=candidates[0]
            assert prior._rpc(conn,'lab_arena_reconcile_deepline_cost_v1(%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                (round_id,lease['run_id'],CALL,candidate['uncertain_entry_id'],REQUEST,
                 'firecrawl_scrape',FINGERPRINT,2000,'0.02'))['status']=='stale'
            for fingerprint,key in (('sha256:'+'9'*64,EXECUTION),(FINGERPRINT,'arena:'+'9'*64)):
                try:
                    result=prior._rpc(conn,'lab_arena_reconcile_deepline_cost_v2(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                        (round_id,lease['run_id'],CALL,candidate['uncertain_entry_id'],REQUEST,
                         'firecrawl_scrape',fingerprint,2000,'0.02',key,NATIVE))
                    assert result['status']=='stale'
                except psycopg.Error as exc:
                    conn.rollback()
                    assert exc.pgcode=='22023'
            assert store.list_ledger(call_identity=CALL)[-1]['entry_kind']=='uncertain'
            transport=ExactBillingGet()
            broker=_broker(store,transport)
            broker._credential_for=lambda *_:'wrong-test-key'
            assert broker.reconcile_deepline_cost(candidate)['status']=='credential_mismatch'
            assert transport.sent==[]
            broker._credential_for=lambda *_:SECRET
            assert broker.reconcile_deepline_cost({**candidate,'request_id':'ctx-tool-'+'9'*32})['status']=='invalid'
            assert transport.sent==[]
            assert broker.reconcile_deepline_cost(candidate)['status']=='settled'
            assert [r['method'] for r in transport.sent]==['GET','GET']
            settled=store.list_ledger(call_identity=CALL)
            assert [r['entry_kind'] for r in settled]==['reservation','dispatch','uncertain','settlement']
            assert settled[-1]['amount_microusd']==2000
            assert settled[-1]['terminal_response']['call_succeeded'] is False
            assert settled[-1]['entry_doc']['deepline_recovered_request_id']==NATIVE
            # Exact production-shaped451 hold plus owned canonical guard:
            # settlement cannot early-reclaim an unexpired captured lease.
            restart=prior.restart
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='oct10_uniform_search_date_recovery',actor_ref='oct10-uniform-evidence-hold451' WHERE singleton")
            conn.commit()
            generation=restart.guard._acquire(conn,restart._generation(conn))['guard_generation']
            conn.commit()
            drained=restart.guard._quiescence(conn,generation=generation)
            assert (drained['captured_count'],drained['still_leased_count'],drained['expired_receipt_count'])==(1,1,0)
            assert prior._expire(conn)=={'status':'ok','expired':0,'retried':0}
            restart._abort(conn,generation)
            # Aborting the guard alone preserves the independent owner hold.
            assert prior._expire(conn)=={'status':'ok','expired':0,'retried':0}
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='' WHERE singleton")
            conn.commit()
            assert prior._expire(conn)=={'status':'ok','expired':1,'retried':1}
            assert store.list_ledger(call_identity=CALL)==settled
            assert prior._rpc(conn,'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)',cost_args)==source_cost
            assert source_cost['competition_sourcing_microusd']==0
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute('DELETE FROM public.lab_arena_runs WHERE assignment_id<>%s',(lease['assignment_id'],))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            retry=prior.recovery._claim(conn,prior.recovery.prior.RUNNER_B,'c')
            assert (retry['status'],retry['attempt'])==('leased',2)
            result=prior._rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
                (retry['run_id'],'sha256:'+'c'*64,'{"terminal_status":"accepted"}','accepted','arena/test/recovered453.json'))
            assert result['status']=='accepted'
            assert store.list_ledger(call_identity=CALL)==settled
        finally:
            store.close()
