"""Initial keyed readback can precede Deepline execution registration.

Synthetic timing controls reproduce the observed first POST timeout at .044s,
404 lookups, and a running execution becoming visible after 2.55s. No network.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from lab_arena import broker as br,deepline_catalog
from tests.lab_arena.deepline_completed_response_recovery_test import NATIVE,record
from tests.lab_arena.deepline_late_response_recovery_test import ResponseStore
from tests.lab_arena.test_deepline_catalog import row
from tests.lab_arena.test_lab_arena_broker import CONTEXT,DL_KEY,make_broker


class Clock:
    def __init__(self):self.now=0.0
    def monotonic(self):return self.now
    def sleep(self,seconds):self.now+=seconds


def catalog():
    return deepline_catalog.freeze_catalog({'tools':[row('exa_contents',provider='exa',categories=['research'],
      inputSchema={'type':'object','properties':{'ids':{'type':'array','items':{'type':'string'}},'text':{'type':'object'}},'required':['ids']})]})


class RegistrationRaceTransport:
    def __init__(self,clock,*,state='running',register_at=2.55,bill=True,change=None,reply='completed',credits=.02,lookup_delay=.184,observed_status=None):
        self.clock=clock;self.state=state;self.register_at=register_at;self.bill=bill;self.change=change
        self.reply=reply;self.credits=credits;self.lookup_delay=lookup_delay;self.observed_status=observed_status
        self.requests=[];self.posts=[];self.lookups=0;self.key=None
        self.tool='exa_contents';self.provider='exa';self.source_url='https://example.com/'
    def send(self,**request):
        self.requests.append({**deepcopy(request),'started':self.clock.now})
        if request['method']=='POST':
            self.posts.append(deepcopy(request))
            if self.key is None:
                wire=json.loads(request['body']);self.tool=wire['operation'];self.provider=wire['provider']
                self.source_url=wire['payload'].get('url','https://example.com/')
                self.key=request['headers']['idempotency-key'];self.clock.sleep(.044)
                if self.reply=='oversized':return br.ProviderResponse(502,{},b'{}','response_too_large')
                if self.reply=='malformed':return br.ProviderResponse(200,{},b'{not-json')
                raise br.ProviderTransportError('ReadTimeout',observed_status=self.observed_status)
            assert request['headers']==self.posts[0]['headers'] and request['body']==self.posts[0]['body']
            assert request['url']==self.posts[0]['url'] and request['method']==self.posts[0]['method']
            assert self.clock.now+request['timeout_seconds']<=self.posts[0]['timeout_seconds']+.001
            if self.reply=='late_completed':self.clock.sleep(request['timeout_seconds']+.001)
            if self.reply=='timeout_again':raise br.ProviderTransportError('ReadTimeout')
            if self.reply=='malformed_body':return br.ProviderResponse(200,{},b'{bad-json')
            if self.reply=='oversized_resume':return br.ProviderResponse(502,{},b'{}','response_too_large')
            body=record()['response'];body['billing']={'credits_charged':self.credits}
            if self.tool=='firecrawl_scrape':
                body['result']={'data':{'rawHtml':'<html>Recovered company page</html>',
                  'metadata':{'url':self.source_url,'sourceURL':self.source_url,'statusCode':200}}}
            elif self.tool=='generic_http_request':
                from tests.lab_arena.firecrawl_content_fallback_test import _generic
                body=_generic(html='<html>Recovered company page</html>',credits=self.credits)
                body['job_id']=NATIVE
            if not self.bill:body.pop('billing')
            if self.reply=='failed':body.update(status='completed',error={'code':'UPSTREAM_FAILURE'});body.pop('result',None)
            if self.reply=='wrong_native':body['job_id']='public-execute:foreign'
            if self.reply=='secret':body['result']={'text':DL_KEY}
            if self.reply=='still_running':body['status']='running'
            return br.ProviderResponse(500 if self.reply=='failed' else 200,{},json.dumps(body).encode())
        assert request['method']=='GET'
        if '/executions/by-key/' in request['url']:
            self.lookups+=1;self.clock.sleep(min(self.lookup_delay,request['timeout_seconds']))
            if self.clock.now<self.register_at:return br.ProviderResponse(404,{},b'{"error":"not_found"}')
            doc=record();doc.update(toolId=self.tool);doc['executionRecovery'].update(idempotencyKey=self.key,state=self.state)
            if self.state!='completed':doc.pop('response');doc.pop('responseStatus')
            else:
                doc['response']['billing']={'credits_charged':self.credits}
                if self.observed_status is not None and self.observed_status>=400:
                    doc['responseStatus']=self.observed_status
                    doc['response'].update(status='completed',error={'code':'EXPECTED_FAILURE'});doc['response'].pop('result',None)
                if not self.bill:doc['response'].pop('billing')
            headers={'x-deepline-idempotency-supported':'true'}
            if self.change=='key':doc['executionRecovery']['idempotencyKey']='arena:'+'f'*64
            elif self.change=='tool':doc['toolId']='other_tool'
            elif self.change=='native':doc['requestId']='ctx-tool-'+'f'*32
            elif self.change=='credential':doc['echo']=DL_KEY
            elif self.change=='unsupported':headers={}
            elif self.change=='403':return br.ProviderResponse(403,{},b'{}')
            elif self.change=='malformed':return br.ProviderResponse(200,headers,b'{invalid')
            return br.ProviderResponse(200,headers,json.dumps(doc).encode())
        assert '/billing/usage?request_id=' in request['url']
        return br.ProviderResponse(200,{},json.dumps({'recent':{'request_id':NATIVE,'entries':[{
          'id':'exact-one-execution','request_id':NATIVE,'provider':self.provider,'operation':self.tool,
          'credits':self.credits,'delta':-self.credits,'charge_state':'posted' if self.bill else 'temporary_hold',
          'charge_finality':'final' if self.bill else 'pending','billing_mode':'deduct_on_settle','metadata':{}}]}}).encode())


def setup(monkeypatch,**options):
    clock=Clock();monkeypatch.setattr(br,'time',clock)
    transport=RegistrationRaceTransport(clock,**options);store=ResponseStore()
    paid,_,_=make_broker(store=store,transport=transport)
    context=replace(CONTEXT,deepline_catalog=catalog())
    return paid,store,transport,clock,context


def execute(paid,context,timeout_ms=30000):
    return paid.execute(context,operation_id='deepline.execute',parameters={'tool':'exa_contents','payload':{'ids':['https://example.com/'],'text':{'maxCharacters':2000}}},action_sequence=0,timeout_ms=timeout_ms)


def test_real_probe_registration_timing_recovers_initial_call_with_one_paid_identity(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch)
    result=execute(paid,context)
    assert result.status==200 and result.call['transport_recovery']=='deepline_execution_lookup'
    assert len(transport.posts)==2 and transport.lookups>=2 and clock.now<30
    assert len(store.calls)==1 and store.log.count('reserve')==store.log.count('dispatch')==1
    assert result.call['actual_microusd']==2000
    assert store.calls[result.call['call_identity']]['terminal']['call_succeeded'] is True


def test_registered_completed_reply_is_get_only_no_second_post(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,state='completed',register_at=0)
    result=execute(paid,context)
    assert result.status==200 and result.call['actual_microusd']==2000
    assert len(transport.posts)==1 and transport.lookups==1


def test_404_then_completed_reply_is_get_only(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,state='completed')
    result=execute(paid,context)
    assert result.status==200 and len(transport.posts)==1 and transport.lookups>=2


@pytest.mark.parametrize('change',['key','tool','native','credential','unsupported','403','malformed'])
def test_unknown_or_foreign_registration_proof_cannot_resume(monkeypatch,change):
    paid,store,transport,clock,context=setup(monkeypatch,change=change,register_at=0,bill=False)
    result=execute(paid,context)
    assert result.status>=400 and len(transport.posts)==1
    assert store.calls[result.call['call_identity']]['kind']=='uncertain'
    assert 'actual_microusd' not in result.call


@pytest.mark.parametrize('options',[{'register_at':999},{'register_at':0,'lookup_delay':1},{'state':'completed','register_at':0,'lookup_delay':1}])
def test_registration_wait_cannot_create_a_new_source_deadline(monkeypatch,options):
    paid,store,transport,clock,context=setup(monkeypatch,bill=False,**options)
    result=execute(paid,context,timeout_ms=500)
    assert result.status>=400 and len(transport.posts)==1
    # Later GET-only billing may continue; it has no source-resume authority.
    assert all(r['method']=='GET' for r in transport.requests[1:])


@pytest.mark.parametrize('reply',['wrong_native','secret','still_running','timeout_again','malformed_body','oversized_resume'])
def test_foreign_or_incomplete_resumed_body_never_becomes_success(monkeypatch,reply):
    paid,store,transport,clock,context=setup(monkeypatch,reply=reply,bill=False)
    result=execute(paid,context)
    assert result.status>=400 and len(transport.posts)==2
    assert store.calls[result.call['call_identity']]['kind']=='uncertain'
    assert 'actual_microusd' not in result.call


def test_completed_resumed_provider_failure_preserves_failure_and_exact_cost(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,reply='failed')
    result=execute(paid,context)
    assert result.status==502 and len(transport.posts)==2 and result.call['actual_microusd']==2000
    assert store.calls[result.call['call_identity']]['terminal']['call_succeeded'] is False


def test_pending_bill_never_becomes_free_after_successful_resume(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,bill=False)
    result=execute(paid,context)
    assert result.status==200 and len(transport.posts)==2
    assert result.call['outcome']=='uncertain' and 'actual_microusd' not in result.call
    assert store.calls[result.call['call_identity']]['kind']=='uncertain'


def test_larger_final_charge_is_recorded_exactly_not_replaced_by_reservation(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,credits=.03)
    result=execute(paid,context)
    assert result.status==200 and len(transport.posts)==2
    assert result.call['actual_microusd']==store.calls[result.call['call_identity']]['actual']==3000


@pytest.mark.parametrize('reply',['oversized','malformed'])
def test_returned_oversize_or_malformed_response_is_not_replayed_as_transport_loss(monkeypatch,reply):
    paid,store,transport,clock,context=setup(monkeypatch,reply=reply)
    result=execute(paid,context)
    assert result.status>=400 and len(transport.posts)==1


def test_registered_running_reply_resumes_immediately_without_registration_poll(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,register_at=0)
    result=execute(paid,context)
    assert result.status==200 and len(transport.posts)==2 and transport.lookups==1
    assert store.log.count('reserve')==store.log.count('dispatch')==1


@pytest.mark.parametrize('status',[401,402,403,409,422,429,500])
def test_response_recovery_does_not_poll_other_http_errors(monkeypatch,status):
    clock=Clock();monkeypatch.setattr(br,'time',clock);requests=[];key='arena:'+'a'*64
    def send(**request):
        requests.append(request)
        return br.ProviderResponse(status,{},b'{}')
    result=br._deepline_execution_response_readback(transport=SimpleNamespace(send=send),secret=DL_KEY,
      execution_key=key,operation='exa_contents',operation_aliases=(),max_response_bytes=4096,deadline=30,
      resume_request={'method':'POST','url':'https://code.deepline.com/api/v2/integrations/exa_contents/execute',
        'headers':{'idempotency-key':key},'body':b'{}','timeout_seconds':30,'max_response_bytes':4096})
    assert result is None and len(requests)==1 and requests[0]['method']=='GET'


def test_unregistered_execution_get_poll_is_bounded_even_with_a_long_caller_deadline(monkeypatch):
    clock=Clock();monkeypatch.setattr(br,'time',clock);requests=[];key='arena:'+'a'*64
    def send(**request):
        requests.append(request);return br.ProviderResponse(404,{},b'{"error":"not_found"}')
    result=br._deepline_execution_response_readback(transport=SimpleNamespace(send=send),secret=DL_KEY,
      execution_key=key,operation='exa_contents',operation_aliases=(),max_response_bytes=4096,deadline=100,
      resume_request={'method':'POST','url':'https://code.deepline.com/api/v2/integrations/exa_contents/execute',
        'headers':{'idempotency-key':key},'body':b'{}','timeout_seconds':100,'max_response_bytes':4096})
    assert result is None and len(requests)==br._DEEPLINE_BILLING_MAX_ATTEMPTS
    assert all(r['method']=='GET' for r in requests) and clock.now<100


def test_late_resume_body_is_not_positive_evidence_but_its_exact_bill_is_accounted(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,reply='late_completed')
    result=execute(paid,context)
    assert clock.now>30 and result.status==502 and len(transport.posts)==2
    assert result.call['actual_microusd']==2000
    assert store.calls[result.call['call_identity']]['terminal']['call_succeeded'] is False
    assert 'transport_recovery' not in result.call


def test_readback_without_a_resume_request_does_not_poll_a_404(monkeypatch):
    clock=Clock();monkeypatch.setattr(br,'time',clock);requests=[]
    def send(**request):
        requests.append(request);return br.ProviderResponse(404,{},b'{"error":"not_found"}')
    result=br._deepline_execution_response_readback(transport=SimpleNamespace(send=send),secret=DL_KEY,
      execution_key='arena:'+'a'*64,operation='exa_contents',operation_aliases=(),max_response_bytes=4096,deadline=30)
    assert result is None and len(requests)==1 and clock.now==0


@pytest.mark.parametrize('status',[403,429])
def test_partial_original_error_headers_cannot_authorize_running_resume(monkeypatch,status):
    paid,store,transport,clock,context=setup(monkeypatch,register_at=0,observed_status=status,bill=False)
    result=execute(paid,context)
    assert result.status>=400 and len(transport.posts)==1
    assert 'transport_recovery' not in result.call
    assert store.calls[result.call['call_identity']]['kind']=='uncertain'


@pytest.mark.parametrize('status',[403,429])
def test_partial_original_error_headers_allow_matching_completed_get_only(monkeypatch,status):
    paid,store,transport,clock,context=setup(monkeypatch,register_at=0,state='completed',observed_status=status)
    result=execute(paid,context)
    assert result.status>=400 and len(transport.posts)==1
    assert result.call['transport_recovery']=='deepline_execution_lookup'
    assert result.call['actual_microusd']==2000
    assert store.calls[result.call['call_identity']]['terminal']['call_succeeded'] is False


def test_initial_routed_firecrawl_registration_race_preserves_envelope_and_source_deadline(monkeypatch):
    from tests.lab_arena.score_compat_capability_test import firecrawl_catalog
    clock=Clock();monkeypatch.setattr(br,'time',clock)
    transport=RegistrationRaceTransport(clock);store=ResponseStore()
    paid,_,_=make_broker(store=store,transport=transport,credential_for=lambda *_:DL_KEY,
      funding_source_for=lambda *_:'miner_key')
    context=replace(CONTEXT,kind='score',round_id='arena-2026-10-18',deepline_catalog=firecrawl_catalog(parsers={'type':'array'}))
    result=paid.execute(context,operation_id='scrapingdog.scrape',parameters={'url':'https://example.com/about'},action_sequence=0,timeout_ms=30000)
    assert result.status==200 and result.body==b'<html>Recovered company page</html>'
    assert result.call['transport_recovery']=='deepline_execution_lookup'
    assert len(transport.posts)==2 and len(store.calls)==1
    assert transport.posts[0]['max_response_bytes']==transport.posts[1]['max_response_bytes']==br._DEEPLINE_FIRECRAWL_ENVELOPE_MAX_BYTES
    assert transport.posts[0]['headers']==transport.posts[1]['headers']
    assert transport.posts[0]['body']==transport.posts[1]['body']
    assert clock.now+transport.posts[1]['timeout_seconds']<=30
    assert result.call['actual_microusd']==2000


def test_initial_generic_html_fallback_registration_race_keeps_raw_v2_and_prior_charge(monkeypatch):
    from tests.lab_arena.firecrawl_content_fallback_test import _execute,_score_context,URL
    from tests.lab_arena.firecrawl_diagnostic_adapter_test import _envelope,DIAGNOSTIC
    clock=Clock();monkeypatch.setattr(br,'time',clock)
    class FallbackRace(RegistrationRaceTransport):
        def send(self,**request):
            if request['method']=='POST' and json.loads(request['body'])['operation']=='firecrawl_scrape':
                self.requests.append({**deepcopy(request),'started':clock.now})
                return br.ProviderResponse(200,{},json.dumps(_envelope(DIAGNOSTIC)).encode())
            return super().send(**request)
    transport=FallbackRace(clock,credits=0);store=ResponseStore()
    paid,_,_=make_broker(store=store,transport=transport,credential_for=lambda *_:DL_KEY,
      funding_source_for=lambda *_:'miner_key')
    result=_execute(paid)
    assert result.status==200 and result.body==b'<html>Recovered company page</html>'
    assert result.call['transport_recovery']=='deepline_execution_lookup'
    assert len(transport.posts)==2 and len(store.calls)==2
    assert transport.posts[0]['headers']['x-deepline-execute-response-contract']=='raw-v2'
    assert transport.posts[0]['headers']==transport.posts[1]['headers']
    assert transport.posts[0]['body']==transport.posts[1]['body']
    assert all(p['max_response_bytes']==br._DEEPLINE_FIRECRAWL_ENVELOPE_MAX_BYTES for p in transport.posts)
    assert sorted(c['actual'] for c in store.calls.values())==[0,2000]
    assert clock.now+transport.posts[1]['timeout_seconds']<=30
