"""A keyed running continuation can conflict while its original job completes.

The real V2 probe observed 404, authenticated running, one identical POST409,
then an authenticated completed GET at 4.07s within its original 30s deadline.
The transient 409 body was not retained; these are synthetic safety controls.
"""
from dataclasses import replace
import json

import pytest
from lab_arena import broker as br
from tests.lab_arena.deepline_registration_race_recovery_test import (
    Clock, RegistrationRaceTransport, catalog, execute,
)
from tests.lab_arena.deepline_late_response_recovery_test import ResponseStore
from tests.lab_arena.deepline_completed_response_recovery_test import NATIVE
from tests.lab_arena.test_lab_arena_broker import CONTEXT, DL_KEY, make_broker


class ConflictTransport(RegistrationRaceTransport):
    def __init__(self, clock, *, complete_at=4.07, after_change=None,
                 conflict_status=409, conflict_body=None, failed=False,
                 late_completed=False, **kwargs):
        super().__init__(clock, **kwargs)
        self.complete_at=complete_at;self.after_change=after_change
        self.conflict_status=conflict_status
        self.conflict_body=conflict_body if conflict_body is not None else {'error':{'code':'synthetic_conflict'}}
        self.failed=failed;self.late_completed=late_completed;self.resumed=False;self.source_lookups=[]

    def send(self, **request):
        if request['method']=='GET' and '/executions/by-key/' in request['url'] and self.resumed:
            self.state='completed' if self.clock.now+self.lookup_delay>=self.complete_at else 'running'
            if self.late_completed and self.state=='completed':
                self.clock.sleep(request['timeout_seconds']+.001)
        resumed=request['method']=='POST' and self.key is not None
        response=super().send(**request)
        if resumed:
            self.resumed=True;self.clock.sleep(.264)
            body=self.conflict_body
            return br.ProviderResponse(self.conflict_status,{},body if isinstance(body,bytes) else json.dumps(body).encode())
        if request['method']=='GET' and '/executions/by-key/' in request['url']:
            self.source_lookups.append((self.clock.now,response))
            if self.resumed and response.status==200:
                doc=json.loads(response.body);headers=dict(response.headers)
                if 'response' in doc and self.tool=='firecrawl_scrape':
                    doc['response']['result']={'data':{'rawHtml':'<html>Recovered company page</html>',
                        'metadata':{'url':self.source_url,'sourceURL':self.source_url,'statusCode':200}}}
                elif 'response' in doc and self.tool=='generic_http_request':
                    from tests.lab_arena.firecrawl_content_fallback_test import _generic
                    doc['response']=_generic(html='<html>Recovered company page</html>',credits=self.credits)
                    doc['response']['job_id']=NATIVE
                if self.after_change=='key':doc['executionRecovery']['idempotencyKey']='arena:'+'f'*64
                elif self.after_change=='native':
                    doc['requestId']='public-execute:99999999-2222-4333-8444-555555555555'
                    if 'response' in doc:doc['response']['job_id']=doc['requestId']
                elif self.after_change=='tool':doc['toolId']='other_tool'
                elif self.after_change=='credential':doc['echo']=DL_KEY
                elif self.after_change=='unsupported':headers={}
                elif self.after_change=='malformed':return br.ProviderResponse(200,headers,b'{invalid')
                elif self.after_change=='oversized':return br.ProviderResponse(502,{},b'{}','response_too_large')
                if self.failed and 'response' in doc:
                    doc['responseStatus']=500
                    doc['response'].update(status='completed',error={'code':'UPSTREAM_FAILURE'})
                    doc['response'].pop('result',None)
                return br.ProviderResponse(response.status,headers,json.dumps(doc).encode())
        return response


def setup(monkeypatch, **kwargs):
    clock=Clock();monkeypatch.setattr(br,'time',clock)
    transport=ConflictTransport(clock,**kwargs);store=ResponseStore()
    paid,_,_=make_broker(store=store,transport=transport)
    return paid,store,transport,clock,replace(CONTEXT,deepline_catalog=catalog())


def test_real_v2_timing_recovers_completed_get_after_one_conflicting_same_key_post(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch)
    result=execute(paid,context)
    assert result.status==200 and result.call['transport_recovery']=='deepline_execution_lookup'
    assert result.call['actual_microusd']==2000 and clock.now<30
    assert len(transport.posts)==2 and transport.posts[0]['body']==transport.posts[1]['body']
    assert transport.posts[0]['headers']==transport.posts[1]['headers']
    assert len(store.calls)==1 and store.log.count('reserve')==store.log.count('dispatch')==1
    before=len(transport.requests);again=execute(paid,context)
    assert again.status==200 and again.body==result.body and len(transport.requests)==before


def test_repeated_running_after_conflict_is_get_only_until_original_execution_completes(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,complete_at=10)
    result=execute(paid,context)
    assert result.status==200 and len(transport.posts)==2 and transport.lookups>=4 and clock.now<30


def test_completed_failure_after_conflict_remains_failed_with_exact_bill(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,failed=True)
    result=execute(paid,context)
    assert result.status==502 and result.call['actual_microusd']==2000
    assert result.call['transport_recovery']=='deepline_execution_lookup'
    assert store.calls[result.call['call_identity']]['terminal']['call_succeeded'] is False
    assert len(transport.posts)==2


@pytest.mark.parametrize('change',['key','native','tool','credential','unsupported','malformed','oversized'])
def test_foreign_or_malformed_completed_get_cannot_replace_original_response(monkeypatch,change):
    paid,store,transport,clock,context=setup(monkeypatch,after_change=change,bill=False)
    result=execute(paid,context)
    assert result.status>=400 and len(transport.posts)==2


def test_conflict_poll_does_not_extend_original_deadline_or_invent_a_free_bill(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,complete_at=999,bill=False)
    result=execute(paid,context)
    assert result.status==502 and result.call.get('actual_microusd') is None
    assert len(transport.posts)==2
    # The separate billing lifecycle may continue; source recovery cannot.
    assert all(r['started']<30 for r in transport.requests if r['method']=='POST')


def test_late_completed_get_is_not_source_success_but_exact_bill_remains_readable(monkeypatch):
    paid,store,transport,clock,context=setup(monkeypatch,late_completed=True)
    result=execute(paid,context,timeout_ms=6000)
    assert result.status==502 and result.call.get('actual_microusd') is None
    assert 'transport_recovery' not in result.call and len(transport.posts)==2
    # A separate billing-only lifecycle may settle the held call later. It
    # does not turn this expired source response into successful evidence.
    transport.late_completed=False
    native_id,cost=br._deepline_exact_readback(transport=transport,secret=DL_KEY,
        request_id=None,execution_key=transport.key,operation='exa_contents',operation_aliases=(),
        provider='exa',reconciliation_deadline=clock.now+120)
    assert native_id and cost.microusd==2000 and result.status==502


@pytest.mark.parametrize('credits',[0,.03])
def test_completed_conflict_recovery_preserves_exact_zero_or_larger_charge(monkeypatch,credits):
    paid,store,transport,clock,context=setup(monkeypatch,credits=credits)
    result=execute(paid,context)
    assert result.status==200 and result.call['actual_microusd']==round(credits*100000)
    assert len(transport.posts)==2


def test_conflict_and_registration_share_one_24_get_attempt_budget(monkeypatch):
    from lab_arena import operations
    clock=Clock();monkeypatch.setattr(br,'time',clock)
    transport=ConflictTransport(clock,complete_at=999)
    outbound=operations.build_outbound_request('deepline.execute',
        {'tool':'exa_contents','payload':{'ids':['https://example.com/'],'text':{'maxCharacters':2000}}},
        deepline_catalog=catalog())
    request={'method':'POST','url':outbound.url,'headers':{'authorization':'Bearer '+DL_KEY,
        'idempotency-key':'arena:'+'a'*64},'body':outbound.body,'timeout_seconds':1000,'max_response_bytes':4096}
    with pytest.raises(br.ProviderTransportError):transport.send(**request)
    result=br._deepline_execution_response_readback(transport=transport,secret=DL_KEY,
        execution_key=request['headers']['idempotency-key'],operation='exa_contents',operation_aliases=(),
        max_response_bytes=4096,resume_request=request,deadline=1000)
    assert result is None and len(transport.posts)==2
    assert transport.lookups==br._DEEPLINE_BILLING_MAX_ATTEMPTS==24


def test_initial_routed_firecrawl_conflict_recovers_with_original_envelope_and_deadline(monkeypatch):
    from tests.lab_arena.score_compat_capability_test import firecrawl_catalog
    clock=Clock();monkeypatch.setattr(br,'time',clock)
    transport=ConflictTransport(clock);store=ResponseStore()
    paid,_,_=make_broker(store=store,transport=transport,credential_for=lambda *_:DL_KEY,
        funding_source_for=lambda *_:'miner_key')
    context=replace(CONTEXT,kind='score',round_id='arena-2026-10-18',
        deepline_catalog=firecrawl_catalog(parsers={'type':'array'}))
    result=paid.execute(context,operation_id='scrapingdog.scrape',parameters={'url':'https://example.com/about'},
        action_sequence=0,timeout_ms=30000)
    assert result.status==200 and result.body==b'<html>Recovered company page</html>'
    assert result.call['actual_microusd']==2000 and len(transport.posts)==2
    assert all(p['max_response_bytes']==br._DEEPLINE_FIRECRAWL_ENVELOPE_MAX_BYTES for p in transport.posts)
    assert clock.now<30 and len(store.calls)==1


def test_initial_raw_html_fallback_conflict_preserves_raw_v2_and_prior_charge(monkeypatch):
    from tests.lab_arena.firecrawl_content_fallback_test import _execute
    from tests.lab_arena.firecrawl_diagnostic_adapter_test import _envelope,DIAGNOSTIC
    clock=Clock();monkeypatch.setattr(br,'time',clock)
    class FallbackConflict(ConflictTransport):
        def send(self,**request):
            if request['method']=='POST' and json.loads(request['body'])['operation']=='firecrawl_scrape':
                return br.ProviderResponse(200,{},json.dumps(_envelope(DIAGNOSTIC)).encode())
            return super().send(**request)
    transport=FallbackConflict(clock,credits=0);store=ResponseStore()
    paid,_,_=make_broker(store=store,transport=transport,credential_for=lambda *_:DL_KEY,
        funding_source_for=lambda *_:'miner_key')
    result=_execute(paid)
    assert result.status==200 and result.body==b'<html>Recovered company page</html>'
    assert len(transport.posts)==2 and len(store.calls)==2 and clock.now<30
    assert transport.posts[0]['headers']['x-deepline-execute-response-contract']=='raw-v2'
    assert sorted(c['actual'] for c in store.calls.values())==[0,2000]


@pytest.mark.parametrize('status',[400,401,403,422,429,500])
def test_other_resumed_http_errors_do_not_start_completion_poll(monkeypatch,status):
    paid,store,transport,clock,context=setup(monkeypatch,conflict_status=status)
    result=execute(paid,context)
    assert result.status==502 and len(transport.posts)==2
    assert 'transport_recovery' not in result.call


@pytest.mark.parametrize('body',[b'{invalid',b'[]',{'echo':DL_KEY},
    {'job_id':'public-execute:99999999-2222-4333-8444-555555555555'},
    {'toolId':'foreign'}, {'executionRecovery':{'idempotencyKey':'arena:'+'f'*64,'state':'running'}}])
def test_malformed_or_foreign_conflict_body_cannot_start_completion_poll(monkeypatch,body):
    paid,store,transport,clock,context=setup(monkeypatch,conflict_body=body)
    result=execute(paid,context)
    assert result.status==502 and len(transport.posts)==2
    assert 'transport_recovery' not in result.call
