"""Keep paid search evidence without leaking provider-owned navigation credentials."""
from copy import deepcopy
from dataclasses import replace
import base64
import json
from urllib.parse import quote
import pytest
from lab_arena import broker as br
from tests.lab_arena.test_lab_arena_broker import CONTEXT, DOG_KEY, FakeTransport, make_broker


def provider_document(secret=DOG_KEY):
    # Nov14 live provider shape; page navigation and click tracking echo api_key.
    return {'organic_results':[{'title':'Example company','link':'https://example.com/about',
        'snippet':'Company evidence that must survive unchanged.',
        'google_goto_link':'https://api.scrapingdog.com/google?api_key='+quote(secret,safe='')+'&query=example'}],
        'scrapingdog_pagination':{'next':'https://api.scrapingdog.com/google?api_key='+quote(secret,safe='')+'&page=2'},
        'pagination':{'next':'https://www.google.com/search?q=example&start=10'},
        'search_information':{'query_displayed':'example'}}


@pytest.mark.parametrize('kind',['execute','score'])
@pytest.mark.parametrize('operation',['scrapingdog.google','scrapingdog.google_news','scrapingdog.google_jobs'])
def test_search_evidence_survives_private_navigation_with_one_paid_dispatch(kind,operation):
    body=provider_document()
    instance,store,transport=make_broker(transport=FakeTransport([(200,body)]))
    context=replace(CONTEXT,kind=kind)
    args=dict(operation_id=operation,parameters={'query':'example'},action_sequence=0,timeout_ms=5000)
    first=instance.execute(context,**args)
    assert first.status==200
    wanted=deepcopy(body);wanted.pop('scrapingdog_pagination');wanted['organic_results'][0].pop('google_goto_link')
    assert json.loads(first.body)==wanted
    assert first.call['outcome']=='settled' and first.call['actual_microusd']>0
    again=instance.execute(context,**args)
    assert again.body==first.body and len(transport.sent)==1
    assert DOG_KEY not in json.dumps(store.calls)
    terminal=store.calls[first.call['call_identity']]['terminal']
    assert json.loads(base64.b64decode(terminal['body_b64']))==wanted


@pytest.mark.parametrize('fault',['snippet','unknown_field','header','duplicate_field','duplicate_nested','malformed'])
def test_unexpected_credentials_remain_blocked(fault):
    body=provider_document()
    if fault=='snippet':body['organic_results'][0]['snippet']=DOG_KEY
    if fault=='unknown_field':body['different_navigation']=DOG_KEY
    raw=json.dumps(body).encode()
    if fault=='duplicate_field':raw=b'{"hidden":"'+DOG_KEY.encode()+b'","hidden":"safe","scrapingdog_pagination":'+json.dumps(body['scrapingdog_pagination']).encode()+b'}'
    if fault=='duplicate_nested':raw=b'{"organic_results":[{"snippet":"'+DOG_KEY.encode()+b'","snippet":"safe","google_goto_link":"'+DOG_KEY.encode()+b'"}]}'
    if fault=='malformed':raw=b'{'+DOG_KEY.encode()
    class Transport(FakeTransport):
        def send(self,**kwargs):
            r=super().send(**kwargs)
            return replace(r,headers={'x-secret':DOG_KEY})if fault=='header'else r
    instance,store,_=make_broker(transport=Transport([(200,raw)]))
    result=instance.execute(CONTEXT,operation_id='scrapingdog.google',parameters={'query':'example'},action_sequence=0,timeout_ms=5000)
    assert result.status==502
    assert DOG_KEY not in json.dumps(store.calls) and DOG_KEY.encode()not in result.body


def test_navigation_without_credentials_is_byte_exact():
    body=provider_document('public-navigation-token')
    raw=json.dumps(body,indent=2).encode()
    instance,_,_=make_broker(transport=FakeTransport([(200,raw)]))
    result=instance.execute(CONTEXT,operation_id='scrapingdog.google',parameters={'query':'example'},action_sequence=0,timeout_ms=5000)
    assert result.status==200 and result.body==raw


def test_other_operations_do_not_gain_navigation_exception():
    instance,_,_=make_broker(transport=FakeTransport([(200,provider_document())]))
    result=instance.execute(CONTEXT,operation_id='scrapingdog.scrape',parameters={'url':'https://example.com'},action_sequence=0,timeout_ms=5000)
    assert result.status==502

@pytest.mark.parametrize('secret',['literal-test-key','synthetic+/=query-key'])
@pytest.mark.parametrize('escaped',[False,True])
def test_encoded_navigation_key_is_removed_but_source_link_survives(secret,escaped):
    doc=provider_document(secret)
    raw=json.dumps(doc).encode()
    if escaped:raw=raw.replace(b'%',b'\\u0025')
    response=br.ProviderResponse(200,{'content-type':'application/json'},raw)
    safe=br._without_private_search_navigation(response,'scrapingdog.google',secret)
    assert not br._response_contains_credential(safe,secret)
    assert json.loads(safe.body)['organic_results'][0]['link']=='https://example.com/about'
    assert safe.status==response.status and safe.headers==response.headers

@pytest.mark.parametrize('status',[302,401,402,403,429,500])
def test_failed_or_synthetic_response_is_not_projected(status):
    response=br.ProviderResponse(status,{},json.dumps(provider_document()).encode())
    assert br._without_private_search_navigation(response,'scrapingdog.google',DOG_KEY)is response
    synthetic=replace(response,status=200,internal_provenance='credential_echo')
    assert br._without_private_search_navigation(synthetic,'scrapingdog.google',DOG_KEY)is synthetic
