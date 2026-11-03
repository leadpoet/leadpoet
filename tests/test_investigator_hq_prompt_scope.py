"""Scope the literal HQ instruction to the original invocation target set."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer
from qualification.scoring.linkedin_company_size import project_structured_linkedin_company_headquarters
from tests.test_company_evidence_investigator import _finding

SNAPSHOT = json.loads((Path(__file__).parent/'fixtures/oct24_hq_prompt_target_scope.json').read_text())
URL = 'https://acme.example/'
INDUSTRY = 'Acme supplies management consulting services to corporate clients.'
HQ = 'Acme is headquartered in London, United Kingdom.'
IDENTITY = {'normalized_name':'Acme','registrable_dns_domain':'acme.example','linkedin_company_slug':'acme'}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('scope tests must stay offline')
    monkeypatch.setattr(socket.socket,'connect',forbidden)


def response(findings,call_id='submit'):
    return {'choices':[{'message':{'tool_calls':[{'id':call_id,'type':'function','function':{
        'name':'submit_findings','arguments':json.dumps({'findings':findings}),
    }}]}}]}


def tool_targets(payload):
    tool = next(t['function'] for t in payload['tools'] if t['function']['name']=='submit_findings')
    return tool['parameters']['properties']['findings']['items']['properties']['target']['enum']


def invoke(monkeypatch,targets,post,**kwargs):
    monkeypatch.setenv('OPENROUTER_API_KEY','offline-placeholder')
    monkeypatch.setenv('EXA_API_KEY','offline-placeholder')
    monkeypatch.setattr(inv,'_post_json',post)
    monkeypatch.setattr(inv,'_search_web',AsyncMock(return_value={'results':[]}))
    monkeypatch.setattr(inv,'_fetch_page',AsyncMock(side_effect=AssertionError('unexpected fetch')))
    options = dict(
        company_locator={'name':'Acme','website':URL,'linkedin':'https://www.linkedin.com/company/acme'},
        targets=targets,requested_attribute='Supplies management consulting services.',
        requested_industry='Management consulting',requested_geography='United Kingdom, London',
        verified_homepage_identity=IDENTITY,
        prior_observations={'submitted_source_urls':[URL],'observed_company_name':'Acme','observed_company_website':URL,
                            'observed_company_linkedin':'https://www.linkedin.com/company/acme'},
        prefetched_pages={URL:{'final_url':URL,'text':INDUSTRY+' '+HQ}},
    )
    options.update(kwargs)
    return asyncio.run(inv.investigate_company_evidence(**options))


def test_generic_system_prompt_remains_exact_main_snapshot():
    assert len(inv._SYSTEM_PROMPT)==SNAPSHOT['main_system_characters']==31626
    assert hashlib.sha256(inv._SYSTEM_PROMPT.encode()).hexdigest()==SNAPSHOT['main_system_sha256']
    assert inv._SYSTEM_PROMPT.count(SNAPSHOT['old_paragraph'])==1
    assert inv._LITERAL_HEADQUARTERS_INSTRUCTION==SNAPSHOT['new_paragraph']


def test_98_original_target_sets_without_literal_hq_keep_exact_main_system_bytes(monkeypatch):
    counts={'main':0}
    assert len(SNAPSHOT['cases'])==len({r['slot_id'] for r in SNAPSHOT['cases']})==98
    for row in SNAPSHOT['cases']:
        requests=[]
        async def post(_session,_url,*,headers,payload):
            requests.append(deepcopy(payload))
            choice=payload.get('tool_choice')
            if isinstance(choice,dict) and choice['function']['name']=='search_web':
                return 200,{'choices':[{'message':{'tool_calls':[{'id':str(len(requests)),
                    'type':'function','function':{'name':'search_web',
                    'arguments':json.dumps({'query':'Acme management consulting'})}}]}}]}
            return 200,response([_finding(t,status='UNPROVEN',observed_value=None,
                evidence_url='',evidence_quote='') for t in tool_targets(payload)],str(len(requests)))
        # These historical contexts supplied no private literal HQ record.
        result=invoke(monkeypatch,row['requested_targets'],post)
        assert result.get('_completed_submit') and requests,(row['slot_id'],result,len(requests))
        actual=requests[0]['messages'][0]['content']
        counts['main']+=1
        assert actual==inv._SYSTEM_PROMPT
        assert hashlib.sha256(actual.encode()).hexdigest()==SNAPSHOT['main_system_sha256']
    assert counts=={'main':98}


@pytest.mark.parametrize('geography',[False,True])
def test_role_correction_target_shrink_keeps_original_invocation_prompt(monkeypatch,geography):
    requests=[]
    monkeypatch.setattr(inv,'MAX_REASONING_TURNS',1)
    original=('industry','geography') if geography else ('industry','required_attribute')
    async def post(_session,_url,*,headers,payload):
        requests.append(deepcopy(payload));targets=tool_targets(payload)
        industry=_finding('industry',observed_industry='Management consulting',
            observed_subindustry='',activity_role='internal_function' if len(requests)==1 else 'supplier_operator',
            evidence_url=URL,evidence_quote=INDUSTRY)
        findings=[industry]
        if 'geography' in targets:
            findings.append(_finding('geography',observed_country='United Kingdom',observed_state='London',
                                     evidence_url=URL,evidence_quote=HQ))
        if 'required_attribute' in targets:
            findings.append(_finding('required_attribute',activity_role='supplier_operator',
                                     evidence_url=URL,evidence_quote=INDUSTRY))
        return 200,response(findings,str(len(requests)))
    result=invoke(monkeypatch,original,post,positive_semantic_review=True)
    assert len(requests)==2 and result.get('_completed_submit'),(result,[tool_targets(r) for r in requests])
    assert set(tool_targets(requests[0]))==set(original)
    assert tool_targets(requests[1])==['industry']
    feedback=json.loads(requests[1]['messages'][-1]['content'])
    assert feedback['rejected_findings'][0]['target']=='industry'
    assert requests[0]['messages'][0]==requests[1]['messages'][0]
    expected=inv._SYSTEM_PROMPT
    assert requests[0]['messages'][0]['content']==expected
    assert result['claims']['industry']['activity_role']=='supplier_operator'
    assert result['claims']['geography' if geography else 'required_attribute']['status']=='VERIFIED'



def literal_context():
    fixture=json.loads((Path(__file__).parent/'fixtures/baringa_r6_structured_hq.json').read_text())
    profile='https://www.linkedin.com/company/baringa'
    home='https://www.baringa.com/'
    identity={'normalized_name':'Baringa','registrable_dns_domain':'baringa.com','linkedin_company_slug':'baringa'}
    source=project_structured_linkedin_company_headquarters('baringa.com',profile,'Baringa',fixture['payload'])
    pages=scorer._investigator_prefetched_pages(
        {},[profile],verified_identity=identity,
        verified_homepage_pages={profile:{'final_url':profile,'text':'Baringa | LinkedIn'+inv._IDENTITY_LINK_CONTEXT_MARKER+' '+home}},
        structured_profile_headquarters_evidence=source,include_structured_headquarters=True,
    )
    return source,dict(
        company_locator={'name':'Baringa','website':home,'linkedin':profile},
        verified_homepage_identity=identity,
        prior_observations={'submitted_source_urls':[profile],'observed_company_name':'Baringa',
                            'observed_company_website':home,'observed_company_linkedin':profile},
        prefetched_pages=pages,structured_headquarters_evidence=source,
    )


@pytest.mark.parametrize('fault',[None,'no-geography','missing-private','wrong-company','wrong-domain',
    'wrong-slug','invalid-record','not-rendered','wrong-final','missing-final','identity-only',
    'invalid-page','not-submitted'])
def test_literal_instruction_requires_the_actual_bound_quotable_provider_record(monkeypatch,fault):
    source,options=literal_context();profile=source['url'];targets=('geography',)
    if fault=='no-geography':targets=('industry',)
    elif fault=='missing-private':options['structured_headquarters_evidence']=None
    elif fault=='wrong-company':options['verified_homepage_identity']['normalized_name']='Other'
    elif fault=='wrong-domain':options['verified_homepage_identity']['registrable_dns_domain']='other.example'
    elif fault=='wrong-slug':options['verified_homepage_identity']['linkedin_company_slug']='other'
    elif fault=='invalid-record':source['text']=source['text'].replace('"headquarter":true','"headquarter":false')
    elif fault=='not-rendered':options['prefetched_pages'][profile]['text']='Baringa | LinkedIn'
    elif fault=='wrong-final':options['prefetched_pages'][profile]['final_url']='https://www.linkedin.com/company/other'
    elif fault=='missing-final':options['prefetched_pages'][profile].pop('final_url')
    elif fault=='identity-only':options['prefetched_pages'][profile]['text']='Baringa | LinkedIn'+inv._IDENTITY_LINK_CONTEXT_MARKER+source['text']
    elif fault=='invalid-page':options['prefetched_pages'][profile]['text']='x'*(inv.MAX_PAGE_CHARACTERS+1)+source['text']
    elif fault=='not-submitted':options['prior_observations']['submitted_source_urls']=[]
    requests=[]
    async def post(_session,_url,*,headers,payload):
        requests.append(deepcopy(payload))
        choice=payload.get('tool_choice')
        if isinstance(choice,dict) and choice['function']['name']=='search_web':
            return 200,{'choices':[{'message':{'tool_calls':[{'id':str(len(requests)),
                'type':'function','function':{'name':'search_web','arguments':json.dumps({'query':'Baringa headquarters'})}}]}}]}
        return 200,response([_finding(t,status='UNPROVEN',observed_value=None,evidence_url='',evidence_quote='')
                             for t in tool_targets(payload)],str(len(requests)))
    result=invoke(monkeypatch,targets,post,**options)
    assert result.get('_completed_submit') and requests,(fault,result)
    expected=inv._SYSTEM_PROMPT
    if fault is None:
        expected=expected.replace(SNAPSHOT['old_paragraph'],SNAPSHOT['new_paragraph'],1)
        assert hashlib.sha256(expected.encode()).hexdigest()==SNAPSHOT['five_ac_system_sha256']
        assert len(expected)==SNAPSHOT['five_ac_system_characters']
    assert all(request['messages'][0]['content']==expected for request in requests),fault


def test_bound_literal_hq_target_shrink_retains_original_prompt(monkeypatch):
    source,options=literal_context();profile=source['url'];requests=[]
    industry='Baringa supplies management consulting services to corporate clients.'
    industry_url=options['company_locator']['website']
    options['prior_observations']['submitted_source_urls'].append(industry_url)
    options['prefetched_pages'][industry_url]={'final_url':industry_url,'text':industry}
    monkeypatch.setattr(inv,'MAX_REASONING_TURNS',1)
    async def post(_session,_url,*,headers,payload):
        requests.append(deepcopy(payload));targets=tool_targets(payload)
        findings=[_finding('industry',observed_industry='Management consulting',observed_subindustry='',
            activity_role='internal_function' if len(requests)==1 else 'supplier_operator',
            evidence_url=industry_url,evidence_quote=industry)]
        if 'geography' in targets:
            findings.append(_finding('geography',observed_country='United Kingdom',observed_state='London',
                                     evidence_url=profile,evidence_quote=source['text']))
        return 200,response(findings,str(len(requests)))
    result=invoke(monkeypatch,('industry','geography'),post,positive_semantic_review=True,**options)
    assert len(requests)==2 and result.get('_completed_submit'),(result,[tool_targets(r) for r in requests])
    assert tool_targets(requests[1])==['industry']
    expected=inv._SYSTEM_PROMPT.replace(SNAPSHOT['old_paragraph'],SNAPSHOT['new_paragraph'],1)
    assert all(request['messages'][0]['content']==expected for request in requests)
    assert result['claims']['geography']['status']=='VERIFIED'
    assert result['claims']['industry']['activity_role']=='supplier_operator'


def test_bound_hq_instruction_composes_with_enabled_industry_review(monkeypatch):
    source, options = literal_context()
    requests = []
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')

    async def review(*args, **kwargs):
        finding = inv._unproven_findings(('industry',), 'Insufficient source text')['industry']
        finding['_industry_review'] = {
            'decision': 'unavailable', 'missing_requirements': [],
            'reason': 'Insufficient source text', 'source_urls': [], 'research_query': '',
        }
        return finding, '', 1

    async def post(_session, _url, *, headers, payload):
        requests.append(deepcopy(payload))
        if payload.get('tool_choice') == {'type': 'function', 'function': {'name': 'search_web'}}:
            return 200, {'choices': [{'message': {'tool_calls': [{
                'id': str(len(requests)), 'type': 'function', 'function': {
                    'name': 'search_web', 'arguments': json.dumps({'query': 'Baringa supplied consulting'})
                }}]}}]}
        return 200, response([
            _finding('industry', status='UNPROVEN', observed_value=None,
                     evidence_url='', evidence_quote=''),
            _finding('geography', observed_country='United Kingdom', observed_state='London',
                     evidence_url=source['url'], evidence_quote=source['text']),
        ])

    monkeypatch.setattr(inv, '_review_industry_sources', review)
    result = invoke(monkeypatch, ('industry', 'geography'), post, **options)
    assert result.get('_completed_submit') and requests
    assert result['claims']['geography']['status'] == 'VERIFIED'
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    for request in requests:
        from lab_arena.operations import validate_operation_request
        validate_operation_request('openrouter.chat', request)
        prompt = ''.join(m['content'] for m in request['messages'] if m['role'] == 'system')
        assert inv._LITERAL_HEADQUARTERS_INSTRUCTION in prompt
        assert 'Submit industry UNPROVEN with empty citations; judge other targets normally.' in prompt
