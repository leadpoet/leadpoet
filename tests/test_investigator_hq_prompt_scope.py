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
    return asyncio.run(inv.investigate_company_evidence(
        company_locator={'name':'Acme','website':URL,'linkedin':'https://www.linkedin.com/company/acme'},
        targets=targets,requested_attribute='Supplies management consulting services.',
        requested_industry='Management consulting',requested_geography='United Kingdom, London',
        verified_homepage_identity=IDENTITY,
        prior_observations={'submitted_source_urls':[URL],'observed_company_name':'Acme','observed_company_website':URL,
                            'observed_company_linkedin':'https://www.linkedin.com/company/acme'},
        prefetched_pages={URL:{'final_url':URL,'text':INDUSTRY+' '+HQ}},**kwargs,
    ))


def test_generic_system_prompt_remains_exact_main_snapshot():
    assert len(inv._SYSTEM_PROMPT)==SNAPSHOT['main_system_characters']==31626
    assert hashlib.sha256(inv._SYSTEM_PROMPT.encode()).hexdigest()==SNAPSHOT['main_system_sha256']
    assert inv._SYSTEM_PROMPT.count(SNAPSHOT['old_paragraph'])==1
    assert inv._LITERAL_HEADQUARTERS_INSTRUCTION==SNAPSHOT['new_paragraph']


def test_98_original_target_sets_keep90_old_and8_exact5ac_system_bytes(monkeypatch):
    counts={'main':0,'5ac':0}
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
        # A private HQ argument is supplied even for non-geography invocations.
        # Its mere presence must not select the geography instruction.
        result=invoke(monkeypatch,row['requested_targets'],post,
                      structured_headquarters_evidence={'url':'https://www.linkedin.com/company/acme'})
        assert result.get('_completed_submit') and requests,(row['slot_id'],result,len(requests))
        actual=requests[0]['messages'][0]['content']
        if 'geography' in row['requested_targets']:
            counts['5ac']+=1
            assert len(actual)==SNAPSHOT['five_ac_system_characters']==31918
            assert hashlib.sha256(actual.encode()).hexdigest()==SNAPSHOT['five_ac_system_sha256']
            assert actual.replace(SNAPSHOT['new_paragraph'],SNAPSHOT['old_paragraph'],1)==inv._SYSTEM_PROMPT
        else:
            counts['main']+=1
            assert actual==inv._SYSTEM_PROMPT
            assert hashlib.sha256(actual.encode()).hexdigest()==SNAPSHOT['main_system_sha256']
    assert counts=={'main':90,'5ac':8}


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
    expected=inv._SYSTEM_PROMPT.replace(SNAPSHOT['old_paragraph'],SNAPSHOT['new_paragraph'],1) if geography else inv._SYSTEM_PROMPT
    assert requests[0]['messages'][0]['content']==expected
    assert result['claims']['industry']['activity_role']=='supplier_operator'
    assert result['claims']['geography' if geography else 'required_attribute']['status']=='VERIFIED'
