"""Stage-only host prefetch faults do not invalidate resolved sibling findings."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from qualification.scoring import company_evidence_investigator as c, lead_scorer
from tests.test_company_evidence_investigator import _complete_verdict, _completed_unproven_investigation
from tests.test_public_stage_submitted_source_prefetch import _tool_response

SAVED = json.loads((Path(__file__).parent/'fixtures/arena_reviewer/p15_stage_fault_public_findings.json').read_text())


def _native(monkeypatch, *, stage='VERIFIED', discovery=True, fail=True, purpose='stage'):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setenv('EXA_API_KEY', 'test-key')
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', '')
    finding = copy.deepcopy(SAVED['stage_finding'])
    if stage != 'VERIFIED':
        finding.update(status=stage, evidence_url='', evidence_quote='', observed_value='')
    findings = [finding, SAVED['industry_finding']]
    async def post(_session, _url, *, headers, payload):
        forced = payload.get('tool_choice')
        if isinstance(forced, dict) and forced.get('function', {}).get('name') == 'search_web':
            return _tool_response('search_web', {'query':'Grid Dynamics mainframe modernization'})
        return _tool_response('submit_findings', {'findings':findings})
    monkeypatch.setattr(c, '_post_json', post)
    monkeypatch.setattr(c, '_fetch_page', AsyncMock(return_value={'ok':False,'error':'http_502','target':'stage','current_stage_search_succeeded':True} if fail else {
        'ok':True,'url':SAVED['failed_url'],'final_url':SAVED['failed_url'],'text':'A company partnership announcement.'} ))
    monkeypatch.setattr(c, '_fetch_exa_exact_page', AsyncMock(return_value={'ok':False,'error':'contents_crawl_error'}))
    monkeypatch.setattr(c, '_search_web', AsyncMock(return_value={'results':[]})
                        if discovery else AsyncMock(side_effect=RuntimeError('search failed')))
    monkeypatch.setattr(c, '_search_industry_locators', AsyncMock(return_value={'results':[]}))
    prior={'submitted_source_urls':[SAVED['failed_url'],SAVED['stage_finding']['evidence_url']],
        'submitted_source_hints':[{'url':SAVED['failed_url'],'text':'Grid Dynamics Holdings, Inc. (Nasdaq: GDYN) announced a partnership.'}]}
    if purpose != 'unknown':
        prior['untrusted_company_stage_evidence']=[{'url':SAVED['failed_url'],'quote':'Grid Dynamics Holdings, Inc. (Nasdaq: GDYN) announced a partnership.'}]
    if purpose == 'self_declared':
        prior.update(stage_only_source_urls=[SAVED['failed_url']],current_stage_search_succeeded=True,
                     fetch_outcomes=[{'target':'stage','ok':False,'error_class':'http_error'}])
    if purpose == 'shared':
        prior['industry_evidence_url']=SAVED['failed_url']
    return asyncio.run(c.investigate_company_evidence(
        company_locator=SAVED['company_locator'],targets=('stage','industry'),
        requested_stage=SAVED['requested_stage'],requested_industry=SAVED['requested_industry'],
        requested_product_service=SAVED['requested_product_service'],
        prior_observations=prior,
        stage_only_source_urls=(SAVED['failed_url'],) if purpose == 'stage' else (),
        prefetched_pages={SAVED['stage_finding']['evidence_url']:{'final_url':SAVED['stage_finding']['evidence_url'],
            'text':SAVED['stage_finding']['evidence_quote']}},
        verified_homepage_identity={'normalized_name':'griddynamics','registrable_dns_domain':'griddynamics.com'},
    ))


def test_saved_p15_stage_prefetch_fault_does_not_taint_completed_industry_unknown(monkeypatch):
    result = _native(monkeypatch)
    assert result['_completed_submit'] is True
    assert result['failure_reason'] == ''
    assert result['claims']['stage']['status'] == 'VERIFIED'
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert [x['error_class']for x in result['usage']['fetch_outcomes']] == ['http_error','fetch_failed']
    assert result['usage']['fetch_calls'] == 2


@pytest.mark.parametrize('stage,discovery,fail,complete',[
    ('VERIFIED',True,False,True),('UNPROVEN',True,True,False),('VERIFIED',False,True,False),
])
def test_healthy_and_unresolved_stage_paths(monkeypatch,stage,discovery,fail,complete):
    result=_native(monkeypatch,stage=stage,discovery=discovery,fail=fail)
    assert result.get('_completed_submit', False) is complete
    if result.get('claims'):
        assert result['claims']['industry']['status']=='UNPROVEN'


def _outcome(*, target='stage', url=None, error='http_error', ok=False):
    return dict(url_sha256=hashlib.sha256((url or SAVED['failed_url']).encode()).hexdigest(),
                ok=ok,error_class='' if ok else error,loaded_text_length=100 if ok else 0,
                **({'target':target} if target is not None else {}))


@pytest.mark.parametrize('mutation,expected',[
    ('resolved_stage',False),('unknown_purpose',True),('industry_purpose',True),
    ('multi_target',True),('no_discovery',True),('missing_discovery',True),
    ('unproven_stage',True),('contradicted_stage',False),('no_stage_quote',True),('same_source_stage',True),
    ('another_industry_fault',True),('earlier_unscoped_fault_same_source',True),
    ('terminal_404',False),('terminal_block',False),('same_source_recovered',False),
    ('different_source_loaded',True),('empty_recovery',True),('unbound_fault',True),
])
def test_shared_fault_check_preserves_real_faults_and_terminal_source_results(mutation,expected):
    outcomes=[_outcome()]; stage=copy.deepcopy(SAVED['stage_finding']); discovered=True
    if mutation=='unknown_purpose':outcomes[0].pop('target')
    elif mutation=='industry_purpose':outcomes[0]['target']='industry'
    elif mutation=='multi_target':outcomes[0]['target']=['stage','industry']
    elif mutation=='no_discovery':discovered=False
    elif mutation=='missing_discovery':discovered=None
    elif mutation=='unproven_stage':stage['status']='UNPROVEN'
    elif mutation=='contradicted_stage':stage['status']='CONTRADICTED'
    elif mutation=='no_stage_quote':stage['evidence_quote']=''
    elif mutation=='same_source_stage':stage['evidence_url']=SAVED['failed_url']
    elif mutation=='another_industry_fault':outcomes.append(_outcome(target=None,url='https://example.com/industry'))
    elif mutation=='earlier_unscoped_fault_same_source':outcomes.insert(0,_outcome(target=None))
    elif mutation=='terminal_404':outcomes=[_outcome(target=None,error='source_not_found')]
    elif mutation=='terminal_block':outcomes=[_outcome(target=None,error='source_access_denied')]
    elif mutation=='same_source_recovered':outcomes=[_outcome(target=None),_outcome(target=None,ok=True)]
    elif mutation=='different_source_loaded':outcomes=[_outcome(target=None),_outcome(target=None,ok=True,url='https://example.com/other')]
    elif mutation=='empty_recovery':outcomes=[_outcome(target=None),dict(_outcome(target=None,ok=True),loaded_text_length=0)]
    elif mutation=='unbound_fault':outcomes[0]['url_sha256']=''
    assert c._has_unresolved_source_fetch_faults(outcomes,stage_finding=stage,
        current_stage_search_succeeded=discovered) is expected
    # The finality consumer must use the same evidence rule.
    receipt=_completed_unproven_investigation('industry')
    receipt['claims']['stage']=stage
    receipt['usage']={'fetch_outcomes':outcomes,'current_stage_search_succeeded':discovered}
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        _complete_verdict(),('industry',),investigation_receipt=receipt) is (not expected)


@pytest.mark.parametrize('purpose',['shared','unknown','self_declared'])
def test_host_prefetch_with_shared_or_unknown_source_purpose_remains_incomplete(monkeypatch,purpose):
    result=_native(monkeypatch,purpose=purpose)
    assert result['_completed_submit'] is False
    assert result['failure_reason']=='provider_error'
    assert result['claims']['stage']['status']=='VERIFIED'
    assert result['claims']['industry']['status']=='UNPROVEN'
    assert all('target' not in outcome for outcome in result['usage']['fetch_outcomes'])


@pytest.mark.parametrize('overlap',['none','required_attribute','legacy_fit','prior_industry','host_cache'])
def test_actual_company_normalization_and_host_caller_bind_stage_source_purpose(monkeypatch,overlap):
    from gateway.qualification.models import CompanyOutput, ICPPrompt
    from qualification.scoring.competition import _normalized_company, _normalized_icp
    values=_normalized_company(SAVED['company'],integrity_policy=True,company_quality=True)
    assert SAVED['failed_url'] in values['fit_evidence_urls']  # V5 stage-derived list.
    if overlap=='required_attribute':
        values['required_attribute']={'text':'Managed infrastructure','passed':True,
            'evidence_url':SAVED['failed_url'],'evidence_quote':'A supplied service.',
            'explanation':'A submitted attribute claim.'}
    elif overlap=='legacy_fit':
        values['intent_details']=None
    company=CompanyOutput.model_validate(values)
    company=CompanyOutput.model_validate_json(company.model_dump_json())
    captured={}
    async def investigate(**kwargs):
        captured.update(kwargs)
        return {'claims':{},'failure_reason':''}
    monkeypatch.setattr(lead_scorer,'investigate_company_evidence',investigate)
    verdict=_complete_verdict(stage_matches=None,stage_evidence_url='',stage_evidence_quote='',
                              required_attribute_evidence_url='',industry_evidence_url='')
    if overlap=='prior_industry':verdict['industry_evidence_url']=SAVED['failed_url']
    asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=company,icp=ICPPrompt.model_validate(_normalized_icp(SAVED['icp'])),verdict=verdict,
        investigation_targets=('stage','industry','required_attribute'),
        icp_attribute=SAVED['requested_attribute'],icp_stage='public',verified_identity={},
        verified_transport_domain='griddynamics.com',structured_employee_size_evidence=None,
        structured_public_company_evidence=None,employee_size_conflict=False,company_quality=True,
        required_attribute_source_cache={SAVED['failed_url']:{'status':'fetched','final_url':SAVED['failed_url'],
            'text':'Previously fetched attribute source.'}} if overlap=='host_cache' else {},
        review_positive_semantics=True))
    assert (SAVED['failed_url'] in captured['stage_only_source_urls']) is (overlap=='none')
    assert SAVED['failed_url'] in captured['prior_observations']['submitted_source_urls']


def test_host_receipt_fields_survive_actual_caller_and_json_round_trip(monkeypatch):
    from gateway.qualification.models import CompanyOutput, ICPPrompt
    from qualification.scoring.competition import _normalized_company, _normalized_icp
    investigation=_native(monkeypatch)
    company=CompanyOutput.model_validate(_normalized_company(SAVED['company'],integrity_policy=True,company_quality=True))
    async def investigate(**kwargs):
        assert SAVED['failed_url'] in kwargs['stage_only_source_urls']
        return investigation
    monkeypatch.setattr(lead_scorer,'investigate_company_evidence',investigate)
    outputs=asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=company,icp=ICPPrompt.model_validate(_normalized_icp(SAVED['icp'])),
        verdict=_complete_verdict(stage_matches=None,stage_evidence_url='',stage_evidence_quote='',
                                  industry_matches=None,industry_evidence_url='',industry_evidence_quote=''),
        investigation_targets=('stage','industry'),icp_attribute='',icp_stage='public',
        verified_identity={},verified_transport_domain='griddynamics.com',
        structured_employee_size_evidence=None,structured_public_company_evidence=None,
        employee_size_conflict=False,company_quality=True,review_positive_semantics=True))
    receipt=outputs[1].details['investigation_receipt']
    restored=json.loads(json.dumps(receipt))
    assert restored==receipt
    assert restored['completed_submitted_findings'] is True
    assert restored['usage']['current_stage_search_succeeded'] is True
    assert all(x['target']=='stage' for x in restored['usage']['fetch_outcomes'])
    assert restored['claims']['industry']['status']=='UNPROVEN'
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        _complete_verdict(),('industry',),investigation_receipt=restored)
    assert outputs[1].decision!='match'


@pytest.mark.parametrize('integrity_policy',[False,True])
@pytest.mark.parametrize('company_quality',[False,True])
@pytest.mark.parametrize('extra_fit_field',['fit_evidence_urls','fit_summary'])
def test_v5_ingress_rejects_explicit_legacy_fit_fields_even_with_stage_urls(
    integrity_policy,company_quality,extra_fit_field,
):
    from pydantic import ValidationError
    from qualification.competition_models import CompetitionCompanyV5
    from qualification.scoring.competition import _normalized_company, CompetitionScorerInputError
    raw=copy.deepcopy(SAVED['company'])
    derived=[x['url']for x in raw['company_stage_evidence']]
    healthy=_normalized_company(raw,integrity_policy=integrity_policy,company_quality=company_quality)
    assert healthy['intent_details'] is not None
    assert healthy['fit_evidence_urls']==derived
    raw[extra_fit_field]=derived if extra_fit_field=='fit_evidence_urls' else 'Explicit general fit claim.'
    with pytest.raises(ValidationError,match='extra_forbidden'):
        CompetitionCompanyV5.model_validate(raw)
    with pytest.raises(CompetitionScorerInputError,match='competition output schema'):
        _normalized_company(raw,integrity_policy=integrity_policy,company_quality=company_quality)


def test_legacy_ingress_cannot_smuggle_intent_details_into_explicit_fit_provenance():
    from qualification.scoring.competition import _normalized_company, CompetitionScorerInputError
    raw=copy.deepcopy(SAVED['company'])
    raw.pop('intent_details')
    raw.pop('company_stage_evidence')
    raw['fit_summary']='A company fit summary.'
    raw['fit_evidence_urls']=[SAVED['failed_url']]
    for signal in raw['intent_signals']:
        signal['snippet']='A supplied evidence excerpt.'
        signal['why_now']='A recent company announcement.'
    legacy=_normalized_company(raw,integrity_policy=True,company_quality=True)
    assert 'intent_details' not in legacy
    assert legacy['fit_evidence_urls']==[SAVED['failed_url']]
    raw['intent_details']=SAVED['company']['intent_details']
    with pytest.raises(CompetitionScorerInputError):
        _normalized_company(raw,integrity_policy=True,company_quality=True)
