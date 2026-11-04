"""Saved Fogelman identity uses a fetched numeric profile, never a guessed alias."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qualification.scoring import company_verification as homepage, lead_scorer as scorer, linkedin_company_size as linkedin
from qualification.scoring.company_fit_decision import company_fit_match
from tests.test_fit_evidence_url_compat import _company, _icp

DATA = json.loads((Path(__file__).parent/'lab_arena/fixtures/nov3_fogelman_identity.json').read_text())


def saved_homepage(monkeypatch):
    async def fetch(*args, **kwargs):
        return 200, DATA['homepage_url'], DATA['homepage_html']
    monkeypatch.setattr(homepage,'_fetch_bounded_html',fetch)
    c=DATA['company']
    return asyncio.run(homepage.verify_company_exists(c['company_name'],c['company_website'],
        company_linkedin=c['company_linkedin'], require_https_transport=True))


def web_receipt():
    return scorer._web_identity_receipt(SimpleNamespace(**DATA['company']),DATA['sonar'],
        verified_homepage_transport_domain='fogelman.com')


def proof(payload=None):
    return linkedin.project_structured_linkedin_company_identity(
        'fogelman.com','https://www.linkedin.com/company/455285',payload or DATA['structured'],
        observed_profile_url='https://www.linkedin.com/company/fogelman-properties',
        expected_company_name='Fogelman Properties')


def test_saved_numeric_profile_resolves_only_the_proved_homepage_conflict(monkeypatch):
    home=saved_homepage(monkeypatch);web=web_receipt()
    assert home.decision=='unavailable'
    assert home.details['observed_homepage_organization_identity']=={
        'normalized_name':'fogelman','registrable_dns_domain':'fogelman.com','linkedin_company_slug':'455285'}
    assert scorer._unresolved_homepage_web_identity_conflict(home,web)
    lookup=scorer._conflicting_homepage_numeric_profile_lookup(home,web)
    assert lookup['requested_profile_url']=='https://www.linkedin.com/company/455285'
    assert lookup['observed_profile_url']=='https://www.linkedin.com/company/fogelman-properties'
    assert proof()['company_id']=='455285'
    assert not scorer._unresolved_homepage_web_identity_conflict(home,{**web,'structured_profile_identity':proof()})


@pytest.mark.parametrize('field,value',[
    ('company_id','999999'),('name','Different Company'),('website','https://unrelated.example/'),
    ('url','https://www.linkedin.com/company/unrelated'),
    ('requested_url','https://www.linkedin.com/company/999999'),
    ('provider','untrusted'),('source_field','submitted_name'),
])
def test_wrong_numeric_profile_cannot_resolve_identity(monkeypatch,field,value):
    home=saved_homepage(monkeypatch);web=web_receipt()
    assert scorer._unresolved_homepage_web_identity_conflict(home,{
        **web,'structured_profile_identity':{**proof(),field:value}})


@pytest.mark.parametrize('field,value',[
    ('observed_domain','unrelated.example'),('submitted_domain','unrelated.example'),
    ('observed_name','differentcompany'),('submitted_linkedin_slug','wrong'),
])
def test_unbound_web_claim_cannot_use_numeric_lookup(monkeypatch,field,value):
    home=saved_homepage(monkeypatch);web={**web_receipt(),field:value}
    assert not scorer._conflicting_homepage_numeric_profile_lookup(home,web)


def test_real_web_path_reuses_one_numeric_structured_request(monkeypatch):
    home=saved_homepage(monkeypatch);requests=[]
    monkeypatch.setenv('OPENROUTER_API_KEY','local-test-placeholder')
    async def request(**kwargs):return deepcopy(DATA['sonar']),''
    async def attribute(verdict,**kwargs):return dict(verdict),None
    async def employee(verdict,**kwargs):return dict(verdict)
    async def structured(domain,url,**kwargs):
        requests.append((domain,url))
        identity=linkedin.project_structured_linkedin_company_identity(domain,url,DATA['structured'],
            observed_profile_url=kwargs['company_identity_observed_profile_url'],
            expected_company_name=kwargs['expected_company_name'])
        kwargs['company_identity_evidence'].update(identity)
        return None
    async def refresh(verdict,*args,**kwargs):return dict(verdict)
    monkeypatch.setattr(scorer,'_request_company_reverify_json',request)
    monkeypatch.setattr(scorer,'_ground_required_attribute_evidence',attribute)
    monkeypatch.setattr(scorer,'_ground_employee_size_observation',employee)
    monkeypatch.setattr(scorer,'fetch_structured_linkedin_company_size',structured)
    monkeypatch.setattr(scorer,'_refresh_linkedin_employee_size_observation',refresh)
    company=_company(**{k:DATA['company'][k] for k in ('company_name','company_website','company_linkedin','employee_count','country')},company_stage='')
    icp=_icp().model_copy(update={'company_stage':'','required_attribute':'','industry':'Real Estate','sub_industry':'Multifamily investment and management','country':'United States','geography':'United States, South','employee_count':'501-1,000'})
    result=asyncio.run(scorer._llm_reverify_company(company,icp,require_company_fit_dimensions=True,
        verified_homepage_identity=home))
    assert requests==[('fogelman.com','https://www.linkedin.com/company/455285')]
    assert result.details['identity_receipt']['structured_profile_identity']==proof()
    assert not scorer._unresolved_homepage_web_identity_conflict(home,result.details['identity_receipt'])


@pytest.mark.parametrize('proof_update,industry_decision,expected', [
    ({}, 'match', 'match'),
    ({'company_id': '999999'}, 'match', 'unavailable'),
    ({'website': 'https://unrelated.example/'}, 'match', 'unavailable'),
    ({}, 'mismatch', 'mismatch'),
    ({}, 'unavailable', 'unavailable'),
])
def test_final_company_fit_assembly_preserves_all_other_dimension_decisions(
    monkeypatch, proof_update, industry_decision, expected,
):
    home=saved_homepage(monkeypatch);web=web_receipt()
    async def identity(*args,**kwargs):return home
    async def precheck(*args,**kwargs):return company_fit_match()
    async def completed(*args,**kwargs):
        # Control all other dimensions; the saved identity path remains real.
        return company_fit_match(details={
            'identity_decision':'match','identity_receipt':{**web,'structured_profile_identity':{**proof(),**proof_update}},
            'dimension_decisions':{**dict.fromkeys(('employee_size','industry','geography','stage'),'match'),
                                   'industry':industry_decision},
            'dimension_evidence':{name:{'url':DATA['homepage_url'],'quote':'Independent dimension control.'}
                for name in ('employee_size','industry','geography')},
            'required_attribute_decision':'match',
        })
    monkeypatch.setattr(scorer,'verify_company_exists',identity)
    monkeypatch.setattr(scorer,'run_company_zero_checks',precheck)
    monkeypatch.setattr(scorer,'_llm_reverify_company',completed)
    c=_company(**{k:DATA['company'][k] for k in ('company_name','company_website','company_linkedin','employee_count','country')},company_stage='')
    icp=_icp().model_copy(update={'company_stage':'','required_attribute':'','employee_count':'501-1,000'})
    result=asyncio.run(scorer._verify_company_fit(c,icp,0,1,set(),require_https_transport=True))
    assert result.decision==expected
    assert result.details['company_fit_dimensions']['industry']==industry_decision
    assert result.details['company_fit_dimensions']['identity']==('unavailable' if proof_update else 'match')
