"""Keep an accepted independent identity when a raw observed URL is empty.

The public strategy quote and blank-URL shape are from native R5 message8;
message9 rejected that exact positive finding only for company identification.
No model/provider call or historical truth label is replaced here.
"""
import asyncio
import json
import os
from pathlib import Path
from copy import deepcopy

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer
from tests.test_positive_semantic_fit_review import _verdict

HOME = 'https://baringa.com/'
STRATEGY = 'https://www.baringa.com/en/capabilities/strategy/'
LINKEDIN = 'https://www.linkedin.com/company/baringa'
QUOTE = ('We work at board and c-suite level across corporations and public sector organisations. '
         'We will help you set sound, future-proof strategies that balance profit, people and planet. '
         'And accelerate the transformation of your value chain, capabilities and culture.')
HQ = 'Baringa headquarters: London, England, United Kingdom. (harvestapi_get_company locations.headquarter=true)'
IDENTITY = {'normalized_name':'baringa', 'registrable_dns_domain':'baringa.com', 'linkedin_company_slug':'baringa'}
PAGES = {HOME:'Baringa is a global consulting firm.', STRATEGY:'Baringa Strategy. '+QUOTE, LINKEDIN:HQ}
FINALS = {HOME:'https://www.baringa.com/en/', STRATEGY:STRATEGY, LINKEDIN:LINKEDIN}
FINDING = {'target':'required_attribute', 'status':'VERIFIED', 'activity_role':'supplier_operator',
    'evidence_url':STRATEGY, 'evidence_quote':QUOTE, 'reason':'Strategy consulting for corporations from a London base.',
    'supporting_evidence_url_1':LINKEDIN, 'supporting_evidence_quote_1':HQ}


def capture_call(monkeypatch, *, raw='', identity=None, transport='baringa.com', receipt_changes=None,
                 observed_name='Baringa', observed_website='https://www.baringa.com'):
    company=CompanyOutput(company_name='Baringa',company_website=HOME,company_linkedin='',
        industry='Management consulting',employee_count='1,001-5,000',country='United Kingdom',state='England',intent_signals=[{'url':'https://job-boards.eu.greenhouse.io/baringa/jobs/4879336101', 'date':'2026-06-08', 'source':'job_board', 'snippet':'Baringa is hiring a Transformation, Change and Delivery Consultant.', 'description':'Baringa is hiring a Transformation, Change and Delivery Consultant.'}])
    icp=ICPPrompt(icp_id='native-r5',prompt='Management consulting in London',industry='Professional Services',
        sub_industry='Management Consulting',employee_count='1,001-5,000',company_stage='',product_service='Management and digital-transformation consulting for corporate clients',
        geography='United Kingdom, London',country='United Kingdom',
        required_attribute='Sells management, strategy, or digital-transformation consulting engagements to corporate clients from a London base.')
    verdict=_verdict(observed_company_name=observed_name,observed_company_website=observed_website,
        observed_company_linkedin=raw,observed_hq_country='United Kingdom',observed_hq_state='England')
    before=deepcopy(verdict);captured={}
    if receipt_changes is not None:
        receipt=scorer._web_identity_receipt(company,verdict,verified_homepage_identity=IDENTITY,
            verified_homepage_transport_domain='baringa.com',company_quality=True)
        receipt.update(receipt_changes)
        monkeypatch.setattr(scorer,'_web_identity_receipt',lambda *a,**kw:dict(receipt))
    async def investigate(**kwargs):
        captured.update(kwargs)
        return {'claims':{},'failure_reason':''}
    monkeypatch.setattr(scorer,'investigate_company_evidence',investigate)
    asyncio.run(scorer._run_targeted_company_evidence_investigation(
        company=company,icp=icp,verdict=verdict,investigation_targets=('required_attribute',),
        icp_attribute=icp.required_attribute,icp_stage='',verified_identity=IDENTITY if identity is None else identity,
        verified_transport_domain=transport,structured_employee_size_evidence=None,structured_public_company_evidence=None,
        employee_size_conflict=False,company_quality=True,review_positive_semantics=True,
        verified_homepage_pages={u:{'text':t,'final_url':FINALS[u]} for u,t in PAGES.items()}))
    assert verdict==before
    return captured


def validate(captured, *, pages=None, finals=None, finding=None):
    loc=captured['company_locator'];obs=captured['prior_observations'];home=captured['verified_homepage_identity']
    anchor={'submitted_name':loc['name'],'submitted_domain':inv._registrable_domain(loc['website']),
        'submitted_linkedin_slug':inv.linkedin_company_page_slug(loc['linkedin']),
        'observed_name':obs['observed_company_name'],'observed_domain':inv._registrable_domain(obs['observed_company_website']),
        'observed_linkedin_slug':inv.linkedin_company_page_slug(obs['observed_company_linkedin']),
        'verified_name':home.get('normalized_name'),'verified_domain':home.get('registrable_dns_domain'),
        'verified_linkedin_slug':home.get('linkedin_company_slug'),'verified_structured_alias_slug':''}
    return inv._validated_findings({'findings':[deepcopy(FINDING if finding is None else finding)]},
        targets=('required_attribute',),fetched_pages=PAGES if pages is None else pages,
        fetched_final_urls=FINALS if finals is None else finals,first_party_domains={'baringa.com'},
        identity_names={'baringa'},identity_anchor=anchor)['required_attribute']


def test_exact_native_r5_blank_observation_preserves_the_accepted_identity(monkeypatch):
    captured=capture_call(monkeypatch)
    assert captured['company_locator']['linkedin']==''  # Optional submission remains unchanged.
    assert captured['prior_observations']['observed_company_linkedin']==LINKEDIN
    assert not inv._quote_names_company(QUOTE,{'baringa'})
    assert validate(captured)['status']=='VERIFIED'


@pytest.mark.parametrize('raw',['https://www.linkedin.com/company/another','not-a-url',
    'https://www.linkedin.com/in/baringa','https://www.linkedin.com/company/baringa/jobs?invalid=1',
    ' ',None])
def test_nonblank_conflicting_malformed_or_missing_raw_values_are_never_overridden(monkeypatch,raw):
    captured=capture_call(monkeypatch,raw=raw)
    assert captured['prior_observations']['observed_company_linkedin']==raw
    if raw not in ['https://www.linkedin.com/company/baringa/jobs?invalid=1']:
        assert validate(captured)['status']=='UNPROVEN'


@pytest.mark.parametrize('changes',[{'decision':'unavailable'},{'decision':'mismatch'},
    {'evidence_source':'submitted_company'}, {'observed_name':'OtherCo'}, {'submitted_name':'OtherCo'},
    {'observed_domain':'other.example'},{'submitted_domain':'other.example'},
    {'observed_linkedin_slug':'otherco'},{'submitted_linkedin_slug':'otherco'},
    {'observed_linkedin_slug':''},{'observed_linkedin_slug':'baringa/jobs'}])
def test_unaccepted_incomplete_or_conflicting_receipts_cannot_fill_a_blank(monkeypatch,changes):
    captured=capture_call(monkeypatch,receipt_changes=changes)
    assert captured['prior_observations']['observed_company_linkedin']==''
    assert validate(captured)['status']=='UNPROVEN'


@pytest.mark.parametrize('identity',[{}, {**IDENTITY,'normalized_name':'otherco'},
    {**IDENTITY,'registrable_dns_domain':'other.example'},{**IDENTITY,'linkedin_company_slug':'otherco'}])
def test_missing_or_conflicting_independent_homepage_identity_cannot_fill_a_blank(monkeypatch,identity):
    captured=capture_call(monkeypatch,identity=identity,receipt_changes={})
    assert captured['prior_observations']['observed_company_linkedin']==''
    assert validate(captured)['status']=='UNPROVEN'


def test_transport_domain_must_agree_with_the_complete_identity(monkeypatch):
    captured=capture_call(monkeypatch,transport='other.example')
    assert captured['prior_observations']['observed_company_linkedin']==''
    assert validate(captured)['status']=='UNPROVEN'


@pytest.mark.parametrize('fault',['final','source','quote'])
def test_propagation_does_not_relax_existing_fetched_source_validation(monkeypatch,fault):
    captured=capture_call(monkeypatch)
    assert captured['prior_observations']['observed_company_linkedin']==LINKEDIN
    pages=dict(PAGES);finals=dict(FINALS);finding=deepcopy(FINDING)
    if fault=='final':finals[STRATEGY]='https://other.example/strategy'
    elif fault=='source':finding['evidence_url']='https://other.example/strategy';pages[finding['evidence_url']]=QUOTE;finals[finding['evidence_url']]=finding['evidence_url']
    else:finding['evidence_quote']='Invented corporate strategy quote.'
    assert validate(captured,pages=pages,finals=finals,finding=finding)['status']=='UNPROVEN'


def test_existing_matching_observed_profile_is_preserved(monkeypatch):
    captured=capture_call(monkeypatch,raw=LINKEDIN)
    assert captured['prior_observations']['observed_company_linkedin']==LINKEDIN
    assert validate(captured)['status']=='VERIFIED'


def test_exact_saved_r5_required_attribute_finding_through_current_caller(monkeypatch):
    path=os.environ.get('ARENA_NATIVE_R5_IDENTITY_CAPTURE')
    if not path:
        pytest.skip('Exact saved native R5 source/finding replay is opt-in')
    rows=json.loads(Path(path).read_text())
    row=next(r for r in rows if r['capture_file']=='000193.json')
    assert row['round_id']=='arena-2026-10-27-hqprojectr5'
    assert row['prior_identity_observations']['observed_company_linkedin']==''
    captured=capture_call(monkeypatch,identity=row['verified_homepage_identity'],
        observed_name=row['prior_identity_observations']['observed_company_name'],
        observed_website=row['prior_identity_observations']['observed_company_website'])
    assert captured['company_locator']==row['company_locator']
    finding=next(f for submission in row['submissions'] if submission['message_index']==8
                 for f in submission['findings'] if f['target']=='required_attribute')
    pages={p['url']:p['text'] for p in row['prefetched_sources']+row['tool_fetches']}
    # These exact final URL bindings are from settled native entries7003925,
    #7003930 and7003948; no unobserved redirect is manufactured.
    finals={HOME:'https://www.baringa.com/en/',
            'https://www.baringa.com/en/about/':'https://www.baringa.com/en/about/',STRATEGY:STRATEGY}
    assert validate(captured,pages=pages,finals=finals,finding=finding)['status']=='VERIFIED'
    original=deepcopy(captured)
    original['prior_observations']['observed_company_linkedin']=''
    before=validate(original,pages=pages,finals=finals,finding=finding)
    assert before['status']=='UNPROVEN'
    assert before['reason']=='source quote did not identify the investigated company'
