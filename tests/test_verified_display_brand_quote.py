"""Public Captify citation and adversarial attribution controls; no provider calls."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
from qualification.scoring import company_evidence_investigator as inv

URL = 'https://captifytechnologies.com/latest/news/sense/'
QUOTE = ('Captify, the global leader in Search Intelligence, today announced the launch of its proprietary '
         'search-powered self-service insights, audience planning and activation platform – Sense – which '
         'is in part developed in partnership with UM, the global media agency network of IPG Mediabrands.')
SMART = ('Through Captify’s Programmatic Search Intelligence (PSI) product, programmatic buyers can activate '
         'dynamic audiences powered by real-time search data at scale by creating custom cross-publisher '
         'deals within Smart’s premium, safe inventory ecosystem')
ANCHOR = dict(submitted_name='Captify® (part of Verve)', verified_name='Captify® (part of Verve)',
              observed_name='Captify Technologies Limited', submitted_domain='captifytechnologies.com',
              observed_domain='captifytechnologies.com', verified_domain='captifytechnologies.com',
              submitted_linkedin_slug='captify', verified_linkedin_slug='captify', observed_linkedin_slug='')
NAMES = {'captifypartofverve', 'captifytechnologieslimited'}


def validate(*, quote=QUOTE, anchor=None, target='required_attribute', url=URL, final=URL,
             text=None, status='VERIFIED', role='supplier_operator'):
    return inv._validated_findings({'findings':[dict(target=target, status=status,
        activity_role=role, observed_industry='Advertising technology', evidence_url=url, evidence_quote=quote,
        observed_value='Supplies programmatic advertising technology', reason='Named first-party product.') ]},
        targets=(target,), fetched_pages={url:quote if text is None else text},
        fetched_final_urls={url:final}, first_party_domains={'captifytechnologies.com'},
        identity_names=NAMES, identity_anchor=ANCHOR if anchor is None else anchor)[target]


@pytest.mark.parametrize('quote', [QUOTE, SMART])
def test_exact_brand_quotes_are_not_rejected_for_the_ownership_display_suffix(quote):
    assert not inv._quote_names_company(quote, NAMES)
    assert not inv._complete_verified_first_party_identity(URL, {'captifytechnologies.com'}, ANCHOR)
    result=validate(quote=quote)
    assert result['status']=='VERIFIED'
    assert result['evidence_quote']==quote


@pytest.mark.parametrize('quote', [
    'Verve supplies a programmatic advertising platform.',
    'Captify’s parent supplies a programmatic advertising platform.',
    'Captify partner supplies a programmatic advertising platform.',
    'Captify’s subsidiary supplies a programmatic advertising platform.',
    'Captifyx supplies a programmatic advertising platform.',
    'Captifyé supplies a platform.', 'Captify-Other supplies a platform.',
    'Captify_parent supplies a platform.', 'Captify.com supplies a platform.',
    'Captify’s ultimate parent supplies a platform.',
    'Captify’s U.S. parent supplies a platform.',
    'Captify’s partners supply a platform.',
    'Captify’s subsidiaries supply a platform.',
    'Captify’s advertising partner supplies a platform.',
    'Captify, whose parent owns the platform, announced a release.',
    'The parent supplies a platform for Captify.',
])
def test_parent_related_company_and_name_prefix_are_not_brand_evidence(quote):
    assert validate(quote=quote)['status']=='UNPROVEN'


@pytest.mark.parametrize('changes', [
    {'submitted_name':'Captify'}, {'verified_name':'Captify (formerly Verve)'},
    {'verified_name':'Captify (part of Verve) (US)'}, {'verified_name':''},
    {'submitted_linkedin_slug':''}, {'submitted_linkedin_slug':'other'},
    {'verified_linkedin_slug':'captify-other'}, {'verified_linkedin_slug':'captify/jobs'},
    {'observed_linkedin_slug':'other'}, {'observed_linkedin_slug':' '},
    {'submitted_domain':'other.example'}, {'verified_domain':'other.example'},
    {'observed_domain':'other.example'}, {'observed_domain':''},
])
def test_missing_or_conflicting_independent_anchors_stay_unproven(changes):
    assert validate(anchor={**ANCHOR, **changes})['status']=='UNPROVEN'


@pytest.mark.parametrize('url,final', [(URL,'https://other.example/news'),
    ('https://other.example/news',URL),(URL,'')])
def test_requested_and_final_urls_both_remain_company_bound(url,final):
    assert validate(url=url,final=final)['status']=='UNPROVEN'


def test_quote_must_still_occur_in_loaded_text():
    assert validate(text='Captify page without the claim')['status']=='UNPROVEN'


@pytest.mark.parametrize('target', ['industry','required_attribute'])
def test_only_activity_targets_gain_named_brand_attribution(target):
    assert validate(target=target)['status']=='VERIFIED'
    assert validate(target=target,status='CONTRADICTED')['status']=='UNPROVEN'


def test_industry_still_requires_supplier_operator():
    assert validate(target='industry',role='customer_user')['status']=='UNPROVEN'


def test_exact_saved_production_request_optional():
    path=os.environ.get('ARENA_CAPTIFY_ATTRIBUTION_REPLAY')
    if not path:pytest.skip('Exact saved private production request is opt-in')
    d=json.loads(Path(path).read_text()); finding=deepcopy(d['second_finding'])
    result=inv._validated_findings({'findings':[finding]},targets=('required_attribute',),
        fetched_pages=d['pages'],fetched_final_urls={u:u for u in d['pages']},
        first_party_domains={'captifytechnologies.com'},identity_names=NAMES,identity_anchor=ANCHOR)
    assert result['required_attribute']['status']=='UNPROVEN'
    assert result['required_attribute']['reason']==inv.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON
    # The optional second, anonymous quote is from another page. Retain the
    # original product quote and named commercial support, then let the
    # ordinary correction submit only that independently sufficient evidence.
    finding['supporting_evidence_url_2']=''
    finding['supporting_evidence_quote_2']=''
    result=inv._validated_findings({'findings':[finding]},targets=('required_attribute',),
        fetched_pages=d['pages'],fetched_final_urls={u:u for u in d['pages']},
        first_party_domains={'captifytechnologies.com'},identity_names=NAMES,identity_anchor=ANCHOR)
    assert result['required_attribute']['status']=='VERIFIED'
    assert result['required_attribute']['evidence_quote']==finding['evidence_quote']
    assert len(result['required_attribute']['supporting_evidence'])==1



def test_bounded_correction_completes_with_named_primary_and_named_support(monkeypatch):
    import asyncio
    from tests.test_company_evidence_judgment_reserve import _Clock, _response
    monkeypatch.setenv('OPENROUTER_API_KEY','test-key')
    monkeypatch.setenv('EXA_API_KEY','test-key')
    clock=_Clock();monkeypatch.setattr(inv,'time',clock)
    support='Captify will be using Smart’s platform on behalf of their advertiser, agency and trading desk clients.'
    other=URL+'other'
    finding=dict(target='required_attribute',status='VERIFIED',activity_role='supplier_operator',
        evidence_url=URL,evidence_quote=QUOTE,observed_value='Supplies programmatic advertising technology',
        reason='Named owned product available to brands and agencies.',
        supporting_evidence_url_1=URL,supporting_evidence_quote_1=support,
        supporting_evidence_url_2=other,supporting_evidence_quote_2='We serve global brands.')
    requests=[]
    async def post(_session,_url,*,headers,payload):
        requests.append(deepcopy(payload))
        clock.now=inv.ADMISSION_DEADLINE_SECONDS-inv.JUDGMENT_ADMISSION_RESERVE_SECONDS
        raw=deepcopy(finding)
        if len(requests)>1:
            feedback=json.loads(payload['messages'][-1]['content'])
            assert feedback['rejected_findings'][0]['supporting_evidence_slot']==2
            raw['supporting_evidence_url_2']='';raw['supporting_evidence_quote_2']=''
        return 200,_response('submit_findings',{'findings':[raw]},call=len(requests))
    async def forbidden(*args,**kwargs):raise AssertionError('No new research needed')
    monkeypatch.setattr(inv,'_post_json',post)
    monkeypatch.setattr(inv,'_search_web',forbidden);monkeypatch.setattr(inv,'_fetch_page',forbidden)
    result=asyncio.run(inv.investigate_company_evidence(
        company_locator=dict(name=ANCHOR['submitted_name'],website='https://captifytechnologies.com/',
            linkedin='https://www.linkedin.com/company/captify/'),targets=('required_attribute',),
        requested_attribute='Sells an advertising technology platform used by brands, agencies, or publishers to buy, sell, or measure digital ad campaigns.',
        positive_semantic_review=True,
        prior_observations=dict(observed_company_name=ANCHOR['observed_name'],
            observed_company_website='captifytechnologies.com',observed_company_linkedin='',submitted_source_urls=[URL]),
        verified_homepage_identity=dict(normalized_name=ANCHOR['verified_name'],
            registrable_dns_domain='captifytechnologies.com',linkedin_company_slug='captify'),
        prefetched_pages={URL:dict(final_url=URL,text=QUOTE+' '+support),
            other:dict(final_url=other,text='We serve global brands.')}))
    assert len(requests)==2
    assert result['claims']['required_attribute']['status']=='VERIFIED'
    assert result['claims']['required_attribute']['supporting_evidence']==[dict(url=URL,quote=support)]
    assert result['usage']['search_calls']==result['usage']['fetch_calls']==0
