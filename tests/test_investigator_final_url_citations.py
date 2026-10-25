"""Citations may use a displayed server-observed final URL, never a guessed alias."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.product_coverage import coverage_matches
from tests.test_company_evidence_investigator import _finding

# Native011423 supplied this requested/final homepage pair. Its exact homepage
# support quotes were repeatedly rejected while the case-study primary passed.
REQUEST='https://thetradedesk.com/'
FINAL='https://www.thetradedesk.com/'
CASE=FINAL+'case-studies/pinnacle-ctv-kokai'
HOME_QUOTE='Our advertising platform helps advertisers find new audiences and grow their brands with cutting-edge technology and access to the world’s best content.'
CASE_QUOTE='Pinnacle devised a strategy that involved running an awareness campaign through The Trade Desk’s Kokai platform experience using CTV inventory.'
CRITERION='Operates a demand-side advertising platform that brands and agencies use to plan, buy, and measure digital ads across connected TV and other channels'
ANCHOR={'submitted_name':'The Trade Desk','observed_name':'The Trade Desk','verified_name':'The Trade Desk',
 'submitted_domain':'thetradedesk.com','observed_domain':'thetradedesk.com','verified_domain':'thetradedesk.com',
 'submitted_linkedin_slug':'the-trade-desk','observed_linkedin_slug':'the-trade-desk','verified_linkedin_slug':'the-trade-desk'}


def finding(*,slot=1):
    raw=_finding('industry',status='VERIFIED',activity_role='supplier_operator',observed_industry='Advertising technology',
      evidence_url=FINAL if slot==0 else CASE,evidence_quote=HOME_QUOTE if slot==0 else CASE_QUOTE)
    raw['product_coverage']='VERIFIED'
    if slot:
        raw[f'supporting_evidence_url_{slot}']=FINAL
        raw[f'supporting_evidence_quote_{slot}']=HOME_QUOTE
    return raw


def validate(raw=None,*,pages=None,finals=None,anchor=None):
    return inv._validated_findings({'findings':[raw or finding()]},targets=('industry',),
      fetched_pages={REQUEST:HOME_QUOTE,CASE:CASE_QUOTE} if pages is None else pages,
      fetched_final_urls={REQUEST:FINAL,CASE:CASE} if finals is None else finals,
      first_party_domains={'thetradedesk.com'},identity_names={'thetradedesk'},identity_anchor=ANCHOR if anchor is None else anchor,
      requested_product_service=CRITERION)['industry']


@pytest.mark.parametrize('slot',[0,1,2])
def test_native_displayed_final_url_resolves_to_requested_source_without_copying_pages(slot):
    pages={REQUEST:HOME_QUOTE,CASE:CASE_QUOTE};finals={REQUEST:FINAL,CASE:CASE}
    before=deepcopy((pages,finals));raw=finding(slot=slot);original=deepcopy(raw)
    claim=validate(raw,pages=pages,finals=finals)
    assert claim['status']=='VERIFIED'
    assert coverage_matches(claim['product_coverage'],CRITERION)
    assert REQUEST in [e['url'] for e in claim['product_coverage']['evidence']]
    assert FINAL not in [e['url'] for e in claim['product_coverage']['evidence']]
    assert (pages,finals)==before and raw==original
    assert len(pages)==2 # A redirect never consumes a new source/page slot.
    assert inv._submitted_finding_rejection(raw,claim) is None


@pytest.mark.parametrize('mode',['missing_provenance','unfetched_request','guessed_www','ambiguous','ambiguous_identical_text',
 'foreign_request','foreign_final','wrong_quote','wrong_entity','wrong_role','missing_identity'])
def test_final_url_resolution_does_not_weaken_existing_source_gates(mode):
    raw=finding(slot=0);pages={REQUEST:HOME_QUOTE};finals={REQUEST:FINAL};anchor=deepcopy(ANCHOR)
    if mode=='missing_provenance':finals={}
    elif mode=='unfetched_request':pages={}
    elif mode=='guessed_www':finals={REQUEST:REQUEST}
    elif mode in {'ambiguous','ambiguous_identical_text'}:
        second=REQUEST+'other-request';pages[second]=HOME_QUOTE if mode.endswith('identical_text') else 'Different content.';finals[second]=FINAL
    elif mode=='foreign_request':
        pages={'https://attacker.example/':HOME_QUOTE};finals={'https://attacker.example/':FINAL}
    elif mode=='foreign_final':
        raw['evidence_url']='https://attacker.example/final';finals[REQUEST]=raw['evidence_url']
    elif mode=='wrong_quote':raw['evidence_quote']='We supply an invented product.'
    elif mode=='wrong_entity':
        pages[REQUEST]=raw['evidence_quote']='Other Company supplies its advertising platform.'
        anchor['observed_name']='Other Company'
    elif mode=='wrong_role':raw['activity_role']='customer_user'
    else:anchor={}
    claim=validate(raw,pages=pages,finals=finals,anchor=anchor)
    assert claim['status']=='UNPROVEN',mode
    assert not coverage_matches(claim.get('product_coverage'),CRITERION)


def test_support_slot_two_rejection_keeps_its_slot_after_url_resolution():
    raw=finding(slot=2);raw['supporting_evidence_quote_2']='We supply an invented product.'
    claim=validate(raw)
    assert claim['status']=='UNPROVEN'
    rejected=claim['untrusted_product_citation_rejections']
    assert len(rejected)==1 and rejected[0]['binding_index']==2
    assert rejected[0]['reason']=='submitted quote was not present in fetched source'


def test_normalized_claim_survives_existing_recovery_final_gate_and_retry_state():
    claim=validate(finding(slot=0));assert claim['status']=='VERIFIED'
    assert claim['evidence_url']==REQUEST
    pages={REQUEST:{'final_url':FINAL,'text':HOME_QUOTE}}
    investigation={inv.PRIVATE_FETCHED_PAGES_KEY:pages,'claims':{'industry':claim},'_completed_submit':True}
    alternate={}
    assert lead_scorer._hydrate_verified_required_attribute_recovery_source(alternate,investigation,claim,
      verified_transport_domain='thetradedesk.com') is True
    assert alternate[REQUEST]['text']==HOME_QUOTE
    evidence={'url':claim['evidence_url'],'quote':claim['evidence_quote'],'product_coverage':claim['product_coverage']}
    fit=company_fit_match('source-bound industry match',details={
      'identity_decision':'match','dimension_decisions':{'industry':'match'},
      'dimension_evidence':{
        'identity':{'decision':'match','web_identity_receipt':{'decision':'match','observed_name':'The Trade Desk',
          'observed_domain':'thetradedesk.com','observed_linkedin_slug':'the-trade-desk','evidence_source':'company_web_reverification'}},
        'industry':{'decision':'match','observed_decision':'match','web_evidence':evidence,'product_coverage':claim['product_coverage']}}})
    assert lead_scorer._final_product_coverage_result(fit,SimpleNamespace(product_service=CRITERION),enabled=True) is fit
    retained={};lead_scorer._retain_matched_investigator_source_contexts(retained,fit,investigation)
    assert retained==pages
    retry={};lead_scorer._retain_matched_company_retry_sources(retry,retained,fit)
    assert retry['pages']==pages
    restored=lead_scorer._matched_company_retry_prefetched_pages(retry,{
      'normalized_name':'The Trade Desk','registrable_dns_domain':'thetradedesk.com','linkedin_company_slug':'the-trade-desk'})
    assert restored==pages
    loaded,finals=inv._validated_prefetched_pages(restored,submitted_source_urls=list(restored))
    assert validate(finding(slot=0),pages=loaded,finals=finals)['status']=='VERIFIED'
