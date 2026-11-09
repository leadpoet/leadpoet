"""Saved Warby observations plus explicit synthetic independent provider proof."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import lead_scorer as s
from qualification.scoring import linkedin_company_size as p

URL = 'https://www.linkedin.com/company/warbyparker'
COMPANY = SimpleNamespace(company_name='Warby Parker', company_website='https://www.warbyparker.com/jobs', company_linkedin=URL)
OBSERVED = {'observed_company_name': 'Warby Parker Inc.', 'observed_company_website': 'https://www.warbyparker.com', 'observed_company_linkedin': ''}
DOMAIN = 'warbyparker.com'
WEB = s._web_identity_receipt(COMPANY, OBSERVED)
# Fixture only: no claim this Warby provider response was captured in production.
PAYLOAD = {'status': 'completed', 'result': {'data': {'status': 200, 'element': {
    'name': 'Warby Parker', 'website': 'https://www.warbyparker.com/jobs',
    'linkedinUrl': URL, 'employeeCountRange': {'start': 1001, 'end': 5000},
}}}}


def proof(payload=PAYLOAD, preserve=True):
    return p.project_structured_linkedin_company_identity(DOMAIN, URL, deepcopy(payload), preserve_identity_website=preserve)


def test_saved_missing_link_observation_can_locate_independent_profile():
    assert WEB['decision'] == 'unavailable'
    assert WEB['observed_linkedin_slug'] == ''
    assert s._alias_unresolved_structured_profile_lookup(WEB, DOMAIN) == {
        'normalized_name': 'warbyparker', 'registrable_dns_domain': DOMAIN,
        'linkedin_company_slug': 'warbyparker'}


@pytest.mark.parametrize('round_trip', [False, True])
def test_exact_independent_profile_binds_missing_homepage_link(round_trip):
    evidence = proof()
    if round_trip:
        evidence = json.loads(json.dumps(evidence))
    result = s._web_identity_receipt(COMPANY, OBSERVED,
        verified_homepage_transport_domain=DOMAIN, verified_structured_identity=evidence)
    assert result['decision'] == 'match'
    assert result['reason_code'] == 'structured_missing_homepage_link_identity_verified'
    assert result['observed_linkedin_slug'] == 'warbyparker'


@pytest.mark.parametrize('fault', ['no_proof', 'no_raw_website', 'provider', 'source_field', 'extra_field',
    'name', 'slug', 'noncompany', 'other_domain', 'child', 'parent', 'sibling', 'credentials',
    'no_transport', 'wrong_transport', 'web_name', 'web_domain', 'existing_conflict', 'numeric_slug'])
def test_missing_link_never_approves_unbound_or_projected_profile(fault):
    evidence = proof();observed = deepcopy(OBSERVED);transport=DOMAIN;company=COMPANY
    if fault == 'no_proof': evidence = None
    elif fault == 'no_raw_website': evidence.pop('provider_website_url')
    elif fault in {'provider', 'source_field'}: evidence[fault] = 'untrusted'
    elif fault == 'extra_field': evidence['miner_assertion'] = True
    elif fault == 'name': evidence['name'] = 'Warby Parker Subsidiary'
    elif fault == 'slug': evidence['url'] = URL+'-parent'
    elif fault == 'noncompany': evidence['url'] = 'https://www.linkedin.com/in/warbyparker'
    elif fault in {'other_domain', 'child', 'parent', 'sibling', 'credentials'}:
        evidence['provider_website_url'] = {'other_domain':'https://other.com','child':'https://careers.warbyparker.com',
            'parent':'https://parker.com','sibling':'https://other.warbyparker.com',
            'credentials':'https://user@warbyparker.com'}[fault]
    elif fault == 'no_transport': transport=''
    elif fault == 'wrong_transport': transport='other.com'
    elif fault == 'web_name': observed['observed_company_name']='Warby Parker Subsidiary'
    elif fault == 'web_domain': observed['observed_company_website']='https://other.com'
    elif fault == 'existing_conflict': observed['observed_company_linkedin']=URL+'-other'
    elif fault == 'numeric_slug': company=SimpleNamespace(**{**vars(COMPANY),'company_linkedin':'https://www.linkedin.com/company/12345'})
    result=s._web_identity_receipt(company,observed,verified_homepage_transport_domain=transport,verified_structured_identity=evidence)
    assert result['decision'] != 'match'


@pytest.mark.parametrize('fetch_failure', [False, True])
def test_existing_flow_fetches_once_and_preserves_raw_website(monkeypatch, fetch_failure):
    from tests.test_company_headcount_evidence import company, icp, verdict, install_no_network
    candidate = company().model_copy(update=vars(COMPANY))
    observed = {**verdict(), **OBSERVED}
    install_no_network(monkeypatch, observed)
    calls = []

    async def fetch(domain, url, **kwargs):
        calls.append((domain, url))
        assert kwargs['preserve_identity_website'] is True
        if fetch_failure:
            return None
        identity = p.project_structured_linkedin_company_identity(
            domain, url, PAYLOAD, preserve_identity_website=kwargs['preserve_identity_website'])
        kwargs['company_identity_evidence'].update(identity)
        return p.project_structured_linkedin_company_size(domain, url, PAYLOAD)

    async def existing_size_fetch(url, **kwargs):
        assert url == URL
        return None

    monkeypatch.setattr(s, 'fetch_current_linkedin_company_size', existing_size_fetch)
    monkeypatch.setattr(s, 'fetch_structured_linkedin_company_size', fetch)
    result = asyncio.run(s._llm_reverify_company(candidate, icp(),
        require_company_fit_dimensions=True,
        verified_homepage_identity=s.company_fit_unavailable('LinkedIn company binding not found',
            details={'verified_homepage_transport_domain': DOMAIN})))
    assert calls == [(DOMAIN, URL)]
    assert result.details['identity_decision'] == ('unavailable' if fetch_failure else 'match')


@pytest.mark.parametrize('field', ['name', 'provider', 'source_field', 'url', 'website', 'provider_website_url'])
def test_incomplete_private_provider_proof_is_never_accepted(field):
    evidence = proof()
    evidence.pop(field)
    assert s._web_identity_receipt(COMPANY, OBSERVED,
        verified_homepage_transport_domain=DOMAIN,
        verified_structured_identity=evidence)['decision'] != 'match'


def test_provider_returning_different_profile_is_not_projected():
    payload = deepcopy(PAYLOAD)
    payload['result']['data']['element']['linkedinUrl'] = URL+'-parent'
    assert proof(payload) is None


def test_saved_match_remains_bound_to_exact_company_claim():
    from qualification.scoring.company_fit_decision import company_quality_receipt_matches_claim
    result = s._web_identity_receipt(COMPANY, OBSERVED,
        verified_homepage_transport_domain=DOMAIN, verified_structured_identity=proof(), company_quality=True)
    saved = json.loads(json.dumps(result))
    assert saved['decision'] == 'match'
    assert company_quality_receipt_matches_claim(saved, vars(COMPANY))
    assert not company_quality_receipt_matches_claim(saved, {**vars(COMPANY), 'company_name':'Warby Parker Subsidiary'})
