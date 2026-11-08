"""Actual Nov7 homepage, with separately labelled controlled provider replies.

No numeric Harvest reply for Simpli.fi was retained. Provider/model controls
below test the native trust boundary and are not a historical score replay.
"""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qualification.scoring import company_verification as homepage, lead_scorer as scorer, linkedin_company_size as linkedin
from qualification.scoring.company_fit_decision import company_fit_match
from tests.test_fit_evidence_url_compat import _company, _icp

HTML = (Path(__file__).parent / 'fixtures/simpli_homepage_identity_excerpt.html').read_text()
NUMERIC = 'https://www.linkedin.com/company/801567'
VANITY = 'https://www.linkedin.com/company/simpli.fi'
UK = 'https://www.linkedin.com/company/simpli-fi'
COMPANY = {'company_name': 'Simpli.fi', 'company_website': 'http://www.simpli.fi/', 'company_linkedin': VANITY + '/'}

# Deliberately wrong model slug matching the observed Nov7 failure. All other
# dimensions here are explicit controls, not a claimed current model verdict.
CONTROLLED_WEB = {
    'observed_company_name': 'Simpli.fi', 'observed_company_website': 'https://simpli.fi/',
    'observed_company_linkedin': UK, 'observed_employee_count': None,
    'employee_size_matches': None, 'employee_size_evidence_url': '', 'employee_size_evidence_quote': '',
    'observed_industry': 'Advertising', 'industry_matches': True,
    'industry_activity_role': 'supplier_operator', 'industry_evidence_url': 'https://simpli.fi/',
    'industry_evidence_quote': 'Simpli.fi supplies programmatic advertising software.',
    'observed_hq_country': 'United States', 'observed_hq_state': 'Texas',
    'geography_matches': True, 'geography_evidence_url': 'https://simpli.fi/',
    'geography_evidence_quote': 'Simpli.fi Headquarters, Fort Worth, Texas.',
    'observed_company_stage': '', 'stage_matches': None,
}
# Synthetic provider boundary control. Exact website, name, ID and supplied
# vanity must all bind; the live numeric response remains unverified.
CONTROLLED_PROFILE = {'status': 200, 'element': {
    'name': 'Simpli.fi', 'id': '801567', 'universalName': 'simpli.fi',
    'linkedinUrl': VANITY + '/', 'website': 'http://www.simpli.fi/',
    'employeeCountRange': {'start': 501, 'end': 1000},
}}


def captured_homepage(monkeypatch, html=HTML):
    async def fetch(*_args, **_kwargs):
        return 200, 'https://simpli.fi/', html
    monkeypatch.setattr(homepage, '_fetch_bounded_html', fetch)
    return asyncio.run(homepage.verify_company_exists(
        **{k: COMPANY[k] for k in ('company_name', 'company_website')},
        company_linkedin=COMPANY['company_linkedin'], require_https_transport=True,
    ))


def web_receipt():
    return scorer._web_identity_receipt(SimpleNamespace(**COMPANY), CONTROLLED_WEB,
        verified_homepage_transport_domain='simpli.fi')


def test_actual_source_selects_exact_name_without_granting_identity(monkeypatch):
    result = captured_homepage(monkeypatch)
    assert result.decision == 'unavailable'
    assert result.details['identity']['observed_name'] == 'simplifi'
    assert result.details['identity']['observed_linkedin_slug'] == '801567'
    # Keep the actual Organization record as independent conflict context.
    assert result.details['observed_homepage_organization_identity']['normalized_name'] == (
        'programmaticdspadvertisingautomationplatformsimplifi')
    lookup = scorer._conflicting_homepage_numeric_profile_lookup(result, web_receipt())
    assert lookup['requested_profile_url'] == NUMERIC
    # Provider ID + universalName bind this lookup. No web-observed alias is
    # supplied, so a wrong provider website cannot add a redirect fetch.
    assert 'observed_profile_url' not in lookup


@pytest.mark.parametrize('html', [
    HTML.replace('Simpli.fi</title>', 'OtherCo</title>'),
    '<title>OtherCo | Our partner Simpli.fi</title><a href="' + NUMERIC + '">LinkedIn</a>',
    HTML + '<a href="https://www.linkedin.com/company/other">Other LinkedIn</a>',
    '<title>Simpli.fi</title><a href="' + NUMERIC + '">LinkedIn</a>',
])
def test_wrong_title_partner_and_multiple_profiles_cannot_choose_numeric_lookup(monkeypatch, html):
    result = captured_homepage(monkeypatch, html)
    assert not scorer._conflicting_homepage_numeric_profile_lookup(result, web_receipt())


def numeric_proof(body=CONTROLLED_PROFILE):
    return linkedin.project_structured_linkedin_company_identity(
        'simpli.fi', NUMERIC, body, observed_profile_url=VANITY, expected_company_name='Simpli.fi')


def test_bound_organization_conflict_remains_until_exact_numeric_proof(monkeypatch):
    result = captured_homepage(monkeypatch)
    correct_web = scorer._web_identity_receipt(SimpleNamespace(**COMPANY), {
        **CONTROLLED_WEB, 'observed_company_linkedin': VANITY,
    }, verified_homepage_transport_domain='simpli.fi')
    assert scorer._unresolved_homepage_web_identity_conflict(result, correct_web)
    assert not scorer._unresolved_homepage_web_identity_conflict(result, {
        **correct_web, 'structured_profile_identity': numeric_proof(),
    })
    assert not scorer._unresolved_homepage_web_identity_conflict(result, {
        **correct_web, 'reason_code': 'structured_homepage_numeric_identity_verified',
        'structured_profile_identity': numeric_proof(),
    })
    for key, value in [('company_id', '999'), ('requested_url', 'https://www.linkedin.com/company/999'),
                       ('url', UK), ('website', 'https://simpli-fi.co.uk/'), ('name', 'OtherCo')]:
        assert scorer._unresolved_homepage_web_identity_conflict(result, {
            **correct_web, 'structured_profile_identity': {**numeric_proof(), key: value},
        })
        assert scorer._unresolved_homepage_web_identity_conflict(result, {
            **correct_web, 'reason_code': 'structured_homepage_numeric_identity_verified',
            'structured_profile_identity': {**numeric_proof(), key: value},
        })


def test_new_numeric_receipt_must_match_homepage_id_even_when_organization_name_matches(monkeypatch):
    result = captured_homepage(monkeypatch)
    result.details['observed_homepage_organization_identity']['normalized_name'] = 'simplifi'
    corrected = {**web_receipt(), 'decision': 'match', 'observed_linkedin_slug': 'simpli.fi',
        'reason_code': 'structured_homepage_numeric_identity_verified',
        'structured_profile_identity': {**numeric_proof(), 'company_id': '999',
            'requested_url': 'https://www.linkedin.com/company/999'}}
    assert scorer._unresolved_homepage_web_identity_conflict(result, corrected)


def _run_native_fit(monkeypatch, profile=CONTROLLED_PROFILE, *, web_updates=None, company_updates=None, quality=False):
    captured_homepage(monkeypatch)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setenv('DEEPLINE_API_KEY', 'test-key')
    monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL', raising=False)
    requests = []

    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def json(self): return deepcopy(profile)
    class Session:
        def __init__(self, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        def post(self, url, *, json, headers):
            assert 'harvestapi_get_company' in url
            requests.append(json)
            return Response()
    async def request(**_kwargs): return {**deepcopy(CONTROLLED_WEB), **(web_updates or {})}, ''
    async def unchanged(verdict, *_args, **_kwargs): return dict(verdict)
    async def redirect(_company, verdict): return dict(verdict)
    async def attribute(verdict, **_kwargs): return dict(verdict), None
    async def precheck(*_args, **_kwargs): return company_fit_match()
    async def size(*_args, **_kwargs):
        return {'outcome': linkedin.CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE}
    monkeypatch.setattr(linkedin.aiohttp, 'ClientSession', Session)
    monkeypatch.setattr(scorer, '_request_company_reverify_json', request)
    monkeypatch.setattr(scorer, '_resolve_observed_linkedin_redirect_alias', redirect)
    monkeypatch.setattr(scorer, '_ground_required_attribute_evidence', attribute)
    monkeypatch.setattr(scorer, '_ground_employee_size_observation', unchanged)
    monkeypatch.setattr(scorer, 'fetch_current_linkedin_company_size', size)
    monkeypatch.setattr(scorer, 'run_company_zero_checks', precheck)
    company = _company(**{**COMPANY, **(company_updates or {})}, employee_count='501-1,000', country='United States', state='Texas', company_stage='')
    icp = _icp().model_copy(update={'company_stage': '', 'required_attribute': '',
        'industry': 'Advertising', 'sub_industry': '', 'product_service': '',
        'country': 'United States', 'geography': 'United States', 'employee_count': '501-1,000'})
    result = asyncio.run(scorer._verify_company_fit(
        company, icp, 0, 1, set(), require_https_transport=True, company_quality=quality))
    return result, requests


@pytest.mark.parametrize('quality', [False, True])
def test_controlled_numeric_reply_repairs_identity_and_size_in_native_full_fit(monkeypatch, quality):
    result, requests = _run_native_fit(monkeypatch, quality=quality)
    assert result.decision == 'match'  # Other fit dimensions are controlled.
    assert requests == [{'payload': {'url': NUMERIC}}]
    assert result.details['company_fit_dimensions']['identity'] == 'match'
    assert result.details['company_fit_dimensions']['employee_size'] == 'match'
    proof = result.details['dimension_evidence']['identity']['web_identity_receipt']['structured_profile_identity']
    assert proof['company_id'] == '801567'
    assert proof['url'] == VANITY
    from qualification.scoring.company_fit_decision import company_quality_receipt_matches_claim
    identity = result.details['dimension_evidence']['identity']['web_identity_receipt']
    assert identity['raw_observed_linkedin_slug'] == 'simpli-fi'
    assert company_quality_receipt_matches_claim(json.loads(json.dumps(identity)), COMPANY)


@pytest.mark.parametrize('key,value', [
    ('id', '999'), ('name', 'Simpli-Fi Wireless Connectivity'),
    ('website', 'https://simpli-fi.co.uk/'), ('linkedinUrl', UK),
    ('universalName', 'simpli-fi'),
    ('id', None), ('id', True), ('id', '８０１５６７'),
])
def test_wrong_entity_or_id_cannot_repair_native_fit(monkeypatch, key, value):
    profile = deepcopy(CONTROLLED_PROFILE)
    profile['element'][key] = value
    result, requests = _run_native_fit(monkeypatch, profile)
    assert len(requests) == 1
    assert result.details['company_fit_dimensions']['identity'] != 'match'


@pytest.mark.parametrize('industry', [False, None])
def test_identity_repair_preserves_unresolved_or_negative_other_dimensions(monkeypatch, industry):
    result, requests = _run_native_fit(monkeypatch, web_updates={'industry_matches': industry})
    assert len(requests) == 1
    assert result.details['company_fit_dimensions']['identity'] == 'match'
    assert result.details['company_fit_dimensions']['industry'] != 'match'
    assert result.decision != 'match'


def test_bad_supplied_wireless_profile_cannot_be_repaired_as_us_advertising(monkeypatch):
    result, requests = _run_native_fit(monkeypatch, company_updates={'company_linkedin': UK})
    assert len(requests) == 1
    assert result.details['company_fit_dimensions']['identity'] != 'match'


def test_multiple_provider_main_profiles_cannot_repair_identity(monkeypatch):
    result, requests = _run_native_fit(monkeypatch, {'status': 200, 'elements': [
        CONTROLLED_PROFILE['element'], CONTROLLED_PROFILE['element'],
    ]})
    assert len(requests) == 1
    assert result.details['company_fit_dimensions']['identity'] != 'match'


def test_selected_exact_title_cannot_hide_unproved_organization_conflict_in_full_fit(monkeypatch):
    profile = deepcopy(CONTROLLED_PROFILE)
    profile['element']['id'] = '999'
    # Even a model match cannot replace the bound Organization's numeric ID.
    result, requests = _run_native_fit(monkeypatch, profile, web_updates={
        'observed_company_linkedin': VANITY,
    })
    assert len(requests) == 1
    identity = result.details['dimension_evidence']['identity']['web_identity_receipt']
    assert identity['decision'] == 'unavailable'
    assert identity['reason_code'] == 'homepage_web_identity_conflict_unresolved'
    assert result.decision != 'match'
