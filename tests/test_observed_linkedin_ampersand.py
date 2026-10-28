"""Observed ampersand slugs reach exact profile proof without miner expansion."""
import asyncio
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlencode

import pytest

from gateway.qualification.models import candidate_linkedin_prompt_slug
from qualification.scoring import company_verification as homepage
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import linkedin_company_size as size
from qualification.scoring.company_fit_decision import _linkedin_slug
from qualification.scoring.lead_scorer import _verified_homepage_identity_anchor
from tests.test_current_linkedin_company_size import _install_exa_bodies

RAW = 'https://www.linkedin.com/company/acme&sons'
ESCAPED = 'https://www.linkedin.com/company/acme%26sons'
SLUG = 'acme%26sons'
TEXT = 'Acme & Sons\n## About us\nWebsite\nhttps://acme.com/\nCompany size\n501-1,000 employees'
HTML = '''<html><head><title>Acme &amp; Sons</title>
<meta property="og:site_name" content="Acme &amp; Sons">
<script type="application/ld+json">{"@type":"Organization","name":"Acme & Sons",
"url":"https://acme.com/","sameAs":["https://www.linkedin.com/company/acme&sons"]}</script>
</head><body><a href="https://www.linkedin.com/company/acme&sons">LinkedIn</a></body></html>'''


@pytest.mark.parametrize('url', [RAW, ESCAPED])
def test_observed_helpers_share_one_escaped_slug(url):
    assert homepage._canonical_linkedin_company_url(url) == ESCAPED
    assert size.linkedin_company_page_slug(url) == SLUG
    assert size._strict_linkedin_company_profile_url(url) == ESCAPED
    assert _linkedin_slug(url) == SLUG
    with pytest.raises(ValueError):
        candidate_linkedin_prompt_slug(url, 'company_linkedin')


@pytest.mark.parametrize('url', [
    'https://www.linkedin.com/company/acme%2526sons',
    'https://www.linkedin.com/company/acme%26sons%2Fposts',
    'https://www.linkedin.com/company/acme%26sons%5Cposts',
    'https://www.linkedin.com/company/acme%26sons%3Furl=evil',
    'https://www.linkedin.com/company/acme%26sons%23ignore',
    'https://www.linkedin.com/company/acme%26sons%0Aignore',
    'https://www.linkedin.com/company/acme&sons?url=https://evil.example/',
    'https://www.linkedin.com/company/acme&sons#other',
    'https://www.linkedin.com/company/acme&sons/about',
    'https://user:pass@www.linkedin.com/company/acme&sons',
    'https://www.linkedin.com:444/company/acme&sons',
    'https://linkedin.com.evil.example/company/acme&sons',
    'https://evil-linkedin.com/company/acme&sons',
    'https://www.linkedin.com/in/acme&sons',
])
def test_strict_fetched_profile_rejects_ambiguous_or_unsafe_shapes(url):
    assert size._strict_linkedin_company_profile_url(url) == ''


def test_homepage_discovery_keeps_existing_tab_and_tracking_normalization():
    assert homepage._canonical_linkedin_company_url(RAW + '/mycompany/?trk=nav#top') == ESCAPED


@pytest.mark.parametrize('returned_url', [RAW, ESCAPED])
def test_homepage_anchor_to_exact_native_profile_and_bound_size(monkeypatch, returned_url):
    monkeypatch.setattr(homepage, '_fetch_bounded_html', AsyncMock(return_value=(200, 'https://acme.com/', HTML)))
    verified = asyncio.run(homepage.verify_company_exists(
        'Acme & Sons', 'https://acme.com/', company_quality=True))
    assert verified.decision == 'match'
    anchor = _verified_homepage_identity_anchor(verified)
    assert anchor['linkedin_company_slug'] == SLUG
    profile = 'https://www.linkedin.com/company/' + anchor['linkedin_company_slug']
    calls, pending = _install_exa_bodies(monkeypatch, {'results': [{'url': returned_url, 'text': TEXT}]})
    generic = AsyncMock(side_effect=AssertionError('observed exact profile must use native route'))
    monkeypatch.setattr(investigator, '_fetch_bounded_html', generic)
    page = asyncio.run(investigator._fetch_page(object(), profile))
    assert page['ok'] is True and page['text'] == TEXT
    assert calls[0][1]['json']['ids'] == [ESCAPED]
    assert len(calls) == 1 and pending == []
    generic.assert_not_awaited()
    identity = {'submitted_name': 'acmesons', 'observed_name': 'acmesons', 'verified_name': 'acmesons',
                'submitted_domain': 'acme.com', 'observed_domain': 'acme.com', 'verified_domain': 'acme.com',
                'submitted_linkedin_slug': '', 'observed_linkedin_slug': SLUG, 'verified_linkedin_slug': SLUG}
    quote = TEXT
    finding = investigator._validated_findings(
        {'findings': [{'target': 'headcount', 'status': 'VERIFIED', 'observed_value': '501-1,000',
                       'evidence_url': profile, 'evidence_quote': quote}]},
        targets=('headcount',), fetched_pages={profile: page['text']},
        fetched_final_urls={profile: page['final_url']}, first_party_domains={'acme.com'},
        identity_names={'acmesons'}, identity_anchor=identity,
    )['headcount']
    assert finding['status'] == 'VERIFIED'
    assert finding['observed_value'] == '501-1,000'
    assert finding['evidence_quote'] == quote
    # API query encoders keep an already canonical URL in one parameter.
    encoded = urlencode({'url': profile, 'dynamic': 'true'})
    assert parse_qs(encoded) == {'url': [ESCAPED], 'dynamic': ['true']}


@pytest.mark.parametrize('url', [RAW, ESCAPED])
def test_structured_profile_sends_canonical_json_url_and_binds_domain(monkeypatch, url):
    payload = {'result': {'data': {'status': 200, 'element': {
        'name': 'Acme & Sons', 'linkedinUrl': RAW, 'website': 'https://acme.com/',
        'employeeCountRange': {'start': 501, 'end': 1000},
    }}}}
    calls, pending = _install_exa_bodies(monkeypatch, payload)
    monkeypatch.setenv('DEEPLINE_API_KEY', 'test-key')
    result = asyncio.run(size.fetch_structured_linkedin_company_size('acme.com', url))
    assert calls[0][1]['json'] == {'payload': {'url': ESCAPED}}
    assert pending == [] and len(calls) == 1
    assert result['employee_count'] == '501-1,000'
    assert result['url'] == ESCAPED


@pytest.mark.parametrize('text', [
    TEXT.replace('Acme & Sons', 'Other Company').replace('Website\nhttps://acme.com/\n', ''),
    TEXT.replace('https://acme.com/', 'https://other.example/'),
    'Acme & Sons\nWebsite\nhttps://acme.com/\n501-1,000 associated members',
    'Acme & Sons\nWebsite\nhttps://acme.com/\nEstimated employees: 800; around 998 professionals',
])
def test_ampersand_never_supplies_missing_company_or_canonical_band_proof(monkeypatch, text):
    calls, _ = _install_exa_bodies(monkeypatch, {'results': [{'url': RAW, 'text': text}]})
    result = asyncio.run(size.fetch_current_linkedin_company_size(
        ESCAPED, expected_company_identity={'normalized_name': 'acmesons',
                                          'registrable_dns_domain': 'acme.com'}))
    assert len(calls) == 1
    assert result['outcome'] == 'insufficient_evidence'
    assert 'employee_count' not in result


@pytest.mark.parametrize('returned_url', [
    'https://www.linkedin.com/company/other',
    'https://other.example/company/acme&sons',
    RAW + '?other=company', RAW + '#other', RAW + '/about',
])
def test_exact_profile_route_rejects_wrong_or_non_main_returned_source(monkeypatch, returned_url):
    _install_exa_bodies(monkeypatch, {'results': [{'url': returned_url, 'text': TEXT}]})
    generic = AsyncMock(side_effect=AssertionError('no hidden generic fallback'))
    monkeypatch.setattr(investigator, '_fetch_bounded_html', generic)
    page = asyncio.run(investigator._fetch_page(object(), ESCAPED))
    assert page['ok'] is False
    assert 'text' not in page
    generic.assert_not_awaited()


@pytest.mark.parametrize('website,profile,range_field', [
    ('https://other.example/', RAW, {'start': 501, 'end': 1000}),
    ('https://acme.com/', 'https://www.linkedin.com/company/other', {'start': 501, 'end': 1000}),
    ('https://acme.com/', RAW, None),
])
def test_structured_band_still_needs_exact_domain_profile_and_range(monkeypatch, website, profile, range_field):
    element = {'name': 'Acme & Sons', 'linkedinUrl': profile, 'website': website,
               'staffCount': 998, 'employeeCount': 800}
    if range_field:
        element['employeeCountRange'] = range_field
    _install_exa_bodies(monkeypatch, {'result': {'data': {'status': 200, 'element': element}}})
    monkeypatch.setenv('DEEPLINE_API_KEY', 'test-key')
    assert asyncio.run(size.fetch_structured_linkedin_company_size('acme.com', ESCAPED)) is None
