"""Exact first-party Organization aliases remain bound through investigation."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_verification as verification
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from tests.test_company_verification_transport import _Response, _Session
from tests.test_company_evidence_investigator import _company, _finding

URL = 'https://www.northstar.example/us/en'
LINKEDIN = 'https://www.linkedin.com/company/northstar'
ORGANIZATION = {
    '@type': 'Organization', 'name': 'Northstar',
    'legalName': 'Northstar Technology Solutions Corp',
    'url': '/content/northstar-dot-com/us/en/home',
    'sameAs': [LINKEDIN + '/'],
}


def _homepage(monkeypatch, name, *, organization=None, company_quality=False):
    record = ORGANIZATION if organization is None else organization
    body = ('<title>Enterprise AI services</title>'
            f'<script type="application/ld+json">{json.dumps(record)}</script>'
            f'<a href="{LINKEDIN}">LinkedIn</a>').encode()
    monkeypatch.setattr(verification.aiohttp, 'ClientSession',
                        lambda **kwargs: _Session(_Response(200, body, URL)))
    result = asyncio.run(verification.verify_company_exists(
        name, 'https://northstar.example/',
        company_linkedin=LINKEDIN if company_quality else '',
        require_https_transport=True, company_quality=company_quality,
    ))
    return result, lead_scorer._verified_homepage_identity_anchor(result)


@pytest.mark.parametrize('name', [
    'Northstar Technology Solutions',
    'Northstar Technology Solutions Corporation',
    'Northstar',
])
@pytest.mark.parametrize('company_quality', [False, True])
def test_exact_homepage_brand_and_legal_name_bind(monkeypatch, name, company_quality):
    result, anchor = _homepage(monkeypatch, name, company_quality=company_quality)
    assert result.decision == 'match'
    assert anchor['verified_brand_name'] == 'Northstar'
    assert anchor['verified_legal_name_aliases'] == ['Northstar Technology Solutions Corp']
    company = _company(name=name, website='https://northstar.example/',
                       linkedin=LINKEDIN if company_quality else '')
    receipt = lead_scorer._web_identity_receipt(company, {
        'observed_company_name': name,
        'observed_company_website': URL,
        'observed_company_linkedin': LINKEDIN,
    }, verified_homepage_identity=anchor, company_quality=company_quality)
    assert receipt['decision'] == 'match'


@pytest.mark.parametrize('record', [
    {**ORGANIZATION, 'name': '---'},
    {**ORGANIZATION, 'legalName': '---'},
    {**ORGANIZATION, 'url': 'https://foreign.example/'},
    {**ORGANIZATION, 'url': '//foreign.example/'},
    {**ORGANIZATION, 'sameAs': ['https://www.linkedin.com/company/unrelated']},
    [ORGANIZATION, {**ORGANIZATION, 'name': 'Unrelated',
                    'legalName': 'Unrelated Legal Corp'}],
    [ORGANIZATION, {**ORGANIZATION, 'legalName': 'Conflicting Legal Corp'}],
    {'@type': 'WebSite', 'publisher': ORGANIZATION},
])
def test_unbound_and_conflicting_organization_never_promote_legal_name(monkeypatch, record):
    result, anchor = _homepage(monkeypatch, 'Northstar Technology Solutions',
                               organization=record)
    assert result.decision != 'match'
    assert anchor == {}


def test_unstated_legal_alias_does_not_match(monkeypatch):
    result, anchor = _homepage(monkeypatch, 'Northstar Technology Partners')
    assert result.decision != 'match'
    assert anchor == {}


def test_verified_brand_reaches_named_investigator_quote(monkeypatch):
    _, anchor = _homepage(monkeypatch, 'Northstar Technology Solutions')
    page = 'https://www.northstar.example/services'
    quote = 'Northstar provides cloud infrastructure services to enterprise customers.'
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        finding = _finding('industry', observed_industry='Technology Services',
                           activity_role='supplier_operator', evidence_url=page,
                           evidence_quote=quote)
        return 200, {'choices': [{'message': {'tool_calls': [{
            'id': 'one', 'type': 'function', 'function': {
                'name': 'submit_findings',
                'arguments': json.dumps({'findings': [finding]}),
            },
        }]}}]}

    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-key')
    monkeypatch.setenv('EXA_API_KEY', 'offline-key')
    monkeypatch.setattr(investigator, '_post_json', post)
    monkeypatch.setattr(investigator, '_search_web', AsyncMock(side_effect=AssertionError('unexpected search')))
    monkeypatch.setattr(investigator, '_fetch_page', AsyncMock(side_effect=AssertionError('unexpected fetch')))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={'name': 'Northstar Technology Solutions',
                         'website': 'https://northstar.example/'},
        targets=('industry',), requested_industry='Technology Services',
        prior_observations={'observed_company_name': 'Northstar Technology Solutions Corporation',
                            'observed_company_website': URL,
                            'observed_company_linkedin': LINKEDIN,
                            'submitted_source_urls': [page]},
        verified_homepage_identity=anchor,
        prefetched_pages={page: {'final_url': page, 'text': quote}},
    ))
    assert result.get('_completed_submit') is True, result
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert len(requests) == 1


@pytest.mark.parametrize('failure_reason,expected', [
    ('source_blocked', True), ('provider_error', False), ('malformed_response', False),
])
def test_bound_brand_retains_exact_headcount_local_recovery(monkeypatch, failure_reason, expected):
    from tests.test_company_evidence_investigator import _icp
    _, identity = _homepage(monkeypatch, 'Northstar Technology Solutions')
    quote = 'Northstar\nCompany size\n10,001+ employees'
    finding = _finding('headcount', observed_value='10,001+',
                       evidence_url=LINKEDIN, evidence_quote=quote)
    investigation = {
        '_completed_submit': True, 'failure_reason': '',
        lead_scorer.PRIVATE_FETCHED_PAGES_KEY: {
            LINKEDIN: {'final_url': LINKEDIN, 'text': quote},
        },
    }
    icp = _icp(employee_count='10,001+')
    projected = lead_scorer._project_investigator_headcount(
        {}, finding, icp=icp, existing_conflict=False,
    )
    cache = {
        'refresh_outcome': 'retryable_failure',
        'profile_url': LINKEDIN,
        'evidence': {'outcome': 'insufficient_evidence', 'url': LINKEDIN},
        'structured_failure': True, 'structured_failure_reason': failure_reason,
        'failure_reason_code': failure_reason,
    }
    assert lead_scorer._settle_recovered_local_headcount_refresh(
        cache, finding, investigation, identity, projected, icp=icp,
    ) is expected
    assert cache['refresh_outcome'] == ('verified' if expected else 'retryable_failure')


def test_profile_domain_conflict_remains_unproved_despite_bound_brand(monkeypatch):
    from qualification.scoring import linkedin_company_size as size
    _, identity = _homepage(monkeypatch, 'Northstar Technology Solutions')
    text = '# Northstar\n## About us\nCompany size 10,001+ employees'
    assert size._profile_body_matches_company_identity(text, identity)
    assert not size._profile_body_matches_company_identity(
        text + '\nWebsite\nhttps://unproved.example/', identity,
    )
    forged = {**identity, 'verified_brand_name': 'Unrelated Brand',
              'verified_legal_name_aliases': ['Unrelated Legal Entity']}
    assert not size._profile_body_matches_company_identity(
        '# Unrelated Brand\n## About us\nCompany size 10,001+ employees', forged,
    )


def test_duplicate_brand_without_legal_name_does_not_erase_explicit_pair(monkeypatch):
    without_legal = {k: v for k, v in ORGANIZATION.items() if k != 'legalName'}
    result, anchor = _homepage(monkeypatch, 'Northstar Technology Solutions',
                               organization=[ORGANIZATION, without_legal])
    assert result.decision == 'match'
    assert anchor['verified_brand_name'] == 'Northstar'


def test_unrelated_root_does_not_erase_existing_brand_discovery(monkeypatch):
    other = {'@type': 'Organization', 'name': 'Unrelated',
             'url': 'https://unrelated.example/', 'sameAs': []}
    result, anchor = _homepage(monkeypatch, 'Northstar',
                               organization=[ORGANIZATION, other])
    assert result.decision == 'match'
    assert anchor['normalized_name'] == 'northstar'


@pytest.mark.parametrize('separate_scripts', [False, True])
def test_eleventh_conflicting_root_cannot_be_silently_ignored(monkeypatch, separate_scripts):
    records = [ORGANIZATION] * 10 + [{**ORGANIZATION, 'name': 'Unrelated',
                                     'legalName': 'Unrelated Legal Corp'}]
    if not separate_scripts:
        result, anchor = _homepage(monkeypatch, 'Northstar Technology Solutions',
                                   organization=records)
    else:
        body = '<title>Enterprise AI services</title>' + ''.join(
            '<script type="application/ld+json">' + json.dumps(record) + '</script>'
            for record in records
        )
        monkeypatch.setattr(verification.aiohttp, 'ClientSession',
                            lambda **kwargs: _Session(_Response(200, body.encode(), URL)))
        result = asyncio.run(verification.verify_company_exists(
            'Northstar Technology Solutions', 'https://northstar.example/',
            require_https_transport=True,
        ))
        anchor = lead_scorer._verified_homepage_identity_anchor(result)
    assert result.decision != 'match'
    assert anchor == {}


def test_model_alias_fields_cannot_create_homepage_authority():
    company = _company(name='Northstar Technology Solutions',
                       website='https://northstar.example/', linkedin=LINKEDIN)
    forged = {
        'observed_company_name': 'Northstar Technology Solutions Corporation',
        'observed_company_website': URL,
        'observed_company_linkedin': LINKEDIN,
        'verified_brand_name': 'Northstar',
        'verified_legal_name_aliases': ['Northstar Technology Solutions Corp'],
    }
    homepage = lead_scorer._verified_homepage_identity_anchor(None)
    receipt = lead_scorer._web_identity_receipt(
        company, forged, verified_homepage_identity=homepage,
    )
    context, _ = lead_scorer._investigator_identity_context(
        LINKEDIN, homepage, receipt, 'northstar.example',
    )
    assert homepage == {}
    assert 'verified_brand_name' not in receipt
    assert 'verified_legal_name_aliases' not in receipt
    assert 'verified_brand_name' not in context
    assert 'verified_legal_name_aliases' not in context


@pytest.mark.parametrize('overflow', [
    [ORGANIZATION] * 10 + [{**ORGANIZATION, 'legalName': 'Conflicting Legal Corp'}],
    [ORGANIZATION] * 21,
    {'@graph': [ORGANIZATION] * 21},
])
@pytest.mark.parametrize('overflow_first', [False, True])
def test_one_valid_script_cannot_erase_other_script_overflow(monkeypatch, overflow, overflow_first):
    scripts = [ORGANIZATION, overflow]
    if overflow_first:
        scripts.reverse()
    body = '<title>Enterprise AI services</title>' + ''.join(
        '<script type="application/ld+json">' + json.dumps(record) + '</script>'
        for record in scripts
    )
    monkeypatch.setattr(verification.aiohttp, 'ClientSession',
                        lambda **kwargs: _Session(_Response(200, body.encode(), URL)))
    result = asyncio.run(verification.verify_company_exists(
        'Northstar Technology Solutions', 'https://northstar.example/',
        require_https_transport=True,
    ))
    assert result.decision != 'match'
    assert lead_scorer._verified_homepage_identity_anchor(result) == {}
