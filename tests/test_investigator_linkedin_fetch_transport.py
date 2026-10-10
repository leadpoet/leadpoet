"""Exact LinkedIn fetches reuse validated source text, never structured quotes."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_current_linkedin_company_size import _install_exa_bodies
from tests.test_linkedin_headcount_conflict_resolution import _caller, _fixture, _investigation

PROFILE = 'https://www.linkedin.com/company/br-dge'
TEXT = 'BR-DGE\nCompany size\n51-200 employees\nHeadquarters\nEdinburgh'
DIAGNOSTIC = json.dumps({'message': (
    'To scrape Linkedin use our dedicated Linkedin API. Docs - '
    'https://docs.scrapingdog.com/linkedin-scraper-api or try directly from your dashboard'
)})


def test_exact_profile_fetch_reuses_raw_validated_exa_text_without_rendered_call(monkeypatch):
    calls, pending = _install_exa_bodies(monkeypatch, {
        'results': [{'url': PROFILE, 'text': TEXT}],
    })
    rendered = AsyncMock(side_effect=AssertionError('exact profile must use Exa contents'))
    monkeypatch.setattr(investigator, '_fetch_bounded_html', rendered)
    monkeypatch.setenv('SCRAPINGDOG_API_KEY', 'test-key')
    result = asyncio.run(investigator._fetch_page(object(), PROFILE))
    assert result == {'ok': True, 'url': PROFILE, 'final_url': PROFILE, 'text': TEXT}
    assert calls[0][0] == 'https://api.exa.ai/contents'
    assert calls[0][1]['json']['ids'] == [PROFILE]
    assert calls[0][1]['json']['maxAgeHours'] == 0
    assert len(calls) == 1 and pending == []
    rendered.assert_not_awaited()


@pytest.mark.parametrize('unsafe', [
    'wrong_profile', 'off_domain', 'access_wall', 'missing_text', 'no_result',
    'too_long', 'provider_diagnostic',
])
def test_exact_profile_cannot_expose_unvalidated_or_diagnostic_source(monkeypatch, unsafe):
    body = {'results': [{'url': PROFILE, 'text': TEXT}]}
    source = body['results'][0]
    if unsafe == 'wrong_profile': source['url'] = 'https://www.linkedin.com/company/other'
    elif unsafe == 'off_domain': source['url'] = 'https://example.com/company/br-dge'
    elif unsafe == 'access_wall': source['text'] = 'Sign in | LinkedIn\nEmail or phone\nPassword'
    elif unsafe == 'missing_text': source.pop('text')
    elif unsafe == 'no_result':
        body = {'results': [], 'statuses': [{'id': PROFILE, 'status': 'error', 'error': {
            'httpStatusCode': 404, 'tag': 'CRAWL_NOT_FOUND',
        }}]}
    elif unsafe == 'too_long': source['text'] = TEXT + ('x' * 4000)
    elif unsafe == 'provider_diagnostic': source['text'] = DIAGNOSTIC
    calls, pending = _install_exa_bodies(monkeypatch, body)
    rendered = AsyncMock(side_effect=AssertionError('no alternate hidden fetch'))
    monkeypatch.setattr(investigator, '_fetch_bounded_html', rendered)
    result = asyncio.run(investigator._fetch_page(object(), PROFILE))
    assert result['ok'] is False
    assert 'text' not in result
    assert len(calls) == 1 and pending == []
    rendered.assert_not_awaited()


def test_profile_without_size_still_exposes_only_its_actual_fetched_text(monkeypatch):
    text = 'BR-DGE\nPayment orchestration\nNo company size published'
    _install_exa_bodies(monkeypatch, {'results': [{'url': PROFILE, 'text': text}]})
    result = asyncio.run(investigator._fetch_page(object(), PROFILE))
    assert result['ok'] is True
    assert result['text'] == text
    assert '51-200' not in result['text']


def test_brdge_caller_resolves_conflict_from_existing_exa_fetch_and_exact_quote(monkeypatch):
    calls, _ = _install_exa_bodies(monkeypatch, {'results': [{'url': PROFILE, 'text': TEXT}]})
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    company, icp, verdict = _fixture()
    company.company_linkedin = ''  # The archived submission omitted this field.
    quote = 'BR-DGE\nCompany size\n51-200 employees'
    finding = _investigation()['claims']['headcount']
    finding['evidence_quote'] = quote
    model_calls = []

    async def post(_session, _url, *, headers, payload):
        model_calls.append(payload)
        name, args = (
            ('fetch_page', {'url': PROFILE}) if len(model_calls) == 1 else
            ('submit_findings', {'findings': [finding]})
        )
        return 200, {'choices': [{'message': {'tool_calls': [{
            'id': str(len(model_calls)), 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)},
        }]}}]}

    monkeypatch.setattr(investigator, '_post_json', post)
    projected, result, *_ = asyncio.run(_caller(company, icp, verdict))
    assert projected['observed_employee_count'] == '51-200'
    assert projected['employee_size_evidence_quote'] == quote
    assert result.details['dimension_decisions']['employee_size'] == 'match'
    assert result.details['investigation_receipt']['usage']['fetch_calls'] == 1
    assert len(calls) == 1


@pytest.mark.parametrize('body,rejected', [
    (DIAGNOSTIC, True),
    ('provider account capacity details redacted.', True),
    ('BR-DGE Company size 51-200 employees', False),
    (json.dumps({'message': 'BR-DGE offers an API for payment orchestration.'}), False),
    ('Our LinkedIn integration is documented at https://docs.scrapingdog.com/linkedin-scraper-api', False),
])
def test_generic_fetch_rejects_only_confirmed_standalone_provider_diagnostic(monkeypatch, body, rejected):
    async def bounded(_session, url): return 200, url, body
    monkeypatch.delenv('SCRAPINGDOG_API_KEY', raising=False)
    monkeypatch.delenv('QUALIFICATION_SCRAPINGDOG_API_KEY', raising=False)
    monkeypatch.setattr(investigator, '_fetch_bounded_html', bounded)
    result = asyncio.run(investigator._fetch_page(object(), 'https://example.com/about'))
    assert result['ok'] is not rejected
    if rejected: assert result['error'] == 'provider_diagnostic_body'


@pytest.mark.parametrize('url', [
    'https://example.com/company/br-dge',
    'https://www.linkedin.com/company/br-dge/about',
    'https://www.linkedin.com/in/br-dge',
])
def test_non_exact_company_profile_does_not_use_company_size_route(monkeypatch, url):
    exa = AsyncMock(side_effect=AssertionError('not an exact LinkedIn company profile'))
    monkeypatch.setattr(investigator, 'fetch_current_linkedin_company_size', exa)
    monkeypatch.delenv('SCRAPINGDOG_API_KEY', raising=False)
    monkeypatch.delenv('QUALIFICATION_SCRAPINGDOG_API_KEY', raising=False)
    monkeypatch.setattr(investigator, '_fetch_bounded_html', AsyncMock(return_value=(200, url, 'Ordinary visible text')))
    result = asyncio.run(investigator._fetch_page(object(), url))
    assert result['ok'] is True
    exa.assert_not_awaited()
