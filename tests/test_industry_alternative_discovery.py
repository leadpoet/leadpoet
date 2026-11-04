"""A second search index supplies locators, never evidence or a verdict."""
import asyncio
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding

SOURCE = 'https://www.coreworks.ai/workflows'
BODY = 'Coreworks supplies account workflow management with its reporting software.'


@pytest.mark.parametrize('formatted, expected', [
    (f'**{SOURCE}**', SOURCE),
    (f'*{SOURCE}*', SOURCE),
    (f'***{SOURCE}***.', SOURCE),
    (f'__{SOURCE}__', SOURCE),
    (f'_{SOURCE}_', SOURCE),
    (f'**{SOURCE}**[1][2]', SOURCE),
    (f'{SOURCE}[1][2].', SOURCE),
    (f'`{SOURCE}`[1]', SOURCE),
    (f'[Workflow docs]({SOURCE})', SOURCE),
    (SOURCE + '/literal*', SOURCE + '/literal*'),
    (SOURCE + '/literal_', SOURCE + '/literal_'),
    (SOURCE + '?path=%2A&version=1', SOURCE + '?path=%2A&version=1'),
    ('**https://coreworks.ai.evil.example/workflows**', None),
    ('**https://user@coreworks.ai/workflows**', None),
])
def test_search_url_formatting_does_not_change_locator_or_authority(monkeypatch, formatted, expected):
    async def post(*args, **kwargs):
        return 200, {'choices': [{'message': {
            'content': f'- {formatted} — Account workflow documentation.',
            'annotations': ([{'type': 'url_citation', 'url_citation': {
                'url': expected, 'title': 'Workflow docs'}}] if expected else []),
        }}]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_industry_locators(
        None, 'account workflows', key='test',
        company_locator={'name': 'Coreworks', 'website': URL},
        include_domains=('coreworks.ai',)))
    assert [r['url'] for r in result['results']] == ([expected] if expected else [])
    if expected:
        assert result['results'][0]['title'] == 'Workflow docs'
        assert 'Account workflow documentation.' in result['results'][0]['discovery_excerpt']
    assert result['notice'] == 'discovery_only_not_evidence'
    assert '_source_pages' not in result


def test_alternative_search_keeps_bound_locators_without_promoting_claims(monkeypatch):
    seen = []
    async def post(session, url, *, headers, payload):
        validate_operation_request('openrouter.chat', payload)
        seen.append(payload)
        urls = [SOURCE, SOURCE, 'https://other.example/facts',
                'https://coreworks.ai.evil.example/facts', 'http://coreworks.ai/facts',
                'https://user@coreworks.ai/facts', 'https://127.0.0.1/facts']
        return 200, {'choices': [{'message': {
            'content': 'VERIFIED. Invented proof: https://coreworks.ai/not-cited',
            'annotations': [{'type': 'url_citation', 'url_citation': {
                'url': value, 'title': 'Workflow documentation'}} for value in urls],
        }}]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_industry_locators(
        None, 'account workflows', key='test',
        company_locator={'name': 'Coreworks', 'website': URL},
        include_domains=('coreworks.ai',)))
    assert [r['url'] for r in result['results']] == [
        'https://coreworks.ai/not-cited', SOURCE]
    assert result['notice'] == 'discovery_only_not_evidence'
    assert '_source_pages' not in result
    assert seen[0]['model'] == 'perplexity/sonar'


@pytest.mark.parametrize('status, body, expected', [
    (503, {}, RuntimeError),
    (200, {'error': 'failed'}, RuntimeError),
    (200, {'choices': []}, ValueError),
    (200, {'choices': [{'message': {'refusal': 'No'}}]}, ValueError),
    (200, {'choices': [{'message': {'annotations': 'invalid'}}]}, ValueError),
])
def test_alternative_search_errors_are_not_negative_company_evidence(monkeypatch, status, body, expected):
    async def post(*args, **kwargs):
        return status, body
    monkeypatch.setattr(c, '_post_json', post)
    with pytest.raises(expected):
        asyncio.run(c._search_industry_locators(
            None, 'account workflows', key='test',
            company_locator={'name': 'Coreworks', 'website': URL},
            include_domains=('coreworks.ai',)))


@pytest.mark.parametrize('fetch_ok', [False, True])
def test_native_exchange_requires_a_fresh_page_before_alternative_locator_can_help(monkeypatch, fetch_ok):
    supplied = {'new': False}
    def assessment(result):
        if not supplied['new']:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])
    run, _, _, reviews = install(monkeypatch, mutation=assessment, locator_transport=True)
    original = c._post_json
    turns, searches, alternatives, fetches = [], [], [], []

    async def search(*args, **kwargs):
        searches.append(kwargs)
        return {'results': [], '_source_pages': {}, 'notice': 'discovery_only_not_evidence'}

    async def fetch(session, url, **kwargs):
        fetches.append(url)
        supplied['new'] = fetch_ok
        if not fetch_ok:
            return {'ok': False, 'error': 'http_403'}
        return {'ok': True, 'url': url, 'final_url': url, 'text': BODY}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original(session, url, headers=headers, payload=payload)
        if payload['model'] == 'perplexity/sonar':
            alternatives.append(payload)
            return 200, {'choices': [{'message': {
                'content': f'{SOURCE} — {BODY}',
                'annotations': [],
            }}]}
        turns.append(payload)
        if len(turns) <= 2:
            return 200, _response('search_web', {'query': 'Coreworks account workflows'})
        if len(turns) == 3:
            return 200, _response('fetch_page', {'url': SOURCE})
        return 200, _response('submit_findings', {'findings': [_finding('industry')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_fetch_page', fetch)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert len(searches) == 2 and len(alternatives) == 1 and fetches == [SOURCE]
    assert BODY in json.dumps(turns[2])  # A search hint can guide source selection.
    # Only the independent fetched body can enter the final evidence packet.
    assert (BODY in json.dumps(reviews[-1])) is fetch_ok
    assert result['claims']['industry']['status'] == ('VERIFIED' if fetch_ok else 'UNPROVEN')
    assert result['usage']['search_calls'] == 3
    assert result['usage']['fetch_calls'] == 1
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    budgets = [json.loads(m['content']) for m in turns[-1]['messages'] if m['role'] == 'tool']
    alternative = next(t for t in budgets if t.get('industry_locator_fallback_used'))
    assert alternative['remaining_fetch_calls'] == c.MAX_FETCH_CALLS - 1
    assert alternative['remaining_research_budget']['remaining_search_calls'] == 0
