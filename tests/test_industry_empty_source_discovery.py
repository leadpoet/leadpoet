"""A failed homepage must not disable the existing bound-source search."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding


@pytest.mark.parametrize('source_kind', ['bound', 'foreign', 'locator_only'])
def test_empty_initial_corpus_still_searches_before_final_judgment(monkeypatch, source_kind):
    run, _, _, reviews = install(monkeypatch)
    original_post = c._post_json
    searches, turns = [], []
    source = 'https://unrelated.example/product' if source_kind == 'foreign' else URL

    async def search(session, query, **kwargs):
        searches.append((query, kwargs))
        return {'results': [{'url': source, 'title': 'Software product'}],
                '_source_pages': {} if source_kind == 'locator_only' else {source: FEATURES},
                'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original_post(session, url, headers=headers, payload=payload)
        turns.append(payload)
        return 200, _response('submit_findings', {'findings': [_finding(
            'industry', status='UNPROVEN', evidence_url='', evidence_quote='')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(requested_product_service='Supplies analysis and reporting software'))
    assert searches and searches[0][1]['read_source_text'] is True
    assert searches[0][1]['include_domains'] == ('coreworks.ai',)
    assert 'Supplies analysis and reporting software' in searches[0][0]
    assert (FEATURES in json.dumps(turns[0])) is (source_kind == 'bound')
    assert result['claims']['industry']['status'] == ('VERIFIED' if source_kind == 'bound' else 'UNPROVEN')
    assert len(reviews) == (1 if source_kind == 'bound' else 0)
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['search_calls'] <= c.MAX_SEARCH_CALLS
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS


@pytest.mark.parametrize('block', ['identity', 'search_budget', 'late_search'])
def test_empty_corpus_discovery_preserves_identity_budget_and_deadline(monkeypatch, block):
    run, _, clock, reviews = install(monkeypatch, identity=(
        {'normalized_name': 'unrelated', 'registrable_dns_domain': 'unrelated.example'}
        if block == 'identity' else None))
    searches, turns = [], []

    async def search(session, query, **options):
        searches.append(options)
        if block == 'late_search':
            clock.now = c.ADMISSION_DEADLINE_SECONDS + 1
        return {'results': [], 'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        turns.append(payload)
        if payload.get('tool_choice') == {
                'type': 'function', 'function': {'name': 'search_web'}}:
            return 200, _response('search_web', {'query': 'Coreworks software'})
        return 200, _response('submit_findings', {'findings': [_finding(
            'industry', status='UNPROVEN', evidence_url='', evidence_quote='')]})

    if block == 'search_budget':
        monkeypatch.setattr(c, 'MAX_SEARCH_CALLS', 0)
    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run())
    assert len(searches) == (0 if block == 'search_budget' else 1)
    if block == 'identity':
        assert not any(options.get('read_source_text') for options in searches)
    assert not reviews
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['search_calls'] == len(searches)
