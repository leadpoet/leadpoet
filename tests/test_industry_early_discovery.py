"""Spend the bounded research window fetching proof, not expanding search bodies."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding

SOURCE = 'https://www.coreworks.ai/workflows'
BODY = 'Coreworks supplies account workflow management with its reporting software.'


@pytest.mark.parametrize('fetch_ok', [False, True])
def test_initial_gap_discovers_locators_then_fetches_before_final_review(monkeypatch, fetch_ok):
    supplied = {'new': False}
    def assessment(result):
        if not supplied['new']:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])
    run, _, _, reviews = install(monkeypatch, mutation=assessment, locator_transport=True)
    original = c._post_json
    searches, fetches, turns = [], [], []

    async def no_full_text_search(*args, **kwargs):
        pytest.fail('Initial industry discovery must leave time to read selected sources')

    async def fetch(session, url, **kwargs):
        fetches.append(url)
        supplied['new'] = fetch_ok
        return ({'ok': True, 'url': url, 'final_url': url, 'text': BODY}
                if fetch_ok else {'ok': False, 'error': 'http_403'})

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original(session, url, headers=headers, payload=payload)
        if payload['model'] == 'perplexity/sonar':
            searches.append(payload)
            return 200, {'choices': [{'message': {
                'content': 'Invented evidence: ' + BODY,
                'annotations': [{'type': 'url_citation', 'url_citation': {'url': SOURCE}}],
            }}]}
        turns.append(payload)
        if len(turns) == 1:
            return 200, _response('fetch_page', {'url': SOURCE})
        return 200, _response('submit_findings', {'findings': [_finding('industry')]})

    monkeypatch.setattr(c, '_search_web', no_full_text_search)
    monkeypatch.setattr(c, '_fetch_page', fetch)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert len(searches) == 1 and fetches == [SOURCE]
    assert len(reviews) == result['usage']['industry_review_calls'] == 2
    assert 'Invented evidence' not in json.dumps(turns + reviews)
    assert BODY not in json.dumps(turns[0])
    assert (BODY in json.dumps(reviews[-1])) is fetch_ok
    assert result['claims']['industry']['status'] == ('VERIFIED' if fetch_ok else 'UNPROVEN')
    assert result['usage']['search_calls'] == result['usage']['fetch_calls'] == 1
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
