"""Newly retrieved proof must replace stale gaps before the next source choice."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding

NEW_URL = 'https://www.coreworks.ai/workflows'
NEW_BODY = 'Coreworks supplies account workflow management with its reporting software.'


@pytest.mark.parametrize('source_case', ['new', 'same', 'foreign', 'late', 'malformed'])
def test_discovery_refreshes_only_new_admissible_evidence_within_existing_budget(monkeypatch, source_case):
    assessments = []
    def assessment(result):
        assessments.append(result)
        result.update(decision='unavailable', missing_requirements=(
            ['Email channel'] if len(assessments) > 1 else ['Account workflows', 'Email channel']))
    run, _, clock, reviews = install(monkeypatch, mutation=assessment)
    original_post = c._post_json
    requests, searches = [], []

    async def search(session, query, **kwargs):
        searches.append(query)
        if source_case == 'late':
            clock.now = c.ADMISSION_DEADLINE_SECONDS - c.JUDGMENT_ADMISSION_RESERVE_SECONDS
        pages = {NEW_URL: NEW_BODY}
        if source_case == 'same':
            pages = {URL: FEATURES}
        if source_case == 'foreign':
            pages = {'https://other.example/workflows': NEW_BODY}
        return {'results': [], '_source_pages': pages, 'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            if source_case == 'malformed' and len(reviews) == 1:
                reviews.append(payload)
                return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': 'invalid'}}]}
            return await original_post(session, url, headers=headers, payload=payload)
        requests.append(payload)
        return 200, _response('submit_findings', {'findings': [_finding('industry')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    document = json.loads(next(m['content'].split('\n', 1)[1] for m in requests[0]['messages']
        if m['role'] == 'user' and m['content'].startswith('Investigate this bounded request.')))
    refreshed = source_case == 'new'
    assert document['industry_source_research_brief']['missing_requirements'] == (
        ['Email channel'] if refreshed else ['Account workflows', 'Email channel'])
    assert bool(document['industry_source_research_brief'].get('refreshed_after_discovery')) is refreshed
    refresh_attempted = source_case in ('new', 'malformed')
    assert document['investigation_limits']['reasoning_turns'] == c.MAX_REASONING_TURNS - 2 - int(refresh_attempted)
    assert len(reviews) == result['usage']['industry_review_calls'] == 2 + int(refresh_attempted)
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert result['usage']['search_calls'] == len(searches) == 1
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert 'https://other.example/workflows' not in result['claims']['industry']['_industry_review']['source_urls']
    if source_case == 'malformed':
        assert document['industry_source_research_brief']['advisory_failure_reason'] == 'malformed_response'


def test_refresh_provider_failure_does_not_become_a_negative_verdict(monkeypatch):
    run, _, _, reviews = install(monkeypatch, decision='unavailable', missing=['Account workflows'])
    original_post = c._post_json
    async def search(*args, **kwargs):
        return {'results': [], '_source_pages': {NEW_URL: NEW_BODY}}
    async def post(session, url, *, headers, payload):
        if len(reviews) == 1:
            return 503, {'error': 'provider unavailable'}
        return await original_post(session, url, headers=headers, payload=payload)
    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['failure_reason'] == 'provider_error'
    assert not result['claims'] and not result.get('_completed_submit')
