"""An unused exact fetch may fund one bounded company-domain source search."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding

NEW_URL = 'https://www.coreworks.ai/workflows'
NEW_BODY = 'Coreworks supplies account workflow management with its analysis and reporting software.'


def test_source_picker_can_rely_on_the_advertised_remaining_search_budget(monkeypatch):
    """A consumed search allowance must not hide the permitted fetch exchange."""
    supplied = {'new': False}
    def assessment(result):
        if not supplied['new']:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])
    run, _, _, _ = install(monkeypatch, mutation=assessment)
    monkeypatch.setattr(c, 'MAX_SEARCH_CALLS', 1)
    monkeypatch.setattr(c, 'MAX_FETCH_CALLS', 1)  # Last read uses full text, not locators.
    original_post = c._post_json
    searches, limits = [], []

    async def search(session, query, **kwargs):
        searches.append(kwargs)
        supplied['new'] = len(searches) > 1
        return {'results': [], '_source_pages': {NEW_URL: NEW_BODY} if supplied['new'] else {}}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original_post(session, url, headers=headers, payload=payload)
        request = next(m['content'] for m in payload['messages']
                       if m['role'] == 'user' and m['content'].startswith('Investigate this bounded request.'))
        budget = json.loads(request.split('\n', 1)[1])['investigation_limits']
        for message in payload['messages']:
            if message['role'] == 'tool':
                budget = json.loads(message['content']).get('remaining_research_budget', budget)
        limits.append(budget)
        if budget['remaining_search_calls'] > 0:
            return 200, _response('search_web', {'query': 'Coreworks account workflows'})
        return 200, _response('submit_findings', {'findings': [_finding('industry')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert limits[0]['remaining_search_calls'] == 1
    assert limits[0]['industry_search_exchange_available'] is True
    assert limits[-1]['remaining_search_calls'] == 0
    assert limits[-1]['industry_search_exchange_available'] is False
    assert limits[-1]['remaining_fetch_calls'] == limits[0]['remaining_fetch_calls'] - 1
    assert len(searches) == 2 and result['claims']['industry']['status'] == 'VERIFIED'


@pytest.mark.parametrize('late, capacity', [(False, None), (True, None), (False, 0), (False, 20)])
def test_native_remaining_gap_can_use_one_unused_fetch_without_extending_deadline(monkeypatch, late, capacity):
    observed = []
    supplied = {'new': False}

    def assessment(result):
        if not supplied['new']:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])

    run, calls, clock, reviews = install(monkeypatch, mutation=assessment)
    monkeypatch.setattr(c, 'MAX_FETCH_CALLS', 1)  # Preserve the final-read Exa path.
    if capacity is not None:
        monkeypatch.setattr(c.industry_classification, 'MAX_TOTAL_CHARS', len(FEATURES) + capacity)
    original_post = c._post_json
    turns = []

    async def search(session, query, **kwargs):
        observed.append(dict(kwargs))
        if len(observed) == 3:
            supplied['new'] = True
            return {'results': [{'url': NEW_URL}], '_source_pages': {NEW_URL: NEW_BODY},
                    'notice': 'discovery_only_not_evidence'}
        return {'results': [], '_source_pages': {}, 'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original_post(session, url, headers=headers, payload=payload)
        turns.append(payload)
        if len(turns) <= 2:
            if late and len(turns) == 2:
                clock.now = c.ADMISSION_DEADLINE_SECONDS - c.JUDGMENT_ADMISSION_RESERVE_SECONDS
            return 200, _response('search_web', {'query': 'Coreworks account workflow product'})
        return 200, _response('submit_findings', {'findings': [_finding('industry')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['_completed_submit'] is True
    exchanged = not late and capacity != 0
    assert len(observed) == (3 if exchanged else 2)
    assert result['usage']['search_calls'] == len(observed)
    assert result['usage']['search_calls'] + result['usage']['fetch_calls'] <= 6
    assert result['usage']['reasoning_turns'] <= 8
    input_message = next(m['content'] for m in turns[0]['messages']
                         if m['role'] == 'user' and m['content'].startswith('Investigate this bounded request.'))
    initial_budget = json.loads(input_message.split('\n', 1)[1])['investigation_limits']
    assert initial_budget['industry_search_exchange_available'] is (capacity != 0)
    for message in turns[-1]['messages']:
        if message['role'] == 'tool':
            tool_budget = json.loads(message['content'])['remaining_research_budget']
    assert tool_budget['industry_search_exchange_available'] is False
    if late:
        assert tool_budget['remaining_search_calls'] == tool_budget['remaining_fetch_calls'] == 0
    if not exchanged:
        assert result['claims']['industry']['status'] == 'UNPROVEN'
    else:
        assert observed[-1]['read_source_text'] is True
        assert observed[-1]['include_domains'] == ('coreworks.ai',)
        assert result['claims']['industry']['status'] == 'VERIFIED'
        assert NEW_URL in result['claims']['industry']['_industry_review']['source_urls']
        expected_body = NEW_BODY if capacity is None else NEW_BODY[:capacity]
        assert expected_body in json.dumps(turns[-1]['messages'])
        final_sources = json.dumps(reviews[-1]['messages'])
        assert FEATURES in final_sources and expected_body in final_sources


def test_exchange_consumes_fetch_allowance_and_cannot_repeat(monkeypatch):
    run, calls, clock, reviews = install(monkeypatch, decision='unavailable', missing=['Account workflows'])
    monkeypatch.setattr(c, 'MAX_FETCH_CALLS', 1)
    original_post = c._post_json
    requests = []
    searches = []
    fetches = []

    async def search(session, query, **kwargs):
        searches.append(kwargs)
        return {'results': [], '_source_pages': {}, 'notice': 'discovery_only_not_evidence'}

    async def fetch(*args, **kwargs):
        fetches.append(args)
        raise AssertionError('The exchanged fetch must not be available again')

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original_post(session, url, headers=headers, payload=payload)
        requests.append(payload)
        n = len(requests)
        if n in (1, 2, 4):
            return 200, _response('search_web', {'query': 'Coreworks account workflow product'})
        if n == 3:
            return 200, _response('fetch_page', {'url': NEW_URL})
        return 200, _response('submit_findings', {'findings': [_finding('industry')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    monkeypatch.setattr(c, '_fetch_page', fetch)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['_completed_submit'] is True
    assert len(searches) == 3 and not fetches
    assert result['usage']['search_calls'] == 3 and result['usage']['fetch_calls'] == 0
    tool_results = [json.loads(m['content']) for m in requests[-1]['messages'] if m['role'] == 'tool']
    assert any(t.get('error') == 'fetch_budget_exhausted' for t in tool_results)
    assert any(t.get('error') == 'search_budget_exhausted' for t in tool_results)
