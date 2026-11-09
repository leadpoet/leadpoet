"""A positive capability review must still resolve a native attribution gap."""
import asyncio
import copy
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_industry_semantic_projection import CAPTURE, DOMAIN, URL, QUOTE


NAMED = 'https://www.aitworldwide.com/about-us/'
NAMED_QUOTE = 'AIT Worldwide Logistics provides truckload brokerage and freight transportation services.'


def _run(monkeypatch, fault='', *, search_cap=2):
    searches, reviews, controller = [], [], []
    initial = None
    original_review = c._review_industry_sources

    async def capture_review(*args, **kwargs):
        nonlocal initial
        result = await original_review(*args, **kwargs)
        if kwargs.get('research') and initial is None:
            initial = copy.deepcopy(result[0])
        return result

    async def search(_session, query, **options):
        searches.append((query, options))
        if fault == 'search_failure':
            raise RuntimeError('controlled search failure')
        body = QUOTE if fault == 'unresolved' else NAMED_QUOTE
        url = 'https://other.example/about/' if fault == 'foreign_source' else NAMED
        return {'results': [{'url': url, 'title': 'Company information'}],
                '_source_pages': {url: body}, 'notice': 'discovery_only_not_evidence'}

    async def fetch(*_args, **_kwargs):
        pytest.fail('Attribution recovery must use the existing source search, not add fetches')

    async def post(_session, _url, *, payload, **kwargs):
        if 'response_format' in payload:
            reviews.append(payload)
            spans = [s for message in payload['messages'][2:]
                     for page in json.loads(message['content'])['untrusted_source_pages']
                     for s in page['spans']]
            proposal = {'observed_activity': 'Truckload brokerage', 'reason': CAPTURE['reason'],
                        'missing_requirements': [], 'role': 'supplier_operator',
                        'evidence_span_ids': [s['span_id'] for s in spans], 'decision': 'match'}
            if 'research_query' in payload['response_format']['json_schema']['schema']['properties']:
                proposal['research_query'] = ''
            return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(proposal)}}]}
        controller.append(payload)
        return 200, _response('submit_findings', {'findings': [_finding('industry',
            status='UNPROVEN', evidence_url='', evidence_quote='', reason='Use source review')]})

    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-key')
    monkeypatch.setenv('EXA_API_KEY', 'offline-key')
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'offline-model')
    monkeypatch.setattr(c, 'MAX_SEARCH_CALLS', search_cap)
    monkeypatch.setattr(c, '_review_industry_sources', capture_review)
    monkeypatch.setattr(c, '_post_json', post)
    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_fetch_page', fetch)
    prior = {'observed_company_name': CAPTURE['company_name'],
             'observed_company_website': f'https://{DOMAIN}/',
             'observed_company_linkedin': 'https://www.linkedin.com/company/ait-worldwide-logistics'}
    if fault == 'missing_identity':
        prior = {}
    prior['submitted_source_urls'] = [URL]
    result = asyncio.run(c.investigate_company_evidence(
        company_locator={'name': CAPTURE['company_name'], 'website': f'https://{DOMAIN}/',
                         'linkedin': 'https://www.linkedin.com/company/ait-worldwide-logistics'},
        targets=('industry',), requested_industry='Logistics', requested_product_service='Freight brokerage',
        prefetched_pages={URL: {'final_url': URL, 'text': QUOTE}}, prior_observations=prior,
        verified_homepage_identity={}, positive_semantic_review=True))
    return result, searches, reviews, controller, initial


def test_saved_positive_passage_recovers_with_one_existing_named_source_search(monkeypatch):
    result, searches, reviews, controller, initial = _run(monkeypatch)
    assert initial['status'] == 'UNPROVEN'
    assert initial['reason'] == 'source quote did not identify the investigated company'
    # The semantic receipt is not changed into a rejection or given invented gaps.
    assert initial['_industry_review']['decision'] == 'match'
    assert initial['_industry_review']['missing_requirements'] == []
    assert initial['_industry_review']['research_query'] == ''
    assert len(searches) == 1
    assert 'first-party passage naming the investigated company' in searches[0][0]
    assert searches[0][1]['include_domains'] == (DOMAIN,)
    claim = result['claims']['industry']
    assert claim['status'] == 'VERIFIED'
    assert claim['evidence_url'] == NAMED and claim['evidence_quote'] == NAMED_QUOTE
    assert {'url': URL, 'quote': QUOTE} in claim['supporting_evidence']
    assert result['_completed_submit'] is True
    assert result['usage']['search_calls'] == 1
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert len(reviews) == 2 and controller
    request = next(message['content'] for message in controller[0]['messages']
                   if message['role'] == 'user' and message['content'].startswith('Investigate this bounded request.'))
    document = json.loads(request.split('\n', 1)[1])
    brief = document['industry_source_research_brief']
    assert brief['decision'] == 'match' and brief['missing_requirements'] == []
    assert brief['projection_rejection'] == initial['reason']


@pytest.mark.parametrize('fault', ['unresolved', 'foreign_source'])
def test_attribution_recovery_cannot_invent_source_or_identity_authority(monkeypatch, fault):
    result, searches, _, _, _ = _run(monkeypatch, fault)
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert len(searches) <= 1
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS


def test_spent_search_budget_does_not_add_an_attribution_retry_loop(monkeypatch):
    result, searches, _, _, _ = _run(monkeypatch, search_cap=0)
    assert not searches
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert result['usage']['search_calls'] == 0
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS


def test_no_independent_identity_never_triggers_attribution_search_or_claim(monkeypatch):
    result, searches, _, _, initial = _run(monkeypatch, 'missing_identity')
    assert initial['status'] == 'UNPROVEN'
    assert not searches
    assert not result['claims']
    assert result['failure_reason'] == 'malformed_response'
