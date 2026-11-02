"""An unrelated failed finding must not erase checked industry source text."""
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as investigator, lead_scorer
from tests.test_matched_company_retry_source_continuity import IDENTITY, PLATFORM_URL, PLATFORM_TEXT, _fit_result
from tests.test_unresolved_industry_retry_sources import run_investigation


@pytest.mark.parametrize('status,decision', [
    ('VERIFIED', 'match'), ('CONTRADICTED', 'mismatch'), ('UNPROVEN', 'unavailable'),
])
@pytest.mark.parametrize('complete,failure', [(True, ''), (False, 'malformed_response')])
def test_completed_industry_review_keeps_all_sources_despite_other_finding_failure(
        monkeypatch, status, decision, complete, failure):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    contrary_url = 'https://unibuddy.com/limitations'
    pages = {
        PLATFORM_URL: {'final_url': PLATFORM_URL, 'text': PLATFORM_TEXT},
        contrary_url: {'final_url': contrary_url, 'text': 'Unibuddy does not supply campus learning delivery.'},
    }
    investigation = {
        '_completed_submit': complete, 'failure_reason': failure,
        'claims': {'industry': {'status': status, '_industry_review': {
            'schema': 'industry_evidence_review_v1', 'model': 'openai/gpt-6.1-sol',
            'packet_sha256': 'a' * 64, 'decision': decision,
            'source_urls': list(pages),
            'evidence': [{'url': PLATFORM_URL, 'quote': PLATFORM_TEXT}],
        }}},
        investigator.PRIVATE_FETCHED_PAGES_KEY: pages,
    }
    result, cache = _fit_result(), {}
    before = deepcopy((result, investigation))
    lead_scorer._retain_unresolved_industry_retry_sources(cache, result, investigation)
    assert lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY) == pages
    assert vars(result) == vars(before[0]) and investigation == before[1]
    # The next invocation receives both sides of the source record. It still
    # returns its own unresolved verdict; no previous positive is inherited.
    next_result, seen = run_investigation(monkeypatch, pages=pages, cache=cache)
    assert {**seen[0]['prefetched_pages'], **seen[0].get('retained_industry_pages', {})} == pages
    assert next_result[0]['industry_matches'] is None


@pytest.mark.parametrize('fault', ['no_claim', 'no_completed_review', 'bad_packet', 'foreign_source'])
def test_failed_investigation_cannot_cache_unreviewed_or_foreign_text(monkeypatch, fault):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    url = 'https://other.example/product' if fault == 'foreign_source' else PLATFORM_URL
    review = {'schema': 'industry_evidence_review_v1', 'model': 'openai/gpt-6.1-sol',
              'decision': 'match', 'packet_sha256': 'a' * 64, 'source_urls': [url]}
    if fault == 'no_completed_review':
        review.pop('packet_sha256')
    if fault == 'bad_packet':
        review['packet_sha256'] = 'invalid'
    investigation = {'_completed_submit': False, 'failure_reason': 'provider_error',
        'claims': {} if fault == 'no_claim' else {'industry': {'status': 'VERIFIED', '_industry_review': review}},
        investigator.PRIVATE_FETCHED_PAGES_KEY: {url: {'final_url': url, 'text': PLATFORM_TEXT}}}
    cache = {}
    lead_scorer._retain_unresolved_industry_retry_sources(cache, _fit_result(), investigation)
    assert cache == {}


def test_actual_targeted_caller_retains_sources_but_keeps_failed_judgment(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    cache = {}
    returned, _ = run_investigation(monkeypatch, cache=cache,
        investigation_changes={'_completed_submit': False, 'failure_reason': 'malformed_response'})
    assert returned[0]['industry_matches'] is None
    receipt = returned[1].details['investigation_receipt']
    assert receipt['completed_submitted_findings'] is False
    assert receipt['failure_reason'] == 'malformed_response'
    retained = lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY)
    assert set(retained) == {PLATFORM_URL, 'https://unibuddy.com/limitations'}
    following, seen = run_investigation(monkeypatch, cache=cache)
    assert {**seen[0]['prefetched_pages'], **seen[0].get('retained_industry_pages', {})} == retained
    assert following[0]['industry_matches'] is None
