"""Spend only the initial advisory skipped for an empty eligible corpus."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_judgment_reserve import _response


def setup(monkeypatch, *, decision='unavailable', source='bound', search_time=0,
          review_time=0, fault='', prefetched=False):
    run, calls, clock, reviews = install(monkeypatch, decision=decision,
                                       missing=['Retail media networks'] if decision != 'match' else [])
    original_post = c._post_json
    searches, researchers, reviews_at = [], [], []
    source_url = 'https://unrelated.example/product' if source == 'foreign' else URL

    async def search(session, query, **kwargs):
        searches.append((query, kwargs))
        clock.now = search_time
        return {'results': [], '_source_pages': (
            {} if source == 'empty' else {source_url: FEATURES}),
            'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            research = 'research_query' in payload['response_format']['json_schema']['schema']['properties']
            reviews_at.append((research, clock.now))
            if research:
                clock.now += review_time
                if fault == 'provider':
                    return 403, {'error': {'code': 'offline_provider_failure'}}
                if fault == 'malformed':
                    return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
            return await original_post(session, url, headers=headers, payload=payload)
        researchers.append(payload)
        return 200, _response('submit_findings', {'findings': [_finding(
            'industry', status='UNPROVEN', evidence_url='', evidence_quote='')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)

    def execute():
        return asyncio.run(run(**({'prefetched_pages': {URL: {'final_url': URL, 'text': FEATURES}},
                                  'prior_observations': {'submitted_source_urls': [URL]}}
                                 if prefetched else {})))

    return execute, searches, researchers, reviews, reviews_at, clock


def metadata(request):
    return json.loads(next(m['content'].split('\n', 1)[1] for m in request['messages']
        if m['role'] == 'user' and m['content'].startswith('Investigate this bounded request.')))


def test_discovered_bound_pages_replace_stale_zero_call_brief(monkeypatch):
    run, searches, researchers, reviews, times, _ = setup(monkeypatch)
    result = run()
    document = metadata(researchers[0])
    brief = document['industry_source_research_brief']
    assert brief['missing_requirements'] == ['Retail media networks']
    assert brief['source_urls'] == [URL]
    assert brief['excluded_source_urls'] == []
    assert brief['packet_sha256']
    assert document['investigation_limits']['reasoning_turns'] == c.MAX_REASONING_TURNS - 2  # Existing final-review reservation + this advisory.
    assert times[0][0] is True  # Advisory precedes researcher, not only final review.
    assert len(searches) == 1  # No automatic second gap search.
    assert len(reviews) == result['usage']['industry_review_calls'] == 2
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    # Advisory and researcher use the same original source bytes; no rewrite.
    assert FEATURES in json.dumps(reviews[0]['messages'])
    assert FEATURES in json.dumps(researchers[0]['messages'])
    def source_pages(messages):
        pages = []
        for message in messages:
            try:
                document = json.loads(message.get('content', ''))
            except (TypeError, ValueError):
                continue
            if isinstance(document, dict):
                pages.extend(document.get('untrusted_source_pages', []))
        return pages
    assert source_pages(reviews[0]['messages']) == source_pages(researchers[0]['messages'])


def test_deferred_completed_match_reuses_unchanged_packet(monkeypatch):
    run, searches, researchers, reviews, times, _ = setup(monkeypatch, decision='match')
    result = run()
    assert len(reviews) == result['usage']['industry_review_calls'] == 1
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['claims']['industry']['_industry_review']['reused_completed_packet'] is True
    assert len(searches) == 1 and times == [(True, 0)]


@pytest.mark.parametrize('source', ['foreign', 'empty'])
def test_discovery_without_eligible_pages_does_not_spend_advisory(monkeypatch, source):
    run, _, researchers, reviews, times, _ = setup(monkeypatch, source=source)
    result = run()
    assert not times and not reviews
    assert metadata(researchers[0])['industry_source_research_brief']['source_urls'] == []
    assert result['usage']['industry_review_calls'] == 0


@pytest.mark.parametrize('search_time', [80, 81, 110])
def test_discovery_at_research_reserve_never_starts_deferred_advisory(monkeypatch, search_time):
    run, _, _, _, times, _ = setup(monkeypatch, search_time=search_time)
    run()
    assert not any(research for research, _ in times)


def test_admitted_slow_advisory_does_not_extend_research_deadline(monkeypatch):
    run, searches, researchers, _, times, clock = setup(monkeypatch, search_time=50, review_time=44)
    result = run()
    assert times[0] == (True, 50)
    assert clock.now == 94  # An admitted 44-second call may consume the research reserve.
    assert len(searches) == 1
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert metadata(researchers[0])['investigation_limits']['remaining_search_calls'] == 0
    assert metadata(researchers[0])['investigation_limits']['remaining_fetch_calls'] == 0


def test_deferred_provider_failure_keeps_initial_failure_policy(monkeypatch):
    run, _, researchers, _, times, _ = setup(monkeypatch, fault='provider')
    result = run()
    assert result == {'claims': {}, 'failure_reason': c.PROVIDER_ERROR_FAILURE_REASON}
    assert times == [(True, 0)] and not researchers


def test_deferred_malformed_advisory_is_not_retried(monkeypatch):
    run, _, researchers, _, times, _ = setup(monkeypatch, fault='malformed')
    run()
    brief = metadata(researchers[0])['industry_source_research_brief']
    assert brief['advisory_failure_reason'] == c.MALFORMED_RESPONSE_FAILURE_REASON
    assert sum(research for research, _ in times) == 1


@pytest.mark.parametrize('fault', ['', 'malformed'])
def test_initial_already_called_advisory_is_not_repeated_after_discovery(monkeypatch, fault):
    run, _, _, _, times, _ = setup(monkeypatch, prefetched=True, fault=fault)
    run()
    assert sum(research for research, _ in times) == 1


@pytest.mark.parametrize('first_failure', [c.MALFORMED_RESPONSE_FAILURE_REASON,
    c.ADMISSION_BUDGET_INTERRUPTED_FAILURE_REASON])
def test_zero_call_failure_is_not_misread_as_no_eligible_source(monkeypatch, first_failure):
    run, _, _, _, times, _ = setup(monkeypatch)
    original = c._review_industry_sources
    first = True

    async def review(*args, **kwargs):
        nonlocal first
        result = await original(*args, **kwargs)
        if first:
            first = False
            return result[0], first_failure, 0
        return result

    monkeypatch.setattr(c, '_review_industry_sources', review)
    result = run()
    assert not any(research for research, _ in times)
    if first_failure != c.MALFORMED_RESPONSE_FAILURE_REASON:
        assert result['failure_reason'] == first_failure
