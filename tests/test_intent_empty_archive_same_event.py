"""An empty archived source retains its exact-target recovery attestation."""
import asyncio

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from tests.test_intent_failed_archive_same_event import install_exact_failures, new_failure
from tests.test_intent_wayback_empty_body import SOURCE, SNAPSHOT, SHELL, ARTICLE_HTML
from tests.test_intent_unavailable_same_event_source import (
    ORIGINAL, fetched_source, judgment, run_verifier,
)


def empty_archive_failure():
    failure = new_failure()
    failure['statuses'][0]['sd_archive_failure']['stage'] = 'wayback_html_empty_body'
    return failure


@pytest.mark.parametrize('body', [SHELL, '<html><body></body></html>'])
def test_completed_empty_archive_retains_bound_receipt_through_exact_source_fallback(monkeypatch, body):
    calls = install_exact_failures(monkeypatch, archive=body)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert result['results'] == []
    assert result['statuses'][0]['sd_archive_failure'] == {
        'requested_url': SOURCE, 'archive_url': SNAPSHOT, 'stage': 'wayback_html_empty_body',
    }
    assert intent._unusable_source_allows_same_event_search(result, SOURCE)
    assert not intent._confirmed_source_absence(result, SOURCE)
    assert [method for method, _ in calls] == ['GET', 'GET', 'GET', 'POST', 'POST']


@pytest.mark.parametrize('final', [
    'https://web.archive.org/web/20261008185541/https://other.example/article',
    'https://web.archive.org.evil.example/web/20261008185541/' + SOURCE,
    'https://web.archive.org/web/20261008185541/' + SOURCE + '?different=1',
])
def test_empty_body_from_foreign_archive_target_does_not_permit_recovery(monkeypatch, final):
    install_exact_failures(monkeypatch, archive=SHELL, final_snapshot=final)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert not result['statuses'][0].get('sd_archive_failure')
    assert not intent._unusable_source_allows_same_event_search(result, SOURCE)


@pytest.mark.parametrize('mutation', [
    {'exa_target_crawl_failure': None},
    {'exa_target_crawl_failure': {'id_matches_requested_url': True, 'confirmed_attempts': 1,
                                'observations': [{'error_tag': 'CRAWL_EMPTY_CONTENT', 'error_http_status': 500}]}},
    {'exa_stage': 'exa_transient_exhausted'},
    {'sd_archive_failure': None},
])
def test_empty_archive_still_requires_two_completed_exact_crawl_failures(monkeypatch, mutation):
    source = empty_archive_failure()
    source['statuses'][0].update(mutation)
    result, fetch, search, judge = asyncio.run(run_verifier(monkeypatch, original=source))
    assert result['decision'] == 'unavailable'
    assert fetch.await_count == 1
    search.assert_not_awaited()
    judge.assert_not_awaited()


@pytest.mark.parametrize('negative', [None, 'wrong_entity', 'wrong_event', 'wrong_date', 'unfetched_quote'])
def test_empty_archive_recovery_keeps_source_entity_event_date_and_quote_guards(monkeypatch, negative):
    response, alternative = judgment(), fetched_source()
    item = response['answer']['signal_evaluations'][0]
    if negative == 'wrong_entity':
        item.update(signal_status='wrong_entity', same_entity_check='fail')
    elif negative == 'wrong_event':
        item['risk_notes'] = ['source_event_date:2026-05-20']
    elif negative == 'wrong_date':
        alternative['results'][0]['text'] = alternative['results'][0]['text'].replace('May 20, 2026', 'May 20, 2027')
    elif negative == 'unfetched_quote':
        alternative['results'][0]['text'] = 'Publisher general news and upcoming events.'
    result, fetch, search, judge = asyncio.run(run_verifier(
        monkeypatch, original=empty_archive_failure(), alternative=alternative, response=response,
    ))
    assert result['client_ready'] is (negative is None)
    assert fetch.await_count == 2 and search.await_count == judge.await_count == 1


def test_readable_archive_is_preserved_without_discovery(monkeypatch):
    calls = install_exact_failures(monkeypatch, archive=ARTICLE_HTML)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert result['results'] and 'agreement' in result['results'][0]['text']
    assert not intent._unusable_source_allows_same_event_search(result, SOURCE)
    assert [method for method, _ in calls] == ['GET', 'GET', 'GET']


@pytest.mark.parametrize('stage', ['wayback_exception', 'wayback_http_error', 'wayback_no_snapshot', [], {}])
def test_archive_transport_or_availability_failure_is_not_an_empty_body(stage):
    source = empty_archive_failure()
    source['statuses'][0]['sd_archive_failure']['stage'] = stage
    assert not intent._unusable_source_allows_same_event_search(source, ORIGINAL)
