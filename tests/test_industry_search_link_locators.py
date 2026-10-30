"""Source links guide an existing fetch; they never supply quote evidence."""
import asyncio
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as c


def search(monkeypatch, results, **options):
    requests = []

    async def post(*args, **kwargs):
        requests.append(kwargs['payload'])
        validate_operation_request('exa.search', kwargs['payload'])
        return 200, {'results': results}

    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_web(None, 'Product applications', key='test',
        include_domains=('example.com', 'jobs.lever.co/example'),
        discovery_highlights=True, read_source_text=True, **options))
    return result, requests


def test_search_preserves_linked_document_locator_without_inventing_its_body(monkeypatch):
    source, linked = 'https://example.com/resources', 'https://example.com/guide.pdf'
    result, requests = search(monkeypatch, [{'url': source, 'text': 'Product resources.',
        'extras': {'links': [linked]}}])
    assert requests[0]['contents']['extras'] == {'links': c.MAX_HOMEPAGE_NAVIGATION_LOCATORS}
    assert [v['url'] for v in result['results']] == [source, linked]
    assert result['_source_pages'] == {source: 'Product resources.'}
    assert result['notice'] == 'discovery_only_not_evidence'
    assert result['results'][1]['title'].startswith('Linked from ')
    assert requests[0]['numResults'] == 10 and len(requests) == 1


def test_links_cannot_expand_company_or_employer_authority(monkeypatch):
    source = 'https://example.com/resources'
    linked = 'https://jobs.lever.co/example/12345'
    result, _ = search(monkeypatch, [{'url': source, 'text': 'Resources.',
        'extras': {'links': [source, linked, linked, 'https://other.example/guide.pdf',
            'https://example.com.evil.org/guide.pdf', 'https://jobs.lever.co/example2/12345',
            'https://jobs.lever.co/other/12345', 'https://127.0.0.1/private',
            'https://user:secret@example.com/private', 'http://example.com/guide.pdf',
            {'url': 'https://example.com/not-a-string'}, 'javascript:alert(1)']}}])
    assert [v['url'] for v in result['results']] == [source, linked]
    assert set(result['_source_pages']) == {source}


def test_links_do_not_displace_search_results_and_obey_shared_navigation_bounds(monkeypatch):
    originals = [f'https://example.com/result/{i}' for i in range(10)]
    rows = [{'url': url, 'text': 'Product facts.', 'extras': {'links': [
        f'https://example.com/linked/{i}/' + 'a' * 160 for i in range(100)]}} for url in originals]
    result, _ = search(monkeypatch, rows)
    locators = result['results']
    assert [x['url'] for x in locators[:10]] == originals
    assert 0 < len(locators) - 10 <= c.MAX_HOMEPAGE_NAVIGATION_LOCATORS
    assert sum(len(x['url']) + len(x['title']) for x in locators[10:]) <= c.MAX_HOMEPAGE_NAVIGATION_TOTAL_CHARACTERS
    assert len(c._bounded_message_json(result)) < c.OPENROUTER_MAX_CONTENT_CHARS


def test_navigation_metadata_cannot_crowd_out_existing_request(monkeypatch):
    rows = [{'url': 'https://example.com/resources', 'text': 'Facts.', 'extras': {'links': [
        f'https://example.com/linked/{i}/' + 'a' * 190 for i in range(40)]}}]
    result, _ = search(monkeypatch, rows)
    result.pop('_source_pages')
    # The same discovery occurs in the first request and in a tool response.
    for document in [result, {'server_industry_discovery': result}]:
        document['existing_metadata'] = 'x' * 12500
        encoded = c._bounded_message_json(document)
        assert len(encoded) < c.OPENROUTER_MAX_CONTENT_CHARS
        assert json.loads(encoded)['existing_metadata'] == 'x' * 12500


@pytest.mark.parametrize('broken', [{'extras': {'links': 'not-a-list'}},
    {'extras': ['not-an-object']}, {'status': 'error'}, {'error': 'crawl failed'}])
def test_invalid_or_failed_source_cannot_contribute_links(monkeypatch, broken):
    row = {'url': 'https://example.com/resources', 'text': 'Resources.',
           'extras': {'links': ['https://example.com/guide.pdf']}, **broken}
    result, _ = search(monkeypatch, [row])
    assert [item['url'] for item in result['results']] == ['https://example.com/resources']


def test_native_loop_fetches_link_before_retaining_its_quote(monkeypatch):
    from tests.test_native_industry_review import install
    from tests.test_company_evidence_refusal_fallback import FEATURES
    from qualification.scoring import lead_scorer

    actual_search = c._search_web
    home, document = 'https://www.coreworks.ai/', 'https://www.coreworks.ai/product'
    fresh = 'Coreworks supplies account workflows, analysis and reporting.'
    reads = []

    def finding(result):
        if not reads:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])

    run, calls, _, reviews = install(monkeypatch, mutation=finding)
    original_post, original_fetch = c._post_json, c._fetch_page

    async def post(session, url, **kwargs):
        if url == 'https://api.exa.ai/search':
            return 200, {'results': [{'url': home, 'text': FEATURES,
                'extras': {'links': [document]}}]}
        return await original_post(session, url, **kwargs)

    async def fetch(session, url, **kwargs):
        if url == document:
            reads.append(url)
            return {'ok': True, 'url': url, 'final_url': url, 'text': fresh}
        return await original_fetch(session, url, **kwargs)

    monkeypatch.setattr(c, '_post_json', post)
    monkeypatch.setattr(c, '_fetch_page', fetch)
    monkeypatch.setattr(c, '_search_web', actual_search)
    result = asyncio.run(run(prior_observations={'submitted_source_urls': [home]},
        prefetched_pages={home: {'final_url': home, 'text': FEATURES}},
        requested_product_service='Supplies analysis, reporting and account workflows'))
    assert reads == [document]
    assert document in json.dumps(calls['model'][0]['messages'])
    assert fresh not in json.dumps(reviews[0])
    claim = result['claims']['industry']
    assert {'url': document, 'quote': fresh} in claim['_industry_review']['evidence']
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][document] == {'final_url': document, 'text': fresh}
    projected = lead_scorer._project_investigator_industry({}, claim)
    assert fresh in json.dumps(lead_scorer._dimension_web_evidence(projected, 'industry'))
    for payload in [*calls['model'], *reviews]:
        validate_operation_request('openrouter.chat', payload)
    assert result['usage']['search_calls'] == 1
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
