"""Search the same observed employer sources that industry verification accepts."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from lab_arena.operations import validate_operation_request
from tests.test_industry_owned_careers_sources import BOARD, POSTING, LOCATORS
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding


@pytest.mark.parametrize('linked', [False, True])
def test_native_discovery_retains_only_observed_employer_body(monkeypatch, linked):
    body = 'Coreworks engineers build our account workflow and reporting product.'

    def assess(value):
        if len(value['evidence_span_ids']) < 2:
            value.update(decision='unavailable', missing_requirements=['Account workflows'])

    run, _, _, reviews = install(monkeypatch, mutation=assess)
    original = c._post_json
    searches = []

    async def search(session, query, **kwargs):
        searches.append(kwargs)
        return {'results': [{'url': POSTING, 'title': 'Own product engineering'}],
                '_source_pages': {POSTING: body}, 'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original(session, url, headers=headers, payload=payload)
        return 200, _response('submit_findings', {'findings': [_finding(
            'industry', status='UNPROVEN', observed_value=None, evidence_url='', evidence_quote='')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(
        prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
        homepage_navigation_locators=LOCATORS if linked else (),
        prior_observations={'submitted_source_urls': [URL]}))
    assert (BOARD.removeprefix('https://') in searches[0]['include_domains']) is linked
    assert (body in json.dumps(reviews[-1])) is linked
    assert result['claims']['industry']['status'] == ('VERIFIED' if linked else 'UNPROVEN')
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['search_calls'] == 1


def test_search_path_filter_preserves_exact_employer_and_first_party(monkeypatch):
    allowed = [POSTING, 'https://www.coreworks.ai/products']
    denied = [POSTING.replace('/coreworks/', '/coreworks-other/'),
              POSTING.replace('/coreworks/', '/another/'),
              POSTING.replace('greenhouse.io', 'greenhouse.io.evil.example'),
              'https://jobs.ashbyhq.com/coreworks/123']
    scopes = ('coreworks.ai', BOARD.removeprefix('https://'))

    async def post(session, url, *, headers, payload):
        validate_operation_request('exa.search', payload)
        assert payload['includeDomains'] == list(scopes)
        assert payload['numResults'] == c.MAX_INDUSTRY_SEARCH_RESULTS
        return 200, {'results': [{'url': u, 'text': 'Fetched own product description.'}
                                 for u in allowed + denied]}

    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_web(None, 'own product', key='offline',
        include_domains=scopes, discovery_highlights=True, read_source_text=True))
    assert list(result['_source_pages']) == allowed
    assert [x['url'] for x in result['results']] == allowed


@pytest.mark.parametrize('changes', [
    {'careers_locators': []},
    {'careers_locators': [{'url': BOARD, 'label': 'Partner'}]},
    {'careers_locators': [{'url': BOARD + '-other', 'label': 'Careers'}]},
    {'identity': {'submitted_domain': 'coreworks.ai'}},
])
def test_unbound_employer_cannot_expand_search_scope(changes):
    from tests.test_industry_owned_careers_sources import IDENTITY
    args = dict(domains={'coreworks.ai'}, identity=IDENTITY, careers_locators=LOCATORS)
    args.update(changes)
    assert all('greenhouse' not in scope for scope in c._industry_search_scopes(**args))


def test_alternative_index_preserves_employer_filter_without_accepting_search_text(monkeypatch):
    async def post(session, url, *, headers, payload):
        validate_operation_request('openrouter.chat', payload)
        return 200, {'choices': [{'message': {
            'content': '', 'annotations': [{'type': 'url_citation', 'url_citation': {
                'url': u, 'title': 'Own product'}} for u in [POSTING, POSTING.replace('/coreworks/', '/another/')]]}}]}

    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_industry_locators(None, 'own product', key='offline',
        company_locator={'name': 'Coreworks'}, include_domains=(BOARD.removeprefix('https://'),)))
    assert [v['url'] for v in result['results']] == [POSTING]
    assert '_source_pages' not in result
