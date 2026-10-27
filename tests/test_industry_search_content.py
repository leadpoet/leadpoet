"""Full search content is a bounded source read, never a search snippet."""
import asyncio
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as c
from qualification.scoring import industry_classification as industry
from tests.test_industry_classification import CRITERION, COMPANY


def test_exact_domain_source_text_is_retained_without_trusting_highlights(monkeypatch):
    requests = []
    async def post(session, url, *, headers, payload):
        requests.append(payload)
        validate_operation_request('exa.search', payload)
        return 200, {'results': [
            {'url':'https://example.com/service','text':'Example supplies account workflows.',
             'highlights':['Invented extra capability'], 'summary':'Invented summary'},
            {'url':'https://docs.example.com/tickets','text':'Example supplies ticket management.'},
            {'url':'https://other.example/service','text':'Wrong company'},
            {'url':'https://example.com.evil.test/service','text':'Lookalike company'},
            {'url':'https://example.com/snippet','highlights':['No full text']},
            {'url':'https://example.com/error','text':'Invented proof','error':'crawl failed'},
            {'url':'https://example.com/pdf','text':'%PDF-1.7 binary'},
        ]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_web(None, 'Example account workflows', key='test-key',
        include_domains=('example.com',), discovery_highlights=True, read_source_text=True))
    assert requests[0]['contents'] == {'text':{'maxCharacters':24000}, 'maxAgeHours':0}
    assert requests[0]['numResults'] == 10
    assert result.pop('_source_pages') == {
        'https://example.com/service':'Example supplies account workflows.',
        'https://docs.example.com/tickets':'Example supplies ticket management.',
    }
    assert result['notice'] == 'discovery_only_not_evidence'
    assert all(set(x) <= {'url','title','discovery_excerpt'} for x in result['results'])
    assert 'Invented' not in json.dumps(result)


def test_source_text_requires_the_host_scoped_industry_route():
    with pytest.raises(ValueError, match='scoped industry'):
        asyncio.run(c._search_web(None, 'query', key='test-key', read_source_text=True))


def test_failed_url_status_cannot_supply_source_text(monkeypatch):
    async def post(*args, **kwargs):
        return 200, {'results':[{'url':'https://example.com/service','text':'Stale proof'}],
                     'statuses':[{'id':'https://example.com/service','status':'error'}]}
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(c._search_web(None,'query',key='test-key',include_domains=('example.com',),
        discovery_highlights=True,read_source_text=True))
    assert result['_source_pages'] == {}


@pytest.mark.parametrize('enabled', [False, True])
def test_larger_private_source_map_is_only_valid_under_new_frozen_policy(monkeypatch, enabled):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', industry.DEFAULT_MODEL if enabled else '')
    pages = {f'https://example.com/{i}':{'final_url':f'https://example.com/{i}', 'text':'Example product'} for i in range(8)}
    actual, _ = c._validated_returned_fetched_pages(pages, submitted_source_urls=tuple(pages))
    assert bool(actual) is enabled
    # The input prefetch limit never changes.
    assert c._validated_prefetched_pages(pages, submitted_source_urls=tuple(pages)) == ({},{})
    oversized = {f'https://example.com/{i}':{'final_url':f'https://example.com/{i}', 'text':'Example product'} for i in range(c.max_returned_fetched_pages() + 1)}
    assert c._validated_returned_fetched_pages(oversized, submitted_source_urls=tuple(oversized)) == ({},{})


def test_larger_source_count_keeps_the_original_total_context_budget():
    pages = [{'source_id':f's{i}','source_url':f'https://example.com/{i}', 'body':'X'*24000} for i in range(industry.MAX_PAGES)]
    packet = industry.prepare(CRITERION, COMPANY, pages)
    assert len(packet.pages) == industry.MAX_PAGES
    assert sum(len(body) for _,body,_ in packet.pages) <= 7*24000
    assert max(len(body) for _,body,_ in packet.pages) - min(len(body) for _,body,_ in packet.pages) <= 1
    validate_operation_request('openrouter.chat', industry.request_payload(packet, model=industry.DEFAULT_MODEL))
    legacy = industry.prepare(CRITERION, COMPANY, pages[:7])
    assert all(len(body) == 24000 for _,body,_ in legacy.pages)


def test_native_loop_reuses_full_search_body_and_retains_its_exact_quote(monkeypatch):
    from tests.test_native_industry_review import install
    from tests.test_company_evidence_refusal_fallback import FEATURES, URL
    from tests.test_company_evidence_investigator import _finding
    from qualification.scoring import lead_scorer
    search = c._search_web
    home = 'https://www.coreworks.ai/'
    service = 'https://www.coreworks.ai/product'
    full_text = 'Coreworks supplies analysis, reporting and account workflow software.'
    seen = []
    def initial_gap(result):
        seen.append(result)
        if len(seen) == 1:
            result.update(decision='unavailable', missing_requirements=['Account workflow offering'])
    finding = _finding('industry',status='VERIFIED',evidence_url=service,evidence_quote=full_text,
                       activity_role='supplier_operator',observed_industry='Software')
    run, calls, clock, reviews = install(monkeypatch, mutation=initial_gap, finding=finding)
    original_post = c._post_json
    paid_searches = []
    async def post(session,url,**kwargs):
        if url == 'https://api.exa.ai/search':
            paid_searches.append(kwargs['payload'])
            return 200, {'results':[{'url':service, 'text':full_text}, {'url':URL,'text':FEATURES}]}
        return await original_post(session,url,**kwargs)
    monkeypatch.setattr(c,'_post_json',post)
    monkeypatch.setattr(c,'_search_web',search)
    result = asyncio.run(run(prior_observations={'submitted_source_urls':[home]},
        prefetched_pages={home:{'final_url':home,'text':FEATURES}},
        requested_product_service='Supplies analysis, reporting and account workflow software'))
    # Source selection must see newly loaded bodies, not only the stale gap.
    assert full_text in json.dumps(calls['model'][0]['messages'])
    assert calls['model'][0]['tool_choice'] == 'required'
    for request in calls['model']:
        validate_operation_request('openrouter.chat', request)
    assert result['_completed_submit'] is True
    claim = result['claims']['industry']
    assert claim['status'] == 'VERIFIED'
    assert any(item['url'] == service and item['quote'] == full_text for item in claim['_industry_review']['evidence'])
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][service] == {'final_url':service,'text':full_text}
    assert calls['primary'] == [] and calls['contents'] == []
    assert len(paid_searches) == result['usage']['search_calls'] == 1
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    projected = lead_scorer._project_investigator_industry({'attribute_satisfied':False},claim)
    assert projected['industry_matches'] is True and projected['attribute_satisfied'] is False
    assert full_text in json.dumps(lead_scorer._dimension_web_evidence(projected,'industry'))


def test_full_loaded_search_corpus_fits_native_selection_request(monkeypatch):
    from tests.test_native_industry_review import install
    from tests.test_company_evidence_refusal_fallback import FEATURES, URL
    seen = []
    def initial_gap(result):
        seen.append(result)
        result['evidence_span_ids'] = result['evidence_span_ids'][:8]
        if len(seen) == 1:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])
    run, calls, _, _ = install(monkeypatch, mutation=initial_gap)
    pages = {f'https://www.coreworks.ai/source/{i}': (FEATURES + ' Detail.' * 4000)[:24000]
             for i in range(10)}
    async def search(*args, **kwargs):
        return {'results': [{'url': url, 'title': 'Company source'} for url in pages],
                '_source_pages': pages, 'notice': 'discovery_only_not_evidence'}
    monkeypatch.setattr(c, '_search_web', search)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['_completed_submit'] is True
    first = calls['model'][0]
    source_parts = [p for m in first['messages'] if m['role'] == 'user'
                    and m['content'].startswith('{"untrusted_source_pages"')
                    for p in json.loads(m['content'])['untrusted_source_pages']]
    assert {p['source_url'] for p in source_parts} == set(pages)
    assert sum(len(s['text']) for p in source_parts for s in p['spans']) == industry.MAX_TOTAL_CHARS
    for request in calls['model']:
        validate_operation_request('openrouter.chat', request)
    assert result['usage']['search_calls'] <= c.MAX_SEARCH_CALLS
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
