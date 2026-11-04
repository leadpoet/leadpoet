"""Initial raw observations use the existing conditional provenance contract.

Hermetic replies establish control flow only, never real ownership semantics.
"""
import asyncio
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from qualification.scoring import company_verification as cv
from qualification.scoring import company_evidence_investigator as c, lead_scorer

FIXTURE = Path(__file__).parent / 'fixtures/initial_homepage_industry_links'
HOME = 'https://www.coreworks.ai/'
CHILD_HOME = 'https://workflowco.example/'
CHILD = CHILD_HOME + 'product'
RELATION = 'Coreworks operates WorkflowCo within its connected commercial software offering.'
FACT = 'WorkflowCo supplies reporting and account workflow software.'
IDENTITY = {'submitted_domain': 'coreworks.ai', 'observed_domain': 'coreworks.ai',
            'verified_domain': 'coreworks.ai', 'observed_name': 'Coreworks'}


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    monkeypatch.setenv('LEADPOET_COMPETITION_EVALUATION_DATE', '2026-10-24')


def initialize(monkeypatch, raw, *, company='Coreworks', url=HOME, final=None,
               linkedin='https://www.linkedin.com/company/coreworks-ai'):
    fetched = []
    async def fetch(session, target, **kwargs):
        fetched.append(target)
        assert target == url
        return 200, final or url, raw
    monkeypatch.setattr(cv, '_fetch_bounded_html', fetch)
    navigation, pages = [], {}
    result = asyncio.run(cv.verify_company_exists(company, url, company_linkedin=linkedin,
        require_https_transport=True, company_quality=False,
        homepage_navigation_locator_sink=navigation, homepage_evidence_sink=pages))
    assert fetched == [url]
    return result, pages, navigation


def html(*, linkedin=True, extra=''):
    return ('<html><head><title>Coreworks</title></head><body><main>' + RELATION
            + '</main><a href="' + CHILD_HOME + '">WorkflowCo</a>'
            + ('<a href="https://www.linkedin.com/company/coreworks-ai">LinkedIn</a>' if linkedin else '')
            + extra + '</body></html>')


def projected(pages):
    return lead_scorer._investigator_prefetched_pages({}, tuple(pages), verified_homepage_pages=pages)


def test_actual_saved_native_initializer_preserves_exact_body_bound_link_record(monkeypatch):
    provenance = json.loads((FIXTURE / 'provenance.json').read_text())
    compressed = (FIXTURE / 'roku-careers-public-homepage.html.gz').read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == provenance['gzip_sha256']
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == provenance['body_sha256']
    result, pages, navigation = initialize(monkeypatch, raw.decode(), company='Roku',
        url='https://www.weareroku.com/', linkedin='https://www.linkedin.com/company/roku/')
    assert result.decision == 'match'
    page = pages['https://www.weareroku.com/']
    observation = page[c.PRIVATE_INDUSTRY_BINDINGS_KEY]
    assert observation['brands'] == {}
    assert observation['evaluation_date'] == '2026-10-24'
    assert observation['parent_domain'] == 'weareroku.com'
    link = observation['links']['https://www.weareroku.com/']
    assert 'https://www.roku.com/' in link['hrefs']
    assert link['body_sha256'] == hashlib.sha256(page['text'].encode()).hexdigest()
    assert all(c._registrable_domain(x['url']) != 'roku.com' for x in navigation)
    assert projected(pages) == pages
    child = 'https://www.roku.com/investor'
    all_pages = {**{u: p['text'] for u, p in pages.items()}, child: 'Roku Investor Relations'}
    finals = {u: u for u in all_pages}
    validated = c._validated_industry_bindings(observation, all_pages, finals,
        domains={'weareroku.com'}, identity={'submitted_domain': 'weareroku.com',
            'observed_domain': 'weareroku.com', 'observed_name': 'Roku, Inc.'},
        evaluation_date_value='2026-10-24')
    assert validated['links'] == observation['links']
    assert validated['brands'] == {}  # A URL does not manufacture ownership.


def test_legacy_initializer_and_projection_keep_existing_page_shape(monkeypatch):
    monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL')
    result, pages, _ = initialize(monkeypatch, html())
    assert result.decision == 'match'
    assert set(pages[HOME]) == {'final_url', 'text'}
    assert projected(pages) == pages


def test_generic_producer_uses_only_observed_visible_bounded_hrefs(monkeypatch):
    extra = ('<a hidden href="https://hidden.example/">Hidden</a>'
             '<script>"https://script.example/"</script>'
             '<a href="javascript:alert(1)">JS</a><a href="http://127.0.0.1/">Private</a>')
    result, pages, _ = initialize(monkeypatch, html(extra=extra))
    assert result.decision == 'match'
    observation = pages[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]
    links = observation['links'][HOME]['hrefs']
    assert CHILD_HOME in links
    assert all('hidden.example' not in u and 'script.example' not in u and '127.0.0.1' not in u for u in links)
    assert 'https://unobserved.example/' not in links
    assert observation['brands'] == {}
    assert projected(pages) == pages


@pytest.mark.parametrize('mutation', ['body', 'final', 'date', 'hash', 'brands'])
def test_stale_metadata_drops_without_erasing_source(monkeypatch, mutation):
    _, pages, _ = initialize(monkeypatch, html())
    changed = deepcopy(pages)
    if mutation == 'body': changed[HOME]['text'] += ' A later read.'
    if mutation == 'final': changed[HOME]['final_url'] = HOME + 'new'
    if mutation == 'date': changed[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]['evaluation_date'] = '2026-10-23'
    if mutation == 'hash': changed[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]['links'][HOME]['body_sha256'] = '0' * 64
    if mutation == 'brands': changed[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]['brands'] = {CHILD: {}}
    result = projected(changed)
    assert result[HOME]['text'] == changed[HOME]['text']
    assert result[HOME]['final_url'] == changed[HOME]['final_url']
    assert c.PRIVATE_INDUSTRY_BINDINGS_KEY not in result[HOME]


def test_real_identity_and_date_still_revalidate_after_private_projection(monkeypatch):
    _, pages, _ = initialize(monkeypatch, html())
    record = projected(pages)[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]
    for changed in [dict(IDENTITY, observed_name='Different Company'),
                    dict(IDENTITY, observed_domain='foreign.example')]:
        result = c._validated_industry_bindings(record, {HOME: pages[HOME]['text']}, {HOME: HOME},
            domains={'coreworks.ai'}, identity=changed, evaluation_date_value='2026-10-24')
        assert result['links'] == {}
    assert not c._validated_industry_bindings(record, {HOME: pages[HOME]['text']}, {HOME: HOME},
        domains={'coreworks.ai'}, identity=IDENTITY, evaluation_date_value='2026-10-25')['links']


@pytest.mark.parametrize('omit_relation', [False, True])
def test_native_prefetched_parent_allows_child_read_without_parent_refetch(monkeypatch, omit_relation):
    from tests import test_industry_conditional_brand_sources as fixture
    original_install = fixture.install
    _, pages, _ = initialize(monkeypatch, html().replace(RELATION, fixture.RELATION), url=fixture.PARENT)
    def install(*args, **kwargs):
        run, *rest = original_install(*args, **kwargs)
        async def with_prefetch(**options):
            options['prefetched_pages'] = projected(pages)
            options['prior_observations'] = {**options.get('prior_observations', {}),
                                            'submitted_source_urls': [fixture.PARENT]}
            return await run(**options)
        return with_prefetch, *rest
    monkeypatch.setattr(fixture, 'install', install)
    result, reads, _, _ = fixture.native_loop(monkeypatch, omit_relation=omit_relation)
    assert reads == [fixture.CHILD]
    assert result['usage']['fetch_calls'] == 1
    assert fixture.PARENT in result[c.PRIVATE_INDUSTRY_BINDINGS_KEY]['links']
    if omit_relation:
        assert result['claims']['industry']['status'] != 'VERIFIED'
        assert result['failure_reason'] == 'malformed_response'
    else:
        assert result['claims']['industry']['status'] == 'VERIFIED'
        assert fixture.CHILD in result[c.PRIVATE_INDUSTRY_BINDINGS_KEY]['brands']


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('branch', ['no_linkedin', 'unresolved_name', 'redirect'])
def test_other_existing_identity_sink_branches_keep_pending_observation_only(monkeypatch, branch, enabled):
    if not enabled:
        monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL')
    body = html(linkedin=branch != 'no_linkedin')
    final = None
    if branch == 'unresolved_name':
        # A single exact LinkedIn remains, but no independently attributable
        # company-name metadata; existing path still retains untrusted body.
        body = body.replace('<title>Coreworks</title>', '<title>Other text</title>')
        monkeypatch.setattr(cv, '_homepage_company_names', lambda *a, **k: ['Other Company'])
    if branch == 'redirect':
        final = 'https://renamed.example/'
    result, pages, _ = initialize(monkeypatch, body, final=final)
    assert result.decision != 'match'
    assert pages and pages[HOME]['text'] == c._plain_text(body)
    if not enabled:
        assert c.PRIVATE_INDUSTRY_BINDINGS_KEY not in pages[HOME]
        return
    assert c.PRIVATE_INDUSTRY_BINDINGS_KEY in pages[HOME]
    record = pages[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]
    assert record['brands'] == {}
    if branch == 'redirect':
        assert c._validated_industry_bindings(record, {HOME: pages[HOME]['text']},
            {HOME: final}, domains={'coreworks.ai'}, identity=IDENTITY,
            evaluation_date_value='2026-10-24')['links'] == {}


def test_existing_locator_count_and_character_limits_are_shared(monkeypatch):
    body = html(extra=''.join('<a href="https://brand%d.example/">Brand</a>' % i for i in range(100)))
    _, pages, _ = initialize(monkeypatch, body)
    record = pages[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]['links'][HOME]
    assert len(record['hrefs']) == c.MAX_HOMEPAGE_NAVIGATION_LOCATORS
    assert sum(map(len, record['hrefs'])) <= c.MAX_HOMEPAGE_NAVIGATION_TOTAL_CHARACTERS


def test_initial_and_fresh_native_fetch_share_the_exact_existing_locator_surface(monkeypatch):
    raw = html(extra='<a href="https://partner.example/">Partner</a>')
    _, pages, _ = initialize(monkeypatch, raw)
    async def fetch(session, url, **kwargs):
        return 200, HOME, raw
    monkeypatch.setattr(c, '_fetch_bounded_html', fetch)
    fresh = asyncio.run(c._fetch_page(None, HOME))
    assert fresh['ok'] is True
    assert fresh['server_industry_source_links'] == pages[HOME][
        c.PRIVATE_INDUSTRY_BINDINGS_KEY]['links'][HOME]['hrefs']
    # A real outgoing partner href is a locator, never verified ownership.
    assert 'https://partner.example/' in fresh['server_industry_source_links']
    assert pages[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]['brands'] == {}
    assert c._visible_quote_surface(fresh['text']) == c._visible_quote_surface(pages[HOME]['text'])


def test_public_tool_arguments_cannot_supply_private_link_authority(monkeypatch):
    from tests.test_company_evidence_judgment_reserve import _response
    _, pages, _ = initialize(monkeypatch, html())
    response = _response('fetch_page', {'url': CHILD, 'industry_brand_source': None,
        c.PRIVATE_INDUSTRY_BINDINGS_KEY: pages[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]})
    assert c._valid_multi_research_reply(response['choices'][0]['message']['tool_calls'],
        required_tool='fetch_page', industry_sources=True) is False


def test_pending_links_alone_leave_ordinary_native_review_request_unchanged(monkeypatch):
    _, pages, _ = initialize(monkeypatch, html())
    requests = []
    async def post(session, url, *, headers, payload, started):
        requests.append(deepcopy(payload))
        span = payload['response_format']['json_schema']['schema']['properties']['evidence_span_ids']['items']['enum'][0]
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'observed_activity': 'Parent operating offering', 'reason': 'Child facts remain unproved',
            'missing_requirements': ['Child capability'], 'role': 'supplier_operator',
            'evidence_span_ids': [span], 'decision': 'unavailable'})}}]}
    monkeypatch.setattr(c, '_post_openrouter_json', post)
    kwargs = dict(model='openai/gpt-6.1-sol', openrouter_key='offline-key', started=c.time.monotonic(),
        company_locator={'name': 'Coreworks', 'website': HOME},
        criteria={'industry': 'Software', 'subindustry': '', 'product_service': 'Reporting workflows'},
        evaluation_date_value='2026-10-24', fetched_pages={HOME: pages[HOME]['text']},
        fetched_final_urls={HOME: HOME}, first_party_domains={'coreworks.ai'},
        identity_anchor=IDENTITY, identity_names={'Coreworks'})
    asyncio.run(c._review_industry_sources(None, **kwargs))
    asyncio.run(c._review_industry_sources(None,
        industry_bindings=pages[HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY], **kwargs))
    assert len(requests) == 2 and requests[0] == requests[1]
