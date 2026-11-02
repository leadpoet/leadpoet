"""Already observed industry context fills unused slots without new authority."""
import asyncio
from copy import deepcopy
import socket

import pytest

from lab_arena import scoring
from qualification.scoring import company_evidence_investigator as c
from qualification.scoring import company_verification as cv
from qualification.scoring import lead_scorer as s
from qualification.scoring.company_fit_decision import company_fit_match
from tests.test_positive_semantic_fit_review import _company, _icp, _verdict

HOME = 'https://peer.example/'


@pytest.fixture(autouse=True)
def offline_policy(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('No network permitted in homepage handoff tests')
    for name in ('create_connection', 'getaddrinfo'):
        monkeypatch.setattr(socket, name, deny)
    monkeypatch.setattr(socket.socket, 'connect', deny)
    for name in scoring.CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-placeholder-no-network')
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    monkeypatch.setenv('LEADPOET_COMPETITION_EVALUATION_DATE', '2026-10-24')


@pytest.mark.parametrize('kind', [
    'industry_only', 'stage_and_industry', 'positive_industry',
    'legacy_industry', 'private_equity_industry_only',
])
def test_native_public_homepage_handoff(monkeypatch, kind):
    company = _company()
    icp = _icp().model_copy(update={
        'company_stage': 'Series A' if kind == 'stage_and_industry' else '',
        'required_attribute': '',
    })
    initial = _verdict(industry_matches=None, industry_activity_role='unresolved',
                       industry_evidence_url='', industry_evidence_quote='')
    if kind == 'positive_industry':
        initial = _verdict()
    if kind == 'private_equity_industry_only':
        icp = icp.model_copy(update={'company_stage': 'Private Equity'})
        initial.update(observed_company_stage='Private Equity', stage_matches=True,
                       stage_evidence_url=HOME + 'ownership',
                       stage_evidence_quote='PeerConnect is controlled by a private equity sponsor.')
    raw = ('<html><title>PeerConnect</title><body>'
           'PeerConnect supplies an enrollment platform. '
           '<a href="https://official-child.example/">Official Child</a>'
           '<a href="https://www.linkedin.com/company/peerconnect">LinkedIn</a>'
           '</body></html>')
    observed, investigations, requests, reads = {}, [], [], []

    async def fetch(*args, **kwargs):
        reads.append(HOME)
        return 200, HOME, raw

    async def homepage(*args, **kwargs):
        result = await cv.verify_company_exists(*args, **kwargs)
        observed['pages'] = deepcopy(kwargs['homepage_evidence_sink'])
        return result

    async def request(**kwargs):
        requests.append(kwargs['telemetry_purpose'])
        return deepcopy(initial), ''

    async def unchanged(verdict, *args, **kwargs):
        return dict(verdict)

    async def refresh(verdict, *args, invocation_cache, **kwargs):
        invocation_cache['refresh_outcome'] = 'verified'
        return dict(verdict)

    async def prechecks(*args, **kwargs):
        return company_fit_match()

    async def investigate(**kwargs):
        investigations.append(kwargs)
        return {'_completed_submit': True, 'claims': {'industry': {
            'target': 'industry', 'status': 'UNPROVEN', 'activity_role': 'unresolved',
            'evidence_url': '', 'evidence_quote': '', 'reason': 'Hermetic source audit',
        }}}

    if kind == 'legacy_industry':
        monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL')
    monkeypatch.setattr(cv, '_fetch_bounded_html', fetch)
    for name, fn in [
        ('run_company_zero_checks', prechecks), ('verify_company_exists', homepage),
        ('_request_company_reverify_json', request),
        ('_ground_employee_size_observation', unchanged),
        ('_refresh_linkedin_employee_size_observation', refresh),
        ('investigate_company_evidence', investigate),
    ]:
        monkeypatch.setattr(s, name, fn)
    monkeypatch.setattr(s, '_resolve_observed_linkedin_redirect_alias',
                        lambda company, verdict: unchanged(verdict))
    asyncio.run(s._verify_company_fit(company, icp, 0., 0., set(),
        require_https_transport=True, evidence_investigator=True, company_quality=True))
    assert reads == [HOME]
    assert requests == ['lead_scorer_reverify']
    assert len(investigations) == 1
    inv = investigations[0]
    assert observed['pages'] and inv['verified_homepage_identity']
    if kind in ('industry_only', 'legacy_industry', 'private_equity_industry_only'):
        assert inv['targets'] == ('industry',)
        assert inv['positive_semantic_review'] is False
    if kind == 'legacy_industry':
        assert inv['prefetched_pages'] == {}
    else:
        assert inv['prefetched_pages'][HOME] == observed['pages'][HOME]
        envelope = inv['prefetched_pages'][HOME][c.PRIVATE_INDUSTRY_BINDINGS_KEY]
        assert envelope['brands'] == {}  # An observed URL proves no ownership.
        assert envelope['links'][HOME]['hrefs'] == [
            'https://official-child.example/',
            'https://www.linkedin.com/company/peerconnect',
        ]


def targeted(monkeypatch, *, homepages, targets=('industry',), **options):
    """Exercise the shared initial/postrepair native target caller."""
    calls = []

    async def capture(**kwargs):
        calls.append(kwargs)
        return {'claims': {}, 'failure_reason': 'Hermetic source audit'}

    monkeypatch.setattr(s, 'investigate_company_evidence', capture)
    identity = {'normalized_name': 'peerconnect',
                'registrable_dns_domain': 'peer.example',
                'linkedin_company_slug': 'peerconnect'}
    kwargs = dict(company=_company(), icp=_icp(), verdict=_verdict(),
                  investigation_targets=targets, icp_attribute='', icp_stage='',
                  verified_identity=identity, verified_transport_domain='peer.example',
                  structured_employee_size_evidence=None,
                  structured_public_company_evidence=None, employee_size_conflict=False,
                  company_quality=True, verified_homepage_pages=homepages)
    kwargs.update(options)
    asyncio.run(s._run_targeted_company_evidence_investigation(**kwargs))
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize('count', [0, 1, 2, 3])
def test_existing_pages_keep_priority_and_three_page_cap(count):
    urls = [HOME + f'source-{i}' for i in range(count)]
    cache = {url: {'status': 'fetched', 'final_url': url, 'text': f'Existing fact {i}.'}
             for i, url in enumerate(urls)}
    home = {HOME: {'final_url': HOME, 'text': 'Existing homepage body.'}}
    existing = s._investigator_prefetched_pages(cache, [*urls, HOME])
    selected = s._investigator_prefetched_pages(cache, [*urls, HOME],
        verified_homepage_pages=home, homepage_fills_unused_slots=True)
    assert list(selected)[:count] == list(existing)
    assert all(selected[url] == page for url, page in existing.items())
    assert len(selected) == min(count + 1, c.MAX_PREFETCHED_PAGES) <= 3
    assert (HOME in selected) is (count < 3)


def test_same_url_keeps_prior_exact_body_without_rebinding_metadata():
    old = {'final_url': HOME, 'text': 'Existing independently retained body.'}
    selected = s._investigator_prefetched_pages({}, [HOME],
        matched_company_retry_pages={HOME: old},
        verified_homepage_pages={HOME: {'final_url': HOME, 'text': 'Different homepage body.'}},
        homepage_fills_unused_slots=True)
    assert selected == {HOME: old}


def test_full_mixed_target_pages_and_url_prefix_unchanged(monkeypatch):
    from tests.test_industry_retry_evidence_continuity import supported_result, retain, IDENTITY
    from tests.test_matched_company_retry_source_continuity import _company as company
    result, pages = supported_result()
    _, retained = retain(result, pages)
    retry = {}
    sink = {}
    s._retain_matched_investigator_source_contexts(sink, result,
        {c.PRIVATE_FETCHED_PAGES_KEY: pages})
    s._retain_matched_company_retry_sources(retry, sink, result)
    homepage = 'https://unibuddy.com/'
    kwargs = dict(company=company(), verified_identity=IDENTITY,
                  matched_company_retry_source_cache=retry)
    before = targeted(monkeypatch, homepages={}, targets=('industry', 'geography', 'required_attribute'), **kwargs)
    after = targeted(monkeypatch, homepages={homepage: {'final_url': homepage, 'text': 'Extra body.'}},
                     targets=('industry', 'geography', 'required_attribute'), **kwargs)
    assert after['prefetched_pages'] == before['prefetched_pages'] == retained
    assert len(retained) == 3
    assert after == before


def test_eight_existing_source_urls_are_not_displaced(monkeypatch):
    urls = [HOME + f'evidence-{i}' for i in range(c.MAX_SUBMITTED_SOURCE_URLS)]
    company = _company().model_copy(update={'fit_evidence_urls': urls})
    alternate_home = 'https://www.peer.example/'
    before = targeted(monkeypatch, homepages={}, company=company)
    after = targeted(monkeypatch, homepages={alternate_home: {
        'final_url': alternate_home, 'text': 'Additional already fetched homepage.'}}, company=company)
    assert after['prior_observations']['submitted_source_urls'] == before['prior_observations']['submitted_source_urls']
    assert len(after['prior_observations']['submitted_source_urls']) == c.MAX_SUBMITTED_SOURCE_URLS
    assert after['prefetched_pages'] == before['prefetched_pages'] == {}
    assert after == before


@pytest.mark.parametrize('fault', ['unbound', 'unsafe_url', 'binary', 'body', 'final', 'date', 'identity'])
def test_invalid_page_or_metadata_never_gains_authority(monkeypatch, fault):
    from tests.test_initial_homepage_industry_link_provenance import initialize, html, HOME as parent
    _, pages, _ = initialize(monkeypatch, html())
    if fault == 'unsafe_url':
        pages = {'https://127.0.0.1/': {'final_url': 'https://127.0.0.1/', 'text': 'Unsafe'}}
    elif fault == 'binary':
        pages[parent]['text'] = '%PDF-1.7 invalid homepage'
    elif fault == 'body':
        pages[parent]['text'] += ' Changed body.'
    elif fault in ('date', 'identity'):
        record = pages[parent][c.PRIVATE_INDUSTRY_BINDINGS_KEY]
        record['evaluation_date' if fault == 'date' else 'parent_domain'] = (
            '2026-10-25' if fault == 'date' else 'foreign.example')
    elif fault == 'final':
        pages[parent]['final_url'] = parent + 'changed'
    options = {
        'company': _company().model_copy(update={'company_name': 'Coreworks',
            'company_website': parent, 'company_linkedin': 'https://www.linkedin.com/company/coreworks-ai'}),
        'verdict': _verdict(observed_company_name='Coreworks', observed_company_website=parent,
            observed_company_linkedin='https://www.linkedin.com/company/coreworks-ai'),
        'verified_identity': {'normalized_name': 'coreworks', 'registrable_dns_domain': 'coreworks.ai',
                              'linkedin_company_slug': 'coreworks-ai'},
        'verified_transport_domain': 'coreworks.ai',
    }
    healthy = targeted(monkeypatch, homepages=deepcopy(initialize(monkeypatch, html())[1]), **options)
    assert healthy['prefetched_pages'][parent][c.PRIVATE_INDUSTRY_BINDINGS_KEY]['links']
    if fault == 'unbound':
        options.update(verified_identity={}, verified_transport_domain='')
    call = targeted(monkeypatch, homepages=pages, **options)
    if fault in ('unbound', 'unsafe_url', 'binary'):
        assert not call['prefetched_pages']
        return
    selected = call['prefetched_pages']
    # The host revalidates exact body/final URL/actual identity/date again at
    # investigator ingress. Metadata is not ownership, even when well formed.
    page_texts = {url: page['text'] for url, page in selected.items()}
    finals = {url: page['final_url'] for url, page in selected.items()}
    for page in selected.values():
        valid = c._validated_industry_bindings(page.get(c.PRIVATE_INDUSTRY_BINDINGS_KEY),
            page_texts, finals, domains={'coreworks.ai'},
            identity={'submitted_domain': 'coreworks.ai', 'observed_domain': 'coreworks.ai',
                      'observed_name': 'Coreworks'}, evaluation_date_value='2026-10-24')
        assert not valid['links'] and not valid['brands']


def test_disputed_hq_and_attribute_pages_keep_priority():
    from tests.test_structured_linkedin_headquarters_reuse import (
        ANCHOR, PROFILE, HQ_TEXT, _project, _body,
    )
    home = 'https://quaise.com/'
    retries = {home + f'retry-{i}': {'final_url': home + f'retry-{i}', 'text': f'Prior fact {i}.'}
               for i in range(2)}
    options = dict(verified_identity=ANCHOR, matched_company_retry_pages=retries,
        structured_profile_headquarters_evidence=_project(_body()),
        include_structured_headquarters=True)
    urls = [*retries, PROFILE, home]
    before = s._investigator_prefetched_pages({}, urls, **options)
    after = s._investigator_prefetched_pages({}, urls, **options,
        verified_homepage_pages={home: {'final_url': home, 'text': 'Optional homepage.'}},
        homepage_fills_unused_slots=True)
    assert after == before
    assert len(after) == 3 and after[PROFILE]['text'] == HQ_TEXT


@pytest.mark.parametrize('omit_relation', [False, True])
def test_new_projection_reuses_bound_parent_without_ownership_inference(monkeypatch, omit_relation):
    from tests import test_industry_conditional_brand_sources as fixture
    from tests.test_initial_homepage_industry_link_provenance import initialize, html, RELATION
    original_install = fixture.install
    _, pages, _ = initialize(monkeypatch, html().replace(RELATION, fixture.RELATION), url=fixture.PARENT)

    def install(*args, **kwargs):
        run, *rest = original_install(*args, **kwargs)

        async def with_prefetch(**options):
            options['prefetched_pages'] = s._investigator_prefetched_pages({}, tuple(pages),
                verified_homepage_pages=pages, homepage_fills_unused_slots=True)
            options['prior_observations'] = {**options.get('prior_observations', {}),
                                            'submitted_source_urls': list(pages)}
            return await run(**options)
        return with_prefetch, *rest

    monkeypatch.setattr(fixture, 'install', install)
    result, reads, _, _ = fixture.native_loop(monkeypatch, omit_relation=omit_relation)
    assert reads == [fixture.CHILD]
    assert result['usage']['fetch_calls'] == 1
    if omit_relation:
        assert result['claims']['industry']['status'] != 'VERIFIED'
    else:
        assert result['claims']['industry']['status'] == 'VERIFIED'


def test_hidden_or_unobserved_links_are_not_added_by_unused_slot_projection(monkeypatch):
    from tests.test_initial_homepage_industry_link_provenance import initialize, html, HOME as parent
    extra = ('<a hidden href="https://hidden.example/">Hidden</a>'
             '<script>"https://script.example/"</script>'
             '<a href="javascript:alert(1)">JS</a>')
    _, pages, _ = initialize(monkeypatch, html(extra=extra))
    selected = s._investigator_prefetched_pages({}, tuple(pages),
        verified_homepage_pages=pages, homepage_fills_unused_slots=True)
    assert selected == pages
    record = selected[parent][c.PRIVATE_INDUSTRY_BINDINGS_KEY]
    assert record['brands'] == {}
    assert all('hidden.example' not in url and 'script.example' not in url
               for url in record['links'][parent]['hrefs'])
    assert 'https://unobserved.example/' not in record['links'][parent]['hrefs']
