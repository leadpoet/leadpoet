"""A successful HTML response with no body must not poison shared evidence."""
import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as c, lead_scorer
from tests.test_native_industry_review import install, MODEL, URL

EMPTY_BODY = '<html><head><title>Coreworks supplies cloud migration</title></head><body></body></html>'


@pytest.mark.parametrize('enabled', [False, True])
def test_title_without_body_is_not_evidence_under_new_policy(monkeypatch, enabled):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL if enabled else '')
    assert bool(c._plain_text(EMPTY_BODY)) is (not enabled)


@pytest.mark.parametrize('body', [
    'Coreworks supplies cloud migration.',
    '<p>Coreworks supplies cloud migration.</p>',
    '<title>Title</title><body><p>Brief product evidence.</p></body>',
    '<title>Title</title><main>Brief product evidence.</main>',
])
def test_short_actual_bodies_are_preserved(monkeypatch, body):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)
    assert c._plain_text(body)


def test_title_only_attribute_read_is_not_cached_as_success(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)
    async def fetch(*args, **kwargs):
        return 200, URL, EMPTY_BODY
    monkeypatch.setattr(lead_scorer, '_fetch_bounded_html', fetch)
    cache = {}
    result, sources = asyncio.run(lead_scorer._ground_required_attribute_evidence(
        {'attribute_satisfied': True, 'required_attribute_evidence_url': URL,
         'required_attribute_evidence_quote': 'Coreworks supplies cloud migration'},
        active_attribute=True, source_cache=cache))
    assert cache[URL]['status'] == 'source_unavailable'
    assert not sources
    assert URL not in lead_scorer._investigator_prefetched_pages_from_attribute_cache(cache, [URL])


def test_native_loop_recovers_empty_body_through_existing_exact_url_path(monkeypatch):
    run, calls, _, _ = install(monkeypatch)
    original = c._fetch_bounded_html
    async def raw(session, url, **kwargs):
        status, final, body = await original(session, url, **kwargs)
        return status, final, EMPTY_BODY if final == URL else body
    monkeypatch.setattr(c, '_fetch_bounded_html', raw)
    result = asyncio.run(run())
    assert len(calls['contents']) == 1
    assert calls['contents'][0]['ids'] == [URL]
    assert result['_completed_submit'] is True
    assert result['usage']['fetch_calls'] == 3
    assert [(x['ok'], x['error_class']) for x in result['usage']['fetch_outcomes']] == [
        (True, ''), (False, 'empty_page'), (True, '')]


@pytest.mark.parametrize('hidden', [
    '<main hidden>Unsupported product claim.</main>',
    '<main aria-hidden="true">Unsupported product claim.</main>',
    '<style>.off {display:none}</style><div class="off">Unsupported claim.</div>',
    '<script>Unsupported product claim.</script>',
])
def test_hidden_body_cannot_make_metadata_usable(monkeypatch, hidden):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)
    assert c._plain_text('<head><title>Company</title></head><body>' + hidden + '</body>') == ''
