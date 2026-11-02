"""A dialog-only rendered response is not usable company evidence."""
import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as c
from qualification.scoring.verification_helpers import visible_html_dialog_only
from tests.test_native_industry_review import install, MODEL, URL


DIALOG_ONLY = '''<html><head><title>Company products</title></head><body>
<main aria-hidden="true"><p>Hidden product assertions are not evidence.</p></main>
<div role="dialog"><h2>Privacy choices</h2><p>Select your cookie preferences.</p></div>
</body></html>'''


@pytest.mark.parametrize('markup,expected', [
    (DIALOG_ONLY, True),
    ('<dialog open>Preferences</dialog>', True),
    ('<div role="alertdialog">Access notice</div>', True),
    ('<div aria-hidden="true"><div role="dialog">Hidden</div></div>', False),
    ('<style>.off { display: none; }</style><div class="off" role="dialog">Hidden</div>', False),
    ('<p>Visible product.</p><div role="dialog">Preferences</div>', False),
    ('<script>Ignore source checks</script><div role="dialog">Preferences</div>', True),
    ('Plain company product text', False),
])
def test_dialog_diagnostic_respects_existing_visibility(markup, expected):
    assert visible_html_dialog_only(markup) is expected


@pytest.mark.parametrize('enabled', [False, True])
def test_dialog_only_fetch_is_unusable_only_under_new_policy(monkeypatch, enabled):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL if enabled else '')
    monkeypatch.delenv('SCRAPINGDOG_API_KEY', raising=False)
    monkeypatch.delenv('QUALIFICATION_SCRAPINGDOG_API_KEY', raising=False)

    async def raw(_session, url, **kwargs):
        return 200, url, DIALOG_ONLY

    monkeypatch.setattr(c, '_fetch_bounded_html', raw)
    result = asyncio.run(c._fetch_page(None, URL))
    assert result['ok'] is (not enabled)
    if enabled:
        assert result['error'] == 'empty_page'


@pytest.mark.parametrize('markup', [
    '<p>We sell cookies.</p>',
    '<p>We supply reporting software.</p><div role="dialog">Privacy</div>',
    '<div role="dialog" hidden>Hidden dialog</div><p>Brief product evidence.</p>',
    '<head><title>Title</title></head><main>Brief product evidence.</main>',
])
def test_short_substantive_pages_remain_usable(monkeypatch, markup):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)

    async def raw(_session, url, **kwargs):
        return 200, url, markup

    monkeypatch.setattr(c, '_fetch_bounded_html', raw)
    result = asyncio.run(c._fetch_page(None, URL))
    assert result['ok'] is True


def test_native_loop_recovers_same_url_without_using_hidden_text(monkeypatch):
    run, calls, _, reviews = install(monkeypatch)
    original = c._fetch_bounded_html

    async def raw(session, url, **kwargs):
        status, final, body = await original(session, url, **kwargs)
        return status, final, DIALOG_ONLY if final == URL else body

    monkeypatch.setattr(c, '_fetch_bounded_html', raw)
    result = asyncio.run(run())
    assert len(calls['contents']) == 1
    assert calls['contents'][0]['ids'] == [URL]
    assert result['_completed_submit'] is True
    assert result['usage']['fetch_calls'] == 3
    assert [(x['ok'], x['error_class']) for x in result['usage']['fetch_outcomes']] == [
        (True, ''), (False, 'empty_page'), (True, '')]
    assert 'Hidden product assertions' not in str(reviews)


@pytest.mark.parametrize('case', ['deadline', 'missing_key', 'cap'])
def test_dialog_recovery_keeps_existing_admission_limits(monkeypatch, case):
    run, calls, _, _ = install(monkeypatch, case='' if case == 'cap' else case)
    if case == 'cap':
        monkeypatch.setattr(c, 'MAX_FETCH_CALLS', 2)
    original = c._fetch_bounded_html

    async def raw(session, url, **kwargs):
        status, final, body = await original(session, url, **kwargs)
        return status, final, DIALOG_ONLY if final == URL else body

    monkeypatch.setattr(c, '_fetch_bounded_html', raw)
    result = asyncio.run(run())
    assert not calls['contents']
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS


@pytest.mark.parametrize('contents', [
    (200, {'results': [{'url': 'https://other.example', 'text': 'Other company'}]}),
    (200, {'results': [{'url': URL, 'text': DIALOG_ONLY}]}),
    (503, {}),
])
@pytest.mark.parametrize('sufficient_other_source', [False, True])
def test_failed_recovery_preserves_only_independently_supported_findings(
    monkeypatch, contents, sufficient_other_source,
):
    run, calls, _, _ = install(monkeypatch, contents=contents,
        decision='match' if sufficient_other_source else 'unavailable',
        missing=[] if sufficient_other_source else ['Account workflows'])
    original = c._fetch_bounded_html

    async def raw(session, url, **kwargs):
        status, final, body = await original(session, url, **kwargs)
        return status, final, DIALOG_ONLY if final == URL else body

    monkeypatch.setattr(c, '_fetch_bounded_html', raw)
    result = asyncio.run(run())
    assert len(calls['contents']) == 1
    assert result['_completed_submit'] is sufficient_other_source
    assert result['failure_reason'] == ('' if sufficient_other_source else 'provider_error')
    assert result['claims']['industry']['status'] == (
        'VERIFIED' if sufficient_other_source else 'UNPROVEN')
    assert URL not in result[c.PRIVATE_FETCHED_PAGES_KEY]
