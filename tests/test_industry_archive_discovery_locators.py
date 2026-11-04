"""Structural archive hrefs are unread discovery leads, never source facts."""
import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as inv


INDEX = 'https://acme.example/press/'
ARTICLE = INDEX + 'acme-launches-new-system'


def archive(anchor, *, wrapper='related-block', extra=''):
    return ('<html><body><main><h1>Press releases</h1>' + extra
            + '<div class="' + wrapper + '">' + anchor
            + '</div></main></body></html>')


def link(url=ARTICLE):
    return '<a href="' + url + '">Acme launches new system</a>'


def fetch(monkeypatch, raw, *, source=INDEX, enabled=True):
    calls = []
    for key in ['SCRAPINGDOG_API_KEY', 'QUALIFICATION_SCRAPINGDOG_API_KEY']:
        monkeypatch.delenv(key, raising=False)
    if enabled:
        monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/test')
    else:
        monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL', raising=False)

    async def transport(_session, url, **kwargs):
        calls.append((url, kwargs))
        return 200, source, raw

    monkeypatch.setattr(inv, '_fetch_bounded_html', transport)
    return asyncio.run(inv._fetch_page(None, source)), calls


def test_related_index_href_is_unread_and_not_quotable(monkeypatch):
    raw = archive(link())
    before = inv._plain_text(raw)
    result, calls = fetch(monkeypatch, raw)
    assert len(calls) == 1
    assert result['text'].startswith(before)
    assert ARTICLE in result['text']
    assert inv._visible_quote_surface(result['text']).strip() == inv._visible_quote_surface(before).strip()
    assert 'Acme launches new system' not in inv._visible_quote_surface(result['text'])
    assert not inv._industry_relationship_quote(ARTICLE, result['text'])
    assert 'server_industry_source_links' not in result


@pytest.mark.parametrize('wrapper', ['related-block', 'related-news', 'blog-index'])
def test_existing_link_only_parser_covers_structural_index(monkeypatch, wrapper):
    result, calls = fetch(monkeypatch, archive(link(), wrapper=wrapper))
    assert ARTICLE in result['text']
    assert len(calls) == 1


@pytest.mark.parametrize('hidden', [
    '<div hidden>{}</div>', '<div aria-hidden="true">{}</div>',
    '<div style="display:none">{}</div>', '<div style="visibility:hidden">{}</div>',
    '<template>{}</template>', '<script type="text/template">{}</script>',
    '<div class="not-visible">{}</div>',
    '<div role="tabpanel" aria-hidden="true">{}</div>',
])
def test_actual_hidden_content_does_not_supply_locator(monkeypatch, hidden):
    raw = archive(hidden.format(link()), extra='<style>.not-visible{display:none}</style>')
    before = inv._plain_text(raw)
    result, _ = fetch(monkeypatch, raw)
    assert result['text'] == before
    assert ARTICLE not in result['text']


def test_tab_without_explicit_visibility_proof_is_locator_only(monkeypatch):
    # An inactive rendered tab may be hidden by external CSS. Its literal href
    # can be an unread lead; neither card text nor visibility becomes a fact.
    raw = archive('<div class="w-tab-pane">' + link() + '</div>')
    result, _ = fetch(monkeypatch, raw)
    assert ARTICLE in result['text']
    assert 'Acme launches new system' not in inv._visible_quote_surface(result['text'])


@pytest.mark.parametrize('url', [
    'https://elsewhere.example/press/acme-launches-new-system',
    'https://tenant.acme.example/press/acme-launches-new-system',
    'https://acme.example:444/press/acme-launches-new-system',
    'https://user@acme.example/press/acme-launches-new-system',
    '//acme.example/press/acme-launches-new-system',
    'http://acme.example/press/acme-launches-new-system',
    'https://acme.example/press/acme-launches-new-system#section',
    'https://acme.example/press/acme-launches-new-system?next=https://www.linkedin.com/company/other',
    'https://acme.example/press/../other/acme-launches-new-system',
    'https://acme.example/press/%2e%2e/other/acme-launches-new-system',
    'https://acme.example/press/%252e%252e/other/acme-launches-new-system',
    'https://acme.example/press/acme%2fother',
    'https://acme.example/press/acme%5cother',
    'https://acme.example/press/https://www.linkedin.com/company/other',
    'https://acme.example/other/acme-launches-new-system',
    'javascript:alert(1)',
    'https://acme.example:bad/press/story',
])
def test_ambiguous_or_different_authority_is_not_new_metadata(monkeypatch, url):
    raw = archive(link(url))
    result, _ = fetch(monkeypatch, raw)
    assert result['text'] == inv._plain_text(raw)


def test_article_related_siblings_and_root_navigation_are_unchanged(monkeypatch):
    raw = archive(link())
    for source in [INDEX + 'older-article', 'https://acme.example/']:
        result, _ = fetch(monkeypatch, raw, source=source)
        assert result['text'] == inv._plain_text(raw)


def test_plain_urls_and_legacy_role_do_not_add_leads(monkeypatch):
    raw = archive('<p>' + ARTICLE + '</p>')
    result, _ = fetch(monkeypatch, raw)
    assert result['text'] == inv._plain_text(raw)
    result, _ = fetch(monkeypatch, archive(link()), enabled=False)
    assert result['text'] == inv._plain_text(archive(link()))


def test_whole_urls_stable_order_and_existing_budgets(monkeypatch):
    urls = [INDEX + 'story-' + str(i) for i in range(80)]
    raw = archive(''.join(link(url) for url in urls) + link(urls[0]))
    result, calls = fetch(monkeypatch, raw)
    added = result['text'].partition(inv._IDENTITY_LINK_CONTEXT_MARKER)[2].split()
    assert added == urls
    assert urls[43] in added  # Same-host archive leads do not inherit the external-brand40 cap.
    assert len(''.join(added)) <= inv.MAX_HOMEPAGE_NAVIGATION_TOTAL_CHARACTERS
    assert len(result['text']) <= inv.MAX_PAGE_CHARACTERS
    assert len(calls) == 1


@pytest.mark.parametrize('filler', ['x' * inv.MAX_PAGE_CHARACTERS,
                                   'x' * (inv.MAX_PAGE_CHARACTERS - 80)])
def test_optional_links_never_trim_old_body_or_emit_partial_url(monkeypatch, filler):
    raw = '<main><p>' + filler + '</p></main>' + archive(link())
    old = inv._plain_text(raw)
    result, _ = fetch(monkeypatch, raw)
    assert result['text'] == old


def test_new_links_follow_existing_metadata_without_reordering(monkeypatch):
    prior = INDEX + 'known-story'
    raw = '<main><a href="' + prior + '">Current news</a></main>' + archive(link())
    before = inv._plain_text(raw)
    result, _ = fetch(monkeypatch, raw)
    assert result['text'] == before + ' ' + ARTICLE


def test_discovery_character_cap_is_a_stable_whole_url_prefix(monkeypatch):
    urls = [INDEX + ('x' * 90) + '-' + str(i) for i in range(100)]
    result, _ = fetch(monkeypatch, archive(''.join(link(url) for url in urls)))
    added = result['text'].partition(inv._IDENTITY_LINK_CONTEXT_MARKER)[2].split()
    expected = []
    for url in urls:
        if sum(map(len, expected)) + len(url) > inv.MAX_HOMEPAGE_NAVIGATION_TOTAL_CHARACTERS:
            break
        expected.append(url)
    assert added == expected
    assert len(added) < len(urls)


def test_existing_parser_anchor_ceiling_remains(monkeypatch):
    raw = archive(''.join(link(INDEX + 'article-' + str(i)) for i in range(600)))
    result, _ = fetch(monkeypatch, raw)
    assert INDEX + 'article-500' not in result['text']


def test_href_after_existing_metadata_uses_only_whole_remaining_url(monkeypatch):
    prior = INDEX + 'known-story'
    suffix = ' ' + ARTICLE
    # Existing metadata survives intact when only a partial new URL could fit.
    def markup(size):
        return '<p>' + ('x' * size) + '</p><a href="' + prior + '">Old</a>' + archive(link())
    size = inv.MAX_PAGE_CHARACTERS - 200
    size += inv.MAX_PAGE_CHARACTERS - len(suffix) + 1 - len(inv._plain_text(markup(size)))
    raw = markup(size)
    before = inv._plain_text(raw)
    assert before.endswith(prior)
    assert len(before) + len(suffix) > inv.MAX_PAGE_CHARACTERS
    result, _ = fetch(monkeypatch, raw)
    assert result['text'] == before


def test_malformed_locator_is_rejected_by_new_helper():
    # The existing external-brand href path has its own malformed-URL behavior.
    # This new discovery helper must never raise on the same untrusted href.
    assert inv._passive_archive_article_locators(
        archive(link('https://[invalid/press/story')), INDEX) == ()


def test_malformed_href_does_not_erase_page_or_other_observed_links(monkeypatch):
    raw = archive(link('https://[invalid/press/story') + link()
                  + '<a href="https://brand.example/">Official brand</a>',
                  extra='<p>Acme manufactures useful hardware.</p>')
    result, calls = fetch(monkeypatch, raw)
    assert len(calls) == 1
    assert 'Acme manufactures useful hardware.' in inv._visible_quote_surface(result['text'])
    assert ARTICLE in result['text']
    assert result['server_industry_source_links'] == ['https://brand.example/']


def test_unread_locators_do_not_turn_an_empty_source_into_loaded_evidence(monkeypatch):
    raw = '<html><body><div class="related-block">' + link() + '</div></body></html>'
    result, calls = fetch(monkeypatch, raw)
    assert calls == [(INDEX, {})]
    assert result == {'ok': False, 'error': 'empty_page'}
