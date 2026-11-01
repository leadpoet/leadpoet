"""Archive interface documents must not supply or hide article metadata."""
import asyncio
import json

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from tests.test_archived_publication_date import article
from tests.test_intent_failed_archive_same_event import install_exact_failures
from tests.test_intent_wayback_empty_body import SOURCE, SNAPSHOT


def toolbar(body, *, identifier="wm-ipp-base"):
    return f'<div id="{identifier}">{body}</div>'


def framed_document(body):
    return ('<div data-original-tag="iframe"><html><head>'
            '<title>Archive interface</title></head><body>'
            + body + '</body></html></div>')


@pytest.mark.parametrize("identifier", ["wm-ipp-base", "wm-ipp-print"])
@pytest.mark.parametrize("framed", [False, True])
def test_archive_chrome_heading_does_not_hide_source_date(monkeypatch, identifier, framed):
    chrome = '<h1>Archive interface unavailable</h1>'
    if framed:
        chrome = framed_document(chrome)
    body = article().replace('<body>', '<body>' + toolbar(chrome, identifier=identifier), 1)
    install_exact_failures(monkeypatch, archive=body)
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    row = fetched['results'][0]
    assert row['source_publication_date'] == '2026-02-16'
    assert intent._project_contents_for_prompt(fetched)['results'][0][
        'source_publication_date'] == '2026-02-16'


def test_archive_chrome_cannot_supply_source_publication():
    node = {'@type': 'NewsArticle', 'mainEntityOfPage': SOURCE,
            'headline': 'A supplier partnership', 'datePublished': '2026-02-16'}
    chrome = framed_document('<script type="application/ld+json">'
                             + json.dumps(node) + '</script>')
    body = ('<html><head></head><body>' + toolbar(chrome)
            + '<h1>A supplier partnership</h1><p>Undated source article.</p></body></html>')
    assert intent._published_date_from_html(body, SOURCE, archive_url=SNAPSHOT) == ''


def test_archive_chrome_conflicting_date_is_not_article_metadata():
    chrome = framed_document('<script type="application/ld+json">' + json.dumps({
        '@type': 'NewsArticle', 'mainEntityOfPage': SOURCE,
        'headline': 'A supplier partnership', 'datePublished': '2025-01-01',
    }) + '</script>')
    body = article().replace('<body>', '<body>' + toolbar(chrome), 1)
    assert intent._published_date_from_html(body, SOURCE, archive_url=SNAPSHOT) == '2026-02-16'


def test_nested_archive_chrome_does_not_remove_source_body():
    chrome = toolbar('<h1>Archive navigation</h1>' + toolbar(
        framed_document('<h1>Print interface</h1>'), identifier='wm-ipp-print'))
    body = article().replace('<body>', '<body>' + chrome, 1)
    assert intent._published_date_from_html(body, SOURCE, archive_url=SNAPSHOT) == '2026-02-16'


@pytest.mark.parametrize('identifier', ['article-toolbar', 'donato-if', 'WM-IPP-BASE'])
def test_unrecognized_source_sections_stay_ambiguous(identifier):
    body = article().replace('<body>', '<body>' + toolbar(
        '<h1>Another source article</h1>', identifier=identifier), 1)
    assert intent._published_date_from_html(body, SOURCE, archive_url=SNAPSHOT) == ''


def test_direct_page_does_not_gain_archive_chrome_removal():
    body = article(canonical=SOURCE).replace('<body>', '<body>' + toolbar(
        '<h1>Another source article</h1>'), 1)
    assert intent._published_date_from_html(body, SOURCE) == ''


@pytest.mark.parametrize('heading', ['<h1>Another source article</h1>',
                                    '<h1>A supplier partnership</h1>'])
def test_multiple_source_headings_still_cannot_bind_date(heading):
    body = article().replace('</body>', heading + '</body>')
    assert intent._published_date_from_html(body, SOURCE, archive_url=SNAPSHOT) == ''


def test_article_identity_guard_still_applies_after_chrome_removal():
    body = article(headline='Different article').replace('<body>', '<body>' + toolbar(
        '<h1>Archive navigation</h1>'), 1)
    assert intent._published_date_from_html(body, SOURCE, archive_url=SNAPSHOT) == ''
