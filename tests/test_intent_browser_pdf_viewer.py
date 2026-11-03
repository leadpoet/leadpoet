"""Browser PDF controls are not the source document they would render."""
from __future__ import annotations

import asyncio
import gzip
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent

CAPTURE_BYTES = gzip.decompress((Path(__file__).parent / 'fixtures/unisys_wayback_pdf_viewer.html.gz').read_bytes())
assert hashlib.sha256(CAPTURE_BYTES).hexdigest() == '7296815d47225123762caf2f13c02cbd8fe41d4f8fabab71260501640f09b389'
CAPTURE = CAPTURE_BYTES.decode('utf-8')
SOURCE = 'https://www.unisys.com/siteassets/collateral/news/nr-052010052-unisys-rafay-cai-partnership.pdf'
SNAPSHOT = 'https://web.archive.org/web/20260606224635/' + SOURCE
QUOTE = 'On May 20, 2026, Unisys partnered with Rafay Systems to serve enterprise cloud workloads.'
ARTICLE = '<article><h1>Partnership announcement</h1><p>' + QUOTE + ' The partners will jointly deliver cloud services.' * 80 + '</p></article>'
VIEWER = ('<div data-original-tag="iframe" src="chrome-extension://browser-owned/index.html" '
          'type="application/pdf"><html><body><pdf-viewer>Document properties '
          'Annotations Present Extracting text from PDF… '
          + 'Page controls ' * 250 + '</pdf-viewer></body></html></div>')


def responses(monkeypatch, *, direct=CAPTURE, archive=CAPTURE, direct_status=200):
    calls = []

    class Client:
        def __init__(self, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def get(self, url, **kwargs):
            calls.append(url)
            request = httpx.Request('GET', url)
            if url == 'https://api.scrapingdog.com/scrape':
                return httpx.Response(direct_status, text=direct, request=request)
            if url == 'https://archive.org/wayback/available?url=' + SOURCE:
                return httpx.Response(200, json={'archived_snapshots': {'closest': {
                    'available': True, 'status': '200', 'timestamp': '20260606224635',
                    'url': SNAPSHOT}}}, request=request)
            assert url == SNAPSHOT
            return httpx.Response(200, text=archive, request=request)

    monkeypatch.setenv('SCRAPINGDOG_API_KEY', 'offline-only-placeholder')
    monkeypatch.setattr(intent.httpx, 'AsyncClient', Client)
    return calls


def test_exact_captured_viewer_is_not_archive_source():
    assert intent._has_wayback_error_body(CAPTURE)


def test_captured_viewer_continues_existing_exact_source_fallback(monkeypatch):
    responses(monkeypatch, direct_status=502)
    fallback = AsyncMock(return_value={'ok': True, 'stage': 'exa_scraped', 'content': QUOTE,
                                       'error': None})
    monkeypatch.setattr(intent, '_scrape_exa', fallback)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    fallback.assert_awaited_once_with(SOURCE)
    assert result['results'][0]['url'] == SOURCE
    assert result['results'][0]['text'] == QUOTE


@pytest.mark.parametrize('viewer', [VIEWER, VIEWER.replace('div data-original-tag="iframe"', 'iframe').replace('</div>', '</iframe>')], ids=['expanded_frame', 'native_iframe'])
def test_direct_viewer_only_body_is_not_source(viewer):
    assert intent._evaluate_sd_response(200, '<html><body>' + viewer + '</body></html>') != 'ok'


def test_real_source_sibling_survives_both_ingress_paths_without_viewer_ui(monkeypatch):
    body = '<html><head><title>Partnership announcement</title></head><body>' + VIEWER + ARTICLE + '</body></html>'
    responses(monkeypatch, direct=body, archive=body)
    direct = asyncio.run(intent._scrape_sd_hardened(SOURCE))
    archive = asyncio.run(intent._try_wayback(SOURCE))
    for result in (direct, archive):
        assert result['ok'] is True and QUOTE in result['content']
        assert 'Extracting text from PDF' not in result['content']
        assert 'chrome-extension://' not in result['content']


@pytest.mark.parametrize('body', [
    QUOTE + '\n\n' + 'Extracted PDF page text. ' * 30,
    '# Partnership announcement\n\n' + QUOTE + ' PDF markdown body. ' * 30,
    '<html><body>' + ARTICLE + '</body></html>',
    '<html><body>' + ARTICLE + '<embed type="application/pdf" src="https://example.com/report.pdf"></body></html>',
    '<html><body>' + ARTICLE + '<p>Our article mentions chrome-extension://example/viewer and PDF viewers.</p></body></html>',
    '<html><body><pdf-viewer>' + ARTICLE + '</pdf-viewer></body></html>',
    '<html><body><iframe src="chrome-extension://example/viewer" type="text/html">' + ARTICLE + '</iframe></body></html>',
    '<html><body><div data-original-tag="iframe" src="https://example.com/viewer" type="application/pdf">' + ARTICLE + '</div></body></html>',
], ids=['plain_pdf_text', 'pdf_markdown', 'ordinary_archive', 'web_pdf_embed', 'extension_prose',
        'unowned_pdf_viewer_label', 'unsupported_mime', 'unowned_frame'])
def test_real_or_unowned_content_stays_usable_and_unchanged(monkeypatch, body):
    responses(monkeypatch, archive=body)
    assert intent._has_wayback_error_body(body) is False
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert result['ok'] is True and result['content'] == body


def test_source_publication_date_is_not_replaced_by_viewer_metadata(monkeypatch):
    viewer = VIEWER.replace('<body>', '<body><meta property="article:published_time" content="2026-10-24">')
    body = ('<html><head><title>Partnership announcement</title>'
            '<meta property="article:published_time" content="2026-05-20">'
            '</head><body>' + viewer + ARTICLE + '</body></html>')
    responses(monkeypatch, direct=body, archive=body)
    for result in (asyncio.run(intent._scrape_sd_hardened(SOURCE)),
                   asyncio.run(intent._try_wayback(SOURCE))):
        assert result['source_publication_date'] == '2026-05-20'


def test_direct_article_with_archive_like_ids_keeps_existing_semantics(monkeypatch):
    body = ('<html><head><title>Ordinary report</title></head><body class="navia">'
            '<div id="wm-ipp-base">Our navigation</div>'
            '<form action="https://web.archive.org/web/submit"></form>'
            + ARTICLE + '</body></html>')
    responses(monkeypatch, direct=body)
    assert intent._evaluate_sd_response(200, body) == 'ok'
    result = asyncio.run(intent._scrape_sd_hardened(SOURCE))
    assert result['stage'] == 'sd:baseline' and QUOTE in result['content']


def test_truncated_owned_frame_does_not_admit_its_ui_or_drop_prior_source():
    truncated = VIEWER[:VIEWER.index('</pdf-viewer>')]
    body = '<html><body>' + truncated
    assert intent._has_wayback_error_body(body)
    assert intent._evaluate_sd_response(200, body) != 'ok'
    with_source = '<html><body>' + ARTICLE + truncated
    cleaned = intent._without_browser_pdf_frames(with_source)
    assert QUOTE in cleaned and 'Extracting text from PDF' not in cleaned
    assert intent._has_wayback_error_body(with_source) is False


def test_native_pdf_bytes_still_use_existing_text_extraction_fallback(monkeypatch):
    responses(monkeypatch, direct='%PDF-1.7\n' + 'binary object ' * 300)
    fallback = AsyncMock(return_value={'ok': True, 'stage': 'exa_scraped', 'content': QUOTE,
                                       'error': None})
    monkeypatch.setattr(intent, '_scrape_exa', fallback)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    fallback.assert_awaited_once_with(SOURCE)
    assert result['results'][0]['text'] == QUOTE


def test_browser_internal_pdf_text_layer_uses_independent_pdf_text_fallback(monkeypatch):
    # Expanded browser DOM is not a supported PDF extraction. Even loaded
    # internal text uses the existing exact-source text retrieval path.
    frame = VIEWER.replace('Document properties', '<div class="textLayer">' + QUOTE + '</div>')
    body = '<html><body>' + frame + '</body></html>'
    responses(monkeypatch, direct=body, archive=body)
    fallback = AsyncMock(return_value={'ok': True, 'stage': 'exa_scraped', 'content': QUOTE,
                                       'error': None})
    monkeypatch.setattr(intent, '_scrape_exa', fallback)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    fallback.assert_awaited_once_with(SOURCE)
    assert result['results'][0]['text'] == QUOTE
