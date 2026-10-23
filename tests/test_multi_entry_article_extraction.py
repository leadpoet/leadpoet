"""A visible archive must not become only its extractor's first article."""

import asyncio
from pathlib import Path

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring import verification_helpers as helpers
from tests.test_scrapingdog_deadline_routing import _HttpxClient


FIXTURE = Path(__file__).with_name("fixtures") / "intent_multi_entry_financing_archive.html"


def _entry(name, date, prose=None):
    prose = prose or (f"{name} completed financing to advance its research programs. " * 5)
    return (f'<article><header><h2><a href="/{name.lower()}">{name} financing</a>'
            f'</h2><time>{date}</time></header><p>{prose}</p></article>')


def _first_entry_extractor(monkeypatch):
    class FirstEntryOnly:
        @staticmethod
        def extract(*_args, **_kwargs):
            return "First financing\nJuly 20, 2026\n" + (
                "First completed financing to advance its research programs. " * 5
            )

    monkeypatch.setattr(helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(helpers, "_trafilatura", FirstEntryOnly, raising=False)


def test_captured_public_archive_keeps_second_company_and_its_own_dates():
    pytest.importorskip("trafilatura")
    body = helpers.extract_article_body(FIXTURE.read_text())

    assert "Memento Medicines" in body
    assert "Alveus Therapeutics" in body
    assert "$159.8 million Series A" in body
    assert body.index("Memento Medicines") < body.index("Alveus Therapeutics")
    alveus = body[body.index("Alveus Therapeutics"):]
    assert "January 26, 2026" in alveus
    assert "January 8, 2026" in alveus
    assert "July 20, 2026" not in alveus
    assert "June 18, 2026" in body[:body.index("Alveus Therapeutics")]
    assert "Home Portfolio News" not in body
    assert "Unrelated sidebar news" not in body


def test_partial_extraction_keeps_visible_archive_entries_in_order(monkeypatch):
    _first_entry_extractor(monkeypatch)
    raw = ("<html><body><main><h1>Company news</h1>"
           + _entry("First", "July 20, 2026")
           + _entry("Second", "January 8, 2026")
           + "</main><div>Unrelated page text outside the articles.</div></body></html>")
    body = helpers.extract_article_body(raw)

    assert body.index("First financing") < body.index("July 20, 2026")
    assert body.index("July 20, 2026") < body.index("Second financing")
    assert body.index("Second financing") < body.index("January 8, 2026")
    assert "Unrelated page text" not in body


def test_archive_uses_same_visible_entries_without_optional_extractor(monkeypatch):
    monkeypatch.setattr(helpers, "_TRAFILATURA_AVAILABLE", False)
    body = helpers.extract_article_body(FIXTURE.read_text())
    assert "Alveus Therapeutics" in body
    assert "Unrelated sidebar news" not in body
    assert "Category: Uncategorized" not in body


@pytest.mark.parametrize("wrapper", [
    "<aside>{}</aside>", "<nav>{}</nav>", "<footer>{}</footer>",
    '<div class="news-sidebar">{}</div>',
    '<div id="sidebar">{}</div>', '<div role="complementary">{}</div>',
    '<div class="newsroom_related">{}</div>',
    '<div class="related-articles">{}</div>',
    '<div hidden>{}</div>', '<div aria-hidden="true">{}</div>',
    '<div style="display:none">{}</div>',
    '<div class="css-hidden">{}</div>',
    "<script>{}</script>", "<template>{}</template>",
])
def test_archive_retains_existing_hidden_sidebar_and_related_exclusions(
    monkeypatch, wrapper,
):
    _first_entry_extractor(monkeypatch)
    raw = ('<html><head><style>.css-hidden { display: none; }</style></head><body>'
           + _entry("First", "July 20, 2026")
           + wrapper.format(_entry("Excluded", "December 31, 2026"))
           + _entry("Second", "January 8, 2026") + "</body></html>")
    body = helpers.extract_article_body(raw)
    assert "Second completed financing" in body
    assert "Excluded" not in body
    assert "December 31" not in body


def test_nested_archive_sidebar_is_excluded_without_changing_general_visible_text(monkeypatch):
    _first_entry_extractor(monkeypatch)
    first = _entry("First", "July 20, 2026").replace(
        "</article>", '<div class="sidebar">Sidebar-only date December 31, 2026.</div></article>',
    )
    raw = "<html><body>" + first + _entry("Second", "January 8, 2026") + "</body></html>"
    assert "Sidebar-only" in helpers.visible_html_text(raw)
    assert "Sidebar-only" not in helpers.extract_article_body(raw)


def test_single_article_with_related_articles_keeps_normal_extraction(monkeypatch):
    _first_entry_extractor(monkeypatch)
    raw = ("<html><body>" + _entry("First", "July 20, 2026")
           + '<div class="related-articles">'
           + _entry("Second", "January 8, 2026") + "</div>"
           + "<p>Unrelated legal text outside the article.</p></body></html>")
    assert helpers.extract_article_body(raw) == helpers._trafilatura.extract(raw)


@pytest.mark.parametrize("with_article", [False, True])
def test_nonarchive_visible_fallback_keeps_existing_sidebar_text(monkeypatch, with_article):
    monkeypatch.setattr(helpers, "_TRAFILATURA_AVAILABLE", False)
    main = _entry("First", "July 20, 2026") if with_article else "<p>Ordinary page body.</p>"
    raw = ('<html><body>' + main
           + '<div class="sidebar">Visible sidebar prose.</div>'
           + '<div role="complementary">Visible complementary prose.</div></body></html>')
    expected = helpers.visible_html_text(raw)
    assert "Visible sidebar prose" in expected
    assert "Visible complementary prose" in expected
    assert helpers.extract_article_body(raw) == expected


def test_single_article_and_sidebar_article_keep_original_fallback(monkeypatch):
    monkeypatch.setattr(helpers, "_TRAFILATURA_AVAILABLE", False)
    raw = ("<html><body>" + _entry("First", "July 20, 2026")
           + '<div class="sidebar">' + _entry("Second", "January 8, 2026")
           + "</div></body></html>")
    assert helpers.extract_article_body(raw) == helpers.visible_html_text(raw)


def test_nested_article_does_not_turn_single_article_into_archive(monkeypatch):
    _first_entry_extractor(monkeypatch)
    nested = _entry("First", "July 20, 2026").replace(
        "</article>", _entry("Comment", "January 8, 2026") + "</article>",
    )
    raw = "<html><body>" + nested + "</body></html>"
    assert helpers.extract_article_body(raw) == helpers._trafilatura.extract(raw)


def test_oversize_archive_keeps_existing_source_return_bound(monkeypatch):
    _first_entry_extractor(monkeypatch)
    raw = ("<html><body>" + _entry("First", "July 20, 2026")
           + _entry("Second", "January 8, 2026", "Second research context. " * 4_000)
           + _entry("BeyondLimit", "March 3, 2026") + "</body></html>")
    client = _HttpxClient([httpx.Response(200, text=raw)])
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    result = asyncio.run(intent._scrape_sd_hardened("https://archive.example/news/"))

    assert result["ok"] is True
    assert "Second financing" in result["content"]
    assert len(result["content"]) == intent.MAX_SCRAPED_CHARS
    assert "BeyondLimit" not in result["content"]
    assert len(client.calls) == 1
