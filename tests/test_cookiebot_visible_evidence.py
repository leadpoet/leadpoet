"""Cookie consent chrome must not displace bounded company-page evidence."""

import json

import pytest

from lab_arena.operations import OPENROUTER_MAX_CONTENT_CHARS
from qualification.scoring.company_evidence_investigator import (
    MAX_PAGE_CHARACTERS,
    _bounded_message_json,
    _plain_text,
    _quote_occurs,
)
from qualification.scoring.verification_helpers import (
    visible_html_links,
    visible_html_text,
)


HQ_QUOTE = "Headquarters 4000 Center at North Hills St Raleigh, NC 27609"


@pytest.mark.parametrize(
    "widget_id",
    ["CybotCookiebotDialog", "cybotcookiebotdialog", "CYBOTCOOKIEBOTDIALOG"],
)
def test_oversized_consent_widget_cannot_displace_page_or_links(widget_id):
    consent = "Consent Details: tracker storage and preferences. " * 1100
    raw = (
        f'<html><body><div id="{widget_id}">'
        '<a href="https://consent.example/tracker">Consent provider</a>'
        f"{consent}</div>"
        '<main><h1>Contact ExampleCo</h1>'
        f"<section><h2>Headquarters</h2>"
        "<p>4000 Center at North Hills St Raleigh, NC 27609</p></section>"
        '<a href="https://example.com/about">About ExampleCo</a>'
        "</main></body></html>"
    )
    visible = visible_html_text(raw, include_scroll_reveal=True)
    text = _plain_text(raw)
    assert "Consent Details" not in visible
    assert "Consent provider" not in text
    assert "Contact ExampleCo" in visible
    assert _quote_occurs(HQ_QUOTE, text)
    assert len(text) <= MAX_PAGE_CHARACTERS == 24_000
    assert visible_html_links(raw) == ("https://example.com/about",)

    request = {
        "requested_targets": ["geography"],
        "prefetched_sources": [
            {"url": "https://example.com/contact", "text": text},
            {"url": "https://example.com/source-a", "text": "a" * 24_000},
            {"url": "https://example.com/source-b", "text": "b" * 24_000},
        ],
    }
    encoded = _bounded_message_json(request)
    assert len(encoded) < OPENROUTER_MAX_CONTENT_CHARS
    first_page = json.loads(encoded)["prefetched_sources"][0]["text"]
    assert _quote_occurs(HQ_QUOTE, first_page)


@pytest.mark.parametrize(
    "content_id",
    ["CybotCookiebotDialogue", "preCybotCookiebotDialog",
     "CybotCookiebotDialog-content"],
)
def test_only_exact_dialog_root_is_excluded(content_id):
    raw = (
        f'<html><body><article id="{content_id}">'
        "ExampleCo explains its headquarters at 4000 Center at North Hills St."
        "</article></body></html>"
    )
    assert "ExampleCo explains its headquarters" in _plain_text(raw)


def test_legitimate_cookie_article_remains_visible():
    raw = (
        "<html><body><article><h1>Cookie security research</h1>"
        "ExampleCo's product detects stolen browser cookies for customers."
        "</article></body></html>"
    )
    assert "Cookie security research" in _plain_text(raw)
    assert "stolen browser cookies" in visible_html_text(raw)
