"""Archive retrieval must spend the evidence bound on visible source content."""
import asyncio

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from tests.test_intent_wayback_empty_body import SOURCE, install_source_responses


@pytest.mark.parametrize("padding", ["style", "script"])
def test_archive_extracts_visible_article_before_the_existing_size_bound(monkeypatch, padding):
    article = (
        "On May 13, 2026, Example Company announced availability of its new service. "
        "The product automates customer care, service requests and order processing. "
        "The announcement describes the release, the supplier and the launch date. "
    )
    hidden = "HIDDEN CLAIM: the company was acquired yesterday. "
    raw = ("<html><head><title>Product release</title><" + padding + ">"
           + "/* unused page template */" * 4000 + "</" + padding + "></head><body>"
           + "<div hidden>" + hidden + "</div><article><h1>Product release</h1>"
           + "<p>" + article + "</p></article></body></html>")
    assert raw.index(article) > intent.MAX_SCRAPED_CHARS
    calls = install_source_responses(monkeypatch, archive=raw)
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert result["ok"] and result["stage"] == "wayback"
    assert article.strip() in result["content"]
    assert "unused page template" not in result["content"]
    assert hidden not in result["content"]
    assert len(result["content"]) <= intent.MAX_SCRAPED_CHARS
    assert len(calls) == 2  # No added paid or unpaid retrieval.


def test_archive_plain_text_preserves_the_existing_bound(monkeypatch):
    text = "A readable source body with factual event details. " * 2000
    install_source_responses(monkeypatch, archive=text)
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert result["ok"]
    assert result["content"] == text[:intent.MAX_SCRAPED_CHARS]


def test_archive_extraction_failure_does_not_admit_raw_template(monkeypatch):
    from qualification.scoring import verification_helpers
    raw = "<html><body><article>" + "Factual source text. " * 4000 + "</article></body></html>"
    install_source_responses(monkeypatch, archive=raw)
    def fail(_raw):
        raise ValueError("synthetic extractor failure")
    monkeypatch.setattr(verification_helpers, "extract_article_body", fail)
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert not result["ok"] and not result["content"]
