"""Article evidence must not inherit dates from linked recommendations."""

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import verification_helpers as helpers


ARTICLE_URL = "https://acme.example/news/acme-raises-series-b"


def _article_html() -> str:
    body = (
        "Acme completed a Series B financing with new investment. "
        "The transaction funds product expansion and international growth. "
    ) * 4
    return f"""<html><body><main>
      <article><h1>Acme raises Series B</h1>
        <p>Company news July 3, 2024</p><p>{body}</p>
        <p>The related-party transaction was disclosed separately.</p>
      </article>
      <div class="newsroom_related">
        <h2>Related articles</h2>
        <div class="news-related_cms-wrapper">
          <a href="/news/acme-new-product">Company news Sep 14, 2026
            Acme launches a new product</a>
        </div>
      </div>
    </main></body></html>"""


def test_nested_related_news_cards_do_not_date_company_article():
    raw = _article_html()
    text = investigator._plain_text(raw)
    assert "Acme raises Series B" in text
    assert "July 3, 2024" in text
    assert "completed a Series B financing" in text
    assert "related-party transaction" in text
    assert "2026" not in investigator._visible_quote_surface(text)
    # Link discovery still sees visible links; they are not article-body facts.
    assert ("/news/acme-new-product", "Company news Sep 14, 2026 Acme launches a new product") in (
        (href, " ".join(label.split()))
        for href, label in helpers.visible_html_link_labels(raw)
    )


@pytest.mark.parametrize("component_class", [
    "newsroom_related", "news-related_cms-wrapper",
    "NEWS-RELATED_CMS-WRAPPER",
])
def test_each_related_component_is_excluded_from_article_text(component_class):
    raw = ("<html><body><article>Acme completed a Series B in 2024.</article>"
           f'<div class="{component_class}">Related card Sep 14, 2026</div>'
           "</body></html>")
    text = investigator._visible_quote_surface(investigator._plain_text(raw))
    assert "Series B in 2024" in text
    assert "2026" not in text


def test_intent_article_extraction_rejects_related_card_from_recall_extractor(
    monkeypatch,
):
    raw = _article_html()

    class RecallExtractor:
        @staticmethod
        def extract(*_args, **_kwargs):
            return (
                "Acme raises Series B\nCompany news July 3, 2024\n"
                + "Acme completed a Series B financing with new investment. " * 5
                + "Related articles Company news Sep 14, 2026 Acme launches a new product"
            )

    monkeypatch.setattr(helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(helpers, "_trafilatura", RecallExtractor, raising=False)
    body = helpers.extract_article_body(raw)
    assert "July 3, 2024" in body
    assert "completed a Series B financing" in body
    assert "2026" not in body


def test_delimiter_aware_filter_keeps_related_party_and_similar_prose():
    raw = """<html><body><main>
      <div class="annual-report-related-party-content">
        Acme disclosed a related-party financing.</div>
      <div class="article_related_party_transactions">
        Acme disclosed another related-party financing.</div>
      <div class="unrelated">An unrelated sale was completed in 2025.</div>
      <div id="correlated">A correlated revenue measure rose.</div>
      <p>These related facts belong to the article body.</p>
      <div class="newsroom_related">Related news dated in 2026.</div>
      <div class="news-related_cms-item">Another linked item dated in 2027.</div>
    </main></body></html>"""
    text = investigator._visible_quote_surface(investigator._plain_text(raw))
    for phrase in (
        "related-party financing", "another related-party financing",
        "unrelated sale", "correlated revenue", "related facts",
    ):
        assert phrase in text
    assert "2026" not in text and "2027" not in text


def test_independent_news_index_can_bind_exact_visible_archive_card():
    raw = f"""<html><body><section class="newsroom_related">
      <a href="{ARTICLE_URL}">Acme raises Series B. Company news Jul 3, 2024</a>
      <a href="/news/other-company">Other Company raises Series B. Jan 1, 2026</a>
      <span hidden><a href="{ARTICLE_URL}">
        Acme raises Series B. Company news Jan 1, 2026</a></span>
    </section></body></html>"""
    assert investigator._visible_stage_archive_cards(
        raw, "https://acme.example/news", (ARTICLE_URL,), ("Acme",),
    ) == [{
        "article_url": ARTICLE_URL,
        "label": "Acme raises Series B. Company news Jul 3, 2024",
        "publication_date": "2024-07-03",
    }]
