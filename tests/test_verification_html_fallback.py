"""Focused regressions for dependency-free source article extraction."""

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring import verification_helpers


SOURCE_URL = (
    "https://ir.truist.com/2026-09-03-"
    "Tory-Sherman-joins-Truist-Wealth-as-Wealth-Brokerage-national-director"
)


def _truist_shaped_html() -> str:
    return f"""<!doctype html>
<html><head>
<style>{'STYLE-BUDGET-NOISE ' * 2_000}</style>
<script>{'SCRIPT-BUDGET-NOISE ' * 2_000}</script>
</head><body>
<nav>{'Navigation item ' * 300}</nav>
<main><article>
<time>Sep 3, 2026</time>
<h1>Tory Sherman joins Truist Wealth as Wealth Brokerage national director</h1>
<p>Truist Financial Corporation announced the appointment of Tory Sherman as
national director of Wealth Brokerage.</p>
<p>{'Visible company context. ' * 300}</p>
</article></main>
<template>{'TEMPLATE-BUDGET-NOISE ' * 2_000}</template>
</body></html>"""


def _row():
    return {
        "id": "truist-signal",
        "company": "Truist Financial Corporation",
        "website": "https://www.truist.com",
        "company_linkedin": "truistfinancialcorporation",
        "contact_linkedin": "",
        "claim": "Truist appointed Tory Sherman to lead Wealth Brokerage.",
        "signal_date": "2026-09-03",
        "signal_type": "intent",
        "claimed_source_urls": [SOURCE_URL],
        "_target_signal_text": "Recent senior leadership appointment",
    }


def test_stdlib_html_fallback_keeps_truist_event_in_final_prompt(monkeypatch):
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", False)

    extracted = verification_helpers.extract_article_body(_truist_shaped_html())
    prompt = intent._build_final_judge_prompt(
        _row(),
        {"results": [{"url": SOURCE_URL, "title": "Truist News", "text": extracted}]},
    )

    assert "Sep 3, 2026" in prompt
    assert "appointment of Tory Sherman" in prompt
    assert "Wealth Brokerage national director" in prompt
    assert "national director of Wealth Brokerage" in prompt
    assert "SCRIPT-BUDGET-NOISE" not in extracted
    assert "STYLE-BUDGET-NOISE" not in extracted
    assert "TEMPLATE-BUDGET-NOISE" not in extracted


def test_non_html_and_already_extracted_bodies_are_unchanged(monkeypatch):
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", False)
    plain_text = "Ordinary source text with an appointment and a date."
    extracted_body = "Title\n\nAlready extracted article body.\n\nPublished 2026-09-03."

    assert verification_helpers.extract_article_body(plain_text) == plain_text
    assert verification_helpers.extract_article_body(extracted_body) == extracted_body


def test_script_text_cannot_become_source_evidence(monkeypatch):
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", False)
    html = """<html><body>
    <script>Acme appointed a Chief Revenue Officer on 2026-09-03.</script>
    <style>.claim::after { content: "Acme completed an acquisition"; }</style>
    <p>Acme provides ordinary customer support information.</p>
    </body></html>"""

    extracted = verification_helpers.extract_article_body(html)

    assert extracted == "Acme provides ordinary customer support information."
    assert "appointed" not in extracted
    assert "acquisition" not in extracted

    hidden_only = """<html><head>
    <script>Acme appointed a Chief Revenue Officer on 2026-09-03.</script>
    <style>.claim::after { content: "Acme completed an acquisition"; }</style>
    </head></html>"""
    assert verification_helpers.extract_article_body(hidden_only) == ""


def test_visible_links_survive_fallback_and_exclude_hidden_targets(monkeypatch):
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", False)
    html = """<html><head><style>
    .css-hidden { display: none; }
    </style></head><body>
    <nav><a href="https://nav.example/parent">Parent</a></nav>
    <script><a href="https://script.example/parent">Script parent</a></script>
    <a hidden href="https://hidden.example/parent">Hidden parent</a>
    <div class="css-hidden">
      <a href="https://css-hidden.example/parent">CSS hidden parent</a>
    </div>
    <a href="https://visible.example/company">Visible company</a>
    <a href="https://visible.example/company">Duplicate company link</a>
    </body></html>"""

    body = verification_helpers.extract_article_body(html)

    assert "Visible company" in body
    assert "Parent" not in body
    assert verification_helpers.visible_html_links(html) == (
        "https://visible.example/company",
    )


def test_failed_trafilatura_uses_same_visible_text_fallback(monkeypatch):
    class _FailingTrafilatura:
        @staticmethod
        def extract(_content, **_kwargs):
            raise RuntimeError("parser unavailable")

    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers, "_trafilatura", _FailingTrafilatura, raising=False
    )

    extracted = verification_helpers.extract_article_body(_truist_shaped_html())

    assert "appointment of Tory Sherman" in extracted
    assert "SCRIPT-BUDGET-NOISE" not in extracted


def test_valid_extraction_keeps_local_context_at_actual_h1(monkeypatch):
    article_body = (
        "Acculon Energy Announces Opening of New 2GWh Battery Manufacturing "
        "Facility in Mason, Ohio\n"
        "Acculon Energy today announced the opening of its operating battery "
        "manufacturing facility in Mason, Ohio. The plant has 2 GWh of annual "
        "capacity and two automated production lines. " * 4
    )

    class _ArticleWithoutHeaderContext:
        @staticmethod
        def extract(_content, **_kwargs):
            return article_body

    html = f"""<html><head>
      <title>Acculon Energy Announces Opening of New 2GWh Battery Manufacturing
      Facility in Mason, Ohio</title>
    </head><body>
      <div>Unrelated archive date: January 3, 2024.</div>
      <nav>Navigation updated February 2, 2025.</nav>
      <main><article>
        <div class="feature-resource_copy">
          <div class="feature-resourece_label-wrapper">
            <span hidden>Hidden revision: March 30, 2026.</span>
            <span>News</span><span>April 21, 2026</span>
          </div>
          <h1>Acculon Energy Announces Opening of New 2GWh Battery Manufacturing
          Facility in Mason, Ohio</h1>
        </div>
        <p>{article_body}</p>
      </article></main>
      <section class="related-articles">Related story: May 18, 2026.</section>
    </body></html>"""
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers,
        "_trafilatura",
        _ArticleWithoutHeaderContext,
        raising=False,
    )

    extracted = verification_helpers.extract_article_body(html)

    assert extracted.startswith("News April 21, 2026\n")
    assert extracted.endswith(article_body)
    assert extracted.count("April 21, 2026") == 1
    assert "January 3, 2024" not in extracted
    assert "February 2, 2025" not in extracted
    assert "March 30, 2026" not in extracted
    assert "May 18, 2026" not in extracted


def test_h1_context_is_bounded_and_existing_context_is_not_duplicated(
    monkeypatch,
):
    heading = "Acme Opens Its Operating Plant"
    raw_context = "START-TO-TRIM " + "OLD-LOCAL-CONTEXT " * 80 + (
        "News April 21, 2026"
    )
    bounded_context = raw_context[-500:]
    bounded_context = bounded_context.split(" ", 1)[1]
    article_body = (
        f"{bounded_context}\n"
        f"{heading}\n"
        + "Acme opened its operating plant and began production that day. " * 5
    )

    class _ArticleWithHeaderContext:
        @staticmethod
        def extract(_content, **_kwargs):
            return article_body

    html = f"""<html><body><main><article><header>
      <div>{raw_context}</div>
      <h1>{heading}</h1>
    </header><p>{article_body}</p></article></main></body></html>"""
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers,
        "_trafilatura",
        _ArticleWithHeaderContext,
        raising=False,
    )

    _visible, _heading, local_context = (
        verification_helpers._visible_html_document(html)
    )
    extracted = verification_helpers.extract_article_body(html)

    assert len(local_context) <= 500
    assert local_context.endswith("News April 21, 2026")
    assert "START-TO-TRIM" not in local_context
    assert extracted == article_body
    assert extracted.count("April 21, 2026") == 1


def test_h1_context_crosses_nested_empty_wrappers_without_global_text():
    html = """<html><body>
      <div>Unrelated archive date: January 3, 2024.</div>
      <article><header><time>April 21, 2026</time>
        <div><div>   <h1>Acme Opens Its Operating Plant</h1></div></div>
      </header></article>
    </body></html>"""

    _visible, heading, local_context = (
        verification_helpers._visible_html_document(html)
    )

    assert heading == "Acme Opens Its Operating Plant"
    assert local_context == "April 21, 2026"


def test_visible_competing_h1_dates_remain_text_not_publication_metadata(
    monkeypatch,
):
    heading = "Acme Opens Its Operating Plant"
    article_body = (
        f"{heading}\n"
        + "Acme opened its operating plant and began production that day. " * 5
    )

    class _ArticleWithoutHeaderContext:
        @staticmethod
        def extract(_content, **_kwargs):
            return article_body

    source_url = "https://example.test/news/acme-opens-plant"
    html = f"""<html><head><script type="application/ld+json">{{
      "@type":"NewsArticle", "datePublished":"2026-05-18",
      "mainEntityOfPage":null
    }}</script></head><body><main><article><header>
      <div>Event date April 21, 2026. Updated May 18, 2026.</div>
      <h1>{heading}</h1>
    </header><p>{article_body}</p></article></main></body></html>"""
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers,
        "_trafilatura",
        _ArticleWithoutHeaderContext,
        raising=False,
    )

    extracted = verification_helpers.extract_article_body(html)

    assert extracted.startswith(
        "Event date April 21, 2026. Updated May 18, 2026.\n"
    )
    assert intent._published_date_from_html(html, source_url) == ""

    conflicting_metadata = html.replace(
        '"mainEntityOfPage":null',
        f'"mainEntityOfPage":{{"@id":"{source_url}"}}',
    ).replace(
        "</head>",
        '<meta property="article:published_time" '
        'content="2026-04-21"></head>',
    )
    assert intent._published_date_from_html(
        conflicting_metadata, source_url
    ) == ""
