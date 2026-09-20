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
