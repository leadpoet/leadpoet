"""Entrance animations apply to containers and headings, not only paragraphs."""

import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as company
from qualification.scoring.verification_helpers import visible_html_text


REVEAL = 'class="wow fadeInUp" style="visibility:hidden;animation-name:none"'
CLAIM = "We provide payment APIs to regulated financial institutions."


@pytest.mark.parametrize("tag", ["div", "section", "h1", "h2", "h3", "span", "li", "article"])
def test_company_evidence_preserves_revealable_element_text(tag):
    raw = f"<{tag} {REVEAL}>{CLAIM}</{tag}>"
    assert company._plain_text(raw) == CLAIM
    # Intent/article callers do not opt into company-page animation recovery.
    assert CLAIM not in visible_html_text(raw)


def test_nested_animated_container_keeps_text_but_not_hidden_descendants():
    raw = (
        f'<div {REVEAL}><h2 {REVEAL}>Payment APIs</h2><p>{CLAIM}</p>'
        '<span hidden>Unpublished claim.</span>'
        '<span aria-hidden="true">Accessibility-hidden claim.</span></div>'
    )
    assert company._plain_text(raw) == "Payment APIs " + CLAIM


@pytest.mark.parametrize("wrapper", [
    '<div hidden>{}</div>', '<section aria-hidden="true">{}</section>',
    '<div style="display:none">{}</div>',
    '<div style="visibility:hidden">{}</div>',
    '<style>.secret {{visibility:hidden}}</style><div class="secret">{}</div>',
    '<script>{}</script>', '<template>{}</template>', '<aside>{}</aside>',
])
def test_animated_container_cannot_override_a_hidden_ancestor(wrapper):
    assert CLAIM not in company._plain_text(wrapper.format(f'<div {REVEAL}>{CLAIM}</div>'))


@pytest.mark.parametrize("attributes", [
    REVEAL + ' hidden', REVEAL + ' aria-hidden="true"',
    REVEAL.replace("visibility:hidden", "display:none;visibility:hidden"),
    REVEAL.replace("animation-name:none", "animation-name:none;opacity:0"),
    REVEAL.replace("fadeInUp", "fadeOut"),
    REVEAL.replace("wow", "arbitrary"),
])
def test_other_hidden_states_on_containers_remain_hidden(attributes):
    assert CLAIM not in company._plain_text(f'<div {attributes}>{CLAIM}</div>')


def test_native_exact_fetch_preserves_revealable_content_without_extra_requests(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    monkeypatch.delenv("SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.delenv("QUALIFICATION_SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.setattr(company, "public_http_url", lambda value: value)
    calls = []

    async def fetch(_session, url, **kwargs):
        calls.append(url)
        return 200, url, f'<main><div {REVEAL}><p>{CLAIM}</p></div></main>'

    monkeypatch.setattr(company, "_fetch_bounded_html", fetch)
    result = asyncio.run(company._fetch_page(None, "https://example.com/product"))
    assert result["ok"] is True
    assert company._quote_occurs(CLAIM, result["text"])
    assert calls == ["https://example.com/product"]
