"""A fetched first-party company page can ground a later client paragraph."""

import asyncio
from types import SimpleNamespace

import pytest

from qualification.scoring import lead_scorer, intent_details
from qualification.scoring.company_evidence_investigator import (
    MAX_PAGE_CHARACTERS, PRIVATE_FETCHED_PAGES_KEY,
)
from qualification.scoring.company_fit_decision import company_fit_match
from tests.test_arena_intent_details_grounding import inputs


ABOUT = "https://www.chemify.example/about"
ROUND = "https://www.businesswire.com/news/home/chemify-series-b"
ABOUT_QUOTE = (
    "Chemify turns digital code into real molecules and makes medicines."
)
ABOUT_PAGE = (
    "About Chemify. Headquartered in Scotland, Chemify has more than 200 "
    "employees. " + ABOUT_QUOTE
)
ROUND_QUOTE = "Chemify raised a Series B to expand digital chemistry."


def _case(*, about_url=ABOUT, about_quote=ABOUT_QUOTE, name="Chemify"):
    company = SimpleNamespace(
        company_name=name,
        company_website="https://www.chemify.example/",
        intent_details=(
            f"{name}'s own site says it is headquartered in Scotland, has "
            "more than 200 employees, and turns code into molecules."
        ),
        required_attribute=SimpleNamespace(
            evidence_url=about_url, evidence_quote=about_quote,
        ),
    )
    fit = company_fit_match(details={"dimension_evidence": {
        "stage": {
            "decision": "match", "observed_decision": "match",
            "web_evidence": {"url": ROUND, "quote": ROUND_QUOTE},
        },
        "identity": {
            "decision": "match",
            "web_identity_receipt": {
                "decision": "match", "observed_name": name,
                "observed_domain": "chemify.example",
                "observed_linkedin_slug": "chemify",
                "evidence_source": "company_web_reverification",
            },
        },
    }})
    return company, fit


def _investigation(page=ABOUT_PAGE, *, final_url=ABOUT):
    return {PRIVATE_FETCHED_PAGES_KEY: {
        ABOUT: {"final_url": final_url, "text": page},
        ROUND: {"final_url": ROUND, "text": ROUND_QUOTE},
    }}


def test_fetched_first_party_page_reaches_bounded_paragraph_evidence():
    company, fit = _case()
    retained = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        retained, fit, _investigation(), company=company,
    )
    assert set(retained) == {ABOUT, ROUND}

    contexts = lead_scorer._matched_company_source_contexts(
        fit, None, None, retained,
        paragraph=company.intent_details, company=company,
    )
    assert len(contexts) <= 2
    assert any(item["url"] == ABOUT for item in contexts)

    paragraph_company, icp, signals, _fit_receipt = inputs()
    paragraph_company.company_name = company.company_name
    paragraph_company.company_website = company.company_website
    paragraph_company.intent_details = company.intent_details
    paragraph_company.required_attribute = company.required_attribute
    document = intent_details.review_evidence(
        paragraph_company, icp, signals, fit.receipt("company_fit"),
        company_source_contexts=contexts,
    )
    sources = [
        item for item in document["admitted_evidence"]
        if item.get("source_url") == ABOUT
    ]
    assert any(
        item["evidence_kind"] == "verified_company_source_context"
        and "more than 200 employees" in item["admitted_text"][0]
        for item in sources
    )
    assert sum(
        len(text.encode("utf-8"))
        for item in document["admitted_evidence"]
        if item["evidence_kind"] in {
            "verified_source_context", "verified_company_source_context",
        }
        for text in item.get("admitted_text", [])
    ) <= intent_details._MAX_SOURCE_CONTEXT_BYTES


@pytest.mark.parametrize("name", ["Io", "Café Labs Ltd."])
def test_produced_short_or_unicode_legal_name_context_is_admitted(name):
    quote = f"{name} turns digital code into real molecules."
    page = (
        f"About {name}. Headquartered in Scotland, {name} has more than "
        f"200 employees. {quote}"
    )
    company, fit = _case(name=name, about_quote=quote)
    retained = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        retained, fit, _investigation(page), company=company,
    )
    contexts = lead_scorer._matched_company_source_contexts(
        fit, None, None, retained,
        paragraph=company.intent_details, company=company,
    )
    assert contexts and any(item["url"] == ABOUT for item in contexts)

    paragraph_company, icp, signals, _fit_receipt = inputs()
    paragraph_company.company_name = name
    paragraph_company.company_website = company.company_website
    paragraph_company.intent_details = company.intent_details
    paragraph_company.required_attribute = company.required_attribute
    document = intent_details.review_evidence(
        paragraph_company, icp, signals, fit.receipt("company_fit"),
        company_source_contexts=contexts,
    )
    assert any(
        item.get("source_url") == ABOUT
        and item["evidence_kind"] == "verified_company_source_context"
        for item in document["admitted_evidence"]
    )


@pytest.mark.parametrize("change", [
    "off_company", "invalid_url", "empty_quote", "unfetched", "wrong_redirect",
    "quote_absent", "name_absent", "overbudget", "too_many_fetches",
    "identity_incomplete",
])
def test_unbound_submitted_page_never_reaches_paragraph(change):
    company, fit = _case()
    investigation = _investigation()
    if change == "off_company":
        company.required_attribute.evidence_url = "https://other.example/about"
    elif change == "invalid_url":
        company.required_attribute.evidence_url = "https://www.chemify.example/%0a"
    elif change == "empty_quote":
        company.required_attribute.evidence_quote = ""
    elif change == "unfetched":
        investigation[PRIVATE_FETCHED_PAGES_KEY].pop(ABOUT)
    elif change == "wrong_redirect":
        investigation[PRIVATE_FETCHED_PAGES_KEY][ABOUT]["final_url"] = (
            "https://other.example/about"
        )
    elif change == "quote_absent":
        investigation[PRIVATE_FETCHED_PAGES_KEY][ABOUT]["text"] = (
            "About Chemify. No platform claim is stated."
        )
    elif change == "name_absent":
        investigation[PRIVATE_FETCHED_PAGES_KEY][ABOUT]["text"] = (
            "An unrelated firm. " + ABOUT_QUOTE.replace("Chemify", "It")
        )
        company.required_attribute.evidence_quote = (
            ABOUT_QUOTE.replace("Chemify", "It")
        )
    elif change == "overbudget":
        investigation[PRIVATE_FETCHED_PAGES_KEY][ABOUT]["text"] = (
            ABOUT_PAGE + "x" * MAX_PAGE_CHARACTERS
        )
    elif change == "too_many_fetches":
        for index in range(2):
            url = f"https://www.chemify.example/other-{index}"
            investigation[PRIVATE_FETCHED_PAGES_KEY][url] = {
                "final_url": url, "text": "Other page",
            }
    elif change == "identity_incomplete":
        fit.details["dimension_evidence"]["identity"]["web_identity_receipt"][
            "observed_linkedin_slug"
        ] = ""

    retained = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        retained, fit, investigation, company=company,
    )
    contexts = lead_scorer._matched_company_source_contexts(
        fit, None, None, retained,
        paragraph=company.intent_details, company=company,
    ) or []
    if change in {"off_company", "invalid_url", "empty_quote"}:
        # The submitted hint is rejected. A separately fetched company page
        # remains admissible through its verified identity, name and overlap.
        assert not any(item["dimension"] == "first_party_company" for item in contexts)
        assert any(
            item["url"] == ABOUT and item["dimension"] == "first_party_context"
            for item in contexts
        )
    else:
        assert not any(item["url"] == ABOUT for item in contexts)


def test_unbound_context_cannot_be_injected_into_paragraph_reviewer():
    company, fit = _case()
    paragraph_company, icp, signals, _fit_receipt = inputs()
    paragraph_company.company_name = company.company_name
    paragraph_company.company_website = company.company_website
    paragraph_company.intent_details = company.intent_details
    paragraph_company.required_attribute = company.required_attribute
    with pytest.raises(ValueError, match="invalid company source context"):
        intent_details.review_evidence(
            paragraph_company, icp, signals, fit.receipt("company_fit"),
            company_source_contexts=[{
                "dimension": "first_party_company",
                "url": ABOUT,
                "text": "Chemify is headquartered in Scotland.",
            }],
        )


def test_real_scorer_hands_fetched_page_to_paragraph_gate(monkeypatch):
    company, icp, signals, _fit_receipt = inputs()
    company.company_name = "Chemify"
    company.company_website = "https://www.chemify.example/"
    company.intent_details = (
        "Chemify's own site says it is headquartered in Scotland, has "
        "more than 200 employees, and turns code into molecules."
    )
    company.required_attribute = SimpleNamespace(
        evidence_url=ABOUT, evidence_quote=ABOUT_QUOTE,
    )
    _case_company, fit = _case()
    seen = []

    async def verify_company(*_args, **kwargs):
        lead_scorer._retain_matched_investigator_source_contexts(
            kwargs["matched_company_source_sink"], fit,
            _investigation(), company=company,
        )
        return fit

    async def verify_signals(*_args, **_kwargs):
        return 54.0, 54.0, 1.0, 100, False, signals

    async def review(_company, _icp, _results, _receipt, **kwargs):
        contexts = kwargs["company_source_contexts"]
        seen.extend(contexts or [])
        return {"gate": "intent_details", "decision": "match"}

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify_company)
    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent_signal",
        verify_signals,
    )
    monkeypatch.setattr(intent_details, "review_intent_details", review)
    breakdown = asyncio.run(lead_scorer.score_company_competition_intent(
        company, icp, 0, 0, set(), integrity_policy=True,
    ))
    assert breakdown.final_score > 0
    assert any(context["url"] == ABOUT for context in seen)
