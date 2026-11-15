"""Optional submitted company pages obey the paragraph's existing boundary."""
from copy import deepcopy

import pytest

from qualification.scoring import intent_details, lead_scorer
from qualification.scoring.company_evidence_investigator import _IDENTITY_LINK_CONTEXT_MARKER
from tests.test_intent_details_fetched_first_party_context import (
    ABOUT, HOME, HOME_QUOTE, _case,
)


def _project(body, *, suffix="", complete_identity=True, final_url=ABOUT):
    company, icp, fit, signals = _case(complete_identity=complete_identity)
    company.required_attribute.evidence_url = ABOUT
    company.required_attribute.evidence_quote = "Accept payments online."
    pages = {
        HOME: {"final_url": HOME, "text": HOME_QUOTE},
        ABOUT: {"final_url": final_url, "text": body + suffix},
    }
    original = deepcopy(pages)
    contexts = lead_scorer._matched_company_source_contexts(
        fit, None, None, pages, company=company, paragraph=company.intent_details,
    )
    assert pages == original
    return company, icp, fit, signals, contexts


@pytest.mark.parametrize("body,suffix", [
    ("Accept payments online.", _IDENTITY_LINK_CONTEXT_MARKER + " Ryft"),
    ("About Ryft.", _IDENTITY_LINK_CONTEXT_MARKER + " Accept payments online."),
])
def test_metadata_only_name_or_quote_cannot_poison_paragraph_preflight(body, suffix):
    company, icp, fit, signals, contexts = _project(body, suffix=suffix)
    assert not any(row["dimension"] == "first_party_company" for row in (contexts or []))
    document = intent_details.review_evidence(
        company, icp, signals, fit.receipt("company_fit"),
        company_source_contexts=contexts,
    )
    assert not any(source.get("source_url") == ABOUT for source in document["admitted_evidence"])
    assert any(source.get("source_url") == HOME for source in document["admitted_evidence"])


@pytest.mark.parametrize("suffix", ["", _IDENTITY_LINK_CONTEXT_MARKER + " unread locator"])
def test_visible_submitted_company_context_keeps_exact_existing_proof(suffix):
    visible = "About Ryft. Accept payments online."
    company, icp, fit, signals, contexts = _project(visible, suffix=suffix)
    selected = next(row for row in contexts if row["dimension"] == "first_party_company")
    assert selected == {"dimension": "first_party_company", "url": ABOUT, "text": visible}
    intent_details.review_evidence(
        company, icp, signals, fit.receipt("company_fit"),
        company_source_contexts=contexts,
    )


@pytest.mark.parametrize("change", ["identity", "redirect"])
def test_unbound_optional_page_stays_excluded(change):
    *_, contexts = _project(
        "About Ryft. Accept payments online.",
        complete_identity=change != "identity",
        final_url="https://other.example/" if change == "redirect" else ABOUT,
    )
    assert not any(row["dimension"] == "first_party_company" for row in (contexts or []))


def test_manual_metadata_only_company_context_still_fails_native_validator():
    company, icp, fit, signals = _case()
    company.required_attribute.evidence_url = ABOUT
    company.required_attribute.evidence_quote = "Accept payments online."
    with pytest.raises(ValueError, match="invalid company source context"):
        intent_details.review_evidence(
            company, icp, signals, fit.receipt("company_fit"),
            company_source_contexts=[{
                "dimension": "first_party_company", "url": ABOUT,
                "text": "Accept payments online." + _IDENTITY_LINK_CONTEXT_MARKER + " Ryft",
            }],
        )
