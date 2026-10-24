"""Quote typography must not change evidence facts or semantic parsing."""
from __future__ import annotations

import pytest

from qualification.scoring import company_evidence_investigator as inv


AMDOCS_URL = (
    "https://www.amdocs.com/products-services/amdocs-studios/cloud-studio/"
    "cloud-migration-modernization"
)
# Exact October 24 captured fetch and submitted finding (call 27b3461f...3a53).
# Only the terminal possessive apostrophe differs. This is not a lead-score test.
FETCHED = (
    "Amdocs’ Mass Migration service enables fast, secure, and seamless migration "
    "of small‑ and medium‑sized applications through a platform‑based, "
    "repeatable approach."
)
SUBMITTED = FETCHED.replace("Amdocs’", "Amdocs", 1)


def test_captured_amdocs_terminal_possessive_is_grounded():
    assert inv._quote_occurs(SUBMITTED, FETCHED)
    assert inv._quote_occurs(FETCHED, FETCHED)


def test_captured_finding_keeps_held_verdict_after_quote_validation():
    finding = {
        "target": "required_attribute", "status": "VERIFIED",
        "observed_value": None, "observed_country": "", "observed_state": "",
        "observed_industry": "", "observed_subindustry": "",
        "activity_role": "supplier_operator", "evidence_url": AMDOCS_URL,
        "evidence_quote": SUBMITTED, "old_name": "", "new_name": "Amdocs",
        "old_domain": "", "new_domain": "amdocs.com",
        "shared_linkedin_slug": "amdocs", "ownership": None,
        "reason": "The source identifies Amdocs' own cloud migration service.",
    }
    result = inv._validated_findings(
        {"findings": [finding]}, targets=("required_attribute",),
        fetched_pages={AMDOCS_URL: FETCHED},
        fetched_final_urls={AMDOCS_URL: AMDOCS_URL},
        first_party_domains={"amdocs.com"}, identity_names={"amdocs"},
    )["required_attribute"]
    assert result["status"] == "VERIFIED"
    assert result["evidence_quote"] == SUBMITTED
    assert result["evidence_url"] == AMDOCS_URL


@pytest.mark.parametrize("quote,page", [
    ("Atlas Partners platform serves customers.",
     "Atlas Partners' platform serves customers."),
    ("James consulting team serves customers.",
     "James’ consulting team serves customers."),
    ("United States leading firms provide assurance.",
     "United States’ leading firms provide assurance."),
    ("We serve customers’", "We serve customers"),
])
def test_terminal_possessive_equivalence_is_general(quote, page):
    assert inv._quote_occurs(quote, page)


@pytest.mark.parametrize("quote,page", [
    ("Acme Partners is publicly listed.",
     "Acme Partners’ is not publicly listed."),
    ("Acme doesnt operate a banking platform.",
     "Acme doesn’t operate a banking platform."),
    ("Companys platform supports enterprise clients.",
     "Company’s platform supports enterprise clients."),
    ("Amdoc platform supports enterprise clients.",
     "Amdoc’s platform supports enterprise clients."),
    ("Acme Partners develops industrial hardware.",
     "Acme Partners’ resells industrial hardware."),
    ("Acme Partners reported 35 million dollars.",
     "Acme Partners’ reported 3.5 million dollars."),
    ("Acme Partners margin was 3.5 percent.",
     "Acme Partners’ margin was -3.5 percent."),
    ("Acme Partners margin was -3.5 percent.",
     "Acme Partners’ margin was −3.5 percent."),
    ("Acme Partners employs 10000 people.",
     "Acme Partners’ employs 10,000 people."),
    ("Acme Partners announced a deal on 2026-10-08.",
     "Acme Partners’ announced a deal on 2026-10-07."),
    ("The 1990s technology supports the platform.",
     "The 1990s’ technology supports the platform."),
    ("The 5 measurement is on the plan.",
     "The 5’ measurement is on the plan."),
    ("Acme Partners provides software and hardware.",
     "Acme Partners’ provides software. Other Company provides hardware."),
    ("Acme Partners provides software ... and hardware.",
     "Acme Partners’ provides software. Other Company provides hardware."),
])
def test_quote_equivalence_cannot_change_facts_or_internal_apostrophes(quote, page):
    assert not inv._quote_occurs(quote, page)


def test_existing_quote_bounds_and_visible_source_boundary_remain():
    oversized = "James’ " * 287  # 2008 characters after whitespace normalization.
    assert len(inv._normalized_span(oversized)) > 2000
    assert not inv._quote_occurs(oversized, oversized)
    assert not inv._quote_occurs("James’", "James’")
    hidden = "Visible page.\n" + inv._IDENTITY_LINK_CONTEXT_MARKER + "\n" + FETCHED
    assert not inv._quote_occurs(SUBMITTED, hidden)


def test_shared_semantic_normalizer_keeps_possessives_and_negation():
    assert inv._normalized_span("Amdocs’ service is not listed at -3.5.") == (
        "amdocs’ service is not listed at -3.5."
    )
