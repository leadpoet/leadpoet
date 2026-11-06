"""Stage persistence, the requested-Public listed-parent route, and scope lists.

Rule A: a proven stage survives later non-material events.
Rule B: only for a requested Public stage, a current wholly or majority owned
subsidiary of a currently listed parent satisfies Public. The route needs both
company-bound ownership text and the parent's current listing.
"""

from __future__ import annotations

import ast
import inspect
import json

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import industry_classification as industry
from qualification.scoring import lead_scorer
from tests.test_company_evidence_investigator import _company, _finding


INVESTIGATOR_PROMPT = " ".join(investigator._SYSTEM_PROMPT.split())


def _module_strings(module) -> str:
    """Return every string literal, with f-string fields shown as {}."""

    parts = []
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            parts.append("".join(
                value.value if isinstance(value, ast.Constant) else "{}"
                for value in node.values
            ))
    return " ".join(" ".join(parts).split())


LEAD_SCORER_PROMPTS = _module_strings(lead_scorer)


# Rule A -------------------------------------------------------------------

def test_investigator_keeps_a_proven_stage_through_non_material_later_events():
    assert (
        "A competing stage event is material only if the source states it "
        "completed and changed control, listing or priced-round series."
    ) in INVESTIGATOR_PROMPT
    assert (
        "Later strategic, minority or unspecified investments, pending or "
        "conditional deals and unfinished research are not; a proven stage stays."
    ) in INVESTIGATOR_PROMPT
    assert (
        "Later private financing keeps Series C+ unless stated to give control "
        "or a listing."
    ) in INVESTIGATOR_PROMPT
    # The possibility of control no longer erases a proven Series C+.
    assert "may have given" not in INVESTIGATOR_PROMPT
    assert "Conflicting labels without chronology are UNPROVEN" not in INVESTIGATOR_PROMPT
    # Debt, grants, supersession and the server-required search are unchanged.
    assert "later debt alone cannot" in INVESTIGATOR_PROMPT
    assert (
        "Debt, grants, planned rounds, or pending deals do not establish that value"
    ) in INVESTIGATOR_PROMPT
    assert "not the requested value by default" in INVESTIGATOR_PROMPT
    assert "complete the server-required current-stage search" in INVESTIGATOR_PROMPT
    assert "If material chronology or transaction type remains unresolved" in INVESTIGATOR_PROMPT


def test_reviewer_prompts_match_the_investigator_stage_persistence_rule():
    sentence = (
        "A proven stage stays proven unless a later completed event concretely "
        "changed control, listing, or the priced-round series; a later strategic, "
        "minority, or unspecified investment, a pending or conditional deal, or "
        "unfinished research does not erase it."
    )
    assert sentence in LEAD_SCORER_PROMPTS
    assert (
        "A later strategic, minority, or unspecified investment, a pending or "
        "conditional deal, or unfinished research does not erase a proven stage."
    ) in LEAD_SCORER_PROMPTS


# Rule B prompt contract ----------------------------------------------------

def test_listed_parent_route_is_stated_only_for_requested_public():
    assert (
        "Never inherit a stage from a parent or subsidiary, except that for "
        "requested Public a current wholly or majority owned subsidiary, brand "
        "or division of a currently listed parent is Public: return Acquired, "
        "VERIFIED, ownership quote primary, parent's current listing support 1."
    ) in INVESTIGATOR_PROMPT
    assert "it is not Public merely because the parent is public" not in LEAD_SCORER_PROMPTS
    assert (
        "Only when the requested stage is Public, a company that is currently a "
        "wholly or majority owned subsidiary, operating brand, or division of a "
        "currently listed parent also satisfies Public"
    ) in LEAD_SCORER_PROMPTS
    assert (
        "For every other requested stage, Acquired still disproves the stage."
    ) in LEAD_SCORER_PROMPTS


# Rule B end to end -----------------------------------------------------------

OWNERSHIP_URL = "https://acmerealty.example/about"
LISTING_URL = "https://investors.parentgroup.example/stock"
OWNERSHIP = "Acme Realty is a wholly owned subsidiary of Parent Group, Inc."
LISTING = (
    "Parent Group, Inc. (Nasdaq: PGRP) is a holding company for residential "
    "brokerage brands."
)


def _decide(ownership, listing, requested, *, validated=True):
    pages = {
        OWNERSHIP_URL: "About us. " + ownership + " Contact us.",
        LISTING_URL: "Stock information. " + listing,
    }
    raw = _finding(
        "stage", status="VERIFIED", observed_value="Acquired",
        evidence_url=OWNERSHIP_URL, evidence_quote=ownership,
        supporting_evidence_url_1=LISTING_URL,
        supporting_evidence_quote_1=listing,
    )
    raw.pop("supporting_evidence")
    finding = investigator._validated_findings(
        {"findings": [raw]}, targets=("stage",), fetched_pages=pages,
        fetched_final_urls={url: url for url in pages},
        first_party_domains={"acmerealty.example"},
        identity_names={"acmerealty"},
        requested_stage=requested,
    )["stage"]
    company = _company(
        name="Acme Realty", website="https://acmerealty.example",
        linkedin="https://www.linkedin.com/company/acme-realty",
    )
    projected = lead_scorer._project_investigator_stage(
        {"observed_company_name": company.company_name}, finding,
        icp_stage=requested,
    )
    # The decision reads the saved receipt, not live Python objects.
    retained = json.loads(json.dumps(projected))
    decision = lead_scorer._decision_from_observed_stage(
        retained, requested,
        validated_stage_finding=finding if validated else None,
        company=company, evidence_attributed=True,
    )
    return finding, retained, decision


def test_subsidiary_of_currently_listed_parent_matches_requested_public():
    finding, retained, decision = _decide(OWNERSHIP, LISTING, "public")

    assert finding["status"] == "VERIFIED"
    assert finding["supporting_evidence"] == [{"url": LISTING_URL, "quote": LISTING}]
    assert retained["observed_company_stage"] == "acquired"
    assert retained["stage_matches"] is True
    assert retained["dimension_evidence"]["stage"]["supporting_evidence"] == [
        {"url": LISTING_URL, "quote": LISTING}
    ]
    assert decision == "match"


def test_listed_parent_route_requires_the_validated_investigator_finding():
    # A saved verdict claiming the route without the bound finding fails closed.
    *_, decision = _decide(OWNERSHIP, LISTING, "public", validated=False)
    assert decision != "match"


def test_same_subsidiary_does_not_match_a_requested_venture_stage():
    finding, retained, decision = _decide(OWNERSHIP, LISTING, "series b")

    assert finding["status"] != "VERIFIED"
    assert not retained.get("stage_matches")
    assert decision != "match"


@pytest.mark.parametrize(("ownership", "listing"), [
    # Parent taken private and delisted.
    (OWNERSHIP, "Parent Group, Inc. was taken private by Sponsor Capital and "
                "its shares were delisted from the Nasdaq."),
    # An unrelated listed company on the same page.
    (OWNERSHIP, "Other Corp (NYSE: OTHR) is a holding company for insurance brands."),
    # Conditional acquisition.
    ("Acme Realty will become a wholly owned subsidiary of Parent Group, Inc. "
     "upon closing, subject to regulatory approval.", LISTING),
    # Minority stake.
    ("Parent Group, Inc. holds a minority stake in Acme Realty.", LISTING),
    # Former ownership.
    ("Acme Realty was formerly a wholly owned subsidiary of Parent Group, Inc.",
     LISTING),
])
def test_listed_parent_controls_stay_not_public(ownership, listing):
    finding, retained, decision = _decide(ownership, listing, "public")

    assert finding["status"] != "VERIFIED"
    assert not retained.get("stage_matches")
    assert decision != "match"


@pytest.mark.parametrize(("ownership", "listings", "expected"), [
    ("Parent Group, Inc. (Nasdaq: PGRP), the parent company of Acme Realty, "
     "reported results.", (), True),
    ("Parent Group, Inc. (NYSE: PGRP) completed its acquisition of Acme Realty.",
     (), True),
    ("Acme Realty, a subsidiary of Parent Group, Inc. (Nasdaq: PGRP), serves "
     "home buyers.", (), True),
    # The ticker belongs to a different parent than the one that owns Acme.
    ("Acme Realty is a subsidiary of Parent Group, Inc.",
     ("Other Holdings, Inc. (Nasdaq: OTHR) reported results.",), False),
    # The investigated company cannot be its own listed parent.
    ("Acme Realty, a subsidiary of Acme Realty (Nasdaq: ACRE).", (), False),
    ("Acme Realty is expected to become a subsidiary of Parent Group, Inc. "
     "(Nasdaq: PGRP).", (), False),
    ("Acme Realty is a subsidiary of Parent Group, Inc. (Nasdaq: PGRP), "
     "pending regulatory approval.", (), False),
])
def test_listed_parent_proof_binds_parent_relation_and_listing(
    ownership, listings, expected,
):
    assert lead_scorer._listed_parent_public_stage_matches(
        "acquired", "public", company_names=("acmerealty",),
        ownership_quote=ownership, listing_quotes=listings,
    ) is expected
    # Never for any other requested stage.
    assert not lead_scorer._listed_parent_public_stage_matches(
        "acquired", "series c+", company_names=("acmerealty",),
        ownership_quote=ownership, listing_quotes=listings,
    )


# Rule C ------------------------------------------------------------------

def test_descriptive_scope_lists_and_delivery_words_keep_required_functions():
    rules = " ".join(industry.CRITERION_RULES.split())

    assert (
        "An ordinary descriptive list of lifecycle stages, disease areas or "
        "indications, markets, regions or use cases states the offering's scope: "
        "proof of an offering within that scope establishes fit without proof of "
        "each listed item, unless must, each, all, both, only or explicit logical "
        "AND requires every item."
    ) in rules
    assert "An offering wholly outside that scope does not fit" in rules
    assert "an antibody company does not supply 'genetic medicines'" in rules
    assert "an oncology-only cell therapy does not serve 'genetic and rare diseases'" in rules
    assert (
        "Read delivery or form words such as cloud, online, digital, web-based, "
        "mobile-first and app-based by the same ordinary meaning"
    ) in rules
    assert "it does not supply a missing function or capability" in rules
    # Mandatory wording and listed functions keep precedence.
    assert "Explicit logical AND and mandatory wording take precedence over descriptive-list guidance" in rules
    assert "'must serve hospitals and clinics' requires both customer types" in rules
    assert "Each listed function or capability of a required product or service remains required" in rules
    # The rules reach both the reviewer and the investigator source selection.
    assert industry.CRITERION_RULES in industry.SYSTEM
    assert rules.index("Explicit logical AND and mandatory wording") < rules.index(
        "An ordinary descriptive list of lifecycle stages"
    )
