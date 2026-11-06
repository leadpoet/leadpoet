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
    # Review fix 5: a later strategic equity investment keeps Series C+
    # (for example a corporate investment after a Series C) but leaves Seed,
    # Series A or Series B unresolved, because most rounds are minority equity.
    assert (
        "A later completed equity investment or round of unstated series, even "
        "one called strategic, keeps Series C+ unless stated to give control or "
        "a listing, but leaves an earlier stage UNPROVEN. A pending or "
        "conditional deal does not erase a proven stage."
    ) in INVESTIGATOR_PROMPT
    assert "minority investments" not in INVESTIGATOR_PROMPT
    assert "unspecified investment" not in INVESTIGATOR_PROMPT
    # Review fix 6: failed or unfinished research is not a stage-keeping
    # reason; the latest completed event rule covers Seed again.
    assert "unfinished research" not in INVESTIGATOR_PROMPT
    assert (
        "For venture stages, compare dated rounds, acquisitions and IPO/listing "
        "events; use the latest completed event, Seed included."
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
    assert (
        "A proven stage stays proven unless a later completed event concretely "
        "changed control, listing, or the priced-round series."
    ) in LEAD_SCORER_PROMPTS
    # An unnamed later round or strategic investment keeps only Series C+.
    assert LEAD_SCORER_PROMPTS.count(
        "A later completed equity investment or round of unstated series, even "
        "one called strategic, keeps Series C+ but leaves an earlier stage "
        "unresolved. A pending or conditional deal does not erase a proven stage."
    ) >= 2  # full-fit and stage-repair prompts (f-string parts are also walked)
    assert "For a venture stage, including Seed, use the latest completed event" in LEAD_SCORER_PROMPTS
    assert "unspecified investment" not in LEAD_SCORER_PROMPTS
    assert "minority investment" not in LEAD_SCORER_PROMPTS
    assert "unfinished research" not in LEAD_SCORER_PROMPTS


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
    # Review fix 2: a completed acquisition is past, not current, ownership.
    ("Parent Group, Inc. (NYSE: PGRP) completed its acquisition of Acme Realty.",
     (), False),
    ("Acme Realty was acquired by Parent Group, Inc. (NYSE: PGRP).", (), False),
    ("Acme Realty is wholly owned by Parent Group, Inc. (NYSE: PGRP).", (), True),
    ("Acme Realty is an operating company of Parent Group, Inc. (NYSE: PGRP).",
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


# Review regressions: each reproduces a false Public found end to end ------

BARCO_LISTING = "Barco Industries (NYSE: BRC) reported results."


@pytest.mark.parametrize(("ownership", "listing"), [
    # Fix 1: the listed issuer must be exactly the parent, not a name that
    # merely contains it.
    ("Acme Realty is a wholly owned subsidiary of Mars.",
     "Marstons PLC (NYSE: MARS) reported results."),
    ("Acme Realty is a wholly owned subsidiary of Beta.",
     "AlphaBeta Corp (NASDAQ: AB) reported results."),
    ("Acme Realty is a wholly owned subsidiary of Samsung.",
     "Samsung Electronics (NYSE: SSNLF) reported results."),
    # Fix 2: a past acquisition followed by a sale, spin-off or buyback.
    ("Acme Realty was acquired by Barco Industries (NYSE: BRC) in 2010; Barco "
     "Industries divested Acme Realty in 2022.", BARCO_LISTING),
    ("Acme Realty was acquired by Barco Industries (NYSE: BRC) in 2010 and is "
     "now an independent company.", BARCO_LISTING),
    ("Acme Realty is a subsidiary of Barco Industries. In 2024 Barco Industries "
     "sold Acme Realty to Qux Corp.", BARCO_LISTING),
    ("Acme Realty is a subsidiary of Barco Industries until 2025.", BARCO_LISTING),
    ("Acme Realty is a subsidiary of Barco Industries; in 2020 management "
     "bought it back.", BARCO_LISTING),
    # Fix 3: joint ownership is not a single listed parent.
    ("Acme Realty is a subsidiary of Barco Industries and Qux Corp.",
     BARCO_LISTING),
    ("Acme Realty is a joint venture; Acme Realty is a subsidiary of Barco "
     "Industries.", BARCO_LISTING),
    # Fix 4: a stated minority percentage or stake.
    ("Acme Realty is 49% owned by Barco Industries. Acme Realty is an "
     "operating company of Barco Industries (NYSE: BRC).", BARCO_LISTING),
    ("Barco Industries holds a 40 percent stake. Acme Realty is a subsidiary "
     "of Barco Industries.", BARCO_LISTING),
])
def test_review_false_public_repros_are_not_public(ownership, listing):
    assert not lead_scorer._listed_parent_public_stage_matches(
        "acquired", "public", company_names=("acmerealty",),
        ownership_quote=ownership, listing_quotes=(listing,),
    )
    finding, retained, decision = _decide(ownership, listing, "public")
    assert finding["status"] != "VERIFIED"
    assert not retained.get("stage_matches")
    assert decision != "match"


@pytest.mark.parametrize(("ownership", "listing"), [
    ("Acme Realty is a wholly owned subsidiary of Barco Industries.",
     BARCO_LISTING),
    # A trailing legal suffix is not part of the name; ticker case is ignored.
    ("Acme Realty is a wholly owned subsidiary of Barco Industries, Inc.",
     "Barco Industries (NYSE: BRC) reported results."),
    ("Acme Realty is a wholly owned subsidiary of Barco Industries.",
     "Shares of Barco Industries Inc. (NYSE: BRC) rose."),
    # A majority percentage is not a minority stake.
    ("Acme Realty is 80% owned by Barco Industries. Acme Realty is a "
     "majority-owned subsidiary of Barco Industries.", BARCO_LISTING),
])
def test_exact_listed_parent_still_matches_requested_public(ownership, listing):
    assert lead_scorer._listed_parent_public_stage_matches(
        "acquired", "public", company_names=("acmerealty",),
        ownership_quote=ownership, listing_quotes=(listing,),
    )
    *_, decision = _decide(ownership, listing, "public")
    assert decision == "match"


@pytest.mark.parametrize("listing", [
    "Barco Industries Holdings (NYSE: BRC) reported results.",
    "Tata Barco Industries (NYSE: BRC) reported results.",
    "Barcos Industries (NYSE: BRC) reported results.",
])
def test_listing_issuer_must_equal_the_parent_name(listing):
    assert not lead_scorer._listed_parent_public_stage_matches(
        "acquired", "public", company_names=("acmerealty",),
        ownership_quote="Acme Realty is a subsidiary of Barco Industries.",
        listing_quotes=(listing,),
    )


def test_direct_public_substring_behavior_is_unchanged():
    # Fix 1 is scoped to the listed-parent route only.
    assert lead_scorer._public_quote_has_bound_market_locator(
        "Marstons PLC (NYSE: MARS) reported results.", ("mars",),
    )


# Rule C ------------------------------------------------------------------

def test_descriptive_scope_lists_and_delivery_words_keep_required_functions():
    rules = " ".join(industry.CRITERION_RULES.split())

    assert (
        "An ordinary descriptive list of lifecycle stages, disease areas or "
        "indications, markets or regions states the offering's scope: "
        "proof of an offering within that scope establishes fit without proof of "
        "each listed item, unless must, each, all, both, only or explicit logical "
        "AND requires every item."
    ) in rules
    # Review fix 7: use cases are not a descriptive scope list, because they
    # can name required functions ("onboarding, payroll and benefits").
    assert "use cases" not in rules
    assert "An offering wholly outside that scope does not fit" in rules
    assert "an antibody company does not supply 'genetic medicines'" in rules
    assert "an oncology-only cell therapy does not serve 'genetic and rare diseases'" in rules
    assert (
        "Read delivery or form words such as cloud, online, digital, web-based, "
        "mobile-first and app-based by the same ordinary meaning"
    ) in rules
    assert "it does not supply a missing function or capability" in rules
    # Web delivery is not mobile-first; each form word keeps its own meaning.
    assert (
        "an offering customers reach over the internet, in a browser or by API "
        "is cloud, online, digital or web-based; one centered on a mobile app "
        "is mobile-first or app-based."
    ) in rules
    # Mandatory wording and listed functions keep precedence.
    assert "Explicit logical AND and mandatory wording take precedence over descriptive-list guidance" in rules
    assert "'must serve hospitals and clinics' requires both customer types" in rules
    assert "Each listed function or capability of a required product or service remains required" in rules
    # The rules reach both the reviewer and the investigator source selection.
    assert industry.CRITERION_RULES in industry.SYSTEM
    assert rules.index("Explicit logical AND and mandatory wording") < rules.index(
        "An ordinary descriptive list of lifecycle stages"
    )
