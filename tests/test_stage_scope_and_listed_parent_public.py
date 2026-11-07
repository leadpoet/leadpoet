"""Stage persistence, the requested-Public listed-parent route, and scope lists.

Rule A: a proven stage survives later non-material events.
Rule B: only for a requested Public stage, a current wholly or majority owned
subsidiary of a currently listed parent satisfies Public. The route needs both
company-bound ownership text and the parent's current listing.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path

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
REAL_CONTRADICTED_PARENT = json.loads(
    (Path(__file__).parent / 'fixtures/listed_parent_public_contradicted_40628.json').read_text()
)


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


def test_stage_observation_contract_is_consistent_at_system_and_tool_fields():
    assert "Use VERIFIED for a proved observation" in INVESTIGATOR_PROMPT
    assert "do not decide requested fit" in INVESTIGATOR_PROMPT
    assert "before comparing requested_stage" not in INVESTIGATOR_PROMPT
    assert "the server computes requested fit and normalizes status" in INVESTIGATOR_PROMPT
    assert "Except for stage's observed-fact contract" in INVESTIGATOR_PROMPT
    assert "Use VERIFIED for a match" not in INVESTIGATOR_PROMPT
    tool = next(value for value in investigator._tools(("stage",))
                if value["name"] == "submit_findings")
    properties = tool["parameters"]["properties"]["findings"]["items"]["properties"]
    assert "even when requested_stage differs" in properties["observed_value"]["description"]
    assert "VERIFIED proves the observed stage" in properties["status"]["description"]
    assert properties["status"]["enum"] == sorted(investigator.STATUSES)
    assert properties["observed_value"]["type"] == ["string", "integer", "null"]
    assert "current private-equity control is Private Equity" in properties["observed_value"]["description"]
    assert "currently listed company stays Public" in properties["observed_value"]["description"]


def test_listing_issuer_scope_does_not_change_required_discovery_company():
    assert (
        "For Public listing checks, issuer means the investigated company for direct "
        "proof or the exact proven parent for listed-parent proof."
    ) in INVESTIGATOR_PROMPT
    assert "Required discovery remains company-bound" in INVESTIGATOR_PROMPT
    assert "the exact company's shares are listed or traded" not in INVESTIGATOR_PROMPT
    assert "structured `Privately Held` evidence about the issuer" in INVESTIGATOR_PROMPT


# Rule B prompt contract ----------------------------------------------------

def test_listed_parent_route_is_stated_only_for_requested_public():
    assert (
        "For requested Public, report Public for direct company listing proof. For a "
        "current wholly or majority owned subsidiary, brand or division of a currently "
        "listed parent, report Acquired with company-bound ownership primary and the "
        "same exact parent's affirmative current listing as support 1."
    ) in INVESTIGATOR_PROMPT
    assert "the child may be private or unlisted" in INVESTIGATOR_PROMPT
    assert "A bare parent ticker is insufficient" in INVESTIGATOR_PROMPT
    assert "No other parent or subsidiary stage inheritance" in INVESTIGATOR_PROMPT
    public_start = INVESTIGATOR_PROMPT.index("For requested Public, report Public")
    parent_route = INVESTIGATOR_PROMPT.index("listed parent, report Acquired", public_start)
    direct_details = INVESTIGATOR_PROMPT.index("For direct proof,", public_start)
    rebrand = INVESTIGATOR_PROMPT.index("A merger, acquisition, ownership change,")
    assert public_start < parent_route < direct_details < rebrand
    assert INVESTIGATOR_PROMPT.count("listed parent, report Acquired") == 1
    assert "Never inherit a stage from a parent or subsidiary" not in INVESTIGATOR_PROMPT
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
    "Parent Group, Inc. (Nasdaq: PGRP) is publicly traded and is a holding company for residential "
    "brokerage brands."
)


def _decide(
    ownership, listing, requested, *, validated=True, listing2="", listing_url=LISTING_URL,
):
    pages = {
        OWNERSHIP_URL: "About us. " + ownership + " Contact us.",
        listing_url: "Stock information. " + listing,
    }
    raw = _finding(
        "stage", status="VERIFIED", observed_value="Acquired",
        evidence_url=OWNERSHIP_URL, evidence_quote=ownership,
        supporting_evidence_url_1=listing_url,
        supporting_evidence_quote_1=listing,
    )
    raw.pop("supporting_evidence")
    if listing2:
        pages[listing_url + "/status"] = listing2
        raw.update(supporting_evidence_url_2=listing_url + "/status",
                   supporting_evidence_quote_2=listing2)
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


@pytest.mark.parametrize("damage", ["ownership", "listing", "missing_support"])
def test_saved_parent_proof_cannot_change_after_validation(damage):
    finding, retained, _ = _decide(OWNERSHIP, LISTING, "public")
    if damage == "ownership":
        retained["stage_evidence_quote"] = "Acme Realty is independent."
    elif damage == "listing":
        retained["dimension_evidence"]["stage"]["supporting_evidence"][0]["quote"] = (
            "Other Corp (NYSE: OTHR) reported results."
        )
    else:
        retained["dimension_evidence"]["stage"].pop("supporting_evidence")
    company = _company(name="Acme Realty", website="https://acmerealty.example")
    assert lead_scorer._decision_from_observed_stage(
        retained, "public", validated_stage_finding=finding,
        company=company, evidence_attributed=True,
    ) != "match"


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
     "reported results. Parent Group is publicly traded.", (), True),
    # Review fix 2: a completed acquisition is past, not current, ownership.
    ("Parent Group, Inc. (NYSE: PGRP) completed its acquisition of Acme Realty.",
     (), False),
    ("Acme Realty was acquired by Parent Group, Inc. (NYSE: PGRP).", (), False),
    ("Acme Realty is wholly owned by Parent Group, Inc. (NYSE: PGRP). "
     "Parent Group is publicly traded.", (), True),
    ("Acme Realty is an operating company of Parent Group, Inc. (NYSE: PGRP).",
     ("Parent Group is publicly traded.",), True),
    ("Acme Realty, a subsidiary of Parent Group, Inc. (Nasdaq: PGRP), serves "
     "home buyers.", ("Parent Group is publicly traded.",), True),
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

BARCO_LISTING = "Barco Industries (NYSE: BRC) is publicly traded."


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
     "Barco Industries (NYSE: BRC) is publicly traded."),
    ("Acme Realty is a wholly owned subsidiary of Barco Industries.",
     "Barco Industries Inc. (NYSE: BRC) is publicly traded."),
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
    "Tata Barco Industries (NYSE: BRC) is publicly traded.",
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


@pytest.mark.parametrize("listing", [
    "Barco Industries (NYSE: BRC) is not listed; the ticker is hypothetical.",
    "Barco Industries (NYSE: BRC) is not a publicly listed company.",
    "Barco Industries (NYSE: BRC) is not listed; Other Corp is publicly traded.",
    "Barco Industries (NYSE: BRC) is planning to list.",
    "Barco Industries (NYSE: BRC) is a hypothetical listing.",
    "Barco Industries (NYSE: BRC) completed its IPO in 2000.",
    "Zqlistedparentissuerzq (NYSE: ZZZ) is unrelated to Barco Industries.",
])
def test_parent_listing_must_be_affirmed_current_fact(listing):
    finding, retained, decision = _decide(
        "Acme Realty is a subsidiary of Barco Industries.", listing, "public",
    )
    assert finding["status"] != "VERIFIED"
    assert not retained.get("stage_matches")
    assert decision != "match"


@pytest.mark.parametrize("separate_quote", [False, True])
@pytest.mark.parametrize("denial", [
    "Samsung is not listed.",
    "Samsung is not publicly traded.",
    "Samsung (NYSE: SSNLF) is not a listed company.",
])
def test_exact_parent_denial_survives_sentence_and_citation_boundaries(separate_quote, denial):
    listing = "Samsung (NYSE: SSNLF) is an example."
    *_, decision = _decide(
        "Acme Realty is a wholly owned subsidiary of Samsung.",
        listing if separate_quote else denial + " " + listing,
        "public", listing2=denial if separate_quote else "",
    )
    assert decision != "match"


def test_another_issuers_nonpublic_status_does_not_erase_the_listed_parent():
    *_, decision = _decide(
        "Acme Realty is a wholly owned subsidiary of Samsung.",
        "Samsung (NYSE: SSNLF) is publicly traded.", "public",
        listing2="Other Corp is not listed.",
    )
    assert decision == "match"


@pytest.mark.parametrize("percentage", ["50%", "50.0 percent", "49%", "101%"])
def test_equal_or_minority_ownership_cannot_establish_majority(percentage):
    ownership = (
        "Acme Realty is a subsidiary of Barco Industries. "
        f"Barco Industries owns {percentage} of Acme Realty."
    )
    *_, decision = _decide(ownership, BARCO_LISTING, "public")
    assert decision != "match"


@pytest.mark.parametrize("percentage", ["50.1%", "80 percent", "100%"])
def test_stated_majority_preserves_valid_parent_route(percentage):
    *_, decision = _decide(
        "Acme Realty is a subsidiary of Barco Industries. "
        f"Barco Industries owns {percentage} of Acme Realty.",
        BARCO_LISTING, "public",
    )
    assert decision == "match"


def test_majority_stake_wording_preserves_valid_parent_route():
    *_, decision = _decide(
        "Acme Realty is a majority-owned subsidiary of Barco Industries. "
        "Barco Industries holds an 80 percent stake in Acme Realty.",
        BARCO_LISTING, "public",
    )
    assert decision == "match"


@pytest.mark.parametrize("sales", [
    "Barco Industries (NYSE: BRC) is publicly traded and reported profits from products sold to customers.",
    "Barco Industries (NYSE: BRC) is publicly traded and sells software to companies.",
])
def test_unrelated_product_sales_do_not_erase_parent_listing(sales):
    *_, decision = _decide(
        "Acme Realty is a wholly owned subsidiary of Barco Industries.",
        sales, "public",
    )
    assert decision == "match"


@pytest.mark.parametrize("search_ok", [True, False])
@pytest.mark.parametrize("submitted_status", ["VERIFIED", "CONTRADICTED"])
def test_native_listed_parent_route_keeps_current_search_and_private_receipt(monkeypatch, search_ok, submitted_status):
    raw = _finding(
        "stage", status=submitted_status, observed_value="Acquired", evidence_url=OWNERSHIP_URL,
        evidence_quote=OWNERSHIP, supporting_evidence_url_1=LISTING_URL,
        supporting_evidence_quote_1=LISTING,
    )
    raw.pop("supporting_evidence")
    searches = []
    turns = []

    async def search(_session, query, *, key):
        searches.append(query)
        if not search_ok:
            raise RuntimeError("offline discovery failure")
        return {"results": []}

    async def submit(_session, _url, *, headers, payload):
        emitted_system = " ".join(" ".join(
            message["content"] for message in payload["messages"]
            if message["role"] == "system"
        ).split())
        assert "For requested Public, report Public" in emitted_system
        assert "Use VERIFIED for a proved observation" in emitted_system
        assert "listed parent, report Acquired with company-bound ownership primary" in emitted_system
        assert "the child may be private or unlisted" in emitted_system
        turns.append(payload)
        if len(turns) <= 2:
            name, arguments = "fetch_page", {"url": (
                OWNERSHIP_URL if len(turns) == 1 else LISTING_URL
            )}
        else:
            name, arguments = "submit_findings", {"findings": [raw]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            },
        }]}}]}

    async def fetch(_session, url):
        return {"ok": True, "url": url, "final_url": url,
                "text": OWNERSHIP if url == OWNERSHIP_URL else LISTING}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_post_json", submit)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme Realty", "website": "https://acmerealty.example"},
        targets=("stage",), requested_stage="Public",
    ))
    assert len(searches) == 1
    finding = result["claims"]["stage"]
    assert finding["status"] == ("VERIFIED" if search_ok else "UNPROVEN")
    if search_ok:
        assert finding["supporting_evidence"] == [{"url": LISTING_URL, "quote": LISTING}]
        assert result["_validated_stage_finding"] == finding
        raw["private_provider_metadata"] = "must-not-enter-stage-receipt"
        validated = investigator._validated_findings(
            {"findings": [raw]}, targets=("stage",),
            fetched_pages={OWNERSHIP_URL: OWNERSHIP, LISTING_URL: LISTING},
            fetched_final_urls={OWNERSHIP_URL: OWNERSHIP_URL, LISTING_URL: LISTING_URL},
            first_party_domains={"acmerealty.example"}, identity_names={"acmerealty"},
            requested_stage="Public",
        )["stage"]
        assert validated["status"] == "VERIFIED"
        assert "must-not-enter-stage-receipt" not in json.dumps(validated)
    else:
        assert result["_completed_submit"] is False


@pytest.mark.parametrize('quantity', [
    'one third', 'a third', 'two fifths', '50-50', '50/50',
    'less than 60%', 'up to 80%', 'at most 80%',
    'not more than 80%', 'no more than 80%', '<=80%', '≤80%',
])
def test_ambiguous_quantitative_ownership_does_not_prove_majority(quantity):
    *_, decision = _decide(
        'Acme Realty is a subsidiary of Barco Industries. '
        f'Barco Industries owns {quantity} of Acme Realty.',
        BARCO_LISTING, 'public',
    )
    assert decision != 'match'


@pytest.mark.parametrize('quantity', ['more than 50%', 'over 50%', 'at least 51%', '>50%', '>=51%'])
def test_clear_lower_bounded_majority_preserves_parent_route(quantity):
    *_, decision = _decide(
        'Acme Realty is a subsidiary of Barco Industries. '
        f'Barco Industries owns {quantity} of Acme Realty.',
        BARCO_LISTING, 'public',
    )
    assert decision == 'match'


@pytest.mark.parametrize('unrelated_fact', [
    'Its board includes independent directors.',
    'Barco Industries serves independent home buyers.',
    'Barco Industries owns 40% of Unrelated Corp.',
    'Barco Industries owns a minority stake in Unrelated Corp.',
    'Its former CEO Jane Doe attended.',
    'Its shares rose 20% this quarter.',
])
def test_unrelated_parent_facts_do_not_change_target_ownership(unrelated_fact):
    *_, decision = _decide(OWNERSHIP.replace('Parent Group, Inc.', 'Barco Industries'),
                          BARCO_LISTING + ' ' + unrelated_fact, 'public')
    assert decision == 'match'


@pytest.mark.parametrize('listing', [
    'Barco Industries (NYSE: BRC) reported results.',
    'Barco Industries (NYSE: BRC) was publicly traded in 2000.',
    'Barco Industries (NYSE: BRC) is a private company.',
    'Barco Industries (NYSE: BRC) is privately held.',
])
def test_ticker_alone_cannot_establish_parent_current_listing(listing):
    *_, decision = _decide('Acme Realty is a wholly owned subsidiary of Barco Industries.',
                          listing, 'public')
    assert decision != 'match'


@pytest.mark.parametrize('denial', ['Barco Industries is a private company.',
                                  'Barco Industries is privately held.'])
def test_private_parent_contradiction_is_kept_across_citations(denial):
    *_, decision = _decide('Acme Realty is a subsidiary of Barco Industries.',
                          BARCO_LISTING, 'public', listing2=denial)
    assert decision != 'match'


@pytest.mark.parametrize('other_owner', ['qux Corp', 'its partner Qux Corp'])
def test_joint_ownership_is_independent_of_casing_and_determiners(other_owner):
    *_, decision = _decide('Acme Realty is a subsidiary of Barco Industries and '
                          + other_owner + '.', BARCO_LISTING, 'public')
    assert decision != 'match'


@pytest.mark.parametrize('url, expected', [
    ('https://www.nyse.com/quote/XNYS:BRC', True),
    ('https://unrelated.example/quote/XNYS:BRC', False),
    ('https://www.sec.gov/Archives/edgar/data/123/report.htm', False),
])
def test_live_parent_exchange_card_requires_exchange_source(url, expected):
    assert lead_scorer._listed_parent_public_stage_matches(
        'acquired', 'public', company_names=('Acme Realty',),
        ownership_quote='Acme Realty is a subsidiary of Barco Industries.',
        listing_quotes=('Barco Industries Inc. New York Stock Exchange: BRC Last 10.00',),
        listing_urls=(url,),
    ) is expected


def test_archived_sec_present_tense_parent_listing_is_not_current_authority():
    assert not lead_scorer._listed_parent_public_stage_matches(
        'acquired', 'public', company_names=('Acme Realty',),
        ownership_quote='Acme Realty is a subsidiary of Barco Industries.',
        listing_quotes=(BARCO_LISTING,),
        listing_urls=('https://www.sec.gov/Archives/edgar/data/123/report.htm',),
    )


@pytest.mark.parametrize('event', ['was taken private by Barco Industries in 2024.',
                                 'was delisted from Nasdaq in 2024.'])
def test_child_former_listing_does_not_disprove_current_parent_listing(event):
    *_, decision = _decide('Acme Realty is a subsidiary of Barco Industries. '
                          'Acme Realty ' + event, BARCO_LISTING, 'public')
    assert decision == 'match'


def test_ownership_source_retains_same_parent_listing_denial_after_sentence():
    *_, decision = _decide('Acme Realty is a wholly owned subsidiary of Barco Industries. '
                          'Barco Industries is not listed.', BARCO_LISTING, 'public')
    assert decision != 'match'


def test_fictional_example_does_not_prove_parent_listing():
    *_, decision = _decide('Acme Realty is a wholly owned subsidiary of Barco Industries.',
                          'Barco Industries (NYSE: BRC) is publicly traded '
                          'in this fictional example.', 'public')
    assert decision != 'match'


@pytest.mark.parametrize('final_url, expected', [
    ('https://www.nyse.com/quote/XNYS:BRC', 'VERIFIED'),
    ('https://unrelated.example/stock', 'CONTRADICTED'),
])
def test_native_parent_exchange_card_requires_final_exchange_authority(final_url, expected):
    url = 'https://www.nyse.com/quote/XNYS:BRC'
    quote = 'Barco Industries Inc. New York Stock Exchange: BRC Last 10.00'
    raw = _finding('stage', observed_value='Acquired', evidence_url=OWNERSHIP_URL,
                   evidence_quote='Acme Realty is a wholly owned subsidiary of Barco Industries.',
                   supporting_evidence_url_1=url, supporting_evidence_quote_1=quote)
    raw.pop('supporting_evidence')
    finding = investigator._validated_findings(
        {'findings': [raw]}, targets=('stage',),
        fetched_pages={OWNERSHIP_URL: raw['evidence_quote'], url: quote},
        fetched_final_urls={OWNERSHIP_URL: OWNERSHIP_URL, url: final_url},
        first_party_domains={'acmerealty.example'}, identity_names={'acmerealty'},
        requested_stage='Public',
    )['stage']
    assert finding['status'] == expected


@pytest.mark.parametrize('denial', ['Barco Industries is not listed.',
                                  'Barco Industries is a private company.'])
def test_live_exchange_card_does_not_override_explicit_same_parent_contradiction(denial):
    assert not lead_scorer._listed_parent_public_stage_matches(
        'acquired', 'public', company_names=('Acme Realty',),
        ownership_quote='Acme Realty is a wholly owned subsidiary of Barco Industries. ' + denial,
        listing_quotes=('Barco Industries Inc. New York Stock Exchange: BRC Last 10.00',),
        listing_urls=('https://www.nyse.com/quote/XNYS:BRC',),
    )


@pytest.mark.parametrize('observed,requested,quote', [
    ('Series A', 'Series B', 'Acme Realty completed a $20 million Series A financing round.'),
    ('Acquired', 'Series C+', 'Acme Realty is a wholly owned subsidiary of Barco Industries.'),
    ('Public', 'Private Equity', 'Acme Realty (NYSE: ACRE) is currently publicly traded.'),
])
def test_proved_stage_fact_normalizes_to_a_different_requested_stage(observed, requested, quote):
    raw = _finding('stage', status='VERIFIED', observed_value=observed,
                   evidence_url=OWNERSHIP_URL, evidence_quote=quote)
    finding = investigator._validated_findings(
        {'findings': [raw]}, targets=('stage',), fetched_pages={OWNERSHIP_URL: quote},
        fetched_final_urls={OWNERSHIP_URL: OWNERSHIP_URL},
        first_party_domains={'acmerealty.example'}, identity_names={'acmerealty'},
        requested_stage=requested,
    )['stage']
    assert finding['status'] == 'CONTRADICTED'
    assert finding['observed_value'] == observed
    requested_value = lead_scorer._normalize_company_stage(requested)
    company = _company(name='Acme Realty', website='https://acmerealty.example')
    projected = lead_scorer._project_investigator_stage(
        {'observed_company_name': company.company_name}, finding, icp_stage=requested_value)
    assert lead_scorer._decision_from_observed_stage(
        json.loads(json.dumps(projected)), requested_value,
        validated_stage_finding=finding, company=company, evidence_attributed=True,
    ) == 'mismatch'


@pytest.mark.parametrize('status', ['UNPROVEN', 'CONTRADICTED'])
def test_empty_stage_claim_cannot_be_promoted_by_fact_normalization(status):
    raw = _finding('stage', status=status, observed_value='Acquired',
                   evidence_url='', evidence_quote='')
    finding = investigator._validated_findings(
        {'findings': [raw]}, targets=('stage',), fetched_pages={},
        first_party_domains={'acmerealty.example'}, identity_names={'acmerealty'},
        requested_stage='Public',
    )['stage']
    assert finding['status'] == 'UNPROVEN'
    projected = lead_scorer._project_investigator_stage({}, finding, icp_stage='public')
    assert lead_scorer._decision_from_observed_stage(
        json.loads(json.dumps(projected)), 'public', validated_stage_finding=finding,
        company=_company(name='Acme Realty', website='https://acmerealty.example'),
    ) == 'unavailable'


def test_child_private_status_does_not_replace_proved_acquired_observation():
    finding, retained, decision = _decide(
        OWNERSHIP + ' Acme Realty is privately held and has no listed shares.',
        LISTING, 'public',
    )
    assert finding['status'] == 'VERIFIED'
    assert retained['observed_company_stage'] == 'acquired'
    assert decision == 'match'


def _validate_recorded_parent(finding=None, pages=None):
    import copy
    fixture = REAL_CONTRADICTED_PARENT
    finding = copy.deepcopy(fixture['finding'] if finding is None else finding)
    source_pages = fixture['native_kwargs']['prefetched_pages']
    pages = ({url: value['text'] for url, value in source_pages.items()}
             if pages is None else pages)
    return investigator._validated_findings(
        {'findings': [finding]}, targets=('stage',), fetched_pages=pages,
        fetched_final_urls={url: url for url in pages},
        first_party_domains={'acme-realty.example'}, identity_names={'acmerealty'},
        requested_stage='Public',
    )['stage']


def test_real_contradicted_parent_facts_normalize_only_after_strict_proof():
    finding = _validate_recorded_parent()
    raw = REAL_CONTRADICTED_PARENT['finding']
    assert raw['status'] == 'CONTRADICTED'
    assert finding['status'] == 'VERIFIED'
    assert finding['observed_value'] == 'Acquired'
    assert finding['evidence_quote'] == raw['evidence_quote']
    assert finding['supporting_evidence'] == [{'url': raw['supporting_evidence_url_1'],
                                             'quote': raw['supporting_evidence_quote_1']}]
    company = _company(name='Acme Realty', website='https://acme-realty.example')
    projected = lead_scorer._project_investigator_stage(
        {'observed_company_name': company.company_name}, finding, icp_stage='public')
    assert lead_scorer._decision_from_observed_stage(
        json.loads(json.dumps(projected)), 'public', validated_stage_finding=finding,
        company=company, evidence_attributed=True,
    ) == 'match'


@pytest.mark.parametrize('damage', ['missing_support', 'wrong_issuer', 'historical', 'private_parent',
                                  'minority', 'source_absent', 'support_source_absent',
                                  'inverse_quote', 'unproven'])
def test_contradicted_parent_label_cannot_promote_unproved_facts(damage):
    import copy
    raw = copy.deepcopy(REAL_CONTRADICTED_PARENT['finding'])
    pages = {url: value['text'] for url, value in
             REAL_CONTRADICTED_PARENT['native_kwargs']['prefetched_pages'].items()}
    if damage == 'missing_support':
        raw['supporting_evidence_url_1'] = raw['supporting_evidence_quote_1'] = ''
    elif damage in {'wrong_issuer', 'historical', 'private_parent'}:
        quote = {'wrong_issuer': 'Other Corp is currently publicly traded.',
                 'historical': 'Barco Industries (NYSE: BRC) was publicly traded in 2020.',
                 'private_parent': 'Barco Industries is currently a private company.'}[damage]
        raw['supporting_evidence_quote_1'] = quote
        pages[raw['supporting_evidence_url_1']] = quote
    elif damage == 'minority':
        raw['evidence_quote'] = ('Acme Realty is a subsidiary of Barco Industries. '
                                 'Barco Industries owns 40% of Acme Realty.')
        pages[raw['evidence_url']] = raw['evidence_quote']
    elif damage == 'source_absent':
        pages[raw['evidence_url']] = 'The submitted ownership quote is absent.'
    elif damage == 'support_source_absent':
        pages.pop(raw['supporting_evidence_url_1'])
    elif damage == 'inverse_quote':
        raw['evidence_quote'] = 'Barco Industries currently owns 100% of Acme Realty.'
    else:
        raw.update(status='UNPROVEN', observed_value=None, evidence_url='', evidence_quote='')
    finding = _validate_recorded_parent(raw, pages)
    assert finding['status'] != 'VERIFIED'
    projected = lead_scorer._project_investigator_stage({}, finding, icp_stage='public')
    assert lead_scorer._decision_from_observed_stage(
        json.loads(json.dumps(projected)), 'public', validated_stage_finding=finding,
        company=_company(name='Acme Realty', website='https://acme-realty.example'),
    ) != 'match'


@pytest.mark.parametrize('observed,requested,quote', [
    ('Series A', 'Series A', 'Acme Realty has not completed a Series A financing round.'),
    ('Public', 'Public', 'Acme Realty (NYSE: ACRE) is currently publicly traded.'),
])
def test_matching_contradicted_direct_or_venture_label_keeps_existing_veto(observed, requested, quote):
    finding = investigator._validated_findings(
        {'findings': [_finding('stage', status='CONTRADICTED', observed_value=observed,
                              evidence_url=OWNERSHIP_URL, evidence_quote=quote)]},
        targets=('stage',), fetched_pages={OWNERSHIP_URL: quote},
        fetched_final_urls={OWNERSHIP_URL: OWNERSHIP_URL},
        first_party_domains={'acmerealty.example'}, identity_names={'acmerealty'},
        requested_stage=requested,
    )['stage']
    assert finding['status'] == 'UNPROVEN'
    assert finding['evidence_url'] == finding['evidence_quote'] == ''


# November 7 independent-review regressions. All assertions use native source
# validation followed by JSON saved projection and the final fit decision.
@pytest.mark.parametrize("suffix", [
    ", if the transaction completes.",
    ", if the agreement closes.",
    ", conditional on regulatory approval.",
    ", but this relationship is unconfirmed.",
    ", but that relationship has not been confirmed.",
    ", but this relationship may be inaccurate.",
    ", may be inaccurate.",
    ", has not been confirmed.",
    ", according to unconfirmed reports.",
    ", though this relationship is unconfirmed.",
    ", although this relationship has not been confirmed.",
    ", but it may be inaccurate.",
    ", only if the agreement closes.",
    ", contingent on regulatory approval.",
])
def test_review_ownership_suffix_is_not_current_proof(suffix):
    *_, decision = _decide(
        "Acme Realty is wholly owned by Barco Industries" + suffix,
        BARCO_LISTING, "public",
    )
    assert decision != "match"


@pytest.mark.parametrize("denial", [
    "Barco Industries is now privately held.",
    "Barco Industries has ceased trading.",
    "Barco Industries has never been publicly traded.",
    "Barco Industries has never been publicly listed.",
    "Barco Industries no longer trades on Nasdaq.",
    "Barco Industries's common shares are no longer listed.",
])
def test_review_same_parent_denial_cannot_be_overridden(denial):
    *_, decision = _decide(
        "Acme Realty is wholly owned by Barco Industries.",
        BARCO_LISTING, "public", listing2=denial,
    )
    assert decision != "match"


@pytest.mark.parametrize("quantity", [
    "one percent", "fifty percent", "forty-nine percent",
    "at least fifty percent", "less than sixty percent",
    "one hundred and one percent", "one hundred sixty percent",
    "fifty point zero percent", "one hundred point one percent",
    "fifty point ten percent", "fifty point point one percent",
    "fifty point percent", "-80%", "1500%", "1,080%",
    "minus sixty percent", "negative eighty percent", "−80%", "1,099%",
    "60% or less", "sixty percent at most", "between 40 and 60 percent",
    "40 to 60 percent", "forty to sixty percent",
])
def test_review_written_or_invalid_quantity_cannot_hide_minority(quantity):
    *_, decision = _decide(
        "Acme Realty is a subsidiary of Barco Industries. "
        f"Barco Industries owns {quantity} of Acme Realty.",
        BARCO_LISTING, "public",
    )
    assert decision != "match"


@pytest.mark.parametrize("quantity", [
    "fifty-one percent", "sixty percent", "one hundred percent",
    "more than fifty percent", "at least fifty-one percent", "80%",
    "fifty point one percent", "ninety-nine point nine percent",
])
def test_review_clear_written_majority_still_matches(quantity):
    *_, decision = _decide(
        "Acme Realty is a subsidiary of Barco Industries. "
        f"Barco Industries owns {quantity} of Acme Realty.",
        BARCO_LISTING, "public",
    )
    assert decision == "match"


@pytest.mark.parametrize("unrelated", [
    ", while Unrelated Corp is not publicly listed.",
    ", but its earnings report has not been confirmed.",
    ". Unrelated Corp may be acquired.",
    " as of May 2026.",
    ". Barco Industries owns fifty percent of Unrelated Corp.",
])
def test_review_unrelated_uncertainty_and_quantities_preserve_parent(unrelated):
    *_, decision = _decide(
        "Acme Realty is wholly owned by Barco Industries" + unrelated,
        BARCO_LISTING, "public",
        listing2="Unrelated Corp is now privately held. Acme Realty is privately held.",
    )
    assert decision == "match"


@pytest.mark.parametrize("issuer, expected", [
    ("Midwest Advanced Research Systems Inc.", False),
    ("Zqlistedparentissuerzq", False),
    ("Mars Inc.", True),
])
def test_review_exchange_card_must_name_exact_parent(issuer, expected):
    *_, decision = _decide(
        "Acme Realty is wholly owned by Mars.",
        issuer + " New York Stock Exchange: MARS Last 10.00", "public",
        listing_url="https://www.nyse.com/quote/XNYS:MARS",
    )
    assert (decision == "match") is expected


@pytest.mark.parametrize("description", [
    "a private label manufacturer", "a private cloud provider",
    "a private markets investment manager",
])
def test_review_private_product_description_is_not_listing_denial(description):
    *_, decision = _decide(
        "Acme Realty is wholly owned by Barco Industries.",
        BARCO_LISTING, "public", listing2=f"Barco Industries is {description}.",
    )
    assert decision == "match"


def test_review_direct_exchange_initials_policy_is_unchanged():
    quote = "Midwest Advanced Research Systems Inc. New York Stock Exchange: MARS Last 10.00"
    assert lead_scorer._current_exchange_profile_names_issuer(quote, ("Mars",))
    assert not lead_scorer._current_exchange_profile_names_issuer(
        quote, ("Mars",), allow_initials=False,
    )
