"""Company-name prefixes must not create source attribution for another entity."""

import asyncio
import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer
from tests.test_company_evidence_investigator import (
    _company, _complete_verdict, _finding, _icp,
)


@pytest.mark.parametrize("quote,names", [
    ("ENGIE Austria sells electricity.", {"engieau"}),
    ("ENGIE Automotive sells electricity.", {"engieau"}),
    ("ENGIE Australia sells electricity.", {"engieau"}),
    ("SuperENGIE AU sells electricity.", {"engieau"}),
    ("ENGIE AUé sells electricity.", {"engieau"}),
    ("Companies in Canada sell AI software.", {"ada"}),
    ("AcmeX supplies software.", {"acme"}),
    ("AnotherAcme supplies software.", {"acme"}),
])
def test_name_fragment_in_another_word_does_not_bind_company(quote, names):
    assert not inv._quote_names_company(quote, names)


@pytest.mark.parametrize("quote,names", [
    ("ENGIE AU sells electricity.", {"engieau"}),
    ("ENGIE-AU sells electricity.", {"engieau"}),
    ("ENGIE AU’s products include electricity.", {"engieau"}),
    ("(Engie A.U.) supplies electricity.", {"engieau"}),
    ("ＥＮＧＩＥ ＡＵ supplies electricity.", {"engieau"}),
    ("AT&T supplies telecommunications services.", {"att"}),
    ("O’Reilly supplies automotive parts.", {"oreilly"}),
    ("ENGIE Australia supplies electricity.", {"engieau", "engieaustralia"}),
])
def test_complete_names_preserve_case_typography_and_admitted_aliases(quote, names):
    assert inv._quote_names_company(quote, names)


@pytest.mark.parametrize("named_company,expected", [
    ("ENGIE AU", "match"),
    ("ENGIE-AU", "match"),
    ("ENGIE Austria", "unavailable"),
    ("ENGIE Automotive", "unavailable"),
    ("Other Energy", "unavailable"),
    ("ENGIE", "unavailable"),
])
def test_native_finding_projection_grounding_and_fit_reject_wrong_entity(
    monkeypatch, named_company, expected,
):
    # The investigated identity is independently valid. The loaded third-party
    # quote must still name this company; a prefix cannot borrow that identity.
    company = _company(name="ENGIE AU", website="https://engie.com.au",
                       linkedin="https://www.linkedin.com/company/engie-au")
    icp = _icp(required_attribute="Sells electricity to customers.")
    identity = {"normalized_name": "ENGIE AU", "registrable_dns_domain": "engie.com.au",
                "linkedin_company_slug": "engie-au"}
    anchor = {
        **dict.fromkeys(("submitted_name", "observed_name", "verified_name"), "ENGIE AU"),
        **dict.fromkeys(("submitted_domain", "observed_domain", "verified_domain"), "engie.com.au"),
        **dict.fromkeys(("submitted_linkedin_slug", "observed_linkedin_slug", "verified_linkedin_slug"), "engie-au"),
    }
    url = "https://independent-news.example/company"
    quote = f"{named_company} sells electricity to residential customers."
    finding = _finding("required_attribute", evidence_url=url, evidence_quote=quote,
                       observed_value="Sells electricity")
    validated = inv._validated_findings(
        {"findings": [finding]}, targets=("required_attribute",),
        fetched_pages={url: quote}, fetched_final_urls={url: url},
        first_party_domains={"engie.com.au"}, identity_names={"engieau"}, identity_anchor=anchor,
    )["required_attribute"]
    assert validated["status"] == ("VERIFIED" if expected == "match" else "UNPROVEN")
    verdict = _complete_verdict(
        observed_company_name="ENGIE AU", observed_company_website="https://engie.com.au",
        observed_company_linkedin="https://www.linkedin.com/company/engie-au",
    )
    for field in ("employee_size_evidence_quote", "industry_evidence_quote", "geography_evidence_quote"):
        verdict[field] = verdict[field].replace("Acme", "ENGIE AU")
    projected = scorer._project_investigator_required_attribute(verdict, validated)

    async def no_network(*args, **kwargs):
        pytest.fail("Source-grounded replay must not make provider calls")

    monkeypatch.setattr(scorer, "_fetch_bounded_html", no_network)
    grounded, _ = asyncio.run(scorer._ground_required_attribute_evidence(
        projected, active_attribute=True,
        source_cache={url: {"status": "fetched", "final_url": url, "text": quote}},
    ))
    result = scorer._reverify_decision(
        grounded, icp.required_attribute, "", icp=icp, company=company,
        verified_homepage_identity=identity,
    )
    assert result.decision == expected
    assert result.details["identity_decision"] == "match"
    assert result.details["required_attribute_decision"] == expected


def test_historical_ada_source_names_stay_bound_without_matching_canada():
    capture = json.loads((Path(__file__).parent / "fixtures" /
                          "intent_details_partial_identity_industry.json").read_text())
    # This is an existing retained source, not a new semantic truth label.
    assert inv._quote_names_company(capture["industry_primary"]["quote"], {"ada"})
    assert not inv._quote_names_company("Canada develops and sells AI platforms.", {"ada"})


@pytest.mark.parametrize("support,expected", [
    ("ENGIE AU sells electricity to customers.", "VERIFIED"),
    ("ENGIE Automotive sells electricity to customers.", "UNPROVEN"),
])
def test_named_primary_does_not_authorize_wrong_entity_support(support, expected):
    url = "https://independent-news.example/company"
    primary = "ENGIE AU supplies electricity."
    finding = _finding(
        "required_attribute", evidence_url=url, evidence_quote=primary,
        supporting_evidence_url_1=url, supporting_evidence_quote_1=support,
    )
    result = inv._validated_findings(
        {"findings": [finding]}, targets=("required_attribute",),
        fetched_pages={url: primary + " " + support}, fetched_final_urls={url: url},
        first_party_domains={"engie.com.au"}, identity_names={"engieau"},
    )["required_attribute"]
    assert result["status"] == expected
