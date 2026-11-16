"""A listed buyer does not become an acquired company by buying another firm."""
import pytest

from qualification.scoring import lead_scorer as scorer
from tests.test_public_issuer_ticker_quotes import _validated_public


@pytest.mark.parametrize("company,domain,quote", [
    ("OneSpan", "onespan.com", "OneSpan Inc. (NASDAQ: OSPN) today announced that it has completed the acquisition of Build38"),
    ("OneSpan", "onespan.com", "BOSTON – March 2, 2026 – OneSpan Inc. (NASDAQ: OSPN) today announced that it has completed the acquisition of Build38"),
    ("Acme", "acme.example", "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company."),
    ("Acme", "acme.example", "Acme Corporation (NASDAQ: ACM) closes its acquisition of Other Company."),
])
def test_issuer_buying_another_company_keeps_its_listing(company, domain, quote):
    result = _validated_public(quote, company=company, domain=domain)
    assert result["status"] == "VERIFIED"
    assert result["evidence_quote"] == quote
    assert scorer._stage_evidence_supports_observation(
        "public", quote, evidence_url=f"https://{domain}/investors",
        evidence_final_url=f"https://{domain}/investors",
        first_party_domains=(domain,), identity_names=(company,),
    )


@pytest.mark.parametrize("quote", [
    "Acme Corporation (NASDAQ: ACM) was acquired by Other Company.",
    "Acme Corporation (NASDAQ: ACM) is now a wholly-owned subsidiary of Other Company.",
    "Other Company completed the acquisition of Acme Corporation (NASDAQ: ACM).",
    "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company. Acme later went private.",
    "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company. Acme was delisted.",
    "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company. Acme ceased trading.",
    "Other Corporation (NASDAQ: OTH) completed the acquisition of Acme.",
    "Other Corporation (NASDAQ: OTH) completed the acquisition of Third Company. Acme is its customer.",
    "Acme Corporation (NASDAQ: ACM) announced the completed acquisition of the Company.",
    "Acme Corporation (NASDAQ: ACM) announced the completed acquisition of its parent.",
    "Acme Corporation (NASDAQ: ACM) announced the completed acquisition of the Group.",
    "Acme Corporation (NASDAQ: ACM) announced the completed acquisition of Company.",
    'Acme Corporation (NASDAQ: ACM) announced the completed acquisition of Different Company ("Acme").',
    "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company and Acme.",
    "Acme Corporation (NASDAQ: ACM) completed the acquisition of Acme Corporation and Other Company.",
    "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company & Acme.",
    "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company, including Acme.",
    "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company or Acme.",
])
def test_target_parent_other_issuer_and_superseded_listing_do_not_pass(quote):
    assert _validated_public(quote)["status"] == "UNPROVEN"


def test_missing_identity_does_not_resolve_ambiguous_listing_and_acquisition():
    quote = "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company."
    assert not scorer._stage_quote_supports_observation("public", quote)


def test_exact_quote_and_source_presence_remain_mandatory():
    quote = "Acme Corporation (NASDAQ: ACM) completed the acquisition of Other Company."
    assert _validated_public(quote, text="The submitted quote is absent.")["status"] == "UNPROVEN"


def test_public_buyer_survives_saved_verdict_projection():
    import json
    from tests.test_company_evidence_investigator import _company

    quote = "OneSpan Inc. (NASDAQ: OSPN) today announced that it has completed the acquisition of Build38"
    finding = _validated_public(quote, company="OneSpan", domain="onespan.com")
    verdict = scorer._project_investigator_stage({}, finding, icp_stage="public")
    saved = json.loads(json.dumps(verdict))
    assert scorer._decision_from_observed_stage(
        saved, "public", company=_company(name="OneSpan", website="https://onespan.com"),
        evidence_attributed=True,
    ) == scorer.COMPANY_FIT_MATCH
