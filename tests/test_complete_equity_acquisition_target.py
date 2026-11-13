"""Complete equity acquisitions bind the company after a location descriptor."""
import json

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer

COMPANY = "AAA Cooper Transportation"
# Public issuer statement; no production run identifiers or retained packet.
QUOTE = ('On Monday, July 5, 2021, Knight-Swift Transportation Holdings Inc. '
         '(NYSE: KNX) ("Knight-Swift"), acquired 100% of Dothan, Alabama-based '
         'AAA Cooper Transportation and an affiliated entity ("AAA Cooper"). '
         'AAA Cooper is a leading less-than-truckload ("LTL") carrier')
URL = "https://parent.example/news/completed-company-acquisition"
SUPPORT_URL = "https://www.aaacooper.com/about-us"
SUPPORT = ('AAA Cooper Transportation (SCAC “AACT”), an independent subsidiary '
           'of Knight-Swift Transportation Holdings (NYSE: KNX), is an '
           'asset-based, multi-regional transportation solutions provider.')


def test_exact_complete_equity_quote_binds_target_not_buyer():
    assert lead_scorer._acquired_stage_quote_supports_names((COMPANY,), QUOTE)
    assert not lead_scorer._acquired_stage_quote_supports_names(
        ("Knight-Swift Transportation Holdings",), QUOTE,
    )
    # The generic guard still sees the buyer's listing. Only the separate
    # company-bound acquisition path can establish the target's stage.
    assert not lead_scorer._stage_quote_supports_observation("acquired", QUOTE)


@pytest.mark.parametrize("equity", [
    "100% of", "100 % of", "100% of the equity in",
    "100% of the outstanding shares of", "all equity in", "all stock of",
    "all shares of", "all outstanding shares of", "all of the equity of",
    "all of the issued and outstanding shares of", "all ownership interests in",
])
@pytest.mark.parametrize("location", ["", "London-based ", "Dothan, Alabama-based "])
def test_complete_equity_forms_preserve_full_target_name(equity, location):
    quote = f"Parent Holdings has acquired {equity} {location}Acme Analytics from Seller."
    assert lead_scorer._acquired_stage_quote_supports_names(("Acme Analytics",), quote)
    assert not lead_scorer._acquired_stage_quote_supports_names(("Parent Holdings", "Seller"), quote)


@pytest.mark.parametrize("quote", [
    "Parent acquired 100% of Acme Analytics assets.",
    "Parent acquired 100% of the assets of Acme Analytics.",
    "Parent acquired all assets of Acme Analytics.",
    "Parent acquired 100% of Acme Analytics's products.",
    "Parent acquired 100% of Acme Analytics, a software product of Other.",
    "Parent acquired all shares of Acme Analytics Systems.",
    "Parent acquired 100% of Dothan, Alabama-based Acme Analytics Systems.",
    "Parent acquired 49% of Acme Analytics.",
    "Parent acquired a minority stake in Acme Analytics.",
    "Parent acquired all of a minority stake in Acme Analytics.",
    "Parent acquired all minority shares of Acme Analytics.",
    "Parent acquired 100% of Other Company. Acme Analytics is a buyer.",
    "Acme Analytics acquired 100% of Other Company.",
    "Parent has not acquired 100% of Acme Analytics.",
    "Parent never acquired all shares of Acme Analytics.",
    "Parent plans to have acquired 100% of Acme Analytics by June.",
    "Parent would have acquired all equity in Acme Analytics if approved.",
    "Parent acquired 100% of Acme Analytics, subject to approval.",
    "Parent acquired all shares of Acme Analytics, but the deal was canceled.",
    "Parent acquired 100% of Acme Analytics. Acme Analytics later became independent.",
])
def test_partial_assets_other_entity_and_uncompleted_proof_stay_rejected(quote):
    assert not lead_scorer._acquired_stage_quote_supports_names(("Acme Analytics",), quote)


def _native_finding(*, requested="public", bad_support=False, quote=QUOTE, source=None):
    finding = {"target": "stage", "status": "VERIFIED", "observed_value": "Acquired",
               "evidence_url": URL, "evidence_quote": quote,
               "supporting_evidence_url_1": SUPPORT_URL, "supporting_evidence_quote_1": SUPPORT,
               "reason": "The company was acquired; its current company page confirms the parent."}
    if bad_support:
        finding.update(supporting_evidence_url_2=SUPPORT_URL,
                       supporting_evidence_quote_2="An unsupported paraphrase of the acquisition.")
    return investigator._validated_findings(
        {"findings": [finding]}, targets=("stage",),
        fetched_pages={URL: quote if source is None else source, SUPPORT_URL: SUPPORT},
        fetched_final_urls={URL: URL, SUPPORT_URL: SUPPORT_URL},
        first_party_domains={"aaacooper.com"},
        identity_names={lead_scorer._compact_company_name(COMPANY)},
        identity_anchor={"submitted_name": COMPANY}, requested_stage=requested,
    )["stage"]


@pytest.mark.parametrize(("requested", "status", "decision"), [
    ("public", "CONTRADICTED", lead_scorer.COMPANY_FIT_MISMATCH),
    ("acquired", "VERIFIED", lead_scorer.COMPANY_FIT_MATCH),
])
def test_complete_equity_finding_survives_native_and_saved_stage(requested, status, decision):
    from tests.test_company_evidence_investigator import _company

    finding = _native_finding(requested=requested)
    assert finding["status"] == status
    verdict = lead_scorer._project_investigator_stage({}, finding, icp_stage=requested)
    restored = json.loads(json.dumps(verdict))
    company = _company(name=COMPANY, website="https://www.aaacooper.com")
    assert lead_scorer._decision_from_observed_stage(
        restored, requested, company=company, evidence_attributed=True,
    ) == decision


def test_unused_missing_support_quote_never_enters_validated_or_saved_evidence():
    finding = _native_finding(bad_support=True)
    # This existing path uses the sufficient primary acquisition proof. Only
    # linked ownership/listed-parent paths consume stage support slots.
    assert finding == _native_finding()
    assert finding["supporting_evidence"] == []
    saved = json.dumps(lead_scorer._project_investigator_stage(
        {}, finding, icp_stage="public",
    ))
    assert "unsupported paraphrase" not in saved


def test_absent_paraphrase_cannot_be_used_as_primary_evidence():
    finding = _native_finding(quote="An unsupported paraphrase of the acquisition.", source=QUOTE)
    assert finding["status"] == "UNPROVEN"
    assert not finding["evidence_quote"]


def test_target_separately_public_or_conflicting_does_not_inherit_buyer_listing():
    finding = _native_finding(quote=QUOTE + " AAA Cooper Transportation later went public.")
    assert finding["status"] == "UNPROVEN"
    # A separate independently listed issuer statement remains usable as Public.
    quote = "AAA Cooper Transportation is publicly traded on NYSE under the ticker AACT."
    assert lead_scorer._stage_evidence_supports_observation(
        "public", quote, evidence_url=SUPPORT_URL,
        evidence_final_url=SUPPORT_URL, first_party_domains=("aaacooper.com",),
        identity_names=(COMPANY,),
    )
