"""Exact target binding for completed acquisitions stated in active voice."""

import json
from urllib.parse import urlsplit

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer


# Exact public-source quote from the November 13 failed stage adjudication.
SAVED_COMPANY = 'AAA Cooper Transportation'
SAVED_SOURCE_URL = 'https://www.aaacooper.com/About'
SAVED_QUOTE = 'A pivotal moment occurred in July 2021, when Knight-Swift Transportation Holdings, Inc. (NYSE: KNX) acquired AAA Cooper Transportation, transitioning ACT from a privately held company to an independent subsidiary.'


def test_saved_active_acquisition_quote_binds_exact_target():
    assert lead_scorer._acquired_stage_quote_supports_names(
        (SAVED_COMPANY,), SAVED_QUOTE,
    )


@pytest.mark.parametrize("target", ["Acme Analytics", "Box", "Acme Analytics Inc."])
@pytest.mark.parametrize("verb", ["acquired", "has acquired"])
def test_active_completed_acquisition_binds_target_not_buyer(target, verb):
    quote = f"Parent Holdings {verb} {target} from Seller Holdings."
    assert lead_scorer._acquired_stage_quote_supports_names((target,), quote)
    assert not lead_scorer._acquired_stage_quote_supports_names(
        ("Parent Holdings", "Seller Holdings"), quote,
    )


@pytest.mark.parametrize("quote", [
    "Acme Analytics acquired Other Holdings.",
    "Parent Holdings acquired Other Holdings. Acme Analytics serves clients.",
    "Parent Holdings announced its acquisition of Acme Analytics.",
    "Parent Holdings announced it will acquire Acme Analytics.",
    "Parent Holdings plans to have acquired Acme Analytics by June.",
    "Parent Holdings would have acquired Acme Analytics if approved.",
    "Parent Holdings has not acquired Acme Analytics.",
    "Parent Holdings never acquired Acme Analytics.",
    "Parent Holdings acquired Acme Analytics, subject to approval.",
    "Parent Holdings acquired Acme Analytics, but the deal was cancelled.",
    "Parent Holdings acquired Acme Analytics assets.",
    "Parent Holdings acquired assets of Acme Analytics.",
    "Parent Holdings acquired Acme Analytics's product.",
    "Parent Holdings acquired Acme Analytics, a software product of Other.",
    "Parent Holdings acquired Acme Analytics as a customer.",
    "Parent Holdings acquired Acme Analytics as a supplier.",
    "Parent Holdings acquired Acme Analytics's assets from Seller Holdings.",
    "Parent Holdings acquired a minority stake in Acme Analytics.",
    "Parent Holdings acquired Acme Analytics minority shares.",
    "Parent Holdings acquired Acme Analytics Systems.",
    "Parent Holdings acquired Acme Analytics. Acme Analytics later became independent.",
])
def test_active_acquisition_does_not_admit_uncompleted_partial_or_unbound_proof(quote):
    assert not lead_scorer._acquired_stage_quote_supports_names(
        ("Acme Analytics",), quote,
    )


@pytest.mark.parametrize("quote", [
    "Acme Analytics was acquired by Parent Holdings.",
    "Parent Holdings completed its acquisition of Acme Analytics.",
    "Acme Analytics is a wholly owned subsidiary of Parent Holdings.",
])
def test_existing_completed_acquisition_patterns_stay_valid(quote):
    assert lead_scorer._acquired_stage_quote_supports_names(("Acme Analytics",), quote)


@pytest.mark.parametrize(("requested", "status", "decision"), [
    ("public", "CONTRADICTED", lead_scorer.COMPANY_FIT_MISMATCH),
    ("acquired", "VERIFIED", lead_scorer.COMPANY_FIT_MATCH),
])
def test_saved_active_acquisition_survives_native_and_saved_stage_validation(
    requested, status, decision,
):
    from tests.test_company_evidence_investigator import _company

    findings = investigator._validated_findings(
        {"findings": [{
            "target": "stage", "status": "VERIFIED", "observed_value": "Acquired",
            "evidence_url": SAVED_SOURCE_URL, "evidence_quote": SAVED_QUOTE,
            "reason": "Completed acquisition names the investigated target.",
        }]},
        targets=("stage",), fetched_pages={SAVED_SOURCE_URL: SAVED_QUOTE},
        fetched_final_urls={SAVED_SOURCE_URL: SAVED_SOURCE_URL},
        first_party_domains=set(),
        identity_names={lead_scorer._compact_company_name(SAVED_COMPANY)},
        identity_anchor={"submitted_name": SAVED_COMPANY}, requested_stage=requested,
    )
    # Public needs separately proved current-parent listing. A completed
    # acquisition alone settles the different stage without inventing that fit.
    assert findings["stage"]["status"] == status
    assert findings["stage"]["evidence_quote"] == SAVED_QUOTE
    verdict = lead_scorer._project_investigator_stage(
        {}, findings["stage"], icp_stage=requested,
    )
    restored = json.loads(json.dumps(verdict))
    company = _company(
        name=SAVED_COMPANY,
        website="https://" + str(urlsplit(SAVED_SOURCE_URL).hostname),
    )
    assert lead_scorer._decision_from_observed_stage(
        restored, requested, company=company, evidence_attributed=True,
    ) == decision
