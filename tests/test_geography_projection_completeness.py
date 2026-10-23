"""Exercise the HQ projector inputs used before and after broad schema repair."""
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer
from tests.test_geography_source_conflict import ANCHOR, FIXTURE, IDENTITY, company, icp, verdict


def us_case():
    return (
        company().model_copy(update={"country": "United States", "state": "California"}),
        icp().model_copy(update={"country": "United States", "geography": "California"}),
    )


def proof(country="United States", state="", status="VERIFIED"):
    location = f"{state}, {country}" if state else country
    quote = f"Airwallex is headquartered in {location}."
    return inv._validated_findings(
        {"findings": [{"target": "geography", "status": status,
                       "observed_country": country, "observed_state": state,
                       "evidence_url": FIXTURE["source_url"], "evidence_quote": quote}]},
        targets=("geography",), fetched_pages={FIXTURE["source_url"]: quote},
        fetched_final_urls={FIXTURE["source_url"]: FIXTURE["source_url"]},
        first_party_domains={"airwallex.com"}, identity_names={"airwallex"},
        identity_anchor=ANCHOR,
    )["geography"]


def broad_matching_state():
    broad = verdict()
    broad.update(observed_hq_country="United States", observed_hq_state="California",
                 geography_matches=True, geography_evidence_url=FIXTURE["source_url"],
                 geography_evidence_quote="Airwallex headquarters: California, United States.")
    return broad


def project(observation, finding, candidate, request):
    return scorer._project_verified_linkedin_headquarters(
        observation, verified_identity=IDENTITY,
        invocation_cache={"structured_profile_headquarters_evidence": deepcopy(FIXTURE["headquarters"])},
        linkedin_profile_source={}, icp=request, company=candidate, company_quality=True,
        validated_geography_finding=finding,
    )


def test_repair_projection_cannot_fill_an_unproved_state_from_broad_verdict():
    candidate, request = us_case()
    finding = proof()
    assert finding["status"] == "VERIFIED"
    projected = project(broad_matching_state(), finding, candidate, request)
    assert projected["observed_hq_country"] == "Hong Kong"
    assert projected["geography_evidence_url"] == FIXTURE["headquarters"]["url"]
    assert scorer._decision_from_observed_geography(
        projected, request, company=candidate, company_quality=True,
        validated_geography_finding=finding,
    ) == "mismatch"


def test_incomplete_us_finding_keeps_initial_linkedin_projection():
    candidate, request = us_case()
    initial = project(broad_matching_state(), None, candidate, request)
    assert initial["observed_hq_country"] == "Hong Kong"
    assert project(initial, proof(), candidate, request) == initial


@pytest.mark.parametrize("state,expected", [("California", "match"), ("Nevada", "mismatch")])
def test_complete_us_state_uses_only_the_validated_projection(state, expected):
    candidate, request = us_case()
    finding = proof(state=state)
    projected = project(broad_matching_state(), finding, candidate, request)
    assert projected["observed_hq_country"] == "United States"
    assert projected["observed_hq_state"] == state
    assert scorer._validated_investigator_geography_matches_verdict(projected, finding)
    assert scorer._decision_from_observed_geography(
        projected, request, company=candidate, company_quality=True,
        validated_geography_finding=finding,
    ) == expected


def test_complete_non_us_headquarters_still_resolves_profile_conflict():
    finding = proof(country="Singapore")
    projected = project(verdict(), finding, company(), icp())
    assert projected["observed_hq_country"] == "Singapore"
    assert scorer._validated_investigator_geography_matches_verdict(projected, finding)
    assert scorer._decision_from_observed_geography(
        projected, icp(), company=company(), company_quality=True,
        validated_geography_finding=finding,
    ) == "match"


@pytest.mark.parametrize("status", ["UNPROVEN", "CONTRADICTED"])
def test_nonverified_finding_cannot_preserve_a_matching_repair(status):
    candidate, request = us_case()
    projected = project(broad_matching_state(), proof(state="California", status=status),
                        candidate, request)
    assert projected["observed_hq_country"] == "Hong Kong"
    assert projected["geography_evidence_url"] == FIXTURE["headquarters"]["url"]
