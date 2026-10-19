"""Exact investigator proof can settle only a source-local profile refresh."""
from __future__ import annotations

import copy

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer as scorer
from tests.test_current_linkedin_company_size import _icp

URL = "https://www.linkedin.com/company/acme"
QUOTE = "Acme | LinkedIn. Company size 51-200 employees"
IDENTITY = {"normalized_name": "Acme", "registrable_dns_domain": "acme.example.com", "linkedin_company_slug": "acme"}


def fixture():
    icp = _icp().model_copy(update={"employee_count": "51-200"})
    raw = {"target": "headcount", "status": "VERIFIED", "observed_value": "51-200",
           "evidence_url": URL, "evidence_quote": QUOTE, "activity_role": "unresolved"}
    finding = investigator._validated_findings(
        {"findings": [raw]}, targets=("headcount",), fetched_pages={URL: QUOTE},
        fetched_final_urls={URL: URL}, first_party_domains={"acme.example.com"}, identity_names={"acme"},
    )["headcount"]
    investigation = {
        "_completed_submit": True, "failure_reason": "", "claims": {"headcount": finding},
        investigator.PRIVATE_FETCHED_PAGES_KEY: {URL: {"final_url": URL, "text": QUOTE}},
    }
    cache = {"refresh_outcome": "retryable_failure", "profile_url": URL,
             "evidence": {"outcome": "insufficient_evidence", "url": URL},
             "structured_failure": True, "structured_failure_reason": "source_blocked",
             "failure_reason_code": "source_blocked"}
    projected = scorer._project_investigator_headcount({}, finding, icp=icp, existing_conflict=False)
    return cache, finding, investigation, projected, icp


def settle(cache, finding, investigation, projected, icp, identity=IDENTITY):
    return scorer._settle_recovered_local_headcount_refresh(
        cache, finding, investigation, identity, projected, icp=icp,
    )


def test_valid_exact_same_profile_settles_refresh_without_changing_qualification():
    cache, finding, investigation, projected, icp = fixture()
    before = copy.deepcopy(projected)
    assert settle(cache, finding, investigation, projected, icp)
    assert cache["refresh_outcome"] == "verified"
    assert "failure_reason_code" not in cache
    assert cache["structured_failure_reason"] == "source_blocked"
    assert projected == before
    assert not settle(cache, finding, investigation, projected, icp)  # Settles once.


@pytest.mark.parametrize("failure", ["provider_error", "malformed_response", "unexpected_verifier_error"])
@pytest.mark.parametrize("source", ["initial", "structured", "investigator"])
def test_actual_provider_auth_transport_or_contract_failure_cannot_be_cleared(failure, source):
    cache, finding, investigation, projected, icp = fixture()
    if source == "initial":
        cache.update(evidence=None, failure_reason_code=failure)
    elif source == "structured":
        cache.update(structured_failure_reason=failure, failure_reason_code=failure)
    else:
        investigation["failure_reason"] = failure
    before = copy.deepcopy(cache)
    assert not settle(cache, finding, investigation, projected, icp)
    assert cache == before


def test_structured_local_failure_cannot_hide_an_initial_provider_failure():
    cache, finding, investigation, projected, icp = fixture()
    cache["evidence"] = None  # Later local diagnostic may have replaced the first failure label.
    assert not settle(cache, finding, investigation, projected, icp)
    assert cache["refresh_outcome"] == "retryable_failure"


@pytest.mark.parametrize("control", [
    "incomplete", "negative_headcount", "missing_quote", "unloaded_quote", "wrong_company",
    "different_profile", "redirected_profile", "unprojected_quote", "unprojected_count", "noncanonical_count",
])
def test_cached_or_unvalidated_headcount_cannot_settle_refresh(control):
    cache, finding, investigation, projected, icp = fixture()
    if control == "incomplete":
        investigation["_completed_submit"] = False
    elif control == "negative_headcount":
        finding["status"] = "UNPROVEN"
    elif control == "missing_quote":
        finding["evidence_quote"] = ""
    elif control == "unloaded_quote":
        investigation[investigator.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] = "Acme has no quoted employee range here."
    elif control == "wrong_company":
        finding["evidence_quote"] = "OtherCo | LinkedIn. Company size 51-200 employees"
        investigation[investigator.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] = finding["evidence_quote"]
        projected["employee_size_evidence_quote"] = finding["evidence_quote"]
    elif control == "different_profile":
        cache["profile_url"] = "https://www.linkedin.com/company/other"
    elif control == "redirected_profile":
        investigation[investigator.PRIVATE_FETCHED_PAGES_KEY][URL]["final_url"] = "https://www.linkedin.com/company/other"
    elif control == "unprojected_quote":
        projected["employee_size_evidence_quote"] = "Acme employs staff."
    elif control == "unprojected_count":
        projected["observed_employee_count"] = None
    elif control == "noncanonical_count":
        finding["observed_value"] = projected["observed_employee_count"] = "approximately 100"
    before = copy.deepcopy(cache)
    assert not settle(cache, finding, investigation, projected, icp)
    assert cache == before


def test_other_healthy_unproven_dimensions_can_finish_after_local_recovery():
    cache, finding, investigation, projected, icp = fixture()
    negative = {target: {"target": target, "status": "UNPROVEN", "evidence_url": "", "evidence_quote": ""}
                for target in ("industry", "stage", "required_attribute")}
    receipt = {"gate": "company_evidence_investigation", "completed_submitted_findings": True,
               "failure_reason": "", "targets": ["headcount", *negative],
               "claims": {"headcount": finding, **negative}, "usage": {"fetch_outcomes": []}}
    projected[scorer._REQUIRED_ATTRIBUTE_GROUNDING] = scorer._required_attribute_source_receipt(status="unproven")
    incomplete = ("industry", "stage", "required_attribute")
    assert not scorer._has_explicitly_unproven_fit_dimensions(
        projected, incomplete, icp=icp, linkedin_refresh_outcome=cache["refresh_outcome"], investigation_receipt=receipt,
    )
    assert settle(cache, finding, investigation, projected, icp)
    assert scorer._has_explicitly_unproven_fit_dimensions(
        projected, incomplete, icp=icp, linkedin_refresh_outcome=cache["refresh_outcome"], investigation_receipt=receipt,
    )
    # A separate unresolved source failure still prevents finality.
    receipt["usage"]["fetch_outcomes"] = [{"ok": False, "error_class": "http_error", "url_sha256": "a" * 64}]
    assert not scorer._has_explicitly_unproven_fit_dimensions(
        projected, incomplete, icp=icp, linkedin_refresh_outcome=cache["refresh_outcome"], investigation_receipt=receipt,
    )


@pytest.mark.parametrize("body,expected", [
    ("# Acme\n## About us\nCompany size 201–500 employees", True),
    ("# OtherCo\n## About us\nAcme is a customer.\nCompany size 201–500 employees", False),
    ("# Acme\n## About us\nWebsite\nhttps://other.example/\nCompany size 201–500 employees", False),
])
def test_already_validated_size_quote_can_use_separate_bound_profile_heading(body, expected):
    cache, finding, investigation, projected, icp = fixture()
    # This control concerns completion of an already validated/projected
    # finding. It does not change the investigator's grading validator.
    quote = "Company size 201–500 employees"
    finding.update(observed_value="201-500", evidence_quote=quote)
    icp = icp.model_copy(update={"employee_count": "201-500"})
    projected.update(observed_employee_count="201-500", employee_size_evidence_quote=quote)
    investigation[investigator.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] = body
    assert settle(cache, finding, investigation, projected, icp) is expected
    assert cache["refresh_outcome"] == ("verified" if expected else "retryable_failure")


def test_body_binding_does_not_replace_a_missing_projected_finding():
    cache, finding, investigation, projected, icp = fixture()
    quote = "Company size 51-200 employees"
    finding["evidence_quote"] = quote
    investigation[investigator.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] = "# Acme\n## About us\n" + quote
    projected["observed_employee_count"] = None
    assert not settle(cache, finding, investigation, projected, icp)
