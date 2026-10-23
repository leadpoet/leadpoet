"""Carry an independently proved homepage ID into existing quote binding."""

from copy import deepcopy

import pytest

from qualification.scoring import lead_scorer
from qualification.scoring.company_evidence_investigator import _validated_findings
from qualification.scoring.linkedin_company_size import (
    project_structured_linkedin_company_identity,
)
from tests.test_structured_numeric_alias_context import BODY, HOME, NAMED, NUMERIC, _web


SOURCE = "https://acme.example/products"
QUOTE = "We provide enterprise software tools to business customers."


def _proof():
    return project_structured_linkedin_company_identity(
        "acme.example", NUMERIC, BODY,
        observed_profile_url=NAMED, expected_company_name="Acme",
    )


def _quote_binding(identity, web, *, source=SOURCE, quote=QUOTE, role="supplier_operator"):
    anchor = {
        "submitted_name": web["submitted_name"],
        "observed_name": web["observed_name"],
        "verified_name": identity["normalized_name"],
        "submitted_domain": web["submitted_domain"],
        "observed_domain": web["observed_domain"],
        "verified_domain": identity["registrable_dns_domain"],
        "submitted_linkedin_slug": web["submitted_linkedin_slug"],
        "observed_linkedin_slug": web["observed_linkedin_slug"],
        "verified_linkedin_slug": identity["linkedin_company_slug"],
    }
    finding = {
        "target": "industry", "status": "VERIFIED", "observed_value": None,
        "observed_industry": "Business software", "activity_role": role,
        "evidence_url": source, "evidence_quote": quote,
    }
    return _validated_findings(
        {"findings": [finding]}, targets=("industry",),
        fetched_pages={SOURCE: QUOTE}, fetched_final_urls={SOURCE: SOURCE},
        first_party_domains={"acme.example"}, identity_names={"acme"},
        identity_anchor=anchor,
    )["industry"]


@pytest.mark.parametrize("submitted", ["", NAMED])
def test_optional_or_vanity_submission_carries_exact_numeric_profile_proof(submitted):
    web = _web("acme" if submitted else "")
    homepage = dict(HOME)
    assert _quote_binding(homepage, web)["status"] == "UNPROVEN"
    resolved, locator = lead_scorer._investigator_identity_context(
        submitted, homepage, web, "acme.example", structured_identity=_proof(),
    )
    assert homepage == HOME
    assert resolved == {**HOME, "linkedin_company_slug": "acme"}
    assert locator == submitted
    result = _quote_binding(resolved, web)
    assert result["status"] == "VERIFIED"
    assert result["evidence_quote"] == QUOTE
    assert result["evidence_url"] == SOURCE


@pytest.mark.parametrize("field,value", [
    ("company_id", "67890"), ("requested_url", NAMED),
    ("requested_url", "https://unrelated.example/company/12345"),
    ("url", "https://www.linkedin.com/company/unrelated"),
    ("url", "https://unrelated.example/company/acme"),
    ("website", "https://unrelated.example"), ("name", "Unrelated"),
    ("provider", "untrusted"), ("source_field", "untrusted"),
    ("company_id", 12345), ("extra", "untrusted"),
])
def test_partial_or_conflicting_profile_proof_cannot_bind_quote(field, value):
    proof = {**_proof(), field: value}
    resolved, locator = lead_scorer._investigator_identity_context(
        "", HOME, _web(), "acme.example", structured_identity=proof,
    )
    assert resolved == HOME
    assert locator == ""
    assert _quote_binding(resolved, _web())["status"] == "UNPROVEN"


@pytest.mark.parametrize("missing", [None, "company_id", "requested_url", "name"])
def test_missing_profile_proof_cannot_bind_quote(missing):
    proof = _proof() if missing else None
    if missing:
        proof.pop(missing)
    resolved, _ = lead_scorer._investigator_identity_context(
        "", HOME, _web(), "acme.example", structured_identity=proof,
    )
    assert resolved == HOME


@pytest.mark.parametrize("field,value", [
    ("decision", "unavailable"), ("evidence_source", "company_homepage"),
    ("submitted_domain", "unrelated.example"),
    ("observed_domain", "unrelated.example"),
    ("submitted_name", "Unrelated"), ("observed_name", "Unrelated"),
    ("submitted_linkedin_slug", "unrelated"),
    ("observed_linkedin_slug", "unrelated"),
])
def test_unbound_web_observation_cannot_supply_alias(field, value):
    web = {**_web(), field: value}
    resolved, _ = lead_scorer._investigator_identity_context(
        "", HOME, web, "acme.example", structured_identity=_proof(),
    )
    assert resolved == HOME


@pytest.mark.parametrize("field,value", [
    ("normalized_name", "Unrelated"),
    ("registrable_dns_domain", "unrelated.example"),
    ("linkedin_company_slug", "67890"),
])
def test_conflicting_homepage_identity_cannot_supply_alias(field, value):
    homepage = {**HOME, field: value}
    resolved, _ = lead_scorer._investigator_identity_context(
        "", homepage, _web(), "acme.example", structured_identity=_proof(),
    )
    assert resolved == homepage


@pytest.mark.parametrize("fault", ["invented_quote", "different_source", "wrong_role"])
def test_proved_alias_keeps_existing_evidence_guards(fault):
    resolved, _ = lead_scorer._investigator_identity_context(
        "", HOME, _web(), "acme.example", structured_identity=_proof(),
    )
    options = {
        "invented_quote": {"quote": "An invented product capability is absent."},
        "different_source": {"source": "https://unrelated.example/products"},
        "wrong_role": {"role": "customer_user"},
    }[fault]
    assert _quote_binding(resolved, _web(), **options)["status"] == "UNPROVEN"


def test_original_structured_response_is_not_modified():
    original = deepcopy(BODY)
    _proof()
    assert BODY == original
