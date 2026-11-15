"""Retain a verified legal name through real identity and research consumers."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as inv, lead_scorer
from tests.test_company_evidence_investigator import _company, _finding, _icp

NAME = "MEC (Mayville Engineering Company, Inc.)"
LEGAL = "Mayville Engineering Company, Inc."
HOME = "https://www.mecinc.com/"
PROFILE = "https://www.linkedin.com/company/mayville-engineering-company"
URL = HOME + "manufacturing/"
QUOTE = "We own and operate manufacturing facilities that produce metal components."
HOMEPAGE = {"normalized_name": NAME, "registrable_dns_domain": "mecinc.com",
            "linkedin_company_slug": "mayville-engineering-company"}


def context(**changes):
    company = _company(name=NAME, website=HOME, linkedin=PROFILE)
    verdict = {"observed_company_name": LEGAL, "observed_company_website": HOME,
               "observed_company_linkedin": "", **changes}
    receipt = lead_scorer._web_identity_receipt(company, verdict,
        verified_homepage_identity=HOMEPAGE,
        verified_homepage_transport_domain="mecinc.com", company_quality=True)
    return company, verdict, receipt


def test_native_receipt_recovery_keeps_exact_legal_name_after_json_roundtrip():
    company, _, receipt = context()
    assert receipt["decision"] == "match"
    assert receipt["verified_parenthetical_legal_name"] == LEGAL
    prior = SimpleNamespace(details=json.loads(json.dumps(
        {"identity_decision": "match", "identity_receipt": receipt})))
    before = deepcopy(prior.details)
    assert lead_scorer._complete_verified_attribute_recovery_identity(
        company, prior, HOMEPAGE, URL)
    assert prior.details == before


@pytest.mark.parametrize("field,value", [
    ("verified_parenthetical_legal_name", None),
    ("verified_parenthetical_legal_name", "Mayville Engineering Holdings, Inc."),
    ("verified_parenthetical_legal_name", "MEC"),
    ("observed_domain", "other.example"),
    ("observed_linkedin_slug", "different-company"),
])
def test_saved_recovery_rejects_missing_or_conflicting_proof(field, value):
    company, _, receipt = context()
    receipt[field] = value
    prior = SimpleNamespace(details={"identity_decision": "match", "identity_receipt": receipt})
    assert not lead_scorer._complete_verified_attribute_recovery_identity(
        company, prior, HOMEPAGE, URL)


@pytest.mark.parametrize("observed,expected", [
    (LEGAL, "VERIFIED"),
    ("Mayville Engineering Holdings, Inc.", "UNPROVEN"),
    ("Mayville Engineering Company Asia, Inc.", "UNPROVEN"),
    ("MEC", "UNPROVEN"),
])
def test_native_targeted_investigator_preserves_proof_not_a_bare_acronym(monkeypatch, observed, expected):
    company, verdict, receipt = context(observed_company_name=observed)
    captured = {}
    native = inv.investigate_company_evidence

    async def investigate(**kwargs):
        captured["handoff"] = kwargs["prior_observations"]["observed_company_linkedin"]
        captured["result"] = await native(**kwargs)
        raise RuntimeError("native boundary complete")

    async def post(*args, **kwargs):
        finding = _finding("required_attribute", status="VERIFIED",
            evidence_url=URL, evidence_quote=QUOTE, activity_role="supplier_operator")
        return 200, {"choices": [{"message": {"tool_calls": [{"id": "offline",
            "type": "function", "function": {"name": "submit_findings",
            "arguments": json.dumps({"findings": [finding]})}}]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", AsyncMock(return_value={"results": []}))
    monkeypatch.setattr(inv, "_fetch_page", AsyncMock(side_effect=AssertionError("source retained")))
    with pytest.raises(RuntimeError, match="native boundary complete"):
        asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
            company=company, icp=_icp(required_attribute="Owns manufacturing plants"),
            verdict=verdict, investigation_targets=("required_attribute",),
            icp_attribute="Owns manufacturing plants", icp_stage="",
            verified_identity=HOMEPAGE, verified_transport_domain="mecinc.com",
            structured_employee_size_evidence=None, structured_public_company_evidence=None,
            employee_size_conflict=False, company_quality=True, review_positive_semantics=True,
            required_attribute_source_cache={URL: {"status": "fetched", "final_url": URL, "text": QUOTE}}))
    assert captured["result"]["claims"]["required_attribute"]["status"] == expected
    assert captured["handoff"] == (PROFILE if expected == "VERIFIED" else "")
    assert captured["result"]["usage"]["fetch_calls"] == 0


def test_approximate_count_and_derived_band_are_still_invalid():
    quote = "We employ approximately 43,000 people."
    assert not inv._quote_supports_headcount(quote, 43000)
    assert not inv._quote_supports_headcount(quote, "10,001+")
