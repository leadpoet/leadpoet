from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring import linkedin_company_size as linkedin
from qualification.scoring.competition import _normalized_company

DOMAIN = "northstarpay.com"
PROFILE_URL = "https://www.linkedin.com/company/northstar-pay"
PAGE_URL = "https://www.northstarpay.com/about"
QUOTE = "We operate a global money transfer and payments platform."


def _company(linkedin_url=""):
    return CompanyOutput(
        company_name="Northstar Pay", company_website=f"https://{DOMAIN}/",
        company_linkedin=linkedin_url, industry="Financial Services",
        employee_count="1,001-5,000", country="United States", intent_signals=[{"description": "A company announcement.", "source": "company_website", "url": PAGE_URL, "date": "2026-08-07", "snippet": "A company announcement."}],
    )


def _payload():
    return {"status": 200, "error": None, "element": {
        "name": "Northstar Pay", "website": f"www.{DOMAIN}",
        "linkedinUrl": PROFILE_URL,
        "employeeCountRange": {"start": 1001, "end": 5000},
        "similarOrganizations": [{
            "name": "Other Company", "website": "othercompany.com",
            "linkedinUrl": "https://www.linkedin.com/company/other-company",
        }],
    }}


def _profile():
    return linkedin.project_structured_linkedin_company_identity(
        DOMAIN, PROFILE_URL, _payload(),
    )


def _verdict():
    return {
        "observed_company_name": "Northstar Pay",
        "observed_company_website": f"https://{DOMAIN}/",
        "observed_company_linkedin": PROFILE_URL,
        "observed_employee_count": "1,001-5,000",
        "employee_size_matches": True,
        "employee_size_evidence_url": PROFILE_URL,
        "employee_size_evidence_quote": "1,001-5,000 employees",
    }


def _web(submitted=""):
    return lead_scorer._web_identity_receipt(
        _company(submitted), _verdict(),
        verified_homepage_transport_domain=DOMAIN,
    )


def _context(*, profile=None, web=None, homepage=None, submitted="", transport=DOMAIN):
    return lead_scorer._investigator_identity_context(
        submitted, homepage or {}, web if web is not None else _web(submitted),
        transport, structured_identity=profile,
    )


def _finding():
    return {
        "target": "industry", "status": "VERIFIED", "observed_value": "",
        "observed_industry": "Financial Services",
        "observed_subindustry": "Banking and payments infrastructure",
        "activity_role": "supplier_operator", "evidence_url": PAGE_URL,
        "evidence_quote": QUOTE, "reason": "Direct platform activity.",
    }


def _validated(identity, *, final_url=PAGE_URL, finding=None):
    anchor = {
        "submitted_name": "Northstar Pay", "observed_name": "Northstar Pay",
        "submitted_domain": DOMAIN, "observed_domain": DOMAIN,
        "submitted_linkedin_slug": "", "observed_linkedin_slug": "northstar-pay",
        "verified_name": identity.get("normalized_name"),
        "verified_domain": identity.get("registrable_dns_domain"),
        "verified_linkedin_slug": identity.get("linkedin_company_slug"),
    }
    return investigator._validated_findings(
        {"findings": [finding or _finding()]}, targets=("industry",),
        fetched_pages={PAGE_URL: "About Us | Northstar Pay. " + QUOTE},
        fetched_final_urls={PAGE_URL: final_url}, first_party_domains={DOMAIN},
        identity_names={"northstarpay"}, identity_anchor=anchor,
    )["industry"]


@pytest.mark.parametrize("submitted", ["", PROFILE_URL])
def test_root_profile_carries_independent_identity_without_changing_locator(submitted):
    identity, locator = _context(profile=_profile(), submitted=submitted)
    assert identity == {
        "normalized_name": "Northstar Pay", "registrable_dns_domain": DOMAIN,
        "linkedin_company_slug": "northstar-pay",
    }
    assert locator == submitted
    assert _validated(identity)["status"] == "VERIFIED"
    assert _validated({})["status"] == "UNPROVEN"


@pytest.mark.parametrize("update", [
    {"name": "Other Company"}, {"name": None}, {"website": "https://othercompany.com/"},
    {"website": "https://northstarpay.com@evil.com/"},
    {"website": "https://127.0.0.1/"},
    {"url": "https://www.linkedin.com/company/other-company"},
    {"url": PROFILE_URL + "?alias=northstar-pay"},
    {"url": "http://www.linkedin.com/company/northstar-pay"},
    {"provider": "unverified"}, {"source_field": "description"},
    {"unverified": True}, {"requested_url": PROFILE_URL},
])
def test_profile_mismatch_or_noncanonical_proof_is_not_promoted(update):
    identity, _ = _context(profile={**_profile(), **update})
    assert identity == {}
    assert _validated(identity)["status"] == "UNPROVEN"


@pytest.mark.parametrize("update", [
    {"decision": "unavailable"}, {"decision": "mismatch"},
    {"evidence_source": "model_only"}, {"submitted_name": "Other Company"},
    {"observed_name": "Other Company"}, {"observed_linkedin_slug": "other-company"},
    {"submitted_domain": "othercompany.com"}, {"observed_domain": "othercompany.com"},
    {"submitted_linkedin_slug": "other-company"},
])
def test_web_receipt_only_constrains_independent_profile(update):
    identity, _ = _context(profile=_profile(), web={**_web(), **update})
    assert identity == {}


@pytest.mark.parametrize("profile", [None, {}, {"name": "Northstar Pay"}])
def test_absent_profile_never_promotes_matching_web_observation(profile):
    assert _context(profile=profile)[0] == {}


def test_existing_homepage_identity_is_never_overridden():
    homepage = {"normalized_name": "Other Company", "registrable_dns_domain": "othercompany.com",
                "linkedin_company_slug": "other-company"}
    identity, _ = _context(profile=_profile(), homepage=homepage)
    assert identity is homepage


@pytest.mark.parametrize("submitted,transport", [
    ("https://www.linkedin.com/company/other-company", DOMAIN),
    ("https://www.linkedin.com/company/12345", DOMAIN),
    ("", "othercompany.com"), ("", ""),
])
def test_conflicting_submitted_slug_and_transport_do_not_bind(submitted, transport):
    assert _context(profile=_profile(), submitted=submitted, transport=transport)[0] == {}


@pytest.mark.parametrize("role", ["customer_user", "third_party", "unresolved"])
def test_proved_identity_does_not_turn_non_supplier_claim_into_qualified_activity(role):
    identity, _ = _context(profile=_profile())
    assert _validated(identity, finding={**_finding(), "activity_role": role})["status"] == "UNPROVEN"


def test_proved_identity_does_not_bind_foreign_final_url():
    identity, _ = _context(profile=_profile())
    assert _validated(identity, final_url="https://othercompany.com/about")["status"] == "UNPROVEN"


@pytest.mark.parametrize("enabled", [True, False])
def test_existing_profile_fetch_collects_root_identity_only_for_investigator(monkeypatch, enabled):
    calls = []

    async def current(*args, **kwargs):
        return {"outcome": linkedin.CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE}

    async def structured(domain, url, **kwargs):
        calls.append((domain, url))
        sink = kwargs.get("company_identity_evidence")
        if sink is not None:
            sink.update(linkedin.project_structured_linkedin_company_identity(domain, url, _payload()))
        return linkedin.project_structured_linkedin_company_size(domain, url, _payload())

    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", current)
    monkeypatch.setattr(lead_scorer, "fetch_structured_linkedin_company_size", structured)
    cache = {}
    for _ in range(2):
        asyncio.run(lead_scorer._refresh_linkedin_employee_size_observation(
            _verdict(), _company(), ICPPrompt(
                icp_id="structured-identity", industry="Financial Services",
                sub_industry="Payments", country="United States", geography="United States",
                company_stage="Public", employee_count="1,001-5,000", product_service="Payment platform",
            ),
            verified_homepage_identity={"normalized_name": "Northstar Pay",
                                        "registrable_dns_domain": DOMAIN,
                                        "linkedin_company_slug": "northstar-pay"},
            invocation_cache=cache, collect_structured_conflict=enabled,
        ))
    assert len(calls) == 1
    assert ("structured_company_identity_evidence" in cache) is enabled
    assert cache["refresh_outcome"] == "verified"


def test_similar_organization_never_supplies_root_identity():
    payload = _payload()
    payload["element"]["name"] = "Other Root Company"
    projected = linkedin.project_structured_linkedin_company_identity(DOMAIN, PROFILE_URL, payload)
    assert _context(profile=projected)[0] == {}
    payload["element"]["website"] = "www.othercompany.com"
    assert linkedin.project_structured_linkedin_company_identity(DOMAIN, PROFILE_URL, payload) is None


def test_private_captured_d13_root_profile_and_exact_quote():
    configured = os.environ.get("ARENA_CAPTURE_REPLAY_ROOT")
    if not configured:
        pytest.skip("Set ARENA_CAPTURE_REPLAY_ROOT for captured packet validation")
    root = Path(configured)
    events = json.loads((root / "recoveryd-target13-score2-events-private.json").read_text())
    responses = {v["content"]["action_sequence"]: v for v in events if v["event_kind"] == "provider.response"}
    heads = {v["call_identity"]: v for v in json.loads((root / "recoveryd-target13-score2-heads-private.json").read_text())}

    def raw(index):
        return base64.b64decode(heads[responses[index]["content"]["call"]["call_identity"]]["body_b64"]).decode()

    company_doc = json.loads((root / "recoveryd-target13-output-private.json").read_text())[0]["document"]["companies"][0]
    company = CompanyOutput(**_normalized_company(company_doc, integrity_policy=True))
    verdict = json.loads(json.loads(raw(46))["choices"][0]["message"]["content"])
    web = lead_scorer._web_identity_receipt(company, verdict, verified_homepage_transport_domain="westernunion.com")
    projected = linkedin.project_structured_linkedin_company_identity(
        "westernunion.com", verdict["observed_company_linkedin"], json.loads(raw(48)),
    )
    identity, locator = lead_scorer._investigator_identity_context(
        company.company_linkedin, {}, web, "westernunion.com", structured_identity=projected,
    )
    assert identity["normalized_name"] == projected["name"]
    assert locator == company.company_linkedin == ""
    arguments = json.loads(json.loads(raw(60))["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    finding = next(v for v in arguments["findings"] if v["target"] == "industry")
    anchor = {**web, "verified_name": identity["normalized_name"],
              "verified_domain": identity["registrable_dns_domain"],
              "verified_linkedin_slug": identity["linkedin_company_slug"]}
    result = investigator._validated_findings(
        {"findings": [finding]}, targets=("industry",),
        fetched_pages={finding["evidence_url"]: investigator._plain_text(raw(54))},
        fetched_final_urls={finding["evidence_url"]: finding["evidence_url"]},
        first_party_domains={"westernunion.com"}, identity_names={"westernunion"}, identity_anchor=anchor,
    )
    assert result["industry"]["status"] == "VERIFIED"
    required = next(v for v in arguments["findings"] if v["target"] == "required_attribute")
    assert required["status"] == "UNPROVEN"


@pytest.mark.parametrize("status", [403, 429, 502, "200", None])
def test_provider_error_or_unverified_wrapper_does_not_project_identity(status):
    payload = {**_payload(), "status": status}
    assert linkedin.project_structured_linkedin_company_identity(DOMAIN, PROFILE_URL, payload) is None


def test_numeric_industry_proof_never_implicitly_qualifies_required_attribute(monkeypatch):
    from tests import test_numeric_alias_investigator_continuity as numeric

    icp, prior = numeric._unproven_attribute_result()
    result, calls = numeric._run_real_investigation(
        monkeypatch, structured_identity=numeric._structured_identity(),
        finding=numeric._industry_finding(), icp=icp, prior_result=prior,
        attribute_source_cache={},
    )
    projected, repaired, claims = result[:3]
    assert claims["industry"]["status"] == "VERIFIED"
    assert "required_attribute" not in claims
    assert projected["attribute_satisfied"] is None
    assert repaired.details["required_attribute_decision"] == "unavailable"
    assert len(calls) == 1
