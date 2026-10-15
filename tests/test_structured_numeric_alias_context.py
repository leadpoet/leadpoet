"""A matched web observation must reach the existing numeric-alias parser."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from qualification.scoring import lead_scorer
from qualification.scoring import linkedin_company_size as linkedin


HOME = {
    "normalized_name": "Acme",
    "registrable_dns_domain": "acme.example",
    "linkedin_company_slug": "12345",
}
NAMED = "https://www.linkedin.com/company/acme"
NUMERIC = "https://www.linkedin.com/company/12345"
BODY = {"result": {"data": {"status": 200, "element": {
    "id": 12345,
    "name": "Acme",
    "website": "https://acme.example",
    "linkedinUrl": NAMED,
    "employeeCountRange": {"start": 51, "end": 200},
}}}}


def _web(submitted_slug=""):
    return lead_scorer._web_identity_receipt(
        SimpleNamespace(
            company_name="Acme", company_website="https://acme.example",
            company_linkedin=(NAMED if submitted_slug else ""),
        ),
        {
            "observed_company_name": "Acme",
            "observed_company_website": "https://acme.example",
            "observed_company_linkedin": NAMED,
        },
        verified_homepage_identity=HOME,
        verified_homepage_transport_domain="acme.example",
    )


def _completed_review():
    return {
        "gate": "company_evidence_investigation",
        "completed_submitted_findings": True,
        "failure_reason": "", "targets": ["geography", "stage"],
        "claims": {target: {
            "target": target, "status": "UNPROVEN",
            "evidence_url": "", "evidence_quote": "",
        } for target in ("geography", "stage")},
        "usage": {"fetch_outcomes": []},
    }


@pytest.mark.parametrize("submitted_slug", ["", "acme"])
@pytest.mark.parametrize("fault", [
    None, "id", "name", "website", "linkedinUrl", "provider_error",
    "auth_wall", "malformed", "multiple",
])
def test_numeric_lookup_recovers_only_existing_bound_structured_proof(
    monkeypatch, submitted_slug, fault,
):
    body = copy.deepcopy(BODY)
    element = body["result"]["data"]["element"]
    if fault == "id":
        element["id"] = 67890
    elif fault == "name":
        element["name"] = "Unrelated"
    elif fault == "website":
        element["website"] = "https://unrelated.example"
    elif fault == "linkedinUrl":
        element["linkedinUrl"] = "https://www.linkedin.com/company/unrelated"
    elif fault == "provider_error":
        body["error"] = "provider_unavailable"
    elif fault == "malformed":
        body = {"result": {"data": {"status": 200, "element": {}}}}
    elif fault == "multiple":
        body = {"status": 200, "elements": [element, dict(element)]}
    requests = []

    class Response:
        status = 401 if fault == "auth_wall" else 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def json(self):
            return body

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def post(self, _url, *, json, headers):
            del headers
            requests.append(json)
            return Response()

    async def unavailable_profile(_url, *, diagnostic, **_kwargs):
        diagnostic[linkedin.VERIFIER_FAILURE_REASON_KEY] = linkedin.PROVIDER_ERROR_FAILURE_REASON
        return None

    async def unproved_website_redirect(*_args):
        return ""

    monkeypatch.setenv("DEEPLINE_API_KEY", "test-key")
    monkeypatch.setattr(linkedin.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(linkedin, "_verified_provider_website_redirect", unproved_website_redirect)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", unavailable_profile)
    anchor = lead_scorer._structured_profile_identity_anchor(HOME, _web(submitted_slug), "acme.example")
    cache = {}
    verdict = {
        "employee_size_evidence_url": NUMERIC,
        "employee_size_evidence_quote": "Unverified prior observation",
    }
    asyncio.run(lead_scorer._refresh_linkedin_employee_size_observation(
        verdict, SimpleNamespace(company_name="Acme", company_website="https://acme.example", company_linkedin=""),
        SimpleNamespace(employee_count="51-200"), verified_homepage_identity=anchor,
        invocation_cache=cache, collect_structured_conflict=True,
    ))
    assert requests == [{"payload": {"url": NUMERIC}}]
    if fault is None:
        assert cache["structured_company_identity_evidence"]["requested_url"] == NUMERIC
        assert cache["structured_company_identity_evidence"]["url"] == NAMED
        assert cache["structured_evidence"]["employee_count"] == "51-200"
        assert cache["refresh_outcome"] == "verified"
        assert lead_scorer.VERIFIER_FAILURE_DETAIL_KEY not in cache
        conflicting = {
            "observed_employee_count": "201-500",
            "employee_size_evidence_url": "https://acme.example/about",
            "employee_size_evidence_quote": "Acme has 300 employees.",
        }
        assert lead_scorer._employee_size_sources_conflict(
            conflicting, cache["structured_evidence"],
        )
        assert lead_scorer._project_investigator_headcount(
            conflicting, {
                "status": "VERIFIED", "observed_value": "51-200",
                "evidence_url": NAMED, "evidence_quote": "Acme: 51-200 employees",
            }, icp=SimpleNamespace(employee_count="51-200"), existing_conflict=True,
        ) == conflicting
    else:
        assert cache["structured_evidence"] is None
        assert cache["refresh_outcome"] == "retryable_failure"
    # This is the actual later assembly guard. A failed source must still
    # block unrelated missing dimensions; recovered proof may finish them.
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("geography", "stage"),
        investigation_receipt=_completed_review(),
        linkedin_refresh_outcome=cache["refresh_outcome"],
    ) is (fault is None)


@pytest.mark.parametrize("field,value", [
    ("decision", "unavailable"),
    ("evidence_source", "company_homepage"),
    ("submitted_domain", "unrelated.example"),
    ("observed_domain", "unrelated.example"),
    ("submitted_name", "unrelated"),
    ("observed_name", "unrelated"),
    ("submitted_linkedin_slug", "unrelated"),
    ("observed_linkedin_slug", "67890"),
])
def test_unbound_web_observation_does_not_add_alias_context(field, value):
    web = _web()
    web[field] = value
    anchor = lead_scorer._structured_profile_identity_anchor(HOME, web, "acme.example")
    assert "requested_profile_url" not in anchor
    assert "observed_profile_url" not in anchor
