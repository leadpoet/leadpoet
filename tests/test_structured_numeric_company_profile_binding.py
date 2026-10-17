"""Numeric company locators require a bound provider ID, never a domain alone."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from qualification.scoring import lead_scorer
from qualification.scoring import linkedin_company_size as linkedin


DOMAIN = "acme.example"
NUMERIC = "https://www.linkedin.com/company/12345"
VANITY = "https://www.linkedin.com/company/acme"
HOME = {
    "normalized_name": "Acme",
    "registrable_dns_domain": DOMAIN,
    "linkedin_company_slug": "12345",
}
BODY = {"result": {"data": {"status": 200, "element": {
    "id": "12345",
    "name": "Acme, Inc.",
    "website": "https://acme.example/",
    "linkedinUrl": VANITY,
    "universalName": "acme",
    "employeeCountRange": {"start": 51, "end": 200},
}}}}


def _body(**updates):
    body = copy.deepcopy(BODY)
    body["result"]["data"]["element"].update(updates)
    return body


def _identity(body, **kwargs):
    return linkedin.project_structured_linkedin_company_identity(
        DOMAIN, NUMERIC, body, expected_company_name="Acme", **kwargs,
    )


def _mock_provider(monkeypatch, body):
    requests = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def json(self):
            return copy.deepcopy(body)

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

    async def no_redirect(*_args):
        raise AssertionError("Numeric identity must not add a provider request")

    monkeypatch.setenv("DEEPLINE_API_KEY", "test-key")
    monkeypatch.setattr(linkedin.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(linkedin, "_verified_provider_website_redirect", no_redirect)
    return requests


@pytest.mark.parametrize("company_id", [12345, "12345"])
def test_provider_id_and_vanity_bind_numeric_homepage_without_web_alias(company_id):
    assert _identity(_body(id=company_id)) == {
        "name": "Acme, Inc.", "provider": "harvestapi_get_company",
        "source_field": "name", "url": VANITY,
        "website": "https://acme.example/", "company_id": "12345",
        "requested_url": NUMERIC,
    }


@pytest.mark.parametrize("updates", [
    {"id": None}, {"id": "67890"}, {"id": True}, {"id": 12345.0},
    {"id": " 12345"}, {"id": "012345"}, {"id": "١٢٣٤٥"},
    {"name": "Unrelated Company"}, {"website": "https://unrelated.example/"},
    {"universalName": None}, {"universalName": "other"},
    {"universalName": " acme"}, {"universalName": 12345},
    {"linkedinUrl": "https://www.linkedin.com/company/67890"},
    {"linkedinUrl": "https://www.linkedin.com/company/unrelated"},
    {"linkedinUrl": "https://www.linkedin.com/in/acme"},
    {"linkedinUrl": VANITY + "?unbound=true"},
])
def test_numeric_alias_rejects_missing_malformed_or_conflicting_identity(updates):
    assert _identity(_body(**updates)) is None


@pytest.mark.parametrize("observed", [
    "https://www.linkedin.com/company/unrelated",
    "https://www.linkedin.com/company/67890", "malformed",
])
def test_provider_vanity_never_overrides_an_independent_observation(observed):
    assert _identity(BODY, observed_profile_url=observed) is None


def test_existing_independent_alias_does_not_need_universal_name():
    body = _body()
    del body["result"]["data"]["element"]["universalName"]
    assert _identity(body, observed_profile_url=VANITY)
    assert _identity(body) is None
    assert _identity(_body(universalName="unrelated"), observed_profile_url=VANITY) is None


def test_numeric_binding_needs_an_independent_expected_name():
    assert linkedin.project_structured_linkedin_company_identity(DOMAIN, NUMERIC, BODY) is None


def test_exact_numeric_response_remains_usable_with_bound_id_and_name(monkeypatch):
    _mock_provider(monkeypatch, _body(linkedinUrl=NUMERIC))
    evidence = asyncio.run(linkedin.fetch_structured_linkedin_company_size(
        DOMAIN, NUMERIC, expected_company_name="Acme",
    ))
    assert evidence["url"] == NUMERIC
    assert evidence["employee_count"] == "51-200"


@pytest.mark.parametrize("expected_name", ["Acme", "Unrelated Company"])
@pytest.mark.parametrize("projector", [
    linkedin.project_structured_linkedin_company_description,
    linkedin.project_structured_linkedin_company_headquarters,
])
def test_exact_numeric_description_and_hq_keep_existing_name_binding(projector, expected_name):
    body = _body(
        linkedinUrl=NUMERIC,
        description="Acme develops payment software.",
        locations=[{"headquarter": True, "country": "United States",
                    "geographicArea": "California", "city": "San Francisco"}],
    )
    evidence = projector(DOMAIN, NUMERIC, expected_name, body)
    if expected_name == "Acme":
        assert evidence["name"] == "Acme, Inc."
        assert evidence["url"] == NUMERIC
        assert evidence["text"]
    else:
        assert evidence is None


def test_nested_organization_and_multiple_roots_cannot_supply_identity():
    body = _body(name="Unrelated Company")
    body["result"]["data"]["element"]["similarOrganizations"] = [
        copy.deepcopy(BODY["result"]["data"]["element"]),
    ]
    assert _identity(body) is None
    element = BODY["result"]["data"]["element"]
    assert _identity({"status": 200, "elements": [element, element]}) is None


@pytest.mark.parametrize("collect_identity", [False, True])
def test_numeric_homepage_fetch_caches_bound_size_once(monkeypatch, collect_identity):
    requests = _mock_provider(monkeypatch, BODY)
    cache = {}
    for _ in range(2):
        asyncio.run(lead_scorer._fetch_structured_linkedin_profile_once(
            HOME, cache, collect_identity=collect_identity,
        ))
    assert requests == [{"payload": {"url": NUMERIC}}]
    assert cache["structured_evidence"]["url"] == VANITY
    assert cache["structured_evidence"]["employee_count"] == "51-200"
    assert "structured_failure" not in cache
    assert ("structured_company_identity_evidence" in cache) is collect_identity


def test_numeric_profile_refresh_allows_only_completed_negative_finality(monkeypatch):
    requests = _mock_provider(monkeypatch, BODY)

    async def insufficient_size(_url, **_kwargs):
        return {"outcome": linkedin.CURRENT_LINKEDIN_SIZE_INSUFFICIENT_EVIDENCE}

    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", insufficient_size)
    cache = {}
    verdict = {"employee_size_evidence_url": NUMERIC}
    asyncio.run(lead_scorer._refresh_linkedin_employee_size_observation(
        verdict,
        SimpleNamespace(company_name="Acme", company_website="https://acme.example", company_linkedin=NUMERIC),
        SimpleNamespace(employee_count="51-200"),
        verified_homepage_identity=HOME, invocation_cache=cache,
        collect_structured_conflict=True,
    ))
    assert requests == [{"payload": {"url": NUMERIC}}]
    assert cache["refresh_outcome"] == "verified"
    receipt = {
        "gate": "company_evidence_investigation",
        "completed_submitted_findings": True, "failure_reason": "",
        "targets": ["industry", "geography"],
        "claims": {target: {"target": target, "status": "UNPROVEN",
                             "evidence_url": "", "evidence_quote": ""}
                   for target in ("industry", "geography")},
        "usage": {"fetch_outcomes": []},
    }
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("industry", "geography"), investigation_receipt=receipt,
        linkedin_refresh_outcome=cache["refresh_outcome"],
    )
    assert all(claim["status"] == "UNPROVEN" for claim in receipt["claims"].values())
    receipt["completed_submitted_findings"] = False
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("industry", "geography"), investigation_receipt=receipt,
        linkedin_refresh_outcome=cache["refresh_outcome"],
    )


@pytest.mark.parametrize("returned_url", [VANITY, NUMERIC])
@pytest.mark.parametrize("company_id", [None, "67890", True])
def test_fetch_cannot_bypass_invalid_numeric_id_through_exact_url(
    monkeypatch, returned_url, company_id,
):
    requests = _mock_provider(monkeypatch, _body(id=company_id, linkedinUrl=returned_url))
    diagnostic, identity, company_type = {}, {}, {}
    evidence = asyncio.run(linkedin.fetch_structured_linkedin_company_size(
        DOMAIN, NUMERIC, expected_company_name="Acme", diagnostic=diagnostic,
        company_identity_evidence=identity, public_company_evidence=company_type,
    ))
    assert requests == [{"payload": {"url": NUMERIC}}]
    assert evidence is None
    assert not identity and not company_type
    assert diagnostic["failure_reason"] == "source_blocked"


@pytest.mark.parametrize("employee_range", [None, {}, {"start": 51, "end": 500}, {"start": True, "end": 200}])
def test_bound_numeric_identity_does_not_invent_missing_or_malformed_size(monkeypatch, employee_range):
    _mock_provider(monkeypatch, _body(employeeCountRange=employee_range))
    diagnostic, identity = {}, {}
    assert asyncio.run(linkedin.fetch_structured_linkedin_company_size(
        DOMAIN, NUMERIC, expected_company_name="Acme", diagnostic=diagnostic,
        company_identity_evidence=identity,
    )) is None
    assert identity["company_id"] == "12345"
    assert diagnostic["failure_reason"] == "source_blocked"


def test_bound_numeric_identity_keeps_unknown_fit_claims_unproven():
    assert _identity(BODY)
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("industry", "geography"), investigation_receipt={},
        linkedin_refresh_outcome="verified",
    )


def test_normal_vanity_identity_does_not_require_numeric_provider_fields(monkeypatch):
    body = _body(id=None, universalName=None)
    _mock_provider(monkeypatch, body)
    identity = {}
    evidence = asyncio.run(linkedin.fetch_structured_linkedin_company_size(
        DOMAIN, VANITY, company_identity_evidence=identity,
    ))
    assert evidence["employee_count"] == "51-200"
    assert identity["url"] == VANITY
    assert "requested_url" not in identity
