"""Public Academy facts and synthetic hostile identity controls; no live calls."""

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_verification, lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
)


PROFILE = "https://www.linkedin.com/company/academy-sports-and-outdoors"
CORPORATE = "https://corporate.academy.com/"
NAVIGATION = [{"url": CORPORATE, "label": "About Us"}]


def _company():
    return CompanyOutput(
        company_name="Academy Sports + Outdoors",
        company_website="https://academy.com/",
        company_linkedin="",
        industry="Commerce and Shopping",
        employee_count="10,001+",
        company_stage="Public",
        country="United States",
        state="Texas",
        intent_signals=[{
            "description": "Academy opened new stores.",
            "source": "company_website",
            "url": "https://investors.academy.com/news/expansion",
            "date": "2026-08-01",
            "snippet": "Academy opened new stores.",
        }],
    )


def _verdict():
    return {
        "observed_company_name": "Academy Sports + Outdoors",
        "observed_company_website": CORPORATE,
        "observed_company_linkedin": PROFILE,
        "observed_employee_count": None,
        "employee_size_matches": None,
        "employee_size_evidence_url": "",
        "employee_size_evidence_quote": "",
        "observed_industry": "Commerce and Shopping",
        "observed_subindustry": "Retail",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": CORPORATE,
        "industry_evidence_quote": (
            "Academy Sports + Outdoors is one of the nation's largest "
            "sporting goods and outdoor stores."
        ),
        "observed_hq_country": "United States",
        "observed_hq_state": "Texas",
        "geography_matches": True,
        "geography_evidence_url": CORPORATE,
        "geography_evidence_quote": "Academy is headquartered out of Katy, TX.",
        "observed_company_stage": "Public",
        "stage_matches": True,
        "stage_evidence_url": "https://investors.academy.com/",
        "stage_evidence_quote": "Academy Sports + Outdoors (NASDAQ: ASO)",
    }


def _profile(**updates):
    return {
        "name": "Academy Sports + Outdoors",
        "provider": "harvestapi_get_company",
        "source_field": "name",
        "url": PROFILE,
        "website": "https://academy.com/",
        "provider_website_url": "http://www.academy.com/",
        **updates,
    }


def _receipt(company=None, verdict=None, navigation=NAVIGATION, profile=None):
    return lead_scorer._web_identity_receipt(
        company or _company(),
        verdict or _verdict(),
        verified_homepage_transport_domain="academy.com",
        verified_homepage_navigation_locators=navigation,
        verified_structured_identity=profile,
        company_quality=True,
    )


def test_official_child_identity_requires_profile_before_and_after_rebrand_review():
    assert _receipt()["decision"] != COMPANY_FIT_MATCH
    receipt = _receipt(profile=_profile())
    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["reason_code"] == "structured_official_child_identity_verified"
    assert receipt["raw_observed_domain"] == "corporate.academy.com"
    assert receipt["observed_domain"] == "academy.com"
    assert receipt["submitted_linkedin_slug"] == ""
    assert receipt["observed_linkedin_slug"] == "academy-sports-and-outdoors"
    reviewed = lead_scorer._web_identity_receipt(
        _company(), _verdict(),
        verified_homepage_transport_domain="academy.com",
        verified_homepage_navigation_locators=NAVIGATION,
        verified_structured_identity=_profile(),
        verified_rebrand_identity={"status": "UNPROVEN"},
        company_quality=True,
    )
    assert reviewed["decision"] == COMPANY_FIT_MATCH


@pytest.mark.parametrize("profile", [
    None, {},
    _profile(name="Different Academy"),
    _profile(website="https://corporate.academy.com/"),
    {key: value for key, value in _profile().items() if key != "provider_website_url"},
    _profile(provider_website_url="https://corporate.academy.com/"),
    _profile(provider_website_url="https://academy.com:444/"),
    _profile(provider_website_url="https://user:pass@academy.com/"),
    _profile(website="https://foreign.example/"),
    _profile(url="https://www.linkedin.com/company/different-academy"),
    _profile(provider="model"),
    _profile(source_field="description"),
    {**_profile(), "extra": "untrusted"},
])
def test_official_child_identity_rejects_missing_or_conflicting_profile(profile):
    assert _receipt(profile=profile)["decision"] != COMPANY_FIT_MATCH


@pytest.mark.parametrize("navigation", [
    [],
    [{"url": "https://investors.academy.com/", "label": "About Us"}],
    [{"url": "https://corporate.academy.com.evil.test/", "label": "About Us"}],
    [{"url": CORPORATE, "label": "Customer"}],
    [{"url": "http://corporate.academy.com/", "label": "About Us"}],
])
def test_official_child_identity_rejects_missing_company_navigation(navigation):
    assert _receipt(navigation=navigation, profile=_profile())["decision"] != COMPANY_FIT_MATCH


@pytest.mark.parametrize("company_update,verdict_update", [
    ({}, {"observed_company_name": "Different Academy"}),
    ({}, {"observed_company_website": "https://corporate.foreign.example/"}),
    ({"company_website": "https://investors.academy.com/"}, {}),
    ({"company_linkedin": "https://www.linkedin.com/company/different-academy"}, {}),
])
def test_official_child_identity_rejects_company_or_claim_conflict(company_update, verdict_update):
    assert _receipt(
        company=_company().model_copy(update=company_update),
        verdict={**_verdict(), **verdict_update},
        profile=_profile(),
    )["decision"] != COMPANY_FIT_MATCH


@pytest.mark.parametrize("root,child", [
    ("github.io", "academy.github.io"),
    ("academy.github.io", "corporate.academy.github.io"),
    ("co.uk", "academy.co.uk"),
])
def test_official_child_lookup_rejects_public_suffix_or_hosted_tenant(root, child):
    company = _company().model_copy(update={"company_website": f"https://{root}/"})
    identity = lead_scorer._web_identity_receipt(company, {
        **_verdict(), "observed_company_website": f"https://{child}/",
    })
    assert not lead_scorer._alias_unresolved_structured_profile_lookup(
        identity, root,
        homepage_navigation_locators=[{"url": f"https://{child}/", "label": "About Us"}],
    )


@pytest.mark.parametrize("transport", ["", "foreign.example", "corporate.academy.com"])
def test_official_child_identity_rejects_missing_or_conflicting_root_transport(transport):
    receipt = lead_scorer._web_identity_receipt(
        _company(), _verdict(), verified_homepage_transport_domain=transport,
        verified_homepage_navigation_locators=NAVIGATION,
        verified_structured_identity=_profile(), company_quality=True,
    )
    assert receipt["decision"] != COMPANY_FIT_MATCH


@pytest.mark.parametrize("case", [
    "verified", "missing_navigation", "wrong_profile_name", "wrong_profile_slug",
    "wrong_profile_website", "profile_child_website", "profile_port", "profile_credentials",
])
def test_missing_homepage_linkedin_retains_navigation_and_recovers_actual_fit(monkeypatch, case):
    """The fetched root link authorizes one existing typed profile lookup."""
    async def homepage_fetch(_session, url):
        assert url == "https://academy.com/"
        return 200, "https://www.academy.com/", (
            "<title>Academy Sports + Outdoors</title>"
            + ('' if case == "missing_navigation" else
               '<a href="https://corporate.academy.com/">About Us</a>')
        )

    monkeypatch.setattr(company_verification, "_fetch_bounded_html", homepage_fetch)
    navigation, pages = [], {}
    homepage = asyncio.run(company_verification.verify_company_exists(
        _company().company_name, _company().company_website,
        require_https_transport=True, company_quality=True,
        homepage_navigation_locator_sink=navigation,
        homepage_evidence_sink=pages,
    ))
    assert homepage.decision == COMPANY_FIT_UNAVAILABLE
    assert navigation == ([] if case == "missing_navigation" else NAVIGATION)
    assert pages
    calls = []

    async def web(**_kwargs):
        return _verdict(), ""

    element = {
        "name": "Academy Sports + Outdoors", "website": "http://www.academy.com/",
        "linkedinUrl": PROFILE, "companyType": "Public Company",
        "employeeCountRange": {"start": 10001, "end": None},
    }
    updates = {
        "wrong_profile_name": {"name": "Different Academy"},
        "wrong_profile_slug": {"linkedinUrl": "https://www.linkedin.com/company/other"},
        "wrong_profile_website": {"website": "https://foreign.example/"},
        "profile_child_website": {"website": CORPORATE},
        "profile_port": {"website": "https://academy.com:444/"},
        "profile_credentials": {"website": "https://user:pass@academy.com/"},
    }
    element.update(updates.get(case, {}))

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return self.body

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, *, json, headers):
            calls.append((url, json))
            if url.endswith("/harvestapi_get_company/execute"):
                assert json == {"payload": {"url": PROFILE}}
                return Response({"status": "completed", "result": {
                    "data": {"status": 200, "element": element},
                }})
            assert url == "https://api.exa.ai/contents"
            assert json["ids"] == [PROFILE]
            return Response({"results": [{"url": PROFILE, "text": (
                "Academy Sports + Outdoors\nAbout us\n"
                "Website\nhttp://www.academy.com/\nCompany size\n10,001+ employees"
            )}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DEEPLINE_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", web)
    monkeypatch.setattr(company_verification.aiohttp, "ClientSession", Session)
    icp = ICPPrompt(
        icp_id="public-retailer", prompt="test", industry="Commerce and Shopping",
        sub_industry="Retail", employee_count="10,001+", company_stage="Public",
        geography="United States", country="United States",
        product_service="Sporting goods retailer",
    )
    result = asyncio.run(lead_scorer._verify_company_fit(
        _company(), icp, 0, 0, set(), require_https_transport=True, company_quality=True,
    ))
    structured_calls = [url for url, _payload in calls if url.endswith("/harvestapi_get_company/execute")]
    assert len(structured_calls) == (0 if case == "missing_navigation" else 1)
    if case != "verified":
        assert result.decision != COMPANY_FIT_MATCH
        return
    assert len(calls) == 2  # Existing typed-profile and current-size paths, once each.
    assert result.decision == COMPANY_FIT_MATCH, result.receipt("company_fit")
    identity = result.details["dimension_evidence"]["identity"]["web_identity_receipt"]
    assert identity["reason_code"] == "structured_official_child_identity_verified"
    assert result.details["company_fit_dimensions"]["employee_size"] == COMPANY_FIT_MATCH
