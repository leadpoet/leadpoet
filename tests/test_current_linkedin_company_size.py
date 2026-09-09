from __future__ import annotations

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
)
from qualification.scoring import linkedin_company_size
from qualification.scoring.competition import (
    scorer_breakdown_has_retryable_infrastructure_failure,
)


def _company(*, linkedin: str = "https://linkedin.com/company/acme") -> CompanyOutput:
    return CompanyOutput(
        company_name="Acme",
        company_website="https://acme.example.com",
        company_linkedin=linkedin,
        industry="Software",
        employee_count="11-50",
        country="United States",
        intent_signals=[
            {
                "description": "Acme launched a product.",
                "source": "news",
                "url": "https://acme.example.com/news",
                "date": "2026-08-01",
                "snippet": "Acme launched a product.",
            }
        ],
    )


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="current-linkedin-size",
        prompt="test",
        industry="Software",
        sub_industry="SaaS",
        employee_count="11-50",
        company_stage="",
        geography="United States",
        country="United States",
        product_service="Workflow software",
    )


def _verdict(
    *,
    observed_size: object = "1",
    size_matches: object = False,
    employee_url: str = "https://www.linkedin.com/company/acme",
) -> dict:
    return {
        "observed_company_name": "Acme",
        "observed_company_website": "https://acme.example.com/about",
        "observed_company_linkedin": "https://linkedin.com/company/acme",
        "observed_employee_count": observed_size,
        "employee_size_matches": size_matches,
        "employee_size_evidence_url": employee_url,
        "employee_size_evidence_quote": "1 employee on an old profile copy.",
        "observed_industry": "Software",
        "observed_subindustry": "SaaS",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://acme.example.com/product",
        "industry_evidence_quote": "Acme provides workflow software.",
        "observed_hq_country": "United States",
        "observed_hq_state": "",
        "geography_matches": True,
        "geography_evidence_url": "https://acme.example.com/about",
        "geography_evidence_quote": "Acme is headquartered in the United States.",
        "reason": "Independent sources support the observations.",
    }


def _homepage_anchor():
    return company_fit_match(
        "homepage identity verified",
        details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "acme",
                "observed_domain": "acme.example.com",
                "observed_linkedin_slug": "acme",
            }
        },
    )


def test_literal_company_size_is_bounded_to_about_section():
    text = """# Acme

## About us

Cloud workflow software.

Company size 11-50 employees

## Employees at Acme

View all 89 employees
Company size 51-200 employees
"""

    assert linkedin_company_size.extract_linkedin_company_size(text) == {
        "employee_count": "11-50",
        "quote": "Company size 11-50 employees",
    }
    assert linkedin_company_size.extract_linkedin_company_size(
        "## About us\nView all 89 employees\n## Employees at Acme"
    ) is None


def test_exa_contents_request_is_uncached_and_bound_to_returned_url(monkeypatch):
    calls = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "results": [
                    {
                        "url": "https://linkedin.com/company/acme/",
                        "text": "## About\n\nCompany size:\n\n11-50 employees",
                    }
                ],
                "statuses": [{"status": "success"}],
            }

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    result = asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://www.linkedin.com/company/acme"
        )
    )

    assert result == {
        "employee_count": "11-50",
        "quote": "Company size:\n\n11-50 employees",
        "url": "https://linkedin.com/company/acme/",
    }
    assert len(calls) == 1
    assert calls[0][0] == "https://api.exa.ai/contents"
    assert calls[0][1]["json"] == {
        "ids": ["https://www.linkedin.com/company/acme"],
        "text": {"maxCharacters": 4000},
        "maxAgeHours": 0,
    }
    assert calls[0][1]["headers"]["x-api-key"] == "test-exa-key"


def test_invalid_requested_profile_url_never_calls_exa(monkeypatch):
    class UnexpectedSession:
        def __init__(self, **_kwargs):
            raise AssertionError("invalid profile URL must fail before transport")

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(
        linkedin_company_size.aiohttp,
        "ClientSession",
        UnexpectedSession,
    )

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/in/acme"
        )
    ) is None


@pytest.mark.parametrize(
    "body",
    [
        {"error": "upstream unavailable", "results": []},
        {
            "statuses": [
                {
                    "status": "error",
                    "error": {
                        "httpStatusCode": 504,
                        "tag": "CRAWL_LIVECRAWL_TIMEOUT",
                    },
                }
            ],
            "results": [],
        },
        {
            "status": "error",
            "results": [
                {
                    "url": "https://linkedin.com/company/acme",
                    "text": "## About\nCompany size 11-50 employees",
                }
            ],
        },
        {
            "statuses": [{"status": "error"}],
            "results": [
                {
                    "url": "https://linkedin.com/company/acme",
                    "text": "## About\nCompany size 11-50 employees",
                }
            ],
        },
        {
            "statuses": [{"status": "success"}],
            "results": [
                {
                    "status": "failed",
                    "error": "stale content retained",
                    "url": "https://linkedin.com/company/acme",
                    "text": "## About\nCompany size 11-50 employees",
                }
            ],
        },
        {
            "statuses": [{"status": "success"}],
            "results": [
                {
                    "url": "https://linkedin.com/company/other",
                    "text": "## About\nCompany size 11-50 employees",
                }
            ],
        },
    ],
)
def test_exa_contents_envelope_or_source_failure_is_unavailable(monkeypatch, body):
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return body

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme"
        )
    ) is None


def test_successful_exact_profile_without_company_size_is_insufficient(
    monkeypatch,
):
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "statuses": [{"status": "success", "source": "crawled"}],
                "results": [
                    {
                        "url": "https://linkedin.com/company/acme",
                        "text": "Acme builds workflow software.\nView all employees",
                    }
                ],
            }

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme"
        )
    ) == {
        "outcome": "insufficient_evidence",
        "url": "https://linkedin.com/company/acme",
    }


@pytest.mark.parametrize(
    ("status_id", "expected"),
    [
        (
            "https://linkedin.com/company/acme",
            {
                "outcome": "insufficient_evidence",
                "url": "https://linkedin.com/company/acme",
            },
        ),
        ("https://linkedin.com/company/other", None),
    ],
)
def test_exact_profile_not_found_requires_requested_identity(
    monkeypatch,
    status_id,
    expected,
):
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "statuses": [
                    {
                        "id": status_id,
                        "status": "error",
                        "error": {
                            "httpStatusCode": 404,
                            "tag": "CRAWL_NOT_FOUND",
                        },
                    }
                ],
                "results": [],
            }

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme"
        )
    ) == expected


def test_exact_profile_empty_text_is_insufficient(monkeypatch):
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "statuses": [{"status": "success", "source": "crawled"}],
                "results": [
                    {
                        "url": "https://linkedin.com/company/acme",
                        "text": "",
                    }
                ],
            }

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme"
        )
    ) == {
        "outcome": "insufficient_evidence",
        "url": "https://linkedin.com/company/acme",
    }


@pytest.mark.parametrize(
    ("sonar_size", "sonar_matches", "current_size", "expected"),
    [
        ("1", False, "11-50", COMPANY_FIT_MATCH),
        ("11-50", True, "51-200", COMPANY_FIT_MISMATCH),
        ("1-10", True, "11-50", COMPANY_FIT_MATCH),
        (None, None, "51-200", COMPANY_FIT_MISMATCH),
    ],
)
def test_current_profile_replaces_stale_linkedin_match_or_mismatch(
    monkeypatch,
    sonar_size,
    sonar_matches,
    current_size,
    expected,
):
    fetches = []

    async def provider(**_kwargs):
        return _verdict(
            observed_size=sonar_size,
            size_matches=sonar_matches,
        ), ""

    async def fetch(url):
        fetches.append(url)
        return {
            "employee_count": current_size,
            "url": url,
            "quote": f"Company size {current_size} employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == expected
    assert result.details["dimension_decisions"]["employee_size"] == expected
    assert result.details["provider_observations"]["observed_employee_count"] == (
        current_size
    )
    assert result.details["dimension_evidence"]["employee_size"] == {
        "url": "https://www.linkedin.com/company/acme",
        "quote": f"Company size {current_size} employees",
    }
    assert fetches == ["https://www.linkedin.com/company/acme"]


def test_failed_refresh_clears_stale_size_and_is_reused_on_schema_repair(monkeypatch):
    provider_calls = []
    fetches = []
    original_verdict = _verdict()

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return original_verdict, ""

    async def fetch(url):
        fetches.append(url)
        return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["provider_observations"]["observed_employee_count"] is None
    assert result.details["dimension_evidence"]["employee_size"] == {
        "url": "",
        "quote": "",
    }
    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert fetches == ["https://www.linkedin.com/company/acme"]
    assert result.details["failure_class"] == (
        "employee_size_verification_failed"
    )
    assert original_verdict["observed_employee_count"] == "1"
    assert original_verdict["employee_size_matches"] is False


@pytest.mark.parametrize("repair_kind", ["invalid_citation", "unbound_identity"])
def test_actual_profile_failure_survives_an_unusable_repair(
    monkeypatch,
    repair_kind,
):
    initial = _verdict()
    repaired = _verdict()
    if repair_kind == "invalid_citation":
        repaired["employee_size_evidence_url"] = (
            "https://www.linkedin.com/in/acme-employee"
        )
    else:
        repaired["observed_company_name"] = "Acme Holdings"
    verdicts = [initial, repaired]
    fetches = []

    async def provider(**_kwargs):
        return verdicts.pop(0), ""

    async def fetch(url):
        fetches.append(url)
        return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(linkedin=""),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert fetches == ["https://www.linkedin.com/company/acme"]
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == (
        "employee_size_verification_failed"
    )


@pytest.mark.parametrize(
    "employee_url",
    [
        "https://www.linkedin.com/company/dewaofficial",
        "https://www.linkedin.com/in/dewa-http-careers-dewa-gov-ae-0a579332",
    ],
)
def test_same_domain_dewa_alias_is_insufficient_without_a_profile_failure(
    monkeypatch,
    employee_url,
):
    company = _company(linkedin="").model_copy(
        update={
            "company_name": "Dubai Electricity and Water Authority",
            "company_website": "https://dewa.gov.ae/",
        }
    )
    verdict = _verdict(
        observed_size="5,001-10,000",
        size_matches=True,
        employee_url=employee_url,
    )
    verdict.update(
        observed_company_name="Dubai Electricity & Water Authority - DEWA",
        observed_company_website="https://dewa.gov.ae",
        observed_company_linkedin=(
            "https://www.linkedin.com/company/dewaofficial"
        ),
        employee_size_evidence_quote="Company size 5,001-10,000 employees",
        observed_company_stage="",
        stage_matches=None,
        stage_evidence_url="",
        stage_evidence_quote="",
    )
    provider_calls = []

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return verdict, ""

    async def unexpected_fetch(_url):
        raise AssertionError("an unbound profile must not be fetched")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        unexpected_fetch,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            company,
            _icp().model_copy(update={"company_stage": "Series A"}),
            require_company_fit_dimensions=True,
        )
    )
    receipt = result.receipt("company_fit")

    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert result.details["identity_decision"] == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["dimension_decisions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert not scorer_breakdown_has_retryable_infrastructure_failure(
        {"verifier_gate_receipts": [receipt]}
    )


def test_identity_insufficient_path_does_not_hide_malformed_evidence():
    complete_alias_receipt = {
        "decision": "unavailable",
        "reason_code": "identity_not_proven",
        "evidence_source": "company_web_reverification",
        "submitted_name": "dubaielectricityandwaterauthority",
        "submitted_domain": "dewa.gov.ae",
        "submitted_linkedin_slug": "",
        "observed_name": "dubaielectricitywaterauthoritydewa",
        "observed_domain": "dewa.gov.ae",
        "observed_linkedin_slug": "dewaofficial",
    }

    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity", "employee_size"),
        linkedin_refresh_outcome="insufficient_evidence",
        identity_receipt={
            "decision": "unavailable",
            "reason_code": "identity_observation_type_invalid",
        },
    )
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity", "employee_size", "industry"),
        linkedin_refresh_outcome="insufficient_evidence",
        identity_receipt=complete_alias_receipt,
    )


def test_direct_size_repair_clears_an_earlier_invalid_linkedin_citation(
    monkeypatch,
):
    initial = _verdict(
        employee_url="https://www.linkedin.com/in/acme-employee",
    )
    repaired = _verdict(
        observed_size="11-50",
        size_matches=True,
        employee_url="https://acme.example.com/about",
    )
    repaired["employee_size_evidence_quote"] = "Acme has 11-50 employees."
    verdicts = [initial, repaired]
    provider_calls = []

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return verdicts.pop(0), ""

    async def unexpected_fetch(_url):
        raise AssertionError("neither citation is a LinkedIn company profile")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        unexpected_fetch,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert result.decision == COMPANY_FIT_MATCH
    assert "failure_class" not in result.details


@pytest.mark.parametrize(
    ("observed_size", "size_matches"),
    [
        ("1", False),
        ("1-10", True),
        (None, None),
    ],
)
def test_successful_profile_without_size_is_reused_as_insufficient(
    monkeypatch,
    observed_size,
    size_matches,
):
    provider_calls = []
    fetches = []

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return _verdict(
            observed_size=observed_size,
            size_matches=size_matches,
        ), ""

    async def fetch(url):
        fetches.append(url)
        return {
            "outcome": "insufficient_evidence",
            "url": url,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert fetches == ["https://www.linkedin.com/company/acme"]


def test_repair_reuses_successful_refresh_and_non_linkedin_evidence_is_unchanged(
    monkeypatch,
):
    verdicts = [_verdict(), _verdict()]
    verdicts[0]["industry_matches"] = None
    fetches = []

    async def provider(**_kwargs):
        return verdicts.pop(0), ""

    async def fetch(url):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )
    assert result.decision == COMPANY_FIT_MATCH
    assert fetches == ["https://www.linkedin.com/company/acme"]

    async def direct_provider(**_kwargs):
        return _verdict(
            observed_size="11-50",
            size_matches=True,
            employee_url="https://acme.example.com/about",
        ), ""

    async def unexpected_fetch(_url):
        raise AssertionError("non-LinkedIn evidence must retain the existing path")

    monkeypatch.setattr(
        lead_scorer,
        "_request_company_reverify_json",
        direct_provider,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        unexpected_fetch,
    )
    direct = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )
    assert direct.decision == COMPANY_FIT_MATCH
    assert direct.details["dimension_evidence"]["employee_size"]["url"] == (
        "https://acme.example.com/about"
    )


def test_submitted_linkedin_url_alone_cannot_authorize_profile_fetch(monkeypatch):
    verdict = _verdict()
    verdict["observed_company_linkedin"] = ""
    fetches = []

    async def provider(**_kwargs):
        return verdict, ""

    async def fetch(url):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert fetches == []


def test_wrong_homepage_linkedin_anchor_cannot_authorize_profile_fetch(monkeypatch):
    fetches = []

    async def provider(**_kwargs):
        return _verdict(), ""

    async def fetch(url):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    wrong_anchor = _homepage_anchor()
    wrong_anchor.details["identity"]["observed_linkedin_slug"] = "other"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=wrong_anchor,
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert fetches == []


def test_current_size_does_not_override_wrong_observed_company_identity(monkeypatch):
    fetches = []
    verdict = _verdict()
    verdict.update(
        observed_company_name="Other",
        observed_company_website="https://other.example.com",
        observed_company_linkedin="https://linkedin.com/company/other",
    )

    async def provider(**_kwargs):
        return verdict, ""

    async def fetch(url):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["identity_receipt"]["reason_code"] == "identity_mismatch"
    assert fetches == ["https://www.linkedin.com/company/acme"]


def test_complete_sonar_identity_can_bind_profile_without_homepage_anchor(monkeypatch):
    fetches = []

    async def provider(**_kwargs):
        return _verdict(), ""

    async def fetch(url):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(linkedin=""),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert fetches == ["https://www.linkedin.com/company/acme"]
