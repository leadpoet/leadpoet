import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.qualification.models import CompanyOutput
from lab_arena import scoring as arena_scoring
from qualification.scoring import company_verification, lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
    company_fit_unavailable,
)
from qualification.scoring.competition import (
    COMPANY_VERIFICATION_EXHAUSTED_FAILURE_CLASS,
    REQUIRED_ATTRIBUTE_QUOTE_ABSENT_FAILURE_CLASS,
    scorer_breakdown_has_company_local_verification_failure,
    scorer_breakdown_has_retryable_infrastructure_failure,
)


def _company(**updates):
    values = {
        "company_name": "Example Company",
        "company_website": "https://example.com/",
        "company_linkedin": "",
        "industry": "Software",
        "sub_industry": "",
        "employee_count": "201-500",
        "company_stage": "Series B",
        "country": "United States",
        "state": "",
        "description": "",
        "intent_details": None,
        "fit_evidence_urls": [],
        "company_stage_evidence": [],
        "intent_signals": [],
        "required_attribute": None,
    }
    values.update(updates)
    return CompanyOutput.model_construct(**values)


def _icp(**updates):
    values = {
        "employee_count": ["201-500"],
        "industry": "Software",
        "sub_industry": "",
        "product_service": "",
        "country": "United States",
        "geography": "United States",
        "company_stage": "Series B",
        "required_attribute": "",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_operating_homepage_navigation_label_is_not_a_parked_domain():
    operating_page = """
        <html>
          <title>Example Company | Industrial systems</title>
          <a href="https://www.linkedin.com/company/example-company">LinkedIn</a>
          <nav><a href="/projects">Under construction</a></nav>
          <main>Example Company builds industrial systems for customers.</main>
        </html>
    """

    with patch.object(
        company_verification,
        "_fetch_bounded_html",
        AsyncMock(return_value=(200, "https://example.com/", operating_page)),
    ):
        result = asyncio.run(company_verification.verify_company_exists(
            "Example Company",
            "https://example.com/",
            company_linkedin=(
                "https://www.linkedin.com/company/example-company"
            ),
            require_https_transport=True,
        ))

    assert result.decision == COMPANY_FIT_MATCH


def test_real_parked_domain_marker_remains_a_mismatch():
    with patch.object(
        company_verification,
        "_fetch_bounded_html",
        AsyncMock(return_value=(
            200,
            "https://example.com/",
            "<html><body>This domain is for sale</body></html>",
        )),
    ):
        result = asyncio.run(company_verification.verify_company_exists(
            "Example Company",
            "https://example.com/",
            require_https_transport=True,
        ))

    assert result.decision == COMPANY_FIT_MISMATCH


def test_verified_homepage_linkedin_completes_common_legal_name_observation():
    company = _company(
        company_name="Crusoe",
        company_website="https://crusoe.ai/",
    )
    verdict = {
        "observed_company_name": "Crusoe Technologies Inc.",
        "observed_company_website": "https://www.crusoe.ai/",
        "observed_company_linkedin": "",
    }
    anchor = {
        "normalized_name": "crusoe",
        "registrable_dns_domain": "crusoe.ai",
        "linkedin_company_slug": "crusoe",
    }

    receipt = lead_scorer._web_identity_receipt(
        company,
        verdict,
        verified_homepage_identity=anchor,
        verified_homepage_transport_domain="crusoe.ai",
    )

    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["observed_linkedin_slug"] == "crusoe"
    assert receipt["linkedin_evidence_source"] == "company_homepage"
    assert receipt["web_observed_linkedin_slug"] == ""

    strict_receipt = lead_scorer._web_identity_receipt(
        company,
        verdict,
        verified_homepage_identity=anchor,
        verified_homepage_transport_domain="crusoe.ai",
        company_quality=True,
    )
    assert strict_receipt["decision"] == COMPANY_FIT_UNAVAILABLE


def test_verified_linkedin_redirect_alias_uses_exact_transport_final_url():
    company = _company(
        company_name="Magnitude Biosciences Ltd",
        company_website="https://magnitudebiosciences.com/",
        company_linkedin="https://www.linkedin.com/company/magnitudebiosciences/",
    )
    verdict = {
        "observed_company_name": "Magnitude Biosciences Ltd",
        "observed_company_website": "https://magnitudebiosciences.com/",
        "observed_company_linkedin": (
            "https://uk.linkedin.com/company/magnitude-biosciences"
        ),
    }
    final_url = "https://uk.linkedin.com/company/magnitudebiosciences"

    with patch.object(
        lead_scorer,
        "_fetch_bounded_html",
        AsyncMock(return_value=(999, final_url, "")),
    ) as fetch:
        resolved = asyncio.run(
            lead_scorer._resolve_observed_linkedin_redirect_alias(
                company,
                verdict,
            )
        )

    receipt = lead_scorer._web_identity_receipt(company, resolved)
    assert fetch.await_count == 1
    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["reason_code"] == "verified_linkedin_redirect_alias"
    assert receipt["requested_observed_linkedin_slug"] == "magnitude-biosciences"
    assert receipt["observed_linkedin_slug"] == "magnitudebiosciences"
    assert receipt["observed_linkedin_requested_url"] == verdict[
        "observed_company_linkedin"
    ]
    assert receipt["observed_linkedin_final_url"] == final_url


@pytest.mark.parametrize(
    "final_url",
    [
        "https://uk.linkedin.com/company/magnitude-bio-sciences",
        "https://example.com/company/magnitudebiosciences",
        "http://uk.linkedin.com/company/magnitudebiosciences",
    ],
)
def test_linkedin_redirect_alias_rejects_noncanonical_or_unsafe_final_url(final_url):
    company = _company(
        company_name="Magnitude Biosciences Ltd",
        company_website="https://magnitudebiosciences.com/",
        company_linkedin="https://www.linkedin.com/company/magnitudebiosciences/",
    )
    verdict = {
        "observed_company_name": "Magnitude Biosciences Ltd",
        "observed_company_website": "https://magnitudebiosciences.com/",
        "observed_company_linkedin": (
            "https://uk.linkedin.com/company/magnitude-biosciences"
        ),
    }

    with patch.object(
        lead_scorer,
        "_fetch_bounded_html",
        AsyncMock(return_value=(200, final_url, "")),
    ):
        resolved = asyncio.run(
            lead_scorer._resolve_observed_linkedin_redirect_alias(
                company,
                verdict,
            )
        )

    assert resolved["observed_company_linkedin"] == verdict[
        "observed_company_linkedin"
    ]
    assert lead_scorer._web_identity_receipt(company, resolved)[
        "decision"
    ] == COMPANY_FIT_MISMATCH


def test_model_declared_linkedin_final_url_is_not_trusted():
    company = _company(
        company_name="Magnitude Biosciences Ltd",
        company_website="https://magnitudebiosciences.com/",
        company_linkedin="https://www.linkedin.com/company/magnitudebiosciences/",
    )
    verdict = {
        "observed_company_name": "Magnitude Biosciences Ltd",
        "observed_company_website": "https://magnitudebiosciences.com/",
        "observed_company_linkedin": (
            "https://uk.linkedin.com/company/magnitude-biosciences"
        ),
        lead_scorer._VERIFIED_LINKEDIN_REDIRECT_REQUESTED: (
            "https://uk.linkedin.com/company/magnitude-biosciences"
        ),
        lead_scorer._VERIFIED_LINKEDIN_REDIRECT_FINAL: (
            "https://uk.linkedin.com/company/magnitudebiosciences"
        ),
    }

    with patch.object(
        lead_scorer,
        "_fetch_bounded_html",
        AsyncMock(side_effect=asyncio.TimeoutError),
    ):
        resolved = asyncio.run(
            lead_scorer._resolve_observed_linkedin_redirect_alias(
                company,
                verdict,
            )
        )

    assert lead_scorer._VERIFIED_LINKEDIN_REDIRECT_REQUESTED not in resolved
    assert lead_scorer._VERIFIED_LINKEDIN_REDIRECT_FINAL not in resolved
    assert lead_scorer._web_identity_receipt(company, resolved)[
        "decision"
    ] == COMPANY_FIT_MISMATCH


def _required_attribute_verdict(*, satisfied, url, quote):
    return {
        "observed_company_name": "First Capital REIT",
        "observed_company_website": "https://fcr.ca/",
        "observed_company_linkedin": "",
        "attribute_satisfied": satisfied,
        "required_attribute_evidence_url": url,
        "required_attribute_evidence_quote": quote,
        "reason": "required attribute check",
    }


def test_first_capital_invented_quote_uses_fetched_body_in_normal_repair(
    monkeypatch,
):
    source_url = "https://fcr.ca/news/acquisition-agreement"
    source_text = (
        "First Capital REIT today announced that it entered into an agreement "
        "to be acquired. Upon close of the transaction, the buyers will acquire "
        "First Capital's assets."
    )
    responses = [
        _required_attribute_verdict(
            satisfied=True,
            url=source_url,
            quote="First Capital REIT completed the acquisition.",
        ),
        _required_attribute_verdict(
            satisfied=False,
            url=source_url,
            quote=(
                "First Capital REIT today announced that it entered into an "
                "agreement to be acquired."
            ),
        ),
    ]
    prompts = []

    async def provider(**kwargs):
        prompts.append(kwargs["prompt"])
        return responses.pop(0), ""

    fetch = AsyncMock(return_value=(200, source_url, source_text))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(
            company_name="First Capital REIT",
            company_website="https://fcr.ca/",
        ),
        _icp(
            company_stage="",
            required_attribute="Completed an acquisition in the last 12 months.",
        ),
    ))

    assert result.decision == COMPANY_FIT_MISMATCH
    assert fetch.await_count == 1
    assert len(prompts) == 2
    assert "<untrusted_required_attribute_source>" in prompts[1]
    assert "entered into an agreement" in prompts[1]
    assert result.details["required_attribute_grounding"]["status"] == "grounded"
    assert "fcr.ca" not in str(result.details["required_attribute_grounding"])


def test_valid_required_attribute_announcement_quote_is_grounded(monkeypatch):
    source_url = "https://example.com/announcement"
    quote = "Example Company announced a new regional office in Phoenix."
    provider = AsyncMock(return_value=(
        _required_attribute_verdict(
            satisfied=True,
            url=source_url,
            quote=quote,
        ) | {
            "observed_company_name": "Example Company",
            "observed_company_website": "https://example.com/",
        },
        "",
    ))
    fetch = AsyncMock(return_value=(200, source_url, f"News {quote} Details"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(),
        _icp(
            company_stage="",
            required_attribute="Announced a new office in the last year.",
        ),
    ))

    assert result.decision == COMPANY_FIT_MATCH
    assert provider.await_count == 1
    assert fetch.await_count == 1
    assert result.details["required_attribute_grounding"]["status"] == "grounded"


def test_successful_attribute_body_survives_outer_retry_without_refetch(
    monkeypatch,
):
    source_url = (
        "https://www.businesswire.com/news/home/20260915525333/en/"
        "TypeSafe-AI-Emerges-From-Stealth-With-$40M-in-Funding"
    )
    quote = (
        "TypeSafe AI emerged from stealth with $40 million in seed funding."
    )
    verdict = _required_attribute_verdict(
        satisfied=True,
        url=source_url,
        quote=quote,
    ) | {
        "observed_company_name": "Example Company",
        "observed_company_website": "https://example.com/",
    }
    provider = AsyncMock(return_value=(verdict, ""))
    fetch = AsyncMock(return_value=(200, source_url, f"News {quote} Details"))
    retry_cache = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)

    first = asyncio.run(lead_scorer._llm_reverify_company(
        _company(),
        _icp(company_stage="", required_attribute="Raised seed funding."),
        required_attribute_retry_source_cache=retry_cache,
    ))
    # This boundary represents an unrelated retryable intent-provider failure.
    # The next company invocation must reuse only the prior successful body.
    fetch.side_effect = AssertionError("retained source must not be refetched")
    second = asyncio.run(lead_scorer._llm_reverify_company(
        _company(),
        _icp(company_stage="", required_attribute="Raised seed funding."),
        required_attribute_retry_source_cache=retry_cache,
    ))

    assert first.decision == second.decision == COMPANY_FIT_MATCH
    assert fetch.await_count == 1
    assert retry_cache[source_url]["text"] == f"News {quote} Details"
    assert retry_cache[source_url][lead_scorer._RETRY_RETAINED_SOURCE] is True
    assert "News" not in str(second.receipt("company_fit"))


def test_lab_scorer_retry_reuses_grounded_body_for_final_paragraph(
    monkeypatch,
):
    from qualification.scoring import intent_details as intent_details_module

    source_url = (
        "https://www.businesswire.com/news/home/20260915525333/en/"
        "TypeSafe-AI-Emerges-From-Stealth-With-$40M-in-Funding"
    )
    quote = (
        "TypeSafe AI emerged from stealth with $40 million in seed funding."
    )
    source_body = (
        f"News {quote} The company also reported 50% revenue growth."
    )
    verdict = {
        "observed_company_name": "TypeSafe AI",
        "observed_company_website": "https://typesafe.example/",
        "observed_company_linkedin": (
            "https://www.linkedin.com/company/typesafe-ai"
        ),
        "observed_employee_count": "11-50",
        "employee_size_matches": True,
        "employee_size_evidence_url": source_url,
        "employee_size_evidence_quote": (
            "TypeSafe AI has between 11 and 50 employees."
        ),
        "observed_industry": "Software",
        "observed_subindustry": "SaaS",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": source_url,
        "industry_evidence_quote": "TypeSafe AI builds software.",
        "observed_hq_country": "United States",
        "observed_hq_state": "California",
        "geography_matches": True,
        "geography_evidence_url": source_url,
        "geography_evidence_quote": (
            "TypeSafe AI is headquartered in California, United States."
        ),
        "attribute_satisfied": True,
        "required_attribute_evidence_url": source_url,
        "required_attribute_evidence_quote": quote,
        "reason": "All requested fit dimensions are verified.",
    }
    company = {
        "company_name": "TypeSafe AI",
        "company_website": "https://typesafe.example/",
        "company_linkedin": "https://www.linkedin.com/company/typesafe-ai",
        "industry": "Software",
        "employee_count": "11-50",
        "company_stage": "",
        "country": "United States",
        "state": "California",
        "intent_details": (
            "TypeSafe AI raised seed funding and reported 50% revenue growth."
        ),
        "intent_signals": [{
            "matched_icp_signal": 0,
            "description": "TypeSafe AI announced a completed funding event.",
            "date": "2026-09-15",
            "url": source_url,
        }],
    }
    icp = {
        "icp_id": "typesafe-retry",
        "prompt": "Find software companies with recent funding.",
        "industry": "Software",
        "sub_industry": "SaaS",
        "employee_count": ["11-50"],
        "company_stage": "",
        "geography": "United States",
        "country": "United States",
        "product_service": "software",
        "required_attribute": "Raised seed funding.",
        "intent_signals": ["Announced a completed funding event"],
        "max_companies": 1,
    }
    provider_calls = []
    fetch_calls = []
    intent_calls = 0
    paragraph_contexts = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_match("homepage identity verified")

    async def company_provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return dict(verdict), ""

    async def source_fetch(_session, url, **_kwargs):
        fetch_calls.append(url)
        if len(fetch_calls) > 1:
            raise asyncio.TimeoutError("source is unavailable on retry")
        return 200, url, source_body

    async def intent_score(*_args, **_kwargs):
        nonlocal intent_calls
        intent_calls += 1
        if intent_calls == 1:
            return 0.0, 0.0, 1.0, 0, True, [{
                "raw": 0.0,
                "after_decay": 0.0,
                "matched_icp_signal": 0,
                "judge_verdict": {
                    "decision": "rejected_verifier_error",
                    "pipeline_decision": "unavailable",
                    "error_class": "ProviderTimeout",
                },
            }]
        return 60.0, 60.0, 1.0, 90, False, [{
            "raw": 60.0,
            "after_decay": 60.0,
            "matched_icp_signal": 0,
            "judge_verdict": {
                "decision": "verified",
                "pipeline_decision": "accept",
                "client_ready": True,
            },
        }]

    async def paragraph_review(
        _company, _icp, _signals, _fit_receipt, *, company_source_contexts=None,
    ):
        paragraph_contexts.append(company_source_contexts)
        return {"gate": "intent_details", "decision": COMPANY_FIT_MATCH}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(
        lead_scorer, "_request_company_reverify_json", company_provider
    )
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", source_fetch)
    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent_signal", intent_score
    )
    monkeypatch.setattr(
        intent_details_module, "review_intent_details", paragraph_review
    )
    scorer = arena_scoring.lab_scorer(
        arena_scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2",
            intent_details=True,
        )
    )

    result = arena_scoring.score_work_item(
        {"scored_run_id": "typesafe-source-retry"},
        icp=icp,
        companies=[company],
        scorer=scorer,
        max_retries=2,
    )

    assert result[0]["final_score"] == 60.0
    assert provider_calls == ["lead_scorer_reverify"] * 2
    assert intent_calls == 2
    assert fetch_calls == [source_url]
    assert paragraph_contexts == [[{
        "dimension": "required_attribute",
        "url": source_url,
        "text": source_body,
    }]]
    assert source_body not in str(result)


def test_failed_attribute_fetch_is_not_retained_across_outer_retry(monkeypatch):
    source_url = "https://example.com/announcement"
    quote = "Example Company announced a new regional office."
    verdict = {
        "attribute_satisfied": True,
        "required_attribute_evidence_url": source_url,
        "required_attribute_evidence_quote": quote,
    }
    retry_cache = {}
    first_local_cache = {}

    with patch.object(
        lead_scorer,
        "_fetch_bounded_html",
        AsyncMock(side_effect=asyncio.TimeoutError),
    ):
        first, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                verdict,
                active_attribute=True,
                source_cache=first_local_cache,
                successful_source_sink=retry_cache,
            )
        )

    assert first[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == (
        "source_unavailable"
    )
    assert retry_cache == {}

    with patch.object(
        lead_scorer,
        "_fetch_bounded_html",
        AsyncMock(return_value=(200, source_url, quote)),
    ):
        second, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                verdict,
                active_attribute=True,
                source_cache=lead_scorer._validated_retry_retained_sources(
                    retry_cache
                ),
                successful_source_sink=retry_cache,
            )
        )

    assert second[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "grounded"
    assert retry_cache[source_url]["status"] == "fetched"


def test_retry_retained_sources_fail_closed_on_provenance_and_bounds():
    source_url = "https://example.com/announcement"
    valid_entry = {
        "status": "fetched",
        "final_url": source_url,
        "text": "Example Company announced a new regional office.",
        lead_scorer._RETRY_RETAINED_SOURCE: True,
    }

    assert lead_scorer._validated_retry_retained_sources(
        {source_url: valid_entry}
    ) == {source_url: valid_entry}
    public_http_url = "http://example.com/announcement"
    public_http_entry = {
        **valid_entry,
        "final_url": public_http_url,
    }
    assert lead_scorer._validated_retry_retained_sources(
        {public_http_url: public_http_entry}
    ) == {public_http_url: public_http_entry}
    assert lead_scorer._validated_retry_retained_sources({
        source_url: {key: value for key, value in valid_entry.items()
                     if key != lead_scorer._RETRY_RETAINED_SOURCE}
    }) == {}
    assert lead_scorer._validated_retry_retained_sources({
        "http://127.0.0.1/private": {
            **valid_entry,
            "final_url": "http://127.0.0.1/private",
        }
    }) == {}
    assert lead_scorer._validated_retry_retained_sources({
        "https://example.com": {
            **valid_entry,
            "final_url": "https://example.com",
        }
    }) == {}
    assert lead_scorer._validated_retry_retained_sources({
        source_url: {
            **valid_entry,
            "text": "x" * (lead_scorer.MAX_PAGE_CHARACTERS + 1),
        }
    }) == {}
    assert lead_scorer._validated_retry_retained_sources({
        f"https://example.com/{index}": {
            **valid_entry,
            "final_url": f"https://example.com/{index}",
        }
        for index in range(lead_scorer._MAX_REQUIRED_ATTRIBUTE_SOURCE_URLS + 1)
    }) == {}


def test_final_matched_attribute_body_is_private_paragraph_context_only():
    source_url = "https://example.com/announcement"
    quote = "Example Company announced a new regional office."
    body = f"News {quote} Revenue grew 50% year over year."
    retry_cache = {}
    lead_scorer._retain_successful_required_attribute_source(
        retry_cache,
        source_url,
        {"status": "fetched", "final_url": source_url, "text": body},
    )
    company_fit = company_fit_match(details={
        "required_attribute_decision": COMPANY_FIT_MATCH,
        "supporting_receipts": [{
            "gate": "required_attribute_source",
            "status": "grounded",
        }],
        "dimension_evidence": {
            "required_attribute": {
                "decision": COMPANY_FIT_MATCH,
                "web_evidence": {"url": source_url, "quote": quote},
            }
        },
    })

    context = lead_scorer._matched_required_attribute_source_context(
        company_fit,
        retry_cache,
    )

    assert context == {"url": source_url, "text": body}
    assert body not in str(company_fit.receipt("company_fit"))
    missing_quote = company_fit_match(details={
        **company_fit.details,
        "dimension_evidence": {
            "required_attribute": {
                "decision": COMPANY_FIT_MATCH,
                "web_evidence": {
                    "url": source_url,
                    "quote": "A claim absent from the fetched body.",
                },
            }
        },
    })
    assert lead_scorer._matched_required_attribute_source_context(
        missing_quote,
        retry_cache,
    ) is None

    public_http_url = "http://example.com/announcement"
    public_http_cache = {}
    lead_scorer._retain_successful_required_attribute_source(
        public_http_cache,
        public_http_url,
        {"status": "fetched", "final_url": public_http_url, "text": body},
    )
    public_http_fit = company_fit_match(details={
        **company_fit.details,
        "dimension_evidence": {
            "required_attribute": {
                "decision": COMPANY_FIT_MATCH,
                "web_evidence": {"url": public_http_url, "quote": quote},
            }
        },
    })
    assert lead_scorer._matched_required_attribute_source_context(
        public_http_fit,
        public_http_cache,
    ) is None


def test_final_attribute_context_allows_only_trusted_exact_final_url_alias():
    requested_url = "https://example.com/news/announcement"
    final_url = "https://example.com/news/announcement/"
    quote = "Example Company announced a new regional office."
    body = f"News {quote} Revenue grew 50% year over year."
    company_fit = company_fit_match(details={
        "required_attribute_decision": COMPANY_FIT_MATCH,
        "supporting_receipts": [{
            "gate": "required_attribute_source",
            "status": "grounded",
        }],
        "dimension_evidence": {
            "required_attribute": {
                "decision": COMPANY_FIT_MATCH,
                "web_evidence": {"url": final_url, "quote": quote},
            }
        },
    })
    retained = {
        requested_url: {
            "status": "fetched",
            "final_url": final_url,
            "text": body,
            lead_scorer._RETRY_RETAINED_SOURCE: True,
            lead_scorer._INVESTIGATOR_HYDRATED_SOURCE: True,
        }
    }

    assert lead_scorer._matched_required_attribute_source_context(
        company_fit,
        retained,
    ) == {"url": final_url, "text": body}
    retained[requested_url].pop(
        lead_scorer._INVESTIGATOR_HYDRATED_SOURCE
    )
    assert lead_scorer._matched_required_attribute_source_context(
        company_fit,
        retained,
    ) is None


def test_final_linkedin_context_requires_matched_quote_and_identity_slug():
    source_url = "https://www.linkedin.com/company/common-wealth"
    quote = "Company size\n11-50 employees"
    body = f"## About us\nCommon Wealth provides retirement software.\n{quote}"
    company_fit = company_fit_match(details={
        "dimension_evidence": {
            "employee_size": {
                "decision": COMPANY_FIT_MATCH,
                "submitted_decision": COMPANY_FIT_MATCH,
                "observed_decision": COMPANY_FIT_MATCH,
                "web_evidence": {"url": source_url, "quote": quote},
            },
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "web_identity_receipt": {
                    "decision": COMPANY_FIT_MATCH,
                    "observed_linkedin_slug": "common-wealth",
                },
            },
        },
    })
    candidate = {"url": source_url, "text": body}

    assert lead_scorer._matched_linkedin_profile_source_context(
        company_fit,
        candidate,
    ) == candidate
    wrong_slug = company_fit_match(details={
        "dimension_evidence": {
            **company_fit.details["dimension_evidence"],
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "web_identity_receipt": {
                    "decision": COMPANY_FIT_MATCH,
                    "observed_linkedin_slug": "other-company",
                },
            },
        },
    })
    assert lead_scorer._matched_linkedin_profile_source_context(
        wrong_slug,
        candidate,
    ) is None
    assert lead_scorer._matched_linkedin_profile_source_context(
        company_fit,
        {"url": source_url, "text": "The exact final quote is absent."},
    ) is None


def test_company_source_contexts_are_bounded_and_dimension_ordered():
    attribute_url = "https://example.com/funding"
    attribute_quote = "Example Company raised a seed round."
    profile_url = "https://www.linkedin.com/company/example-company"
    profile_quote = "Company size\n11-50 employees"
    company_fit = company_fit_match(details={
        "required_attribute_decision": COMPANY_FIT_MATCH,
        "supporting_receipts": [{
            "gate": "required_attribute_source",
            "status": "grounded",
        }],
        "dimension_evidence": {
            "required_attribute": {
                "decision": COMPANY_FIT_MATCH,
                "web_evidence": {
                    "url": attribute_url,
                    "quote": attribute_quote,
                },
            },
            "employee_size": {
                "decision": COMPANY_FIT_MATCH,
                "submitted_decision": COMPANY_FIT_MATCH,
                "observed_decision": COMPANY_FIT_MATCH,
                "web_evidence": {
                    "url": profile_url,
                    "quote": profile_quote,
                },
            },
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "web_identity_receipt": {
                    "decision": COMPANY_FIT_MATCH,
                    "observed_linkedin_slug": "example-company",
                },
            },
        },
    })
    attribute_cache = {}
    lead_scorer._retain_successful_required_attribute_source(
        attribute_cache,
        attribute_url,
        {
            "status": "fetched",
            "final_url": attribute_url,
            "text": f"News {attribute_quote}",
        },
    )
    profile_candidate = {
        "url": profile_url,
        "text": f"## About\n{profile_quote}",
    }

    assert lead_scorer._matched_company_source_contexts(
        company_fit,
        attribute_cache,
        profile_candidate,
    ) == [
        {
            "dimension": "required_attribute",
            "url": attribute_url,
            "text": f"News {attribute_quote}",
        },
        {
            "dimension": "employee_size",
            "url": profile_url,
            "text": f"## About\n{profile_quote}",
        },
    ]


def test_required_attribute_source_outage_is_unavailable_and_cached(monkeypatch):
    source_url = "https://example.com/announcement"
    verdict = _required_attribute_verdict(
        satisfied=True,
        url=source_url,
        quote="Example Company announced a new office.",
    ) | {
        "observed_company_name": "Example Company",
        "observed_company_website": "https://example.com/",
        "dimension_evidence": {
            "required_attribute": {
                "url": source_url,
                "quote": "Stale nested quote must not survive.",
            }
        },
    }
    provider = AsyncMock(side_effect=[(dict(verdict), ""), (dict(verdict), "")])
    fetch = AsyncMock(side_effect=asyncio.TimeoutError)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(),
        _icp(company_stage="", required_attribute="Announced a new office."),
    ))

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert provider.await_count == 2
    assert fetch.await_count == 1
    grounding = result.details["required_attribute_grounding"]
    assert grounding["status"] == "source_unavailable"
    assert grounding["cache_hit"] is True
    assert result.details["failure_reason_code"] == "provider_error"
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
    assert result.details["dimension_evidence"]["required_attribute"] == {
        "url": "",
        "quote": "",
    }
    assert scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })


def test_repeated_ungrounded_attribute_quote_is_retryable_company_local(
    monkeypatch,
):
    source_url = "https://example.com/announcement"
    verdict = _required_attribute_verdict(
        satisfied=True,
        url=source_url,
        quote="Example Company completed the acquisition.",
    ) | {
        "observed_company_name": "Example Company",
        "observed_company_website": "https://example.com/",
    }
    provider = AsyncMock(side_effect=[(dict(verdict), ""), (dict(verdict), "")])
    fetch = AsyncMock(return_value=(
        200,
        source_url,
        "Example Company entered into an acquisition agreement.",
    ))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(),
        _icp(company_stage="", required_attribute="Completed an acquisition."),
    ))

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert provider.await_count == 2
    assert fetch.await_count == 1
    assert result.details["failure_class"] == (
        REQUIRED_ATTRIBUTE_QUOTE_ABSENT_FAILURE_CLASS
    )
    assert "failure_reason_code" not in result.details
    assert result.details["required_attribute_grounding"]["status"] == (
        "quote_absent"
    )
    assert result.details["dimension_evidence"]["required_attribute"] == {
        "url": "",
        "quote": "",
    }
    breakdown = {"verifier_gate_receipts": [result.receipt("company_fit")]}
    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)
    assert scorer_breakdown_has_company_local_verification_failure(breakdown)


def _complete_attribute_verdict(source_url):
    return {
        "observed_company_name": "Example Company",
        "observed_company_website": "https://example.com/",
        "observed_company_linkedin": "",
        "observed_employee_count": "201-500",
        "employee_size_matches": True,
        "employee_size_evidence_url": "https://example.com/about",
        "employee_size_evidence_quote": "Example Company has 250 employees.",
        "observed_industry": "Software",
        "observed_subindustry": "Business software",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://example.com/about",
        "industry_evidence_quote": "Example Company builds business software.",
        "observed_hq_country": "United States",
        "observed_hq_state": "California",
        "geography_matches": True,
        "geography_evidence_url": "https://example.com/about",
        "geography_evidence_quote": "Headquartered in California.",
        "observed_company_stage": "",
        "stage_matches": None,
        "stage_evidence_url": "",
        "stage_evidence_quote": "",
        "attribute_satisfied": True,
        "required_attribute_evidence_url": source_url,
        "required_attribute_evidence_quote": (
            "Example Company completed the acquisition."
        ),
        "reason": "required attribute check",
    }


def test_quote_absent_does_not_mask_retryable_linkedin_refresh(monkeypatch):
    source_url = "https://example.com/announcement"
    verdict = _complete_attribute_verdict(source_url)
    provider = AsyncMock(side_effect=[(dict(verdict), ""), (dict(verdict), "")])
    fetch = AsyncMock(return_value=(
        200,
        source_url,
        "Example Company entered into an acquisition agreement.",
    ))

    async def retryable_refresh(
        value,
        _company_value,
        _icp_value,
        *,
        invocation_cache,
        **_kwargs,
    ):
        invocation_cache["refresh_outcome"] = "retryable_failure"
        unavailable = dict(value)
        unavailable.update(
            observed_employee_count=None,
            employee_size_matches=None,
            employee_size_evidence_url="",
            employee_size_evidence_quote="",
        )
        return unavailable

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        retryable_refresh,
    )

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(),
        _icp(company_stage="", required_attribute="Completed an acquisition."),
        require_company_fit_dimensions=True,
    ))

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
    assert result.details["failure_reason_code"] == "malformed_response"
    assert scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })


def test_quote_absent_does_not_mask_retryable_homepage_identity(monkeypatch):
    source_url = "https://example.com/announcement"
    verdict = _required_attribute_verdict(
        satisfied=True,
        url=source_url,
        quote="Example Company completed the acquisition.",
    ) | {
        "observed_company_name": "Example Company",
        "observed_company_website": "https://example.com/",
    }
    provider = AsyncMock(side_effect=[(dict(verdict), ""), (dict(verdict), "")])
    fetch = AsyncMock(return_value=(
        200,
        source_url,
        "Example Company entered into an acquisition agreement.",
    ))

    def unresolved_identity(*_args, **_kwargs):
        return {
            "decision": COMPANY_FIT_UNAVAILABLE,
            "reason_code": "identity_not_proven",
            "evidence_source": "company_web_reverification",
            "submitted_name": "example company",
            "submitted_domain": "example.com",
            "submitted_linkedin_slug": "",
            "observed_name": "example company",
            "observed_domain": "example.com",
            "observed_linkedin_slug": "",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(lead_scorer, "_web_identity_receipt", unresolved_identity)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(),
        _icp(company_stage="", required_attribute="Completed an acquisition."),
        verified_homepage_identity=company_fit_unavailable(
            "website returned HTTP 502"
        ),
    ))

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
    assert result.details["failure_reason_code"] == "malformed_response"
    assert scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })


@pytest.mark.parametrize(
    ("failure_reason", "error"),
    [
        ("malformed_response", "provider response JSON was malformed"),
        ("provider_error", "provider HTTP 502"),
    ],
)
def test_required_attribute_request_failures_remain_retryable(
    monkeypatch,
    failure_reason,
    error,
):
    async def failed_request(**kwargs):
        kwargs["diagnostic"][lead_scorer.VERIFIER_FAILURE_REASON_KEY] = (
            failure_reason
        )
        return None, error

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        lead_scorer,
        "_request_company_reverify_json",
        failed_request,
    )

    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(),
        _icp(company_stage="", required_attribute="Completed an acquisition."),
    ))

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_reason_code"] == failure_reason
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
    assert scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })


def _quote_absent_web_result(identity_decision):
    return company_fit_unavailable(
        "required attribute quote absent after repair",
        details={
            "identity_decision": identity_decision,
            "identity_receipt": {
                "decision": identity_decision,
                "reason_code": (
                    "identity_not_proven"
                    if identity_decision == COMPANY_FIT_UNAVAILABLE
                    else ""
                ),
                "evidence_source": "company_web_reverification",
                "submitted_name": "example company",
                "submitted_domain": "example.com",
                "submitted_linkedin_slug": "",
                "observed_name": "example company",
                "observed_domain": "example.com",
                "observed_linkedin_slug": "",
            },
            "dimension_decisions": {
                "employee_size": COMPANY_FIT_MATCH,
                "industry": COMPANY_FIT_MATCH,
                "geography": COMPANY_FIT_MATCH,
                "stage": COMPANY_FIT_MATCH,
            },
            "dimension_evidence": {
                dimension: {
                    "url": f"https://example.com/{dimension}",
                    "quote": f"Example Company {dimension} evidence.",
                }
                for dimension in ("employee_size", "industry", "geography")
            } | {"required_attribute": {"url": "", "quote": ""}},
            "required_attribute_decision": COMPANY_FIT_UNAVAILABLE,
            "required_attribute_grounding": {"status": "quote_absent"},
            "failure_class": REQUIRED_ATTRIBUTE_QUOTE_ABSENT_FAILURE_CLASS,
        },
    )


def test_full_company_fit_receipt_preserves_quote_absent_class(monkeypatch):
    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_match("homepage identity verified")

    async def web(*_args, **_kwargs):
        return _quote_absent_web_result(COMPANY_FIT_MATCH)

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web)

    result = asyncio.run(lead_scorer._verify_company_fit(
        _company(),
        _icp(company_stage="", required_attribute="Completed an acquisition."),
        0.0,
        0.0,
        set(),
        require_https_transport=True,
    ))
    receipt = result.receipt("company_fit")
    breakdown = {"verifier_gate_receipts": [receipt]}

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert receipt["failure_class"] == (
        REQUIRED_ATTRIBUTE_QUOTE_ABSENT_FAILURE_CLASS
    )
    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)
    assert scorer_breakdown_has_company_local_verification_failure(breakdown)


def test_full_company_fit_drops_quote_class_for_retryable_homepage(monkeypatch):
    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable("website returned HTTP 502")

    async def web(*_args, **_kwargs):
        return _quote_absent_web_result(COMPANY_FIT_UNAVAILABLE)

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web)

    result = asyncio.run(lead_scorer._verify_company_fit(
        _company(),
        _icp(company_stage="", required_attribute="Completed an acquisition."),
        0.0,
        0.0,
        set(),
        require_https_transport=True,
    ))
    receipt = result.receipt("company_fit")
    breakdown = {"verifier_gate_receipts": [receipt]}

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert receipt.get("failure_class") != (
        REQUIRED_ATTRIBUTE_QUOTE_ABSENT_FAILURE_CLASS
    )
    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)
    assert not scorer_breakdown_has_company_local_verification_failure(breakdown)


def _score_work_item_companies():
    return [
        {"company_name": name, "employee_count": "201-500"}
        for name in ("Accepted One", "Just Ice Tea", "Accepted Two")
    ]


def _scored_breakdown(score):
    return {
        "final_score": score,
        "verifier_gate_receipts": [
            {"gate": "company_fit", "decision": "match"}
        ],
    }


def _quote_absent_breakdown():
    return {
        "final_score": 0.0,
        "failure_reason": (
            "Company fit unavailable: unproven dimensions: required_attribute"
        ),
        "verifier_gate_receipts": [{
            "gate": "company_fit",
            "decision": "unavailable",
            "failure_class": REQUIRED_ATTRIBUTE_QUOTE_ABSENT_FAILURE_CLASS,
        }],
    }


def test_score_work_item_recovers_quote_absent_company_on_outer_retry():
    calls = []

    def scorer(invoked, _icp_value, _is_reference_model):
        names = [company["company_name"] for company in invoked]
        calls.append(names)
        if len(calls) == 1:
            return [
                _scored_breakdown(40.0),
                _quote_absent_breakdown(),
                _scored_breakdown(55.0),
            ]
        assert names == ["Just Ice Tea"]
        return [_scored_breakdown(54.0)]

    result = arena_scoring.score_work_item(
        {"scored_run_id": "quote-absent-recovers"},
        icp={"employee_count": ["201-500"], "max_companies": 3},
        companies=_score_work_item_companies(),
        scorer=scorer,
        max_retries=3,
    )

    assert calls == [
        ["Accepted One", "Just Ice Tea", "Accepted Two"],
        ["Just Ice Tea"],
    ]
    assert [row["final_score"] for row in result] == [40.0, 54.0, 55.0]


def test_score_work_item_exhausts_only_persistent_quote_absent_company():
    calls = []

    def scorer(invoked, _icp_value, _is_reference_model):
        names = [company["company_name"] for company in invoked]
        calls.append(names)
        if len(calls) == 1:
            return [
                _scored_breakdown(40.0),
                _quote_absent_breakdown(),
                _scored_breakdown(55.0),
            ]
        assert names == ["Just Ice Tea"]
        return [_quote_absent_breakdown()]

    result = arena_scoring.score_work_item(
        {"scored_run_id": "quote-absent-exhausted"},
        icp={"employee_count": ["201-500"], "max_companies": 3},
        companies=_score_work_item_companies(),
        scorer=scorer,
        max_retries=3,
    )

    assert calls == [
        ["Accepted One", "Just Ice Tea", "Accepted Two"],
        ["Just Ice Tea"],
        ["Just Ice Tea"],
    ]
    assert [row["final_score"] for row in result] == [40.0, 0.0, 55.0]
    assert result[1]["verifier_gate_receipts"][0]["failure_class"] == (
        COMPANY_VERIFICATION_EXHAUSTED_FAILURE_CLASS
    )
    assert not scorer_breakdown_has_retryable_infrastructure_failure(result[1])


def test_score_work_item_mixed_provider_failure_still_errors():
    calls = []

    def scorer(invoked, _icp_value, _is_reference_model):
        names = [company["company_name"] for company in invoked]
        calls.append(names)
        rows = []
        for name in names:
            if name == "Accepted One":
                rows.append(_scored_breakdown(40.0))
            elif name == "Just Ice Tea":
                rows.append(_quote_absent_breakdown())
            else:
                rows.append({
                    "final_score": 0.0,
                    "failure_reason": "Company fit unavailable: provider HTTP 502",
                    "verifier_gate_receipts": [{
                        "gate": "company_fit",
                        "decision": "unavailable",
                        "failure_reason_code": "provider_error",
                    }],
                })
        return rows

    with pytest.raises(arena_scoring.ScoringError):
        arena_scoring.score_work_item(
            {"scored_run_id": "quote-absent-with-provider-failure"},
            icp={"employee_count": ["201-500"], "max_companies": 3},
            companies=_score_work_item_companies(),
            scorer=scorer,
            max_retries=3,
        )

    assert calls == [
        ["Accepted One", "Just Ice Tea", "Accepted Two"],
        ["Just Ice Tea", "Accepted Two"],
        ["Just Ice Tea", "Accepted Two"],
    ]


def test_required_attribute_grounding_cache_bounds_distinct_urls():
    cache = {}
    fetched_urls = []

    async def fetch(_session, url):
        fetched_urls.append(url)
        return 200, url, "Example Company announced a new office."

    async def run(url):
        verdict = {
            "attribute_satisfied": True,
            "required_attribute_evidence_url": url,
            "required_attribute_evidence_quote": (
                "Example Company announced a new office."
            ),
        }
        return await lead_scorer._ground_required_attribute_evidence(
            verdict,
            active_attribute=True,
            source_cache=cache,
        )

    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        first, _ = asyncio.run(run("https://one.example/news"))
        repeated, _ = asyncio.run(run("https://one.example/news"))
        second, _ = asyncio.run(run("https://two.example/news"))
        limited, _ = asyncio.run(run("https://three.example/news"))

    assert len(fetched_urls) == 2
    assert first[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "grounded"
    assert repeated[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["cache_hit"] is True
    assert second[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "grounded"
    assert limited[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == "url_limit"
    assert limited["attribute_satisfied"] is None


def test_inactive_required_attribute_never_fetches_source():
    fetch = AsyncMock(side_effect=AssertionError("source fetch must not run"))
    verdict = {
        "attribute_satisfied": True,
        "required_attribute_evidence_url": "https://example.com/news",
        "required_attribute_evidence_quote": "Example Company announced news.",
        lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING: {"status": "grounded"},
    }

    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        resolved, repair_source = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                verdict,
                active_attribute=False,
                source_cache={},
            )
        )

    assert fetch.await_count == 0
    assert repair_source == {}
    assert lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING not in resolved


@pytest.mark.parametrize(
    ("observed", "quote", "supported"),
    [
        (
            "acquired",
            "Vibe.co is a Walmart company, acquired by Walmart in August 2026.",
            True,
        ),
        (
            "acquired",
            "Vibe.co announced that it will be acquired by Walmart, subject to closing.",
            False,
        ),
        (
            "acquired",
            "Walmart invested in Vibe.co during its Series B round.",
            False,
        ),
        (
            "public",
            "Acme is a subsidiary of NYSE-listed Parent Corp.",
            False,
        ),
        (
            "acquired",
            "Acme is now a wholly owned subsidiary of Parent Corp.",
            True,
        ),
        (
            "series c+",
            "AliveCor announced that it has closed a Series F financing round.",
            True,
        ),
        (
            "series c+",
            "Forus announced that it has closed its Series C financing.",
            True,
        ),
        (
            "series c+",
            "|5. Later Stage VC (Series C)|14-Apr-2016|Completed|",
            False,
        ),
        (
            "series c+",
            "Privately Held. Backed by leading investors.",
            False,
        ),
    ],
)
def test_stage_evidence_keeps_current_ownership_and_completed_round_distinctions(
    observed, quote, supported
):
    assert lead_scorer._stage_quote_supports_observation(
        observed, quote
    ) is supported


@pytest.mark.parametrize(
    ("observed", "quote"),
    [
        (
            "series c+",
            "We're excited to announce a Series E funding round of $100 million.",
        ),
        (
            "series c+",
            "Devoted Health is a Series F company based in Eagan. "
            "Series F, Jan 30, 2026, $317M.",
        ),
        (
            "series f",
            "Anduril's Series F funding round that raised $1.5 billion.",
        ),
        (
            "series a",
            "A $10.3M Series A funding round has just closed for Learnosity.",
        ),
    ],
)
def test_saved_completed_round_word_orders_are_supported(observed, quote):
    assert lead_scorer._stage_quote_supports_observation(observed, quote) is True


@pytest.mark.parametrize(
    ("observed", "quote"),
    [
        (
            "series c+",
            "We're excited to announce plans for a Series E funding round.",
        ),
        ("series c+", "Devoted Health is not a Series F company."),
        (
            "series c+",
            "Anduril's Series F funding round could raise $1.5 billion.",
        ),
        ("series a", "A Series A funding round has not closed for Learnosity."),
    ],
)
def test_new_round_word_orders_reject_uncompleted_or_negated_claims(observed, quote):
    assert lead_scorer._stage_quote_supports_observation(observed, quote) is False


def test_saved_completed_round_word_orders_match_series_c_plus_icp():
    cases = [
        (
            "Spring Health",
            "Series C+",
            "We're excited to announce a Series E funding round of $100 million.",
        ),
        (
            "Devoted Health",
            "Series C+",
            "Devoted Health is a Series F company based in Eagan. "
            "Series F, Jan 30, 2026, $317M.",
        ),
        (
            "Anduril Industries",
            "Series F",
            "Anduril's Series F funding round that raised $1.5 billion.",
        ),
    ]
    for name, observed, quote in cases:
        verdict = {
            "observed_company_name": name,
            "observed_company_stage": observed,
            "stage_matches": True,
            "stage_evidence_url": "https://example.com/funding",
            "stage_evidence_quote": quote,
        }
        assert lead_scorer._decision_from_observed_stage(
            verdict,
            "series c+",
            company=_company(company_name=name, company_stage=observed),
        ) == COMPANY_FIT_MATCH


def test_learnosity_round_syntax_does_not_override_current_acquisition():
    verdict = {
        "observed_company_name": "Learnosity",
        "observed_company_stage": "Series A",
        "stage_matches": True,
        "stage_evidence_url": "https://learnosity.com/series-a",
        "stage_evidence_quote": (
            "A $10.3M Series A funding round has just closed for Learnosity."
        ),
        "required_attribute_evidence_url": "https://learnosity.com/company",
        "required_attribute_evidence_quote": (
            "Learnosity was acquired by Leeds Equity Partners in January 2025."
        ),
    }

    assert lead_scorer._decision_from_observed_stage(
        verdict,
        "series a",
        company=_company(
            company_name="Learnosity",
            company_website="https://learnosity.com/",
            company_stage="Series A",
        ),
    ) == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize(
    "requested",
    ["seed", "series a", "series b", "series c+", "private equity", "public"],
)
def test_acquired_observation_never_auto_matches_an_allowed_icp_stage(requested):
    assert lead_scorer._company_stage_matches("acquired", requested) is False


@pytest.mark.parametrize(
    ("quote", "supported"),
    [
        ("Walmart completed the acquisition of Vibe.", True),
        ("Walmart completed the acquisition of Another Company.", False),
        ("Walmart announced that it will acquire Vibe.", False),
        ("Walmart completed an investment in Vibe.", False),
        ("Walmart completed the acquisition of a minority stake in Vibe.", False),
        ("Walmart completed the acquisition of Vibe's minority shares.", False),
        ("Walmart has not completed the acquisition of Vibe.", False),
        ("Walmart will have completed the acquisition of Vibe.", False),
        (
            "Walmart completed the acquisition of Vibe subject to final approval.",
            False,
        ),
        ("Vibe was formerly a subsidiary of Walmart.", False),
        ("Vibe was acquired by Walmart but is now independent.", False),
        (
            "Another Company was acquired by Walmart. "
            "Vibe has not been acquired by Walmart.",
            False,
        ),
        (
            "Walmart completed the acquisition of Vibe. Vibe later went public.",
            False,
        ),
        (
            "Walmart completed the acquisition of Vibe. Vibe is now listed on NYSE.",
            False,
        ),
    ],
)
def test_completed_acquisition_proof_is_bound_to_the_exact_subject(
    quote, supported
):
    assert lead_scorer._acquired_stage_quote_supports_company(
        _company(company_name="Vibe", company_website="https://vibe.co/"),
        "Vibe.co",
        quote,
    ) is supported


def test_exact_completed_acquisition_stage_contradicts_an_older_venture_round():
    company = _company(
        company_name="Vibe",
        company_website="https://vibe.co/",
        company_stage="Series B",
        industry="Advertising",
    )
    icp = _icp(
        industry="Advertising",
        company_stage="Series B",
        required_attribute=(
            "Sells advertising technology and has a recent public signal of "
            "product launch, market expansion, or partnership."
        ),
    )
    verdict = {
        "observed_company_name": "Vibe",
        "observed_company_website": "https://vibe.co/",
        "observed_company_linkedin": "https://linkedin.com/company/vibedotco",
        "observed_employee_count": "201-500",
        "employee_size_matches": True,
        "employee_size_evidence_url": "https://linkedin.com/company/vibedotco",
        "employee_size_evidence_quote": "Company size: 201-500 employees.",
        "observed_industry": "Advertising",
        "observed_subindustry": "Connected TV advertising platform",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://vibe.co/",
        "industry_evidence_quote": "Vibe operates a connected TV ad platform.",
        "observed_hq_country": "United States",
        "observed_hq_state": "New York",
        "geography_matches": True,
        "geography_evidence_url": "https://vibe.co/about",
        "geography_evidence_quote": "Vibe is headquartered in New York.",
        "observed_company_stage": "Acquired",
        "stage_matches": False,
        "stage_evidence_url": "https://corporate.example/acquisition-complete",
        "stage_evidence_quote": "Walmart completed the acquisition of Vibe.",
        "attribute_satisfied": True,
        "required_attribute_evidence_url": "https://vibe.co/company",
        "required_attribute_evidence_quote": (
            "Vibe is a Walmart company, acquired by Walmart in August 2026."
        ),
    }

    result = lead_scorer._reverify_decision(
        verdict,
        icp.required_attribute,
        "series b",
        icp=icp,
        company=company,
    )

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MISMATCH
    assert result.details["dimension_evidence"]["stage"] == {
        "url": "https://corporate.example/acquisition-complete",
        "quote": "Walmart completed the acquisition of Vibe.",
    }


@pytest.mark.parametrize(
    ("url", "quote", "conflicts"),
    [
        (
            "https://vibe.co/llms.txt",
            "Vibe.co is a Walmart company, acquired by Walmart in August 2026.",
            True,
        ),
        (
            "https://vibe.co/llms.txt",
            "Another Company was acquired by Walmart in August 2026.",
            False,
        ),
        (
            "https://news.example/acquisition",
            "Vibe.co was acquired by Walmart in August 2026.",
            False,
        ),
        (
            "https://vibe.co/llms.txt",
            "Walmart made a strategic minority investment in Vibe.co.",
            False,
        ),
    ],
)
def test_first_party_acquisition_only_invalidates_a_conflicting_venture_stage(
    url, quote, conflicts
):
    verdict = {
        "observed_company_name": "Vibe.co",
        "required_attribute_evidence_url": url,
        "required_attribute_evidence_quote": quote,
    }

    assert lead_scorer._first_party_acquisition_conflicts_with_venture_stage(
        _company(company_name="Vibe", company_website="https://vibe.co/"),
        verdict,
        "series b",
    ) is conflicts


def test_stale_venture_stage_with_bound_current_owner_proof_requires_repair():
    verdict = {
        "observed_company_name": "Vibe.co",
        "observed_company_stage": "Series B",
        "stage_matches": True,
        "stage_evidence_url": "https://news.example/series-b",
        "stage_evidence_quote": (
            "Vibe.co announced that it has closed $50 million in Series B financing."
        ),
        "required_attribute_evidence_url": "https://vibe.co/llms.txt",
        "required_attribute_evidence_quote": (
            "Vibe.co is a Walmart company, acquired by Walmart in August 2026."
        ),
    }

    decision = lead_scorer._decision_from_observed_stage(
        verdict,
        "series b",
        company=_company(company_name="Vibe", company_website="https://vibe.co/"),
    )

    assert decision == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        (
            "Bain Capital completed its take-private acquisition of Envestnet.",
            COMPANY_FIT_UNAVAILABLE,
        ),
        (
            "Envestnet has been taken private by Bain Capital.",
            COMPANY_FIT_UNAVAILABLE,
        ),
        (
            "Envestnet is no longer publicly listed after the transaction closed.",
            COMPANY_FIT_UNAVAILABLE,
        ),
        (
            "Bain Capital completes acquisition of Envestnet.",
            COMPANY_FIT_MATCH,
        ),
        (
            "Envestnet will be taken private if the proposed transaction closes.",
            COMPANY_FIT_MATCH,
        ),
        (
            "Another Company has been taken private. Envestnet remains listed.",
            COMPANY_FIT_MATCH,
        ),
    ],
)
def test_public_stage_reopens_only_for_bound_completed_take_private_or_delisting(
    quote, expected
):
    verdict = {
        "observed_company_name": "Envestnet",
        "observed_company_stage": "Public",
        "stage_matches": True,
        "stage_evidence_url": "https://envestnet.com/old-listing",
        "stage_evidence_quote": "Envestnet, Inc. (NYSE: ENV) is publicly traded.",
        "required_attribute_evidence_url": "https://envestnet.com/transaction",
        "required_attribute_evidence_quote": quote,
    }

    assert lead_scorer._decision_from_observed_stage(
        verdict,
        "public",
        company=_company(
            company_name="Envestnet",
            company_website="https://envestnet.com/",
            company_stage="Public",
        ),
    ) == expected


def test_investigator_receipt_cannot_bypass_acquired_subject_binding():
    quote = "Another Company was acquired by Walmart."
    verdict = {
        "observed_company_name": "Vibe.co",
        "observed_company_stage": "Acquired",
        "stage_matches": False,
        "stage_evidence_url": "https://vibe.co/company",
        "stage_evidence_quote": quote,
    }
    finding = {
        "target": "stage",
        "status": "CONTRADICTED",
        "observed_value": "Acquired",
        "evidence_url": "https://vibe.co/company",
        "evidence_quote": quote,
    }

    decision = lead_scorer._decision_from_observed_stage(
        verdict,
        "series b",
        validated_stage_finding=finding,
        company=_company(company_name="Vibe", company_website="https://vibe.co/"),
    )

    assert decision == COMPANY_FIT_UNAVAILABLE


def test_company_research_prompt_selects_full_activity_and_current_stage(
    monkeypatch,
):
    prompts = []

    async def provider(**kwargs):
        prompts.append(kwargs["prompt"])
        return {
            "observed_company_name": "Example Company",
            "observed_company_website": "https://example.com/",
            "observed_company_linkedin": (
                "https://linkedin.com/company/example-company"
            ),
            "observed_employee_count": "201-500",
            "employee_size_matches": True,
            "employee_size_evidence_url": (
                "https://linkedin.com/company/example-company"
            ),
            "employee_size_evidence_quote": "Company size: 201-500.",
            "observed_industry": "AI infrastructure",
            "observed_subindustry": "Electrical equipment manufacturing",
            "industry_matches": True,
            "industry_activity_role": "supplier_operator",
            "industry_evidence_url": "https://example.com/manufacturing",
            "industry_evidence_quote": (
                "We manufacture switchgear and industrial controls."
            ),
            "observed_hq_country": "United States",
            "observed_hq_state": "New York",
            "geography_matches": True,
            "geography_evidence_url": "https://example.com/about",
            "geography_evidence_quote": (
                "Example Company is headquartered in New York."
            ),
            "observed_company_stage": "Acquired",
            "stage_matches": False,
            "stage_evidence_url": "https://parent.example/acquisition",
            "stage_evidence_quote": (
                "Parent completed the acquisition of Example Company."
            ),
            "attribute_satisfied": True,
            "required_attribute_evidence_url": (
                "https://example.com/manufacturing"
            ),
            "required_attribute_evidence_quote": (
                "Example Company opened a new switchgear factory."
            ),
        }, ""

    async def keep_observation(verdict, *_args, **_kwargs):
        return verdict

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        lead_scorer, "_request_company_reverify_json", provider
    )
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_observation,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_fetch_bounded_html",
        AsyncMock(return_value=(
            200,
            "https://example.com/manufacturing",
            "Example Company opened a new switchgear factory.",
        )),
    )
    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(industry="Hardware"),
        _icp(
            industry="Hardware",
            required_attribute=(
                "Designs physical hardware and has a recent facility opening."
            ),
        ),
        require_company_fit_dimensions=True,
    ))

    assert result.decision == COMPANY_FIT_MISMATCH
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "broad positioning label as exclusive" in prompt
    assert "Prefer that direct full-body activity quote" in prompt
    assert "Score this attribute independently from employee size" in prompt
    assert "When one page discusses multiple companies" in prompt
    assert "Do not stop when you find a venture round" in prompt
    assert "Check chronology before selecting evidence" in prompt
    assert "completed acquisition/current parent" in prompt
    assert "first-party completed-round announcement and its full body" in prompt
    assert "Public, or Acquired" in prompt
