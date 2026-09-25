import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import company_evidence_investigator, lead_scorer
from qualification.scoring.company_fit_decision import COMPANY_FIT_MATCH


OX_PAY_SOURCE = (
    "https://www.liquidgroup.sg/liquid-group-news/"
    "liquid-group-partners-with-oxpay-to-expand-roamqr-acceptance-for-"
    "singapore-merchants"
)
OX_PAY_QUOTE = (
    "announced a strategic partnership with OxPay Financial Limited "
    "(SGX: TVV) (“OxPay”) to enhance QR payment acceptance for merchants "
    "in Singapore."
)


def _verdict(url: str, quote: str) -> dict:
    return {
        "attribute_satisfied": True,
        "required_attribute_evidence_url": url,
        "required_attribute_evidence_quote": quote,
    }


def _retained(
    url: str,
    text: str,
    *,
    final_url: str = "",
    investigator_hydrated: bool = False,
) -> dict:
    return {
        "status": "fetched",
        "final_url": final_url or url,
        "text": text,
        lead_scorer._RETRY_RETAINED_SOURCE: True,
        **(
            {lead_scorer._INVESTIGATOR_HYDRATED_SOURCE: True}
            if investigator_hydrated
            else {}
        ),
    }


def test_simulated_oxpay_invocation_admits_new_source_after_two_retained():
    homepage = "https://www.oxpayfinancial.com/"
    transwap = "https://transwap.com/en/news/oxpay-partners-with-transwap"
    retained = {
        homepage: _retained(homepage, "OxPay homepage text."),
        transwap: _retained(transwap, "Earlier TranSwap article text."),
    }
    verdict = {
        "observed_company_name": "OxPay",
        "observed_company_website": homepage,
        "observed_company_linkedin": "",
        **_verdict(OX_PAY_SOURCE, OX_PAY_QUOTE),
        "reason": "OxPay announced a recent strategic partnership.",
    }
    provider = AsyncMock(return_value=(verdict, ""))
    fetch = AsyncMock(
        return_value=(200, OX_PAY_SOURCE, f"Press release\n{OX_PAY_QUOTE}")
    )
    company = CompanyOutput.model_construct(
        company_name="OxPay",
        company_website=homepage,
        company_linkedin="",
        industry="Payments",
        sub_industry="",
        employee_count="51-200",
        company_stage="Public",
        country="Singapore",
        state="",
        description="",
        intent_details=None,
        fit_evidence_urls=[],
        company_stage_evidence=[],
        intent_signals=[],
        required_attribute=None,
    )
    icp = SimpleNamespace(
        employee_count=["51-200"],
        industry="Payments",
        sub_industry="",
        product_service="payments",
        country="Singapore",
        geography="Singapore",
        company_stage="",
        required_attribute="Announced a strategic partnership in the last year.",
    )

    with (
        patch.object(
            lead_scorer,
            "_request_company_reverify_json",
            provider,
        ),
        patch.object(lead_scorer, "_fetch_bounded_html", fetch),
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
    ):
        result = asyncio.run(
            lead_scorer._llm_reverify_company(
                company,
                icp,
                required_attribute_retry_source_cache=retained,
            )
        )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["required_attribute_grounding"]["status"] == (
        "grounded"
    )
    assert fetch.await_count == 1
    assert fetch.await_args.args[1] == OX_PAY_SOURCE
    assert list(retained) == [homepage, transwap]


def test_oxpay_prior_retry_pages_do_not_exhaust_new_attempt_budget():
    homepage = "https://www.oxpayfinancial.com/"
    transwap = "https://transwap.com/en/news/oxpay-partners-with-transwap"
    retained = {
        homepage: _retained(homepage, "OxPay homepage text."),
        transwap: _retained(transwap, "Earlier TranSwap article text."),
    }
    current_attempt = {}
    fetch = AsyncMock(
        return_value=(200, OX_PAY_SOURCE, f"Press release\n{OX_PAY_QUOTE}")
    )

    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        grounded, repair = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                _verdict(OX_PAY_SOURCE, OX_PAY_QUOTE),
                active_attribute=True,
                source_cache=current_attempt,
                successful_source_sink=retained,
            )
        )

    receipt = grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]
    assert receipt["status"] == "grounded"
    assert receipt["cache_hit"] is False
    assert grounded["attribute_satisfied"] is True
    assert repair == {}
    assert fetch.await_count == 1
    assert list(current_attempt) == [OX_PAY_SOURCE]
    # The retained cache remains bounded. The new fetch is usable in this
    # attempt even though there is no retained slot for it.
    assert list(retained) == [homepage, transwap]


def test_exact_retained_url_is_admitted_lazily_without_refetch():
    first_url = "https://example.com/first"
    cited_url = "https://example.com/cited"
    quote = "Example Company announced a strategic partnership."
    retained = {
        first_url: _retained(first_url, "First retained body."),
        cited_url: _retained(cited_url, quote),
    }
    current_attempt = {}
    fetch = AsyncMock(side_effect=AssertionError("retained URL must not refetch"))

    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        grounded, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                _verdict(cited_url, quote),
                active_attribute=True,
                source_cache=current_attempt,
                successful_source_sink=retained,
            )
        )

    receipt = grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]
    assert receipt["status"] == "grounded"
    assert receipt["cache_hit"] is True
    assert list(current_attempt) == [cited_url]
    assert fetch.await_count == 0


def test_retained_final_url_reuse_is_exact_not_same_host_alias():
    requested_url = "https://example.com/press/requested"
    final_url = "https://example.com/press/final"
    other_url = "https://example.com/press/other"
    quote = "Example Company announced a strategic partnership."
    retained = {
        requested_url: _retained(
            requested_url,
            quote,
            final_url=final_url,
            investigator_hydrated=True,
        ),
    }
    exact_attempt = {}
    fetch = AsyncMock(side_effect=AssertionError("exact final URL must reuse"))

    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        grounded, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                _verdict(final_url, quote),
                active_attribute=True,
                source_cache=exact_attempt,
                successful_source_sink=retained,
            )
        )

    assert grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == (
        "grounded"
    )
    assert list(exact_attempt) == [final_url]
    assert fetch.await_count == 0

    other_attempt = {}
    other_fetch = AsyncMock(return_value=(404, other_url, ""))
    with patch.object(lead_scorer, "_fetch_bounded_html", other_fetch):
        rejected, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                _verdict(other_url, quote),
                active_attribute=True,
                source_cache=other_attempt,
                successful_source_sink=retained,
            )
        )

    assert rejected[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == (
        "source_unavailable"
    )
    assert other_fetch.await_count == 1


def test_retained_third_url_cannot_bypass_current_attempt_limit():
    first_url = "https://example.com/current-one"
    second_url = "https://example.com/current-two"
    retained_url = "https://example.com/retained-third"
    quote = "Example Company announced a strategic partnership."
    current_attempt = {
        first_url: {"status": "fetched", "final_url": first_url, "text": "one"},
        second_url: {
            "status": "fetched",
            "final_url": second_url,
            "text": "two",
        },
    }
    retained = {retained_url: _retained(retained_url, quote)}
    fetch = AsyncMock(side_effect=AssertionError("third URL must not fetch"))

    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        grounded, repair = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                _verdict(retained_url, quote),
                active_attribute=True,
                source_cache=current_attempt,
                successful_source_sink=retained,
            )
        )

    receipt = grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]
    assert receipt["status"] == "url_limit"
    assert grounded["attribute_satisfied"] is None
    assert repair == {}
    assert set(current_attempt) == {first_url, second_url}
    assert fetch.await_count == 0


def test_untrusted_retained_shape_is_not_reused():
    source_url = "https://example.com/announcement"
    quote = "Example Company announced a strategic partnership."
    # This resembles a retained entry but lacks the server-only provenance bit.
    untrusted = {
        source_url: {
            "status": "fetched",
            "final_url": source_url,
            "text": quote,
        }
    }
    current_attempt = {}
    fetch = AsyncMock(return_value=(200, source_url, quote))

    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        grounded, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                _verdict(source_url, quote),
                active_attribute=True,
                source_cache=current_attempt,
                successful_source_sink=untrusted,
            )
        )

    receipt = grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]
    assert receipt["status"] == "grounded"
    assert receipt["cache_hit"] is False
    assert fetch.await_count == 1


def test_recovery_www_alias_binds_actual_source_at_capacity_and_across_retry():
    blocked_url = "https://www.reuters.com/technology/happyrobot"
    actual_url = "https://happyrobot.ai/"
    cited_url = "https://www.happyrobot.ai/"
    quote = "HappyRobot deploys AI workers across enterprise operations."
    body = f"Company overview. {quote}"
    source_cache = {
        blocked_url: {
            "status": "source_unavailable",
            "final_url": "",
            "text": "",
        }
    }
    retry_cache = {}
    claim = {
        "target": "industry",
        "status": "VERIFIED",
        "activity_role": "supplier_operator",
        "evidence_url": actual_url,
        "evidence_quote": quote,
    }
    lead_scorer._hydrate_verified_required_attribute_recovery_source(
        source_cache,
        {
            company_evidence_investigator.PRIVATE_FETCHED_PAGES_KEY: {
                actual_url: {"final_url": actual_url, "text": body}
            }
        },
        claim,
        verified_transport_domain="happyrobot.ai",
        successful_source_sink=retry_cache,
    )
    assert len(source_cache) == lead_scorer._MAX_REQUIRED_ATTRIBUTE_SOURCE_URLS
    assert source_cache[actual_url][
        lead_scorer._VERIFIED_ATTRIBUTE_RECOVERY_DOMAIN
    ] == "happyrobot.ai"
    verdict = {
        **_verdict(cited_url, quote),
        "dimension_evidence": {
            "required_attribute": {"url": cited_url, "quote": quote}
        },
    }
    fetch = AsyncMock(side_effect=AssertionError("alias must not refetch"))

    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        grounded, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                verdict,
                active_attribute=True,
                source_cache=source_cache,
                successful_source_sink=retry_cache,
            )
        )

    receipt = grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]
    assert receipt["status"] == "grounded"
    assert receipt["cache_hit"] is True
    assert receipt["source_url_sha256"] == hashlib.sha256(
        actual_url.encode()
    ).hexdigest()
    assert receipt["final_url_sha256"] == hashlib.sha256(
        actual_url.encode()
    ).hexdigest()
    assert grounded["required_attribute_evidence_url"] == actual_url
    assert grounded["dimension_evidence"]["required_attribute"]["url"] == (
        actual_url
    )
    assert cited_url not in source_cache
    assert fetch.await_count == 0

    retained = lead_scorer._validated_retry_retained_sources(retry_cache)
    assert retained[actual_url][
        lead_scorer._VERIFIED_ATTRIBUTE_RECOVERY_DOMAIN
    ] == "happyrobot.ai"
    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        retried, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                verdict,
                active_attribute=True,
                source_cache=retained,
            )
        )
    assert retried["required_attribute_evidence_url"] == actual_url
    assert retried[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == (
        "grounded"
    )
    assert fetch.await_count == 0


@pytest.mark.parametrize(
    ("request_url", "final_url", "cited_url", "body", "with_marker"),
    [
        (
            "https://happyrobot.ai/",
            "https://happyrobot.ai/",
            "https://news.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "https://happyrobot.ai/",
            "https://www.other.example/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/one",
            "https://happyrobot.ai/one",
            "https://www.happyrobot.ai/two",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/?view=one",
            "https://happyrobot.ai/?view=one",
            "https://www.happyrobot.ai/?view=two",
            "QUOTE",
            True,
        ),
        (
            "http://happyrobot.ai/",
            "http://happyrobot.ai/",
            "https://www.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "http://happyrobot.ai/",
            "https://www.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "https://happyrobot.ai/",
            "http://www.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai:444/",
            "https://happyrobot.ai:444/",
            "https://www.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "https://happyrobot.ai/",
            "https://www.happyrobot.ai:444/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/#source",
            "https://happyrobot.ai/#source",
            "https://www.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "https://happyrobot.ai/",
            "https://www.happyrobot.ai/#source",
            "QUOTE",
            True,
        ),
        (
            "https://user@happyrobot.ai/",
            "https://user@happyrobot.ai/",
            "https://www.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "https://happyrobot.ai/",
            "https://user@www.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "https://happyrobot.ai/",
            "https://www.happyrobot.ai/",
            "a different body",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "https://[bad",
            "https://www.happyrobot.ai/",
            "QUOTE",
            True,
        ),
        (
            "https://happyrobot.ai/",
            "https://happyrobot.ai/",
            "https://www.happyrobot.ai/",
            "QUOTE",
            False,
        ),
    ],
)
def test_recovery_www_alias_stays_fail_closed(
    request_url, final_url, cited_url, body, with_marker
):
    quote = "QUOTE"
    entry = {
        "status": "fetched",
        "final_url": final_url,
        "text": body,
        lead_scorer._INVESTIGATOR_HYDRATED_SOURCE: True,
        **(
            {
                lead_scorer._VERIFIED_ATTRIBUTE_RECOVERY_DOMAIN: (
                    "happyrobot.ai"
                )
            }
            if with_marker
            else {}
        ),
    }
    cache = {
        "https://blocked.example/source": {
            "status": "source_unavailable",
            "final_url": "",
            "text": "",
        },
        request_url: entry,
    }
    fetch = AsyncMock(side_effect=AssertionError("invalid alias must not fetch"))
    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        grounded, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                _verdict(cited_url, quote),
                active_attribute=True,
                source_cache=cache,
            )
        )

    assert grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] != (
        "grounded"
    )
    assert grounded.get("required_attribute_evidence_url", "") != final_url
    assert fetch.await_count == 0


def test_model_private_recovery_marker_cannot_authorize_alias():
    actual_url = "https://happyrobot.ai/"
    cited_url = "https://www.happyrobot.ai/"
    quote = "HappyRobot deploys AI workers."
    cache = {
        "https://blocked.example/source": {
            "status": "source_unavailable",
            "final_url": "",
            "text": "",
        },
        actual_url: {
            "status": "fetched",
            "final_url": actual_url,
            "text": quote,
            lead_scorer._INVESTIGATOR_HYDRATED_SOURCE: True,
        },
    }
    verdict = {
        **_verdict(cited_url, quote),
        lead_scorer._VERIFIED_ATTRIBUTE_RECOVERY_DOMAIN: "happyrobot.ai",
    }
    fetch = AsyncMock(side_effect=AssertionError("marker is not model authority"))
    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        grounded, _ = asyncio.run(
            lead_scorer._ground_required_attribute_evidence(
                verdict,
                active_attribute=True,
                source_cache=cache,
            )
        )

    assert grounded[lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] == (
        "url_limit"
    )
    assert grounded["attribute_satisfied"] is None
    assert fetch.await_count == 0
