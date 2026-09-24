import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.qualification.models import CompanyOutput
from qualification.scoring import lead_scorer
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
