"""Public-stage rereview must reconsider a still-open compound attribute."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as investigator
from tests.product_coverage_fixtures import native_product_post


STAGE_URL = "https://www.tenable.com/news/current-results"
STAGE_QUOTE = "Tenable Holdings, Inc. (NASDAQ: TENB), today announced quarterly results."
PRODUCT_URL = "https://www.tenable.com/products/exposure-management"
PRODUCT_QUOTE = "Tenable sells exposure management software to enterprise customers."
ATTRIBUTE = "Sells exposure management software to enterprises and trades on a public stock exchange"


def _finding(target, status="UNPROVEN", *, url="", quote="", role="unresolved"):
    return {
        "target": target, "status": status,
        "observed_value": "Public" if target == "stage" and status == "VERIFIED" else None,
        "activity_role": role, "evidence_url": url, "evidence_quote": quote,
        "reason": "Evaluate the complete requested fact from fetched company sources.",
    }


def _verified_attribute(*, quote=PRODUCT_QUOTE, role="supplier_operator"):
    return {
        **_finding("required_attribute", "VERIFIED", url=PRODUCT_URL, quote=quote, role=role),
        "supporting_evidence_url_1": STAGE_URL,
        "supporting_evidence_quote_1": STAGE_QUOTE,
    }


@pytest.mark.parametrize("case", ["valid", "invalid_quote", "missing_capability", "customer_use", "delisted"])
def test_public_stage_rereview_includes_open_attribute_without_auto_upgrade(monkeypatch, case):
    stage_text = STAGE_QUOTE
    if case == "delisted":
        stage_text += " Tenable common stock ceased trading on Nasdaq after a completed delisting."
    product_text = (
        "Tenable uses another vendor's exposure management software internally."
        if case == "customer_use" else PRODUCT_QUOTE
    )
    if case == "missing_capability":
        product_text = "Tenable provides general technology services."
    first = [_finding("stage"), _finding("required_attribute")]
    if case == "delisted":
        second = first
    else:
        second = [
            _finding("stage", "VERIFIED", url=STAGE_URL, quote=STAGE_QUOTE),
            (_finding("required_attribute") if case in {"missing_capability", "customer_use"}
             else _verified_attribute(quote="Unfetched product claim." if case == "invalid_quote" else PRODUCT_QUOTE)),
        ]
    requests = []

    async def model(_session, _url, *, headers, payload):
        del headers
        assert validate_operation_request("openrouter.chat", payload)["messages"] == payload["messages"]
        requests.append(payload)
        findings = first if len(requests) == 1 else second
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({"findings": findings})},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "_post_json", model)
    search = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(investigator, "_search_web", search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com"},
        targets=("stage", "required_attribute"), requested_stage="Public",
        requested_attribute=ATTRIBUTE,
        prior_observations={"submitted_source_urls": [STAGE_URL, PRODUCT_URL]},
        prefetched_pages={
            STAGE_URL: {"final_url": STAGE_URL, "text": stage_text},
            PRODUCT_URL: {"final_url": PRODUCT_URL, "text": product_text},
        },
    ))

    assert len(requests) >= 2
    submit = next(tool for tool in requests[1]["tools"] if tool["function"]["name"] == "submit_findings")
    items = submit["function"]["parameters"]["properties"]["findings"]
    assert items["minItems"] == items["maxItems"] == 2
    assert set(items["items"]["properties"]["target"]["enum"]) == {
        "stage", "required_attribute",
    }
    feedback = json.loads(requests[1]["messages"][-1]["content"])
    assert feedback["error"] == "fetched_public_stage_evidence_not_adjudicated"
    assert "complete requested attribute" in feedback["instruction"]
    assert "required_attribute" in result["claims"], result
    assert result["claims"]["required_attribute"]["status"] == (
        "VERIFIED" if case == "valid" else "UNPROVEN"
    )
    assert result["claims"]["stage"]["status"] == (
        "UNPROVEN" if case == "delisted" else "VERIFIED"
    )
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0
    assert len(requests) <= 3


def test_public_stage_rereview_preserves_contradicted_attribute(monkeypatch):
    requests = []
    contradicted = _finding(
        "required_attribute", "CONTRADICTED", url=PRODUCT_URL,
        quote="Tenable does not sell exposure management software.", role="customer_user",
    )

    async def model(_session, _url, *, headers, payload):
        del headers
        assert validate_operation_request("openrouter.chat", payload)["messages"] == payload["messages"]
        requests.append(payload)
        if len(requests) == 2:
            submit = next(tool for tool in payload["tools"] if tool["function"]["name"] == "submit_findings")
            items = submit["function"]["parameters"]["properties"]["findings"]
            assert items["minItems"] == items["maxItems"] == 1
            assert items["items"]["properties"]["target"]["enum"] == ["stage"]
        findings = ([_finding("stage"), contradicted] if len(requests) == 1
                    else [_finding("stage", "VERIFIED", url=STAGE_URL, quote=STAGE_QUOTE)])
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({"findings": findings})},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "_post_json", model)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com"},
        targets=("stage", "required_attribute"), requested_stage="Public",
        requested_attribute=ATTRIBUTE,
        prior_observations={"submitted_source_urls": [STAGE_URL, PRODUCT_URL]},
        prefetched_pages={
            STAGE_URL: {"final_url": STAGE_URL, "text": STAGE_QUOTE},
            PRODUCT_URL: {"final_url": PRODUCT_URL, "text": contradicted["evidence_quote"]},
        },
    ))
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["status"] == "CONTRADICTED"
    assert len(requests) == 2


def test_combined_product_industry_remains_verified_during_joint_rereview(monkeypatch):
    requests = []
    industry = {
        **_finding(
            "industry", "VERIFIED", url=PRODUCT_URL, quote=PRODUCT_QUOTE,
            role="supplier_operator",
        ),
        "observed_industry": "Software",
        "observed_subindustry": "Exposure management software",
    }

    async def model(_session, _url, *, headers, payload):
        del headers
        assert validate_operation_request("openrouter.chat", payload)["messages"] == payload["messages"]
        requests.append(payload)
        findings = (
            [_finding("stage"), industry, _finding("required_attribute")]
            if len(requests) == 1 else [
                _finding("stage", "VERIFIED", url=STAGE_URL, quote=STAGE_QUOTE),
                _verified_attribute(),
            ]
        )
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({"findings": findings})},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "_post_json", native_product_post(model))
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com"},
        targets=("stage", "industry", "required_attribute"),
        requested_stage="Public", requested_industry="Software",
        requested_product_service="Provides exposure management software to enterprises",
        requested_attribute=ATTRIBUTE,
        prior_observations={"submitted_source_urls": [STAGE_URL, PRODUCT_URL]},
        verified_homepage_identity={
            "normalized_name": "Tenable", "registrable_dns_domain": "tenable.com",
        },
        prefetched_pages={
            STAGE_URL: {"final_url": STAGE_URL, "text": STAGE_QUOTE},
            PRODUCT_URL: {"final_url": PRODUCT_URL, "text": PRODUCT_QUOTE},
        },
    ))

    assert len(requests) == 2
    submit = next(tool for tool in requests[1]["tools"] if tool["function"]["name"] == "submit_findings")
    items = submit["function"]["parameters"]["properties"]["findings"]
    assert items["minItems"] == items["maxItems"] == 2
    assert set(items["items"]["properties"]["target"]["enum"]) == {
        "stage", "required_attribute",
    }
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["product_coverage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0
