from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lab_arena import operations as arena_operations
from qualification.scoring import company_evidence_investigator as investigator


PRODUCT_URL = "https://rapid7.com/products/insightcloudsec/"
PRODUCT_QUOTE = (
    "Rapid7 InsightCloudSec provides cloud infrastructure entitlement "
    "management and real-time cloud threat detection."
)
NASDAQ_URL = "https://www.nasdaq.com/press-release/rapid7-leadership"
NASDAQ_QUOTE = (
    "Rapid7, Inc. (NASDAQ: RPD), a global leader in managed cybersecurity "
    "operations, announced a leadership transition."
)
SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/1560327/"
    "000156032726000043/rp-20260807.htm"
)
SEC_PAGE = (
    "Rapid7, Inc. (Exact name of registrant as specified in its charter) "
    "Check the appropriate box below if the Form 8-K filing is intended to "
    "simultaneously satisfy the filing obligation of the registrant under any "
    "of the following provisions. Securities registered pursuant to Section "
    "12(b) of the Securities Exchange Act of 1934: Title of each class Trading "
    "symbol(s) Name of each exchange on which registered Common Stock, $0.01 "
    "par value per share RPD The Nasdaq Global Market."
)
MALFORMED_SEC_QUOTE = SEC_PAGE.replace(" of the registrant", "", 1)
ACQUIRED_QUOTE = (
    "Rapid7 has been acquired by Parent Corp and is now part of its platform."
)
HISTORICAL_IPO_QUOTE = (
    "Rapid7 completed its initial public offering in 2015."
)


def _finding(target: str, **overrides):
    finding = {
        "target": target,
        "status": "VERIFIED",
        "observed_value": "Public",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": NASDAQ_URL,
        "evidence_quote": NASDAQ_QUOTE,
        "old_name": "Rapid7",
        "new_name": "Rapid7",
        "old_domain": "rapid7.com",
        "new_domain": "rapid7.com",
        "shared_linkedin_slug": "rapid7",
        "reason": "The exact source supports this finding.",
    }
    finding.update(overrides)
    return finding


def _industry_finding(**overrides):
    values = {
        "observed_value": "Cloud security software",
        "observed_industry": "Security software and services",
        "observed_subindustry": "Cloud security and identity protection",
        "activity_role": "supplier_operator",
        "evidence_url": PRODUCT_URL,
        "evidence_quote": PRODUCT_QUOTE,
    }
    values.update(overrides)
    return _finding("industry", **values)


def _run_scoped_correction(
    monkeypatch, responses, *, max_turns=2, reserve_after_first=True,
):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        if len(requests) == 1 and reserve_after_first:
            monkeypatch.setattr(
                investigator,
                "JUDGMENT_ADMISSION_RESERVE_SECONDS",
                investigator.ADMISSION_DEADLINE_SECONDS,
            )
        findings = responses[len(requests) - 1]
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": findings}),
            },
        }]}}]}

    async def fake_search(_session, _query, *, key):
        del key
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", max_turns)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Rapid7",
            "website": "https://rapid7.com",
            "linkedin": "https://www.linkedin.com/company/rapid7",
        },
        targets=("stage", "industry"),
        requested_stage="Public",
        requested_industry="Security software and services",
        requested_subindustry="Cloud security and identity protection",
        requested_product_service="Security software and services",
        requested_attribute=(
            "Protects accounts, infrastructure, endpoints, or data"
        ),
        prior_observations={
            "observed_company_name": "Rapid7",
            "observed_company_website": "https://rapid7.com",
            "observed_company_linkedin": (
                "https://www.linkedin.com/company/rapid7"
            ),
            "submitted_source_urls": [NASDAQ_URL, SEC_URL, PRODUCT_URL],
        },
        verified_homepage_identity={
            "normalized_name": "Rapid7",
            "registrable_dns_domain": "rapid7.com",
            "linkedin_company_slug": "rapid7",
        },
        prefetched_pages={
            NASDAQ_URL: {
                "final_url": NASDAQ_URL,
                "text": f"{NASDAQ_QUOTE} OtherCo, Inc. (NASDAQ: OTHR).",
            },
            SEC_URL: {
                "final_url": SEC_URL,
                "text": f"{SEC_PAGE} {ACQUIRED_QUOTE}",
            },
            PRODUCT_URL: {
                "final_url": PRODUCT_URL,
                "text": PRODUCT_QUOTE,
            },
        },
    ))
    return result, requests


def _initial_findings():
    return [
        _finding(
            "stage",
            evidence_url=SEC_URL,
            evidence_quote=MALFORMED_SEC_QUOTE,
        ),
        _industry_finding(),
    ]


def _submit_tool(payload):
    assert len(payload["tools"]) == 1
    tool = payload["tools"][0]["function"]
    assert tool["name"] == "submit_findings"
    return tool


def test_submit_only_correction_scopes_and_merges_valid_pending_target(monkeypatch):
    result, requests = _run_scoped_correction(
        monkeypatch,
        [_initial_findings(), [_finding("stage")]],
    )

    assert len(requests) == 2
    assert requests[0]["tool_choice"] == "required"
    assert requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }
    correction_tool = _submit_tool(requests[1])
    findings_schema = correction_tool["parameters"]["properties"]["findings"]
    assert findings_schema["minItems"] == findings_schema["maxItems"] == 1
    assert findings_schema["items"]["properties"]["target"]["enum"] == [
        "stage"
    ]
    prior_feedback = json.loads(requests[1]["messages"][-2]["content"])
    assert prior_feedback["untrusted_non_rejected_source_context"][0][
        "target"
    ] == "industry"
    scope = json.loads(requests[1]["messages"][-1]["content"])
    assert scope["correction_targets"] == ["stage"]
    assert "server retained unrelated findings" in scope["instruction"]
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["industry"] == investigator._validated_findings(
        {"findings": [_industry_finding()]},
        targets=("industry",),
        fetched_pages={PRODUCT_URL: PRODUCT_QUOTE},
        fetched_final_urls={PRODUCT_URL: PRODUCT_URL},
        first_party_domains={"rapid7.com"},
        identity_names={"rapid7"},
        identity_anchor={
            "submitted_name": "Rapid7",
            "submitted_domain": "rapid7.com",
            "submitted_linkedin_slug": "rapid7",
            "observed_name": "Rapid7",
            "observed_domain": "rapid7.com",
            "observed_linkedin_slug": "rapid7",
            "verified_name": "Rapid7",
            "verified_domain": "rapid7.com",
            "verified_linkedin_slug": "rapid7",
        },
    )["industry"]


def test_malformed_then_unproven_corrections_do_not_erase_valid_industry(
    monkeypatch,
):
    malformed_unproven = _finding(
        "stage",
        status="UNPROVEN",
        observed_value=None,
        evidence_url=SEC_URL,
        evidence_quote="Rapid7 filed a current report.",
        reason="Public stage remains unproven.",
    )
    final_unproven = _finding(
        "stage",
        status="UNPROVEN",
        observed_value=None,
        evidence_url="",
        evidence_quote="",
        reason="Public stage remains unproven.",
    )
    result, requests = _run_scoped_correction(
        monkeypatch,
        [_initial_findings(), [malformed_unproven], [final_unproven]],
    )

    assert len(requests) == 3
    assert requests[0]["tool_choice"] == "required"
    assert all(
        _submit_tool(request)["parameters"]["properties"]["findings"]
        ["items"]["properties"]["target"]["enum"] == ["stage"]
        for request in requests[1:]
    )
    second_feedback = json.loads(requests[2]["messages"][-1]["content"])
    assert len(second_feedback["rejected_findings"]) == 1
    rejected = second_feedback["rejected_findings"][0]
    assert rejected["target"] == "stage"
    assert rejected["reason"] == (
        "UNPROVEN must have empty evidence_url and evidence_quote fields"
    )
    assert rejected["source_url"] == SEC_URL
    assert rejected["source_context"].startswith("Rapid7, Inc.")
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_quote"] == PRODUCT_QUOTE


@pytest.mark.parametrize("retry_completes", [True, False])
def test_scoped_incomplete_submit_retry_uses_pending_schema_once(
    monkeypatch, retry_completes,
):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            monkeypatch.setattr(
                investigator,
                "JUDGMENT_ADMISSION_RESERVE_SECONDS",
                investigator.ADMISSION_DEADLINE_SECONDS,
            )
            arguments = json.dumps({"findings": _initial_findings()})
            finish_reason = "tool_calls"
        elif turn == 2 or not retry_completes:
            arguments = '{"findings":[{"target":"stage"'
            finish_reason = "length"
        else:
            arguments = json.dumps({"findings": [_finding("stage")]})
            finish_reason = "tool_calls"
        return 200, {"choices": [{
            "finish_reason": finish_reason,
            "message": {"tool_calls": [{
                "id": f"call-{turn}",
                "type": "function",
                "function": {
                    "name": "submit_findings",
                    "arguments": arguments,
                },
            }]},
        }]}

    async def fake_search(_session, _query, *, key):
        del key
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 3)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Rapid7",
            "website": "https://rapid7.com",
            "linkedin": "https://www.linkedin.com/company/rapid7",
        },
        targets=("stage", "industry"),
        requested_stage="Public",
        requested_industry="Security software and services",
        requested_subindustry="Cloud security and identity protection",
        positive_semantic_review=True,
        prior_observations={
            "observed_company_name": "Rapid7",
            "observed_company_website": "https://rapid7.com",
            "observed_company_linkedin": (
                "https://www.linkedin.com/company/rapid7"
            ),
            "submitted_source_urls": [NASDAQ_URL, SEC_URL, PRODUCT_URL],
        },
        verified_homepage_identity={
            "normalized_name": "Rapid7",
            "registrable_dns_domain": "rapid7.com",
            "linkedin_company_slug": "rapid7",
        },
        prefetched_pages={
            NASDAQ_URL: {"final_url": NASDAQ_URL, "text": NASDAQ_QUOTE},
            SEC_URL: {"final_url": SEC_URL, "text": SEC_PAGE},
            PRODUCT_URL: {"final_url": PRODUCT_URL, "text": PRODUCT_QUOTE},
        },
    ))

    assert len(requests) == 3
    assert requests[0]["tool_choice"] == "required"
    for request in requests[1:]:
        schema = _submit_tool(request)["parameters"]["properties"][
            "findings"
        ]
        assert schema["minItems"] == schema["maxItems"] == 1
        assert schema["items"]["properties"]["target"]["enum"] == [
            "stage"
        ]
    assert requests[2]["reasoning"] == {"effort": "low"}
    retry_feedback = json.loads(requests[2]["messages"][-1]["content"])
    assert retry_feedback["error"] == "incomplete_submit_findings"
    assert "Correction targets: stage." in retry_feedback["instruction"]
    assert "every requested target" not in retry_feedback["instruction"]
    if retry_completes:
        assert result["claims"]["stage"]["status"] == "VERIFIED"
        assert result["claims"]["industry"]["status"] == "VERIFIED"
    else:
        assert result == {
            "claims": {},
            "failure_reason": investigator.MALFORMED_RESPONSE_FAILURE_REASON,
        }


@pytest.mark.parametrize(
    "invalid_stage",
    [
        pytest.param(
            _finding(
                "stage",
                evidence_url=SEC_URL,
                evidence_quote=MALFORMED_SEC_QUOTE,
            ),
            id="nonexistent-quote",
        ),
        pytest.param(
            _finding("stage", evidence_quote=HISTORICAL_IPO_QUOTE),
            id="stale-ipo",
        ),
    ],
)
def test_scoped_candidate_preserves_valid_finding_on_admission_timeout(
    monkeypatch, invalid_stage,
):
    requests = []
    clock = {"now": 0.0}

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        clock["now"] = investigator.ADMISSION_DEADLINE_SECONDS
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({
                    "findings": [invalid_stage, _industry_finding()]
                }),
            },
        }]}}]}

    async def fake_search(_session, _query, *, key):
        del key
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 3)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(
        investigator,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"]),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Rapid7",
            "website": "https://rapid7.com",
            "linkedin": "https://www.linkedin.com/company/rapid7",
        },
        targets=("stage", "industry"),
        requested_stage="Public",
        prior_observations={
            "observed_company_name": "Rapid7",
            "observed_company_website": "https://rapid7.com",
            "observed_company_linkedin": (
                "https://www.linkedin.com/company/rapid7"
            ),
            "submitted_source_urls": [NASDAQ_URL, SEC_URL, PRODUCT_URL],
        },
        verified_homepage_identity={
            "normalized_name": "Rapid7",
            "registrable_dns_domain": "rapid7.com",
            "linkedin_company_slug": "rapid7",
        },
        prefetched_pages={
            NASDAQ_URL: {
                "final_url": NASDAQ_URL,
                "text": f"{NASDAQ_QUOTE} {HISTORICAL_IPO_QUOTE}",
            },
            SEC_URL: {"final_url": SEC_URL, "text": SEC_PAGE},
            PRODUCT_URL: {"final_url": PRODUCT_URL, "text": PRODUCT_QUOTE},
        },
    ))

    assert len(requests) == 1
    assert requests[0]["tool_choice"] == "required"
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["stage"]["reason"] == (
        "investigation admission budget exhausted"
    )
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_quote"] == PRODUCT_QUOTE
    assert result["_validated_stage_finding"] == {}
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == (
        investigator.ADMISSION_BUDGET_INTERRUPTED_FAILURE_REASON
    )


def test_timeout_preserves_only_server_validated_stage_metadata(monkeypatch):
    requests = []
    clock = {"now": 0.0}
    invalid_industry = _industry_finding(
        evidence_quote="Cloud security wording absent from the fetched page."
    )

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        clock["now"] = investigator.ADMISSION_DEADLINE_SECONDS
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({
                    "findings": [_finding("stage"), invalid_industry]
                }),
            },
        }]}}]}

    async def fake_search(_session, _query, *, key):
        del key
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 3)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(
        investigator,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"]),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Rapid7",
            "website": "https://rapid7.com",
            "linkedin": "https://www.linkedin.com/company/rapid7",
        },
        targets=("stage", "industry"),
        requested_stage="Public",
        prior_observations={
            "observed_company_name": "Rapid7",
            "observed_company_website": "https://rapid7.com",
            "observed_company_linkedin": (
                "https://www.linkedin.com/company/rapid7"
            ),
            "submitted_source_urls": [NASDAQ_URL, PRODUCT_URL],
        },
        verified_homepage_identity={
            "normalized_name": "Rapid7",
            "registrable_dns_domain": "rapid7.com",
            "linkedin_company_slug": "rapid7",
        },
        prefetched_pages={
            NASDAQ_URL: {"final_url": NASDAQ_URL, "text": NASDAQ_QUOTE},
            PRODUCT_URL: {
                "final_url": PRODUCT_URL,
                "text": PRODUCT_QUOTE,
            },
        },
    ))

    assert len(requests) == 1
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["claims"]["industry"]["reason"] == (
        "investigation admission budget exhausted"
    )
    assert result["_validated_stage_finding"] == result["claims"]["stage"]
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == (
        investigator.ADMISSION_BUDGET_INTERRUPTED_FAILURE_REASON
    )


def test_scoped_correction_rejects_unrelated_extra_target(monkeypatch):
    result, requests = _run_scoped_correction(
        monkeypatch,
        [_initial_findings(), [_finding("stage"), _industry_finding()]],
    )

    assert len(requests) == 2
    assert result == {
        "claims": {},
        "failure_reason": investigator.MALFORMED_RESPONSE_FAILURE_REASON,
    }


@pytest.mark.parametrize(
    ("corrected_stage", "expected_status", "expected_value"),
    [
        (
            _finding(
                "stage",
                status="CONTRADICTED",
                observed_value="Acquired",
                evidence_url=SEC_URL,
                evidence_quote=ACQUIRED_QUOTE,
            ),
            "CONTRADICTED",
            "Acquired",
        ),
        (
            _finding(
                "stage",
                evidence_quote="OtherCo, Inc. (NASDAQ: OTHR).",
            ),
            "UNPROVEN",
            None,
        ),
    ],
    ids=("grounded_contradiction", "wrong_entity"),
)
def test_scoped_correction_keeps_stage_identity_and_semantic_gates(
    monkeypatch, corrected_stage, expected_status, expected_value,
):
    responses = [_initial_findings(), [corrected_stage]]
    if expected_status == "UNPROVEN":
        responses.append([_finding(
            "stage",
            status="UNPROVEN",
            observed_value=None,
            evidence_url="",
            evidence_quote="",
        )])
    result, _requests = _run_scoped_correction(
        monkeypatch,
        responses,
    )

    assert result["claims"]["stage"]["status"] == expected_status
    assert result["claims"]["stage"]["observed_value"] == expected_value
    assert result["claims"]["industry"]["status"] == "VERIFIED"


def test_ordinary_new_evidence_can_contradict_unrelated_prior_finding(monkeypatch):
    old_geography_url = "https://rapid7.com/company/old-office"
    old_geography_quote = (
        "Rapid7 is headquartered in Boston, Massachusetts, United States."
    )
    new_url = "https://rapid7.com/company/current-headquarters"
    new_quote = "Rapid7 is headquartered in London, United Kingdom."
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, findings_or_arguments = "submit_findings", [
                _finding(
                    "stage",
                    evidence_url=SEC_URL,
                    evidence_quote=MALFORMED_SEC_QUOTE,
                ),
                _finding(
                    "geography",
                    observed_value="Boston, Massachusetts, United States",
                    observed_country="United States",
                    observed_state="Massachusetts",
                    evidence_url=old_geography_url,
                    evidence_quote=old_geography_quote,
                ),
            ]
        elif turn == 2:
            name, findings_or_arguments = "fetch_page", {"url": new_url}
        else:
            name, findings_or_arguments = "submit_findings", [
                _finding(
                    "stage",
                    status="UNPROVEN",
                    observed_value=None,
                    evidence_url="",
                    evidence_quote="",
                ),
                _finding(
                    "geography",
                    status="CONTRADICTED",
                    observed_value="London, United Kingdom",
                    observed_country="United Kingdom",
                    observed_state="",
                    evidence_url=new_url,
                    evidence_quote=new_quote,
                ),
            ]
        arguments = (
            {"findings": findings_or_arguments}
            if name == "submit_findings"
            else findings_or_arguments
        )
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, url):
        assert url == new_url
        return {"ok": True, "url": url, "final_url": url, "text": new_quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 4)
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Rapid7", "website": "https://rapid7.com"},
        targets=("stage", "geography"),
        requested_geography="Massachusetts, United States",
        prior_observations={
            "observed_company_name": "Rapid7",
            "observed_company_website": "https://rapid7.com",
            "submitted_source_urls": [SEC_URL, old_geography_url],
        },
        verified_homepage_identity={
            "normalized_name": "Rapid7",
            "registrable_dns_domain": "rapid7.com",
        },
        prefetched_pages={
            SEC_URL: {
                "final_url": SEC_URL,
                "text": SEC_PAGE,
            },
            old_geography_url: {
                "final_url": old_geography_url,
                "text": old_geography_quote,
            },
        },
    ))

    assert len(requests) == 3
    assert requests[1]["tool_choice"] == "required"
    assert any(
        tool["function"]["name"] == "fetch_page"
        for tool in requests[1]["tools"]
    )
    final_schema = next(
        tool["function"]["parameters"]
        for tool in requests[2]["tools"]
        if tool["function"]["name"] == "submit_findings"
    )
    assert final_schema["properties"]["findings"]["items"]["properties"][
        "target"
    ]["enum"] == ["geography", "stage"]
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["geography"]["status"] == "CONTRADICTED"
    assert result["claims"]["geography"]["evidence_quote"] == new_quote
