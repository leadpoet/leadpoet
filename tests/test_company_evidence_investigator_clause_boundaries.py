"""Offline protocol/validation checks; these do not measure model accuracy."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from lab_arena.operations import (
    OPENROUTER_MAX_CONTENT_CHARS,
    validate_operation_request,
)
from qualification.scoring import company_evidence_investigator as investigator


# Exact quote retained in the October 21 audit, lead 38. The submitted funding
# intent is separately stale; a valid stage finding must not repair that event.
ELEMENT_URL = (
    "https://www.elementbiosciences.com/news/"
    "element-biosciences-announces-upsized-series-e-to-accelerate-global-growth-"
    "across-ecosystem-of-genomic-multiomic-and-clinical-research-solutions"
)
ELEMENT_QUOTE = (
    "The financing, subject to regulatory approval, follows Element's $277 "
    "million Series D funding round announced in July of 2024."
)


def _stage_finding():
    return {
        "target": "stage",
        "status": "VERIFIED",
        "observed_value": "Series C+",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": ELEMENT_URL,
        "evidence_quote": ELEMENT_QUOTE,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "The completed D proves C+ regardless of pending E completion.",
    }


def test_original_prompt_preserves_conflict_and_qualifier_limits():
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())

    assert "pending Series E still proves Series C+" not in prompt
    assert "Debt, grants, planned rounds" in prompt
    assert "review every one before preserving the older matching stage" in prompt
    assert "complete the server-required current-stage search" in prompt
    assert "prove one allowed alternative and all shared qualifiers" in prompt
    assert "for AND, prove every conjunct for the same company and product" in prompt
    assert "a supplier does not satisfy a criterion that explicitly requires" in prompt
    assert len(investigator._SYSTEM_PROMPT) <= OPENROUTER_MAX_CONTENT_CHARS


@pytest.mark.parametrize("targets", [("industry",), ("industry", "required_attribute"), ("stage",)])
def test_product_coverage_precedes_global_status_without_changing_fields(targets):
    submit = next(t for t in investigator._tools(targets) if t["name"] == "submit_findings")
    item = submit["parameters"]["properties"]["findings"]["items"]
    order = list(item["properties"])
    expected = ["target", "product_coverage", "status"] if "industry" in targets else ["target", "status"]
    assert order[:len(expected)] == expected
    assert item["required"][:len(expected)] == expected
    if "industry" in targets:
        field = item["properties"]["product_coverage"]
        assert field["type"] == ["string", "null"] and field["maxLength"] == 24000
        assert "id, evidence, activity_role, status" in field["description"]
    else:
        assert "product_coverage" not in item["properties"]


@pytest.mark.parametrize(
    ("requested_stage", "quote", "loaded_text", "expected"),
    [
        ("Series C+", ELEMENT_QUOTE, ELEMENT_QUOTE, "VERIFIED"),
        ("Seed", ELEMENT_QUOTE, ELEMENT_QUOTE, "CONTRADICTED"),
        ("Series C+", ELEMENT_QUOTE, "Different fetched text.", "UNPROVEN"),
        ("Series C+", "A different company completed a Series D.",
         "A different company completed a Series D.", "UNPROVEN"),
        ("Series C+", "Element develops sequencing technology.",
         "Element develops sequencing technology.", "UNPROVEN"),
    ],
)
def test_saved_stage_quote_keeps_exact_source_identity_and_stage_guards(
    requested_stage, quote, loaded_text, expected,
):
    finding = {**_stage_finding(), "evidence_quote": quote}
    result = investigator._validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={ELEMENT_URL: loaded_text},
        first_party_domains={"elementbiosciences.com"},
        identity_names={"element"},
        requested_stage=requested_stage,
    )

    assert result["stage"]["status"] == expected


@pytest.mark.parametrize(
    ("requested_stage", "target", "expected_status"),
    [
        ("Series C+", "stage", "VERIFIED"),
        ("Series B", "stage", "CONTRADICTED"),
        ("Seed", "stage", "CONTRADICTED"),
        ("Public", "stage", "CONTRADICTED"),
        ("Series C+", "required_attribute", "VERIFIED"),
    ],
)
def test_original_message_path_keeps_existing_discovery(
    monkeypatch, requested_stage, target, expected_status,
):
    requests = []
    finding = _stage_finding()
    page_text = ELEMENT_QUOTE
    if target == "required_attribute":
        quote = "Element Biosciences supplies genomic life sciences platforms."
        page_text += " " + quote
        finding.update(
            target=target, observed_value="Genomic life sciences platforms",
            activity_role="supplier_operator", evidence_quote=quote,
        )

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "stage-submit", "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": [finding]}),
            },
        }]}}]}

    search = AsyncMock(return_value={"results": []})
    fetch = AsyncMock()
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Element Biosciences",
            "website": "https://www.elementbiosciences.com",
        },
        targets=(target,),
        requested_stage=requested_stage,
        requested_attribute="Supplies genomic life sciences platforms.",
        prior_observations={
            "submitted_source_urls": [ELEMENT_URL],
            "stage_dispute_urls": [ELEMENT_URL],
            "observed_company_website": "https://www.elementbiosciences.com",
        },
        prefetched_pages={
            ELEMENT_URL: {"final_url": ELEMENT_URL, "text": page_text},
        },
    ))

    assert result["claims"][target]["status"] == expected_status
    assert result["_validated_stage_finding"] == (
        result["claims"]["stage"] if target == "stage" else {}
    )
    assert result["usage"]["search_calls"] == int(target == "stage")
    assert result["usage"]["fetch_calls"] == 0
    if target == "stage":
        search.assert_awaited_once()
        if requested_stage != "Public":
            assert "latest funding round acquisition IPO" in search.call_args.args[1]
    else:
        search.assert_not_awaited()
    fetch.assert_not_awaited()
    assert len(requests) == 1
    messages = requests[0]["messages"]
    base_system = {"role": "system", "content": investigator._SYSTEM_PROMPT}
    assert messages[0] == base_system
    systems = [message for message in messages if message["role"] == "system"]
    assert systems == [base_system]
    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    user_message = next(message for message in messages if message["role"] == "user")
    document = json.loads(user_message["content"].split("\n", 1)[1])
    assert document["requested_stage"] == requested_stage
    assert document["requested_targets"] == [target]
    assert document["investigation_limits"]["search_calls"] == 2
    assert document["investigation_limits"]["fetch_calls"] == 4
