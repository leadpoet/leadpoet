"""Loaded-evidence reuse stays inside the existing bounded attribute review.

Mock replies test context, validation and admission, not model accuracy. The
saved-evidence real-model comparison is a separate release check.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lab_arena import operations as arena_operations
from qualification.scoring import company_evidence_investigator as investigator


URL = "https://changefinancial.com/"
QUOTE = (
    "Vertexon our payments platform, seamlessly integrates with your core "
    "systems enabling you to easily deliver physical and digital card "
    "solutions to your customers."
)
OTHER_URL = "https://regulator.example/subsidiary"
OTHER_QUOTE = "Subsidiary Payments Pty Ltd provides payment services."
ATTRIBUTE = (
    "Provides digital banking, payments, or money movement services to "
    "consumers or businesses."
)
PRODUCT = "Digital banking and payments infrastructure"


def _finding(target, status="VERIFIED", *, url=URL, quote=QUOTE):
    positive = status != "UNPROVEN"
    return {
        "target": target,
        "status": status,
        "observed_value": "Payments platform" if positive else None,
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "Payments" if target == "industry" else "",
        "observed_subindustry": "Digital payments infrastructure" if target == "industry" else "",
        "activity_role": "supplier_operator" if positive else "unresolved",
        "evidence_url": url if positive else "",
        "evidence_quote": quote if positive else "",
        "old_name": "Change Financial",
        "new_name": "Change Financial",
        "old_domain": "changefinancial.com",
        "new_domain": "changefinancial.com",
        "shared_linkedin_slug": "change-financial",
        "reason": "The loaded facts support this result.",
    }


def _run(monkeypatch, replies, *, attribute=ATTRIBUTE, product=PRODUCT, extra_text=""):
    requests = []

    async def post(_session, _url, *, headers, payload):
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        reply = replies[len(requests) - 1]
        if reply == "provider_failure":
            return 401, {"error": "provider unavailable"}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({"findings": reply})},
        }]}}]}

    async def no_research(*_args, **_kwargs):
        raise AssertionError("This final loaded-evidence judgment must not add research")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 1)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", no_research)
    monkeypatch.setattr(investigator, "_fetch_page", no_research)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Change Financial", "website": URL,
            "linkedin": "https://www.linkedin.com/company/change-financial",
        },
        targets=("industry", "required_attribute"),
        requested_industry="Financial Services",
        requested_subindustry="Digital banking and payments",
        requested_product_service=product,
        requested_attribute=attribute,
        positive_semantic_review=True,
        prior_observations={
            "observed_company_name": "Change Financial",
            "observed_company_website": URL,
            "observed_company_linkedin": "https://www.linkedin.com/company/change-financial",
            "submitted_source_urls": [URL, OTHER_URL],
        },
        verified_homepage_identity={
            "normalized_name": "Change Financial",
            "registrable_dns_domain": "changefinancial.com",
            "linkedin_company_slug": "change-financial",
        },
        prefetched_pages={
            URL: {"final_url": URL, "text": QUOTE + extra_text},
            OTHER_URL: {"final_url": OTHER_URL, "text": OTHER_QUOTE},
        },
    ))
    return result, requests


def _feedback(request):
    return next(
        json.loads(message["content"])
        for message in reversed(request["messages"])
        if message.get("role") == "tool"
    )


@pytest.mark.parametrize("rejected_source", ["subsidiary", "invalid_quote"])
def test_existing_correction_can_replace_rejected_attribute_with_own_industry_quote(
    monkeypatch, rejected_source,
):
    rejected = _finding(
        "required_attribute",
        url=OTHER_URL if rejected_source == "subsidiary" else URL,
        quote=OTHER_QUOTE if rejected_source == "subsidiary" else "This text was invented.",
    )
    result, requests = _run(monkeypatch, [
        [_finding("industry"), rejected],
        [_finding("required_attribute")],
    ])

    assert len(requests) == 2  # Original judgment and the existing correction.
    feedback = _feedback(requests[1])
    assert feedback["error"] == "deterministic_evidence_validation_failed"
    assert feedback["rejected_findings"][0]["target"] == "required_attribute"
    assert feedback["untrusted_non_rejected_source_context"][0]["evidence_quote"] == QUOTE
    assert "Rejection of the submitted attribute citation does not reject independent evidence" in feedback["instruction"]
    assert "preserve every required qualifier and condition" in feedback["instruction"]
    assert requests[1]["tools"][0]["function"]["parameters"]["properties"]["findings"]["items"]["properties"]["target"]["enum"] == ["required_attribute"]
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["evidence_quote"] == QUOTE
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0


@pytest.mark.parametrize("attribute,extra_text", [
    (ATTRIBUTE + " Sold on a recurring subscription.", ""),
    (ATTRIBUTE + " Serves hospitals only.", ""),
    (ATTRIBUTE + " Available to customers in Japan.", ""),
    (ATTRIBUTE + " Launched in 2026.", ""),
    ("Provides digital banking AND payment services to businesses.", ""),
    ("Provides payments or money movement services to hospitals in Japan.", ""),
    (ATTRIBUTE, " The payments product has been discontinued and is no longer supplied."),
])
def test_industry_pass_does_not_override_semantic_negative_or_retry_it(
    monkeypatch, attribute, extra_text,
):
    result, requests = _run(monkeypatch, [[
        _finding("industry"), _finding("required_attribute", "UNPROVEN"),
    ]], attribute=attribute, extra_text=extra_text)
    assert len(requests) == 1
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
    sent = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert sent["requested_attribute"] == attribute
    assert sent["requested_product_service"] == PRODUCT
    assert extra_text in sent["prefetched_sources"][0]["text"] or not extra_text
    prompt = " ".join(requests[0]["messages"][0]["content"].split())
    assert "every mandatory condition still needs proof there" in prompt
    assert len(requests[0]["messages"][0]["content"]) < arena_operations.OPENROUTER_MAX_CONTENT_CHARS
    assert "Do not add conditions absent from the criterion" in prompt
    assert "unresolved conflicting facts mean UNPROVEN" in prompt


@pytest.mark.parametrize("final", ["subsidiary", "invalid_quote", "unproven", "provider_failure"])
def test_attribute_correction_does_not_repeat_or_relax_evidence_validation(monkeypatch, final):
    initial = _finding("required_attribute", url=OTHER_URL, quote=OTHER_QUOTE)
    corrected = (
        "provider_failure" if final == "provider_failure" else
        [_finding("required_attribute", "UNPROVEN")] if final == "unproven" else
        [_finding("required_attribute", url=OTHER_URL, quote=OTHER_QUOTE)] if final == "subsidiary" else
        [_finding("required_attribute", quote="This text was invented.")]
    )
    result, requests = _run(monkeypatch, [[_finding("industry"), initial], corrected])
    assert len(requests) == 2
    if final == "provider_failure":
        assert result["claims"] == {}
        assert result["failure_reason"] == "provider_error"
        assert result.get("_completed_submit") is not True
    else:
        assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
        assert result["claims"]["required_attribute"]["evidence_url"] == ""
        assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
        assert result["_completed_submit"] is (final == "unproven")
        assert result["failure_reason"] == (
            "" if final == "unproven" else "malformed_response"
        )


def test_simple_or_can_reuse_quote_without_a_correction(monkeypatch):
    result, requests = _run(monkeypatch, [[
        _finding("industry"), _finding("required_attribute"),
    ]])
    assert len(requests) == 1
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_quote"] == result["claims"]["required_attribute"]["evidence_quote"]


def test_rejected_other_identity_is_not_carried_as_candidate_context(monkeypatch):
    result, requests = _run(monkeypatch, [
        [_finding("industry", url=OTHER_URL, quote=OTHER_QUOTE),
         _finding("required_attribute", url=OTHER_URL, quote=OTHER_QUOTE)],
        [_finding("industry", "UNPROVEN"), _finding("required_attribute", "UNPROVEN")],
    ])
    assert len(requests) == 2
    assert "untrusted_non_rejected_source_context" not in _feedback(requests[1])
    assert all(finding["status"] == "UNPROVEN" for finding in result["claims"].values())


def test_exhausted_admission_does_not_start_an_attribute_review(monkeypatch):
    reads = 0

    def monotonic():
        nonlocal reads
        reads += 1
        return 0.0 if reads == 1 else investigator.ADMISSION_DEADLINE_SECONDS + 1

    monkeypatch.setattr(investigator, "time", SimpleNamespace(monotonic=monotonic))
    result, requests = _run(monkeypatch, [])
    assert requests == []
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == "admission_budget_interrupted"
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
