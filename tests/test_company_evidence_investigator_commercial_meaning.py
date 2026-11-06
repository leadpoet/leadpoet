"""Offline prompt and protocol checks; these do not measure model accuracy."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as investigator


PROMPT = " ".join(investigator._SYSTEM_PROMPT.split())
ACME_URL = "https://www.acmemanagedit.com/services"
ACME_QUOTE = (
    "Acme Managed IT runs managed IT and security services for 300 mid-market "
    "clients across North America."
)
ACME_ATTRIBUTE = (
    "Provides managed IT or managed security services to mid-market business "
    "customers under recurring contracts."
)


def test_commercial_qualifiers_use_ordinary_meaning_within_existing_limits():
    assert (
        "Read subscription, recurring, and fee-based qualifiers by ordinary "
        "meaning."
    ) in PROMPT
    # The offering and its clients or paid plans need exact quotes, and only
    # the three named ordinary terms follow from them.
    assert (
        "Exact company-bound quotes proving the company now provides the "
        "requested offering to clients or through paid plans prove only these "
        "terms: hosted business software, subscription; managed services, "
        "ongoing contracts; professional firms, fees."
    ) in PROMPT
    assert (
        "Not when the criterion contrasts another model or loaded text shows "
        "another model or an unreleased, waitlist, pilot, or free-only offer."
    ) in PROMPT
    assert "A pricing contact alone proves no sale." in PROMPT
    assert (
        "Price or billing cadence is needed only if the criterion requires it."
    ) in PROMPT
    assert "purchase path" not in PROMPT
    assert "Each other clause still needs source proof." in PROMPT
    # The rule applies only to requested commercial conditions and keeps every
    # existing limit on labels, licenses, free offers and unreleased products.
    assert (
        "Apply the following subscription and commercial-sale evidence rules only "
        "when the requested criteria require those conditions."
    ) in PROMPT
    assert (
        "Do not infer a commercial model from a product description or company "
        "category alone."
    ) in PROMPT
    assert (
        "a generic SaaS or product label, another vendor's subscription, and a "
        "pay-once license alone do not prove paid product subscriptions"
    ) in PROMPT
    assert "generic license by itself does not prove a subscription" in PROMPT
    assert (
        "A priced monthly or annual platform or workspace plan supports a paid "
        "subscription."
    ) in PROMPT
    assert "free plan, research study, prototype, pilot, demonstration" in PROMPT
    assert "Bind the commercial fact to the exact investigated company" in PROMPT


@pytest.mark.parametrize(
    "absolute_rule",
    [
        "never proves recurring terms by inference",
        "Do not infer subscriptions from software, customers",
        'Do not infer recurring subscription terms from the word "licensed."',
    ],
)
def test_absolute_commercial_bans_are_replaced(absolute_rule):
    assert absolute_rule not in PROMPT


def test_later_untyped_round_keeps_only_a_proven_series_c_plus():
    assert (
        "Later private financing keeps Series C+ unless stated to give control "
        "or a listing."
    ) in PROMPT
    assert "may have given" not in PROMPT
    # Debt, chronology and the current-stage search keep their rules.
    assert "Debt, grants, planned rounds" in PROMPT
    assert "later debt alone cannot" in PROMPT
    assert "complete the server-required current-stage search" in PROMPT


@pytest.mark.parametrize(
    ("quote", "expected_status"),
    [
        # No literal "recurring contract" wording: the model's ordinary-meaning
        # judgment is accepted, while exact quote and identity checks still apply.
        (ACME_QUOTE, "VERIFIED"),
        (
            "Other Corp runs managed IT and security services for 300 mid-market "
            "clients across North America.",
            "UNPROVEN",
        ),
    ],
)
def test_ordinary_meaning_finding_keeps_quote_and_identity_validation(
    monkeypatch, quote, expected_status,
):
    finding = {
        "target": "required_attribute",
        "status": "VERIFIED",
        "observed_value": "Managed IT and security services",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "supplier_operator",
        "evidence_url": ACME_URL,
        "evidence_quote": quote,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "Managed services for businesses ordinarily run on ongoing contracts.",
    }
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "attribute-submit", "type": "function",
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
            "name": "Acme Managed IT",
            "website": "https://www.acmemanagedit.com",
        },
        targets=("required_attribute",),
        requested_attribute=ACME_ATTRIBUTE,
        prior_observations={
            "submitted_source_urls": [ACME_URL],
            "observed_company_website": "https://www.acmemanagedit.com",
        },
        prefetched_pages={
            ACME_URL: {"final_url": ACME_URL, "text": ACME_QUOTE + " " + quote},
        },
    ))

    assert result["claims"]["required_attribute"]["status"] == expected_status
    # A rejected citation uses the existing bounded correction turns.
    assert len(requests) == 1 if expected_status == "VERIFIED" else len(requests) > 1
    search.assert_not_awaited()
    fetch.assert_not_awaited()
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["requested_attribute"] == ACME_ATTRIBUTE
