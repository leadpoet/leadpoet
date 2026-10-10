"""Launch and hiring hints choose a fetched source without proving a fit."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer

HOME = "https://secureframe.com/"
EVENT = "https://secureframe.com/newsroom/announcing-secureframe-defense-for-cmmc"
STAGE = "https://secureframe.com/newsroom/secureframe-raises-56m"
ATTRIBUTE = (
    "Sells privacy or security software and has either launched a major "
    "capability or is actively hiring for security, compliance, or platform roles."
)
DESCRIPTION = (
    "Secureframe, the leading AI-powered cybersecurity compliance platform, "
    "today announced Secureframe Defense, the only end-to-end solution for "
    "CMMC certification."
)
STAGE_QUOTE = "Secureframe announced it has raised a $56 million Series B round."


def _select(*, url=EVENT, text=DESCRIPTION, attribute=ATTRIBUTE, targets=("required_attribute",)):
    return investigator._priority_submitted_attribute_event_source(
        targets=targets, requested_attribute=attribute,
        submitted_source_urls=(url,), submitted_stage_source_urls=(STAGE,),
        submitted_source_hints={url: [text]},
        first_party_domains={"secureframe.com"}, identity_names={"secureframe"},
    )


def test_launch_and_hiring_hints_are_eligible_locators():
    assert _select() == EVENT
    assert _select(text="Secureframe is hiring security and platform engineers.") == EVENT


@pytest.mark.parametrize("kwargs", [
    {"text": "OtherCo announced its new CMMC platform."},
    {"url": "https://unrelated.example/secureframe-launch"},
    {"text": "Secureframe provides a compliance platform."},
    {"url": STAGE, "text": "Secureframe announced its Series B funding round."},
    {"attribute": "Sells subscription security software."},
    {"targets": ("geography",)},
])
def test_unrelated_off_domain_unsupported_or_wrong_target_hint_is_not_selected(kwargs):
    assert _select(**kwargs) == ""


def _run(monkeypatch, *, fetched_text=DESCRIPTION, fetch_ok=True, dispute=False):
    requests = []
    findings = [
        {"target": "stage", "status": "VERIFIED", "observed_value": "Series B",
         "evidence_url": STAGE, "evidence_quote": STAGE_QUOTE},
        {"target": "geography", "status": "UNPROVEN", "reason": "HQ unresolved"},
        {"target": "required_attribute", "status": "VERIFIED", "observed_value": None,
         "activity_role": "supplier_operator", "evidence_url": EVENT,
         "evidence_quote": DESCRIPTION, "reason": "First-party launch of security software"},
    ]

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function", "function": {
                "name": "submit_findings", "arguments": json.dumps({"findings": findings}),
            },
        }]}}]}

    fetch = AsyncMock(return_value={
        "ok": fetch_ok, "url": EVENT, "final_url": EVENT,
        "text": fetched_text if fetch_ok else "", "error": "fetch_failed" if not fetch_ok else "",
    })
    search = AsyncMock(return_value={"results": []})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)
    prior = {
        "submitted_source_urls": [HOME, STAGE, EVENT],
        "untrusted_company_stage_evidence": [{"url": STAGE, "quote": STAGE_QUOTE}],
        "submitted_source_hints": [{"url": EVENT, "text": DESCRIPTION}],
    }
    if dispute:
        prior["stage_dispute_urls"] = [STAGE]
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Secureframe", "website": HOME},
        targets=("stage", "required_attribute", "geography"),
        requested_stage="Series B", requested_attribute=ATTRIBUTE,
        positive_semantic_review=True, prior_observations=prior,
        verified_homepage_identity={
            "normalized_name": "secureframe", "registrable_dns_domain": "secureframe.com",
            "linkedin_company_slug": "secureframe",
        },
        homepage_navigation_locators=[
            {"url": "https://secureframe.com/contact", "label": "Contact"},
            {"url": "https://secureframe.com/about", "label": "About"},
        ],
        prefetched_pages={
            HOME: {"final_url": HOME, "text": "Secureframe provides compliance software."},
            STAGE: {"final_url": STAGE, "text": STAGE_QUOTE},
        },
    ))
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    return result, document, fetch, search


def test_submitted_launch_is_fetched_before_generic_contact_with_same_budget(monkeypatch):
    result, document, fetch, search = _run(monkeypatch)
    assert fetch.await_args.args[1] == EVENT
    fetch.assert_awaited_once()
    search.assert_awaited_once()
    assert document["server_priority_submitted_source"]["url"] == EVENT
    assert document["server_priority_submitted_source"]["notice"].endswith("untrusted_evidence")
    assert document["prior_observations"]["submitted_source_hints"] == [
        {"url": EVENT, "text": DESCRIPTION},
    ]
    assert document["server_current_stage_discovery"]["ok"] is True
    assert document["investigation_limits"]["remaining_fetch_calls"] == 2
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["usage"]["fetch_calls"] == 1
    assert result["usage"]["search_calls"] == 1


@pytest.mark.parametrize("fetch_ok", [True, False])
def test_submitted_launch_hint_cannot_replace_fetched_quote(monkeypatch, fetch_ok):
    result, _document, _fetch, _search = _run(
        monkeypatch, fetched_text="Secureframe provides compliance software.", fetch_ok=fetch_ok,
    )
    assert (result.get("claims", {}).get("required_attribute") or {}).get("status") != "VERIFIED"


def test_non_public_fit_handoff_keeps_intent_description_as_untrusted_hint(monkeypatch):
    from tests.test_company_evidence_investigator import _company, _icp

    company = _company(name="Secureframe", website=HOME, linkedin="https://www.linkedin.com/company/secureframe")
    signal = company.intent_signals[0].model_copy(update={"url": EVENT, "description": DESCRIPTION})
    company = company.model_copy(update={"intent_signals": [signal]})

    async def capture(**kwargs):
        assert kwargs["prior_observations"]["submitted_source_hints"] == [
            {"url": EVENT, "text": DESCRIPTION},
        ]
        assert EVENT in kwargs["prior_observations"]["submitted_source_urls"]
        raise RuntimeError("captured bounded handoff")

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", capture)
    with pytest.raises(RuntimeError, match="captured bounded handoff"):
        asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
            company=company, icp=_icp(required_attribute=ATTRIBUTE), verdict={},
            investigation_targets=("required_attribute",), icp_attribute=ATTRIBUTE,
            icp_stage="Series B", verified_identity={}, verified_transport_domain="secureframe.com",
            structured_employee_size_evidence=None, structured_public_company_evidence=None,
            employee_size_conflict=False, company_quality=True,
        ))
