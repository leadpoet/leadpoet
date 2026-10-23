"""A citation rejection must remain specific to its target and source."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_investigator import _finding


STAGE_URL = "https://news.example/latus-series-a"
ABOUT_URL = "https://latusbio.com/about-us"
STAGE_QUOTE = (
    "Latus Bio, Inc. (Latus), a biotechnology company engineering scalable "
    "gene therapies for broader patient populations, today announced the "
    "closing of a $97 million Series A financing"
)
REJECTED_ATTRIBUTE_QUOTE = (
    "Proceeds will also advance additional preclinical programs that "
    "utilize the company’s proprietary AAV discovery platform."
)
REPAIRED_ATTRIBUTE_QUOTE = (
    "Latus Bio, Inc. (Latus), a biotechnology company engineering scalable "
    "gene therapies for broader patient populations, today announced the "
    "closing of a $97 million Series A financing. Proceeds will also advance "
    "additional preclinical programs that utilize the company’s proprietary "
    "AAV discovery platform."
)
LAB_QUOTE = (
    "Latus Bio operates from offices and labs in Philadelphia, PA. Our team is "
    "committed to research and development of gene therapies."
)


def _stage(**changes):
    fields = dict(
        observed_value="Series A", evidence_url=STAGE_URL,
        evidence_quote=STAGE_QUOTE, old_name="Latus Bio", new_name="Latus Bio",
        old_domain="latusbio.com", new_domain="latusbio.com",
        reason="Latus Bio closed its Series A financing.",
    )
    fields.update(changes)
    return _finding("stage", **fields)


def _attribute(*, repaired=False, **changes):
    fields = dict(
        observed_value=None,
        evidence_url=STAGE_URL,
        evidence_quote=(
            REPAIRED_ATTRIBUTE_QUOTE if repaired else REJECTED_ATTRIBUTE_QUOTE
        ),
        supporting_evidence_url_1=ABOUT_URL,
        supporting_evidence_quote_1=LAB_QUOTE,
        old_name="Latus Bio", new_name="Latus Bio",
        old_domain="latusbio.com", new_domain="latusbio.com",
        reason="Latus develops preclinical therapies and operates labs.",
    )
    fields.update(changes)
    return _finding("required_attribute", **fields)


def _run(monkeypatch, *, revised_stage=None, extra_page=""):
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            findings = [_stage(), _attribute()]
        else:
            findings = [revised_stage or _stage(), _attribute(repaired=True)]
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": findings}),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(
        investigator, "_search_web",
        AsyncMock(return_value={"results": [{"url": STAGE_URL}]}),
    )
    monkeypatch.setattr(
        investigator, "_fetch_page",
        AsyncMock(side_effect=AssertionError("unexpected fetch")),
    )
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Latus Bio", "website": "https://latusbio.com"},
        targets=("stage", "required_attribute"),
        requested_stage="Series A",
        requested_attribute=(
            "Develops proprietary therapeutic candidates in preclinical or "
            "clinical stages with a laboratory or R&D operation."
        ),
        prior_observations={
            "submitted_source_urls": [STAGE_URL, ABOUT_URL],
        },
        prefetched_pages={
            STAGE_URL: {
                "final_url": STAGE_URL,
                "text": REPAIRED_ATTRIBUTE_QUOTE + " " + extra_page,
            },
            ABOUT_URL: {"final_url": ABOUT_URL, "text": LAB_QUOTE},
        },
        verified_homepage_identity={
            "normalized_name": "Latus Bio",
            "registrable_dns_domain": "latusbio.com",
        },
    ))
    return result, requests


def test_latus_citation_correction_keeps_independent_stage(monkeypatch):
    result, requests = _run(monkeypatch)
    assert len(requests) == 2, (result, [r["messages"][-1]["content"][:500] for r in requests])
    feedback = json.loads(requests[1]["messages"][-1]["content"])
    assert [item["target"] for item in feedback["rejected_findings"]] == [
        "required_attribute"
    ]
    assert feedback["untrusted_non_rejected_source_context"][0]["target"] == "stage"
    assert STAGE_QUOTE in feedback["untrusted_non_rejected_source_context"][0]["source_context"]
    assert "Only the targets in rejected_findings failed validation" in feedback["instruction"]
    assert "specific source fact and URL" in feedback["instruction"]
    assert result["_completed_submit"] is True
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0


def test_correction_can_revise_stage_when_loaded_page_contradicts_it(monkeypatch):
    contrary = (
        "Update: On October 1, 2026, Latus Bio closed a Series B financing "
        "that superseded its earlier Series A round."
    )
    revised = _finding(
        "stage", status="UNPROVEN", observed_value=None,
        evidence_url="", evidence_quote="",
        reason=(
            f"The earlier Series A citation at {STAGE_URL} is historical. "
            f"The same source now states: {contrary}"
        ),
    )
    result, requests = _run(
        monkeypatch, revised_stage=revised, extra_page=contrary,
    )
    assert len(requests) == 2, (result, requests[-1]["messages"][-1]["content"])
    assert result["_completed_submit"] is True
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"


def test_wrong_company_quote_does_not_prove_attribute():
    quote = (
        "Other Bio operates laboratories and develops proprietary "
        "therapeutic candidates in preclinical stages."
    )
    finding = _attribute(
        repaired=True, evidence_url="https://other.example/about",
        evidence_quote=quote, supporting_evidence_url_1="",
        supporting_evidence_quote_1="",
    )
    validated = investigator._validated_findings(
        {"findings": [finding]}, targets=("required_attribute",),
        fetched_pages={"https://other.example/about": quote},
        first_party_domains={"latusbio.com"}, identity_names={"latus bio"},
    )
    assert validated["required_attribute"]["status"] == "UNPROVEN"
