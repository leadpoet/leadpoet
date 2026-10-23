"""Guidance changes model context only; existing evidence gates remain active."""

import asyncio
import copy
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as inv
from tests.test_company_evidence_citation_repair_controls import STAGE_QUOTE, STAGE_URL, stage
from tests.test_company_evidence_judgment_reserve import _Clock, _finding, _response, QUOTE, URL


def run(monkeypatch, post, *, requested="Series A", targets=("stage",),
        quote=STAGE_QUOTE, discovery_fails=False, turns=8):
    searches, requests = [], []

    async def search(*args, **kwargs):
        searches.append(args)
        if discovery_fails:
            raise RuntimeError("discovery unavailable")
        return {"results": []}

    async def checked_post(session, url, *, headers, payload):
        validate_operation_request("openrouter.chat", payload)
        requests.append(copy.deepcopy(payload))
        return await post(len(requests), payload)

    async def fetch(session, url):
        assert url == STAGE_URL
        return {"ok": True, "url": url, "final_url": url, "text": quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline")
    monkeypatch.setenv("EXA_API_KEY", "offline")
    monkeypatch.setattr(inv, "time", _Clock())
    monkeypatch.setattr(inv, "MAX_REASONING_TURNS", turns)
    monkeypatch.setattr(inv, "_post_json", checked_post)
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "_fetch_page", fetch)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=targets, requested_stage=requested,
        requested_industry="Education", requested_attribute="Supplies enrollment software to universities",
        requested_subindustry="Higher education services",
        requested_product_service="Enrollment software for universities",
        prior_observations={"submitted_source_urls": [URL, STAGE_URL]},
        verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example",
                                    "linkedin_company_slug": "acme"},
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE},
                          STAGE_URL: {"final_url": STAGE_URL, "text": quote}},
    ))
    return result, requests, searches


@pytest.mark.parametrize("requested,targets,has_guidance", [
    ("Seed", ("stage",), True), ("Series A", ("stage",), True),
    ("Series B", ("stage",), True), ("Series C+", ("stage",), True),
    ("Public", ("stage",), False), ("Private Equity", ("stage",), False),
    ("", ("stage",), False), ("Series A", ("industry",), False),
])
def test_only_active_venture_stage_gets_guidance(monkeypatch, requested, targets, has_guidance):
    async def post(turn, payload):
        assert turn == 1
        raw = stage("UNPROVEN" if requested else "VERIFIED") if "stage" in targets else _finding()
        return 200, _response("submit_findings", {"findings": [raw]})

    _, requests, _ = run(monkeypatch, post, requested=requested, targets=targets, turns=1)
    assert len(requests) == 1
    messages = requests[0]["messages"]
    assert (messages[-1]["content"] == inv._VENTURE_STAGE_DECISION_GUIDANCE) == has_guidance
    assert all(len(m.get("content") or "") <= inv.OPENROUTER_MAX_CONTENT_CHARS for m in messages)
    properties = requests[0]["tools"][-1]["function"]["parameters"]["properties"]["findings"]["items"]["properties"]
    assert "stage_currentness" not in properties
    assert "competing_stage_event" not in properties


def test_ordinary_research_messages_do_not_get_guidance(monkeypatch):
    async def post(turn, payload):
        assert all(m.get("content") != inv._VENTURE_STAGE_DECISION_GUIDANCE for m in payload["messages"])
        if turn == 1:
            assert any(t["function"]["name"] == "fetch_page" for t in payload["tools"])
            return 200, _response("fetch_page", {"url": STAGE_URL})
        assert turn == 2
        return 200, _response("submit_findings", {"findings": [stage()]}, call=2)

    result, requests, searches = run(monkeypatch, post)
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert len(requests) == 2 and len(searches) == 1
    assert result["usage"]["fetch_calls"] == 0  # Existing prefetched source is reused.


def test_existing_scoped_attribute_repair_retains_stage_and_removes_guidance(monkeypatch):
    async def post(turn, payload):
        if turn == 1:
            bad = dict(_finding(quote="Acme manufactures airplanes."), target="required_attribute")
            return 200, _response("submit_findings", {"findings": [stage(), _finding(), bad]})
        assert turn == 2
        assert all(m.get("content") != inv._VENTURE_STAGE_DECISION_GUIDANCE for m in payload["messages"])
        feedback = json.loads(next(m["content"] for m in reversed(payload["messages"]) if m["role"] == "tool"))
        assert "required_attribute" in feedback["instruction"]
        assert payload["tools"][-1]["function"]["parameters"]["properties"]["findings"]["maxItems"] == 1
        repaired = dict(_finding(), target="required_attribute")
        return 200, _response("submit_findings", {"findings": [repaired]}, call=2)

    result, requests, searches = run(monkeypatch, post, targets=("stage", "industry", "required_attribute"), turns=1)
    assert len(requests) == 2 and len(searches) == 1
    assert result["usage"]["fetch_calls"] == 0
    assert result["_completed_submit"] is True
    assert all(f["status"] == "VERIFIED" for f in result["claims"].values())
    assert result["_validated_stage_finding"]["evidence_quote"] == STAGE_QUOTE


@pytest.mark.parametrize("reason", [
    "No fetched stage proof.", "A later subject acquisition remains unresolved.",
    "Material event chronology is unresolved.",
])
def test_guidance_never_promotes_unproven_model_findings(monkeypatch, reason):
    async def post(turn, payload):
        assert turn == 1
        return 200, _response("submit_findings", {"findings": [dict(stage("UNPROVEN"), reason=reason)]})

    result, _, _ = run(monkeypatch, post, turns=1)
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert not result["_validated_stage_finding"]


def test_required_discovery_failure_still_blocks_supported_stage(monkeypatch):
    async def post(turn, payload):
        return 200, _response("submit_findings", {"findings": [stage()]})

    result, _, searches = run(monkeypatch, post, discovery_fails=True, turns=1)
    assert len(searches) == 1
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert not result["_validated_stage_finding"]
    assert not result["_completed_submit"]


@pytest.mark.parametrize("observed,quote", [
    ("Acquired", "Buyer completed its acquisition of Acme."),
    ("Public", "Acme common stock is currently listed on NASDAQ under ticker ACME."),
])
def test_proven_subject_stage_change_still_contradicts_venture_request(monkeypatch, observed, quote):
    async def post(turn, payload):
        raw = dict(stage(), status="CONTRADICTED", observed_value=observed, evidence_quote=quote)
        return 200, _response("submit_findings", {"findings": [raw]})

    result, _, _ = run(monkeypatch, post, quote=quote, turns=1)
    assert result["claims"]["stage"]["status"] == "CONTRADICTED"
    assert result["_validated_stage_finding"]["observed_value"] == observed
