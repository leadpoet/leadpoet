"""An exhausted forced eighth judgment can use only the existing final slot."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_company_evidence_judgment_reserve import (
    URL, QUOTE, _Clock, _finding, _response,
)


CAPTURES = json.loads((Path(__file__).parent / "fixtures" /
    "ahf_eighth_output_exhaustion_20261025.json").read_text())


def run(monkeypatch, *, form="empty", final="verified", positive=True,
        settled_at=120.0, case="", scoped=False):
    clock = _Clock()
    requests, research = [], []
    requested = ("industry", "required_attribute") if scoped else ("industry",)
    if case == "nonforced":
        requested = ("stage", "industry")

    def findings(status="VERIFIED"):
        industry = _finding(status=status)
        if not scoped:
            return [industry]
        attribute = {**_finding(status="UNPROVEN"), "target": "required_attribute"}
        return [industry, attribute]

    async def search(*_args, **_kwargs):
        research.append("search")
        url = URL if case == "used_retry" else "https://acme.example/new-product"
        return {"results": [{"url": url, "title": "Product"}]}

    async def fetch(_session, url, **_kwargs):
        research.append("fetch")
        return {"ok": False, "error": "http_503"}

    async def fallback(*_args, **_kwargs):
        research.append("fallback")
        return {"ok": False, "error": "http_503"}

    async def post(_session, _url, *, headers, payload):
        requests.append(deepcopy(payload))
        turn = len(requests)
        assert turn <= 9, "no tenth reasoning request"
        if case == "used_retry" and turn == 2:
            return 200, deepcopy(CAPTURES[form]["response"])
        if case == "used_retry" and turn == 3:
            return 200, _response("submit_findings", {"findings": findings("UNPROVEN")}, call=turn)
        if case == "used_retry" and turn == 4:
            return 200, _response("search_web", {"query": "Acme enrollment software"}, call=turn)
        if turn == 8:
            clock.now = settled_at
            return 200, deepcopy(CAPTURES[form]["response"])
        if turn == 9:
            assert payload["max_tokens"] == 3000
            assert payload["reasoning"] == {"effort": "low"}
            assert [t["function"]["name"] for t in payload["tools"]] == ["submit_findings"]
            assert payload["tool_choice"]["function"]["name"] == "submit_findings"
            if final == "provider_failure":
                return 503, {"error": "provider unavailable"}
            if final in {"empty", "truncated"}:
                return 200, deepcopy(CAPTURES[final]["response"])
            if final == "malformed":
                return 200, _response("submit_findings", {"findings": []}, call=turn)
            if final == "research":
                return 200, _response("fetch_page", {"url": "https://acme.example/late"}, call=turn)
            if final == "fabricated":
                packet = [_finding(quote="Acme manufactures aircraft.")]
            elif final == "missing_citation":
                packet = [_finding(url="", quote="")]
            else:
                packet = findings("UNPROVEN" if final == "unproven" else "VERIFIED")
            if scoped:
                packet = [packet[-1]]
            return 200, _response("submit_findings", {"findings": packet}, call=turn)
        if scoped and turn == 7:
            packet = findings()
            packet[-1] = {**_finding(), "target": "required_attribute",
                          "evidence_quote": "Acme sells recurring subscriptions."}
            return 200, _response("submit_findings", {"findings": packet}, call=turn)
        if case == "required_research":
            if turn == 5:
                return 200, _response("submit_findings", {"findings": findings("UNPROVEN")}, call=turn)
            if turn == 6:
                assert payload["tool_choice"]["function"]["name"] == "search_web"
                return 200, _response("search_web", {"query": "Acme enrollment software"}, call=turn)
            if turn == 7:
                assert payload["tool_choice"]["function"]["name"] == "fetch_page"
                clock.now = 80.0
                return 200, _response("fetch_page", {"url": "https://acme.example/new-product"}, call=turn)
        if case == "nonforced":
            choice = payload.get("tool_choice")
            forced = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
            if forced == "search_web":
                return 200, _response("search_web", {"query": "Acme enrollment software"}, call=turn)
            if turn == 7:
                stage = {**_finding(), "target": "stage", "observed_value": "Private Equity",
                         "ownership": {"control_status": "current_control",
                                       "controlling_owner": "Example Capital", "owner_type": "private_equity"}}
                return 200, _response("submit_findings", {"findings": [stage, _finding()]}, call=turn)
        target = "https://acme.example/fault" if case == "source_fault" and turn == 1 else URL
        return 200, _response("fetch_page", {"url": target}, call=turn)

    for key in ("OPENROUTER_API_KEY", "EXA_API_KEY"):
        monkeypatch.setenv(key, "offline-placeholder")
    monkeypatch.setattr(c, "time", clock)
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_search_web", search)
    monkeypatch.setattr(c, "_fetch_page", fetch)
    monkeypatch.setattr(c, "_fetch_exa_exact_page", fallback)
    result = asyncio.run(c.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=requested, requested_stage="Private Equity" if case == "nonforced" else "",
        requested_industry="Education", requested_product_service="Enrollment software for universities",
        requested_attribute="Provides recurring subscriptions" if scoped else "",
        positive_semantic_review=positive,
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={"normalized_name": "acme",
                                    "registrable_dns_domain": "acme.example", "linkedin_company_slug": "acme"},
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}}))
    return result, requests, research


@pytest.mark.parametrize("form", ["empty", "truncated"])
@pytest.mark.parametrize("final", ["verified", "unproven"])
@pytest.mark.parametrize("settled_at", [80.0, 120.0, 139.999])
def test_actual_exhaustion_forms_get_one_bounded_serialization_request(monkeypatch, form, final, settled_at):
    result, requests, research = run(monkeypatch, form=form, final=final, settled_at=settled_at)
    assert len(requests) == result["usage"]["reasoning_turns"] == 9
    assert result["_completed_submit"] is True and result["failure_reason"] == ""
    assert result["claims"]["industry"]["status"] == ("UNPROVEN" if final == "unproven" else "VERIFIED")
    assert not research


@pytest.mark.parametrize("form", ["empty", "truncated"])
@pytest.mark.parametrize("case,positive,settled_at", [
    ("", False, 120.0), ("", True, 140.0), ("", True, 140.001),
    ("used_retry", True, 120.0), ("nonforced", True, 20.0),
])
def test_final_repair_requires_positive_forced_unused_budget_and_grace(monkeypatch,
        form, case, positive, settled_at):
    result, requests, _ = run(monkeypatch, form=form, case=case,
                              positive=positive, settled_at=settled_at)
    assert len(requests) == 8
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == "malformed_response"


@pytest.mark.parametrize("final,reason", [
    ("empty", "malformed_response"), ("truncated", "malformed_response"),
    ("provider_failure", "provider_error"),
    ("malformed", "malformed_response"), ("research", "malformed_response"),
    ("fabricated", "malformed_response"), ("missing_citation", "malformed_response"),
])
def test_ninth_response_never_gets_a_tenth_or_bypasses_proof(monkeypatch, final, reason):
    result, requests, research = run(monkeypatch, final=final)
    assert len(requests) == 9 and not research
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == reason


@pytest.mark.parametrize("case,reason", [
    ("source_fault", "provider_error"), ("required_research", "admission_budget_interrupted"),
])
def test_final_negative_does_not_clear_source_or_required_research_fault(monkeypatch, case, reason):
    result, requests, _ = run(monkeypatch, final="unproven", case=case)
    assert len(requests) == 9
    assert result["failure_reason"] == reason
    assert result["_completed_submit"] is False


@pytest.mark.parametrize("form", ["empty", "truncated"])
def test_serialization_keeps_active_targets_and_validated_sibling(monkeypatch, form):
    result, requests, research = run(monkeypatch, form=form, final="unproven", scoped=True)
    assert len(requests) == 9 and not research
    assert result["failure_reason"] == "" and result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_quote"] == QUOTE
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
    schema = requests[-1]["tools"][0]["function"]["parameters"]
    assert schema["properties"]["findings"]["items"]["properties"]["target"]["enum"] == ["required_attribute"]
