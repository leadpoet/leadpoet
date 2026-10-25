"""Only terminal explicit negatives can discard unused string citations."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.product_coverage_fixtures import native_product_post


URL = "https://acme.example/platform"
QUOTE = "Acme supplies enrollment software to universities worldwide."
OTHER_LISTING = "OtherCo (NASDAQ: OTHR) announced current quarterly results."
TARGETS = ("industry", "required_attribute")


def _finding(target, status="UNPROVEN", *, citations=True, quote=QUOTE):
    return {
        "target": target, "status": status,
        "observed_value": None, "observed_country": "", "observed_state": "",
        "observed_industry": "Enrollment software" if status == "VERIFIED" else "",
        "observed_subindustry": "Education software" if status == "VERIFIED" else "",
        "activity_role": "supplier_operator" if status == "VERIFIED" else "unresolved",
        "evidence_url": URL if citations else "",
        "evidence_quote": quote if citations else "",
        "reason": "The loaded evidence does not establish every requested condition.",
    }


def _response(name, arguments, call):
    return {"choices": [{"message": {"tool_calls": [{
        "id": f"offline-{call}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}}]}


def _run(monkeypatch, *, final="negative", mixed=False, stage_failure=False,
         source_failure=False, ordinary_search=False, admission_timeout=False,
         invalid_stage_identity=False):
    requests, searches, fetches = [], [], []
    clock = SimpleNamespace(now=0.0, reads=0)

    def monotonic():
        clock.reads += 1
        return 0.0 if clock.reads == 1 else clock.now

    if not ordinary_search and not source_failure and not admission_timeout:
        clock.now = 80.0
    monkeypatch.setattr(investigator, "time", SimpleNamespace(monotonic=monotonic))
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 5)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    targets = (("stage",) if stage_failure else
               ("stage", "industry") if invalid_stage_identity else TARGETS)

    async def search(_session, query, *, key):
        searches.append(query)
        if stage_failure:
            raise TimeoutError("offline required discovery failure")
        clock.now = 80.0
        return {"results": []}

    async def fetch(_session, url, **_kwargs):
        fetches.append(url)
        assert source_failure
        clock.now = 80.0
        return {"ok": False, "url": url, "error": "provider_error"}

    async def post(_session, _url, *, headers, payload):
        requests.append(deepcopy(payload))
        turn = len(requests)
        if source_failure and turn == 1:
            return 200, _response("fetch_page", {"url": "https://acme.example/missing"}, turn)
        if ordinary_search and turn == 2:
            # The first clean ordinary negative must trigger existing discovery.
            assert payload["tool_choice"]["function"]["name"] == "search_web"
            return 200, _response("search_web", {"query": "Acme enrollment software"}, turn)
        if admission_timeout:
            clock.now = 200.0
        submit = next(t["function"] for t in payload["tools"]
                      if t["function"]["name"] == "submit_findings")
        active = submit["parameters"]["properties"]["findings"]["items"]["properties"]["target"]["enum"]
        findings = [_finding(t, "VERIFIED" if mixed and t == "industry" else "UNPROVEN")
                    for t in active]
        if ordinary_search and turn == 1:
            findings = [_finding(t, citations=False) for t in active]
        final_turn = turn > (2 if ordinary_search or source_failure else 1)
        if final == "clean_negative":
            findings = [_finding(t, citations=False) for t in active]
        elif final == "positive" and final_turn:
            findings = [_finding(t, "VERIFIED") for t in active]
        elif final == "invalid_positive" and final_turn:
            findings = [_finding(t, "VERIFIED", quote="Acme manufactures aircraft.") for t in active]
        elif invalid_stage_identity and final_turn:
            findings = [
                {**_finding(t, "VERIFIED", quote=OTHER_LISTING), "observed_value": "Public"}
                if t == "stage" else _finding(t)
                for t in active
            ]
        elif final == "unknown_status" and final_turn:
            findings = [_finding(t, "NOT_PROVEN") for t in active]
        elif final == "nonstring_url" and final_turn:
            for f in findings:
                f["evidence_url"] = [URL]
        elif final == "nonstring_quote" and final_turn:
            for f in findings:
                f["evidence_quote"] = 123
        elif final == "provider_failure" and final_turn:
            return 503, {"error": "offline provider failure"}
        return 200, _response("submit_findings", {"findings": findings}, turn)

    monkeypatch.setattr(investigator, "_post_json", native_product_post(post))
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=targets, requested_stage=("Series B" if stage_failure else
                                         "Public" if invalid_stage_identity else ""),
        requested_industry="Education", requested_subindustry="Education software",
        requested_product_service="Enrollment software for universities",
        requested_attribute="Supplies enrollment software to universities",
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={"normalized_name": "acme",
                                   "registrable_dns_domain": "acme.example",
                                   "linkedin_company_slug": "acme"},
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE + " " + OTHER_LISTING}},
    ))
    return result, requests, searches, fetches


def test_all_negative_terminal_repair_clears_only_unused_citations(monkeypatch):
    result, requests, searches, fetches = _run(monkeypatch)
    assert len(requests) == 2
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert searches == fetches == []
    for finding in result["claims"].values():
        assert finding["status"] == "UNPROVEN"
        assert finding["evidence_url"] == finding["evidence_quote"] == ""
    feedback = json.loads(requests[1]["messages"][-1]["content"])
    assert {f["target"] for f in feedback["rejected_findings"]} == set(TARGETS)


def test_clean_all_negative_still_completes_without_extra_repair(monkeypatch):
    result, requests, *_ = _run(monkeypatch, final="clean_negative")
    assert len(requests) == 1
    assert result["_completed_submit"] is True


def test_mixed_valid_positive_is_preserved_during_terminal_negative_repair(monkeypatch):
    result, requests, *_ = _run(monkeypatch, mixed=True)
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_quote"] == QUOTE
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
    assert len(requests) == 2


def test_terminal_repair_can_still_supply_valid_positive_evidence(monkeypatch):
    result, requests, *_ = _run(monkeypatch, final="positive")
    assert len(requests) == 2
    assert result["_completed_submit"] is True
    assert all(f["status"] == "VERIFIED" for f in result["claims"].values())


@pytest.mark.parametrize("final", ["invalid_positive", "unknown_status", "nonstring_url", "nonstring_quote"])
def test_terminal_negative_cleanup_does_not_rescue_other_invalid_packets(monkeypatch, final):
    result, requests, *_ = _run(monkeypatch, final=final)
    assert len(requests) == 2
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == "malformed_response"


def test_exact_fetched_other_company_stage_proof_still_fails_attribution(monkeypatch):
    result, requests, searches, _ = _run(monkeypatch, invalid_stage_identity=True)
    assert len(requests) == 2
    assert len(searches) == 1
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == "malformed_response"
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["stage"]["reason"] == "source quote did not prove current public listing"


def test_ordinary_negative_still_performs_targeted_search_before_terminal_repair(monkeypatch):
    result, requests, searches, fetches = _run(monkeypatch, ordinary_search=True)
    assert len(requests) == 4
    assert searches == ["Acme enrollment software"]
    assert fetches == []
    assert result["_completed_submit"] is True


@pytest.mark.parametrize("failure", ["stage_failure", "source_failure", "admission_timeout"])
def test_negative_citation_cleanup_keeps_discovery_transport_and_admission_failures(monkeypatch, failure):
    result, *_ = _run(monkeypatch, **{failure: True})
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == (
        "admission_budget_interrupted" if failure == "admission_timeout" else "provider_error"
    )


def test_final_provider_failure_remains_incomplete(monkeypatch):
    result, *_ = _run(monkeypatch, final="provider_failure")
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == "provider_error"
