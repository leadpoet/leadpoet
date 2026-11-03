"""A pending final correction must settle once at its next existing turn."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.arena_reviewer_admission_seam import convert_admission_seam
from scripts.arena_reviewer_budget_seam import (
    BudgetSeamError, _NATIVE_NEW as _NEW, _NATIVE_OLD as _OLD, convert_budget_seam,
)
from tests.test_company_evidence_investigator import _finding
from tests.reviewer_frozen_sources import frozen_source


SOURCE = Path(__file__).resolve().parents[1] / "qualification/scoring/company_evidence_investigator.py"
STAGE_URL = "https://acme.example/shareholder-information"
ATTRIBUTE_URL = "https://acme.example/asset-management"
STAGE_QUOTE = "Acme plc listed on the London Stock Exchange on 21 October 2019."
ATTRIBUTE_QUOTE = "Acme manages assets on behalf of individual and professional investors."


def _source(*, repaired):
    source = SOURCE.read_bytes()
    assert source.count(_NEW.encode()) == 1
    return source if repaired else source.replace(_NEW.encode(), _OLD.encode(), 1)


def _replay(monkeypatch, *, repaired=True, settle=111.0,
            correction="negative", source_failure=False, search_failure=False):
    module = ModuleType("pending_budget_replay")
    exec(compile(_source(repaired=repaired), "pending_budget_replay.py", "exec"), module.__dict__)
    clock = SimpleNamespace(now=0.0)
    requests, fetches, searches = [], [], []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    async def fetch(_session, url):
        fetches.append(url)
        if source_failure and url == ATTRIBUTE_URL:
            return {"ok": False, "error": "fetch_failed"}
        return {"ok": True, "url": url, "final_url": url,
                "text": STAGE_QUOTE if url == STAGE_URL else ATTRIBUTE_QUOTE}

    async def search(_session, query, *, key):
        searches.append(query)
        if search_failure and len(searches) == 1:
            raise RuntimeError("offline provider failure")
        return {"results": []}

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        turn = len(requests)
        actions = [
            ("fetch_page", {"url": STAGE_URL}),
            ("search_web", {"query": "Acme investment vehicles"}),
            ("fetch_page", {"url": ATTRIBUTE_URL}),
            ("search_web", {"query": "Acme capital raises"}),
            ("fetch_page", {"url": "https://acme.example/contact"}),
        ]
        if turn <= len(actions):
            name, args = actions[turn - 1]
            if turn == 5:
                clock.now = 81.0
        elif turn == 6:
            clock.now = settle
            name, args = "submit_findings", {"findings": [
                _finding("stage", observed_value="Public", evidence_url=STAGE_URL,
                         evidence_quote=STAGE_QUOTE),
                _finding("headcount", status="UNPROVEN", observed_value=None,
                         evidence_url="", evidence_quote=""),
                # The original live failure returned a valid negative finding
                # with forbidden leftover citations. It needs envelope repair,
                # not more research or a different factual conclusion.
                _finding("required_attribute", status="UNPROVEN", observed_value=None,
                         evidence_url=ATTRIBUTE_URL, evidence_quote=ATTRIBUTE_QUOTE),
            ]}
        else:
            assert turn == 7, "a final correction must not repeat"
            if correction == "research":
                name, args = "fetch_page", {"url": ATTRIBUTE_URL}
            else:
                targets = next(tool["function"] for tool in payload["tools"]
                               if tool["function"]["name"] == "submit_findings")
                active = targets["parameters"]["properties"]["findings"]["items"]["properties"]["target"]["enum"]
                findings = [_finding(target, status="UNPROVEN", observed_value=None,
                                     evidence_url="", evidence_quote="")
                            for target in active]
                if correction == "invalid_positive":
                    findings[-1].update(status="VERIFIED", evidence_url=ATTRIBUTE_URL,
                                        evidence_quote="Invented factual proof.")
                name, args = "submit_findings", {"findings": findings}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"c{turn}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("network forbidden"))
    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda **_: Session())
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(module, "_post_json", post)
    monkeypatch.setattr(module, "_fetch_page", fetch)
    monkeypatch.setattr(module, "_search_web", search)
    result = asyncio.run(module.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage", "headcount", "required_attribute"),
        requested_stage="Public", requested_employee_buckets=("5,001-10,000",),
        requested_attribute="Raises and deploys capital through investment vehicles.",
        positive_semantic_review=True,
    ))
    return result, requests, fetches, searches


def test_original_live_sequence_reproduces_lost_pending_correction(monkeypatch):
    result, requests, fetches, searches = _replay(monkeypatch, repaired=False)
    assert len(requests) == 6
    assert result["failure_reason"] == "admission_budget_interrupted"
    assert result["_completed_submit"] is False
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert len(fetches) == len(searches) == 2


@pytest.mark.parametrize("settle", [100.0, 111.0, 180.0])
def test_pending_correction_finishes_once_without_research(monkeypatch, settle):
    result, requests, fetches, searches = _replay(monkeypatch, settle=settle)
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
    assert result["claims"]["required_attribute"]["evidence_quote"] == ""
    assert len(requests) == result["usage"]["reasoning_turns"] == 7
    assert len(requests) <= 9
    assert len(fetches) == result["usage"]["fetch_calls"] == 2
    assert len(searches) == result["usage"]["search_calls"] == 2
    assert [tool["function"]["name"] for tool in requests[-1]["tools"]] == ["submit_findings"]
    assert requests[-1]["tool_choice"]["function"]["name"] == "submit_findings"


@pytest.mark.parametrize("fault", ["source_failure", "search_failure"])
def test_pending_correction_preserves_provider_failures(monkeypatch, fault):
    result, requests, fetches, searches = _replay(monkeypatch, **{fault: True})
    assert len(requests) == 7
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == "provider_error"
    assert len(fetches) == len(searches) == 2


@pytest.mark.parametrize("correction", ["research", "invalid_positive"])
def test_final_correction_cannot_research_or_accept_fabricated_proof(monkeypatch, correction):
    result, requests, fetches, searches = _replay(monkeypatch, correction=correction)
    assert len(requests) == 7
    assert len(fetches) == len(searches) == 2
    if correction == "research":
        assert result["failure_reason"] == "malformed_response"
    else:
        assert result["_completed_submit"] is False
        assert result["failure_reason"] == "malformed_response"
        assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
        assert result["claims"]["required_attribute"]["evidence_quote"] == ""


def test_budget_seam_changes_only_reviewed_predicate_and_is_idempotent():
    # Historical admission conversion must use its reviewed source. The
    # behavioral replays above still exercise the current investigator.
    repaired = frozen_source(
        "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
    )
    assert repaired.count(_NEW.encode()) == 1
    original = repaired.replace(_NEW.encode(), _OLD.encode(), 1)
    corrected = convert_budget_seam(convert_admission_seam(original))
    assert corrected == repaired
    assert convert_budget_seam(corrected) == corrected
    assert convert_admission_seam(corrected) == corrected


@pytest.mark.parametrize("corruption", ["unknown", "duplicate", "missing_hook", "partial", "crlf"])
def test_budget_seam_rejects_unknown_or_partial_shapes(corruption):
    source = _source(repaired=False)
    if corruption == "unknown":
        source = source.replace(b"correction_turn = _turn == reasoning_turn_limit", b"correction_turn = False", 1)
    elif corruption == "duplicate":
        source = source.replace(_OLD.encode(), (_OLD + _OLD).encode(), 1)
    elif corruption == "missing_hook":
        source = source.replace(b"turn_admission = admission.next_turn(", b"turn_admission = other.next_turn(", 1)
    elif corruption == "partial":
        source = source.replace(b"correction_turn = _turn == reasoning_turn_limit", b"correction_turn = final_correction_pending", 1)
    else:
        source = source.replace(b"\n", b"\r\n")
    with pytest.raises(BudgetSeamError):
        convert_budget_seam(source)
