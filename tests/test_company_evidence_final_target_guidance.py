"""Final feedback helps express a judgment without changing acceptance rules."""

import json
import asyncio
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_terminal_negative import _run, _finding, _response, URL, QUOTE


@pytest.mark.parametrize("final,complete", [
    ("negative", True), ("positive", True), ("invalid_positive", False),
])
def test_request_instruction_repair_preserves_native_outcomes(monkeypatch, final, complete):
    result, requests, searches, fetches = _run(monkeypatch, final=final)
    guidance = json.loads(requests[1]["messages"][-1]["content"])
    assert "single final submit-only" in guidance["instruction"]
    assert result["_completed_submit"] is complete
    assert searches == fetches == []
    if final == "positive":
        assert all(value["status"] == "VERIFIED" for value in result["claims"].values())
    elif final == "invalid_positive":
        assert result["failure_reason"] == "malformed_response"


def test_captured76_final_admission_removes_stale_fetch_instruction():
    # Exact trailing research instruction in the captured76 sequence80 request,
    # whose tool_choice already requires submit_findings.
    feedback = {
        "error": "deterministic_evidence_validation_failed",
        "instruction": (
            "First repair the quote from an already fetched page. Fetch another useful "
            "source only if no continuous company-bound span on that page proves the fact. "
            "Submit one complete finding for every requested target."
        ),
        "rejected_findings": [{"target": "stage", "reason": "source quote did not identify the investigated company"}],
    }
    corrected = investigator._final_submission_feedback(feedback)
    assert "Fetch another useful source" not in corrected["instruction"]
    assert "preserve all existing evidence rules" in corrected["instruction"]
    assert corrected["rejected_findings"] == feedback["rejected_findings"]
    assert "now submit-only" not in feedback["instruction"]


def test_ordinary_rejection_becomes_submit_only_on_next_admission(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    requests = []
    monkeypatch.setattr(investigator, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 5)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            clock.now = 80.0
            return 200, _response("submit_findings", {"findings": [
                _finding("required_attribute", "VERIFIED", quote="Wrong company makes aircraft."),
            ]}, 1)
        assert payload["tool_choice"]["function"]["name"] == "submit_findings"
        return 200, _response("submit_findings", {"findings": [
            _finding("required_attribute", citations=False),
        ]}, 2)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("No extra research is authorized by terminal guidance")

    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_fetch_page", forbidden)
    monkeypatch.setattr(investigator, "_search_web", forbidden)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("required_attribute",),
        requested_attribute="Supplies enrollment software to universities",
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))
    assert len(requests) == 2
    feedback = json.loads(requests[1]["messages"][-1]["content"])
    assert "now submit-only" in feedback["instruction"]
    assert "Fetch another useful source" not in feedback["instruction"]
    assert result["_completed_submit"] is True
