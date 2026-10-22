from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from lab_arena.operations import OPENROUTER_MAX_CONTENT_CHARS, validate_operation_request
from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_judgment_reserve import _Clock, _response


CASE = json.loads((Path(__file__).parent / "fixtures/company_evidence/"
                   "bighat_invalid_attribute_support.json").read_text())
FINDING = CASE["finding"]
URL = CASE["source"]["url"]
PRIMARY = FINDING["evidence_quote"]
SUPPORT_REASON = investigator.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON
COMPOUND = "Develops protein therapeutics AND has dosed a Phase 1 trial patient."
# Synthetic control only; this is not a new statement of real company facts.
COMPOUND_SUPPORT = "BigHat Biosciences has dosed a Phase 1 trial patient."


def _without_support(finding):
    return {**finding, **{
        f"supporting_evidence_{field}_{slot}": ""
        for slot in (1, 2) for field in ("url", "quote")
    }}


def _validate(finding, text=None):
    return investigator._validated_findings(
        {"findings": [finding]}, targets=("required_attribute",),
        fetched_pages={URL: CASE["source"]["text"] if text is None else text},
        fetched_final_urls={URL: CASE["source"]["final_url"]},
        first_party_domains={"bighatbio.com"},
        identity_names={"bighatbiosciences"},
        identity_anchor={
            "submitted_name": CASE["company_locator"]["name"],
            "submitted_domain": "bighatbio.com",
            "observed_name": CASE["prior_observations"]["observed_company_name"],
            "observed_domain": "bighatbio.com",
            "observed_linkedin_slug": "bighat-biosciences",
        },
    )["required_attribute"]


def _primary_only():
    return {
        **_without_support(FINDING),
        "observed_value": "developing next-generation protein therapeutics",
        "reason": "The named company designs and develops protein therapeutics.",
    }


def test_exact_bighat_support_failure_does_not_reject_primary_or_drop_support():
    assert FINDING["observed_value"] == "clinical-stage therapeutic candidates"
    assert investigator._quote_occurs(PRIMARY, CASE["source"]["text"])
    assert investigator._quote_occurs(
        FINDING["supporting_evidence_quote_1"], CASE["source"]["text"],
    )
    finding = _validate(FINDING)
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == SUPPORT_REASON
    assert finding["evidence_url"] == finding["evidence_quote"] == ""
    assert finding["supporting_evidence"] == []

    # The model must narrow details that depended on the omitted Phase 1 quote.
    corrected = _validate(_primary_only())
    assert corrected["status"] == "VERIFIED"
    assert corrected["evidence_quote"] == PRIMARY
    assert corrected["observed_value"] == _primary_only()["observed_value"]
    assert "clinical-stage" not in corrected["observed_value"]


@pytest.mark.parametrize("quote", ["", "BigHat Biosciences has invented proof."])
def test_missing_or_nonexistent_compound_support_stays_unproven(quote):
    rejected = _validate({**FINDING, "supporting_evidence_quote_1": quote})
    assert rejected["status"] == "UNPROVEN"
    assert rejected["reason"] == SUPPORT_REASON


@pytest.mark.parametrize("quote,text,reason", [
    ("Other Company develops protein therapeutics.",
     "Other Company develops protein therapeutics.",
     "source quote did not identify the investigated company"),
    (PRIMARY + " Invented proof.", None,
     "submitted quote was not present in fetched source"),
])
def test_invalid_primary_is_not_misreported_as_support_failure(quote, text, reason):
    rejected = _validate({**FINDING, "evidence_quote": quote}, text)
    assert rejected["status"] == "UNPROVEN"
    assert rejected["reason"] == reason
    assert rejected["reason"] != SUPPORT_REASON


def test_valid_company_bound_compound_support_still_validates():
    raw = {**FINDING, "supporting_evidence_quote_1": COMPOUND_SUPPORT}
    finding = _validate(raw, CASE["source"]["text"] + " " + COMPOUND_SUPPORT)
    assert finding["status"] == "VERIFIED"
    assert finding["supporting_evidence"] == [{"url": URL, "quote": COMPOUND_SUPPORT}]


def _run(monkeypatch, post, *, attribute=None, text=None, clock=None):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "time", clock or _Clock())

    async def no_research(*_args, **_kwargs):
        raise AssertionError("Loaded-citation repair must not force new research")

    monkeypatch.setattr(investigator, "_fetch_page", no_research)
    monkeypatch.setattr(investigator, "_search_web", no_research)
    return asyncio.run(investigator.investigate_company_evidence(
        company_locator=CASE["company_locator"], targets=("required_attribute",),
        requested_attribute=attribute or CASE["requested_attribute"],
        positive_semantic_review=True,
        prior_observations={**CASE["prior_observations"], "submitted_source_urls": [URL]},
        verified_homepage_identity=CASE["verified_homepage_identity"],
        prefetched_pages={URL: {
            "final_url": CASE["source"]["final_url"],
            "text": CASE["source"]["text"] if text is None else text,
        }},
    ))


@pytest.mark.parametrize("repair", ["primary_only", "compound", "unproven"])
def test_support_feedback_preserves_primary_and_requires_whole_criterion(
    monkeypatch, repair,
):
    requests = []
    clock = _Clock()

    async def post(_session, _url, *, headers, payload):
        validate_operation_request("openrouter.chat", payload)
        requests.append(copy.deepcopy(payload))
        if len(requests) == 1:
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            finding = FINDING
        else:
            feedback = next(
                json.loads(m["content"]) for m in reversed(payload["messages"])
                if m.get("role") == "tool"
                and "rejected_findings" in json.loads(m["content"])
            )
            assert feedback["rejected_findings"][0]["reason"] == SUPPORT_REASON
            instruction = feedback["instruction"]
            assert "supporting pair failed validation; the primary citation passed" in instruction
            assert "reuse that primary URL and exact quote" in instruction
            assert "independently prove the whole criterion" in instruction
            assert "revise observed-value and reason details" in instruction
            assert "repair the required supporting proof or return UNPROVEN" in instruction
            assert "Never repeat a rejected quote" not in instruction
            assert "Do not repeat an unchanged rejected finding" in instruction
            finding = _primary_only()
            if repair == "compound":
                finding = {**FINDING, "supporting_evidence_quote_1": COMPOUND_SUPPORT}
            elif repair == "unproven":
                finding = {**_without_support(FINDING), "status": "UNPROVEN",
                           "evidence_url": "", "evidence_quote": "",
                           "reason": "The required trial milestone is not proved."}
        return 200, _response("submit_findings", {"findings": [finding]}, call=len(requests))

    result = _run(
        monkeypatch, post, clock=clock,
        attribute=CASE["requested_attribute"] if repair == "primary_only" else COMPOUND,
        text=(CASE["source"]["text"] + " " + COMPOUND_SUPPORT
              if repair == "compound" else None),
    )
    assert len(requests) == 2
    finding = result["claims"]["required_attribute"]
    assert finding["status"] == ("UNPROVEN" if repair == "unproven" else "VERIFIED")
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
    if repair == "primary_only":
        assert finding["evidence_quote"] == PRIMARY
        assert finding["observed_value"] == _primary_only()["observed_value"]
        assert finding["reason"] == _primary_only()["reason"]


@pytest.mark.parametrize("expired", [False, True])
def test_repeated_invalid_support_never_extends_turn_or_deadline_bounds(monkeypatch, expired):
    clock = _Clock()
    requests = []

    async def post(_session, _url, *, headers, payload):
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        clock.now = (investigator.ADMISSION_DEADLINE_SECONDS + 1 if expired else
                     investigator.ADMISSION_DEADLINE_SECONDS
                     - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS)
        return 200, _response("submit_findings", {"findings": [FINDING]}, call=len(requests))

    result = _run(monkeypatch, post, clock=clock)
    assert 1 <= len(requests) <= investigator.MAX_REASONING_TURNS + 1
    if expired:
        assert len(requests) == 1
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
    assert len(investigator._SYSTEM_PROMPT) <= OPENROUTER_MAX_CONTENT_CHARS
