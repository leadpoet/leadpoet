"""Existing Public rereview may use admitted text during the judgment reserve."""

import asyncio
import json
from types import SimpleNamespace

from qualification.scoring import company_evidence_investigator as investigator


URL = "https://repay.com/current-results/"
LISTING = (
    "ATLANTA--(BUSINESS WIRE)--Aug. 26, 2026-- Repay Holdings Corporation "
    "(NASDAQ: RPAY) (“REPAY”)"
)
ACTIVITY = "REPAY supplies payment processing software to businesses."


def _stage(status="UNPROVEN", quote="", value=None):
    return {
        "target": "stage", "status": status, "observed_value": value,
        "evidence_url": URL if quote else "", "evidence_quote": quote,
        "reason": "Stage remains unresolved" if status == "UNPROVEN" else "Exact proof",
    }


def _run(monkeypatch, findings, *, text=LISTING, first_finish=99.0,
         max_turns=8, sibling=False, search_finish=90.0,
         cross_before_next_iteration=False, sibling_status="VERIFIED"):
    clock = {"now": 0.0}
    requests = []
    industry = {
        "target": "industry", "status": "VERIFIED",
        "observed_value": "Payment processing software", "observed_industry": "Payments",
        "observed_subindustry": "Payment processing software",
        "activity_role": "supplier_operator", "evidence_url": URL,
        "evidence_quote": ACTIVITY, "reason": "Own payment processing software",
    }
    if sibling_status == "UNPROVEN":
        industry.update(status="UNPROVEN", observed_value=None, evidence_url="",
                        evidence_quote="", observed_industry="", observed_subindustry="",
                        activity_role="unresolved")

    def monotonic():
        now = clock["now"]
        if cross_before_next_iteration and requests and now == first_finish:
            clock["now"] = investigator.ADMISSION_DEADLINE_SECONDS
        return now

    async def search(*_args, **_kwargs):
        clock["now"] = search_finish
        return {"results": []}

    async def post(_session, _url, *, headers, payload):
        requests.append(json.loads(json.dumps(payload)))
        if len(requests) == 1:
            clock["now"] = first_finish
        stage = findings[min(len(requests) - 1, len(findings) - 1)]
        submitted = (
            [stage, industry]
            if sibling and len(requests) == 1
            else [stage]
        )
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": submitted,
            })},
        }]}}]}

    async def no_fetch(*_args, **_kwargs):
        raise AssertionError("Rereview must use already admitted evidence")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "time", SimpleNamespace(monotonic=monotonic))
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", max_turns)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", no_fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "REPAY", "website": "https://repay.com/"},
        targets=("stage", "industry") if sibling else ("stage",),
        requested_stage="Public", requested_industry="Payments",
        requested_product_service="Payment processing software",
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={
            "normalized_name": "REPAY", "registrable_dns_domain": "repay.com",
            "linkedin_company_slug": "repay",
        },
        prefetched_pages={URL: {"final_url": URL, "text": text + " " + ACTIVITY}},
    ))
    return result, requests


def test_reserve_allows_existing_rereview_and_keeps_valid_sibling(monkeypatch):
    result, requests = _run(monkeypatch, [
        _stage(), _stage("VERIFIED", LISTING, "Public"),
    ], sibling=True)
    assert len(requests) == 2
    assert all(r["tool_choice"]["function"]["name"] == "submit_findings" for r in requests)
    assert "fetched_public_stage_evidence_not_adjudicated" in requests[1]["messages"][-1]["content"]
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0


def test_completed_acquisition_remains_contradicted(monkeypatch):
    acquired = "REPAY has been acquired by Parent Corp and is now part of its platform."
    result, requests = _run(monkeypatch, [
        _stage("CONTRADICTED", acquired, "Acquired"),
    ], text=LISTING + " " + acquired)
    assert len(requests) == 1
    assert result["claims"]["stage"]["status"] == "CONTRADICTED"


def test_rereview_accepts_completed_acquisition_contradiction(monkeypatch):
    acquired = "REPAY has been acquired by Parent Corp and is now part of its platform."
    result, requests = _run(monkeypatch, [
        _stage(), _stage("CONTRADICTED", acquired, "Acquired"),
    ], text=LISTING + " " + acquired)
    assert len(requests) == 2
    assert result["claims"]["stage"]["status"] == "CONTRADICTED"


def test_missing_listing_does_not_get_rereview(monkeypatch):
    result, requests = _run(monkeypatch, [_stage()], text="REPAY provides payment software.")
    assert len(requests) == 1
    assert result["claims"]["stage"]["status"] == "UNPROVEN"


def test_expired_verdict_does_not_admit_rereview_or_erase_sibling(monkeypatch):
    result, requests = _run(monkeypatch, [_stage()], first_finish=110.0, sibling=True)
    assert len(requests) == 1
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["industry"]["status"] == "VERIFIED"


def test_already_expired_deadline_admits_no_judgment(monkeypatch):
    result, requests = _run(monkeypatch, [_stage()], search_finish=110.0)
    assert len(requests) == 0
    assert result["claims"]["stage"]["status"] == "UNPROVEN"


def test_last_reasoning_turn_adds_no_rereview(monkeypatch):
    result, requests = _run(monkeypatch, [_stage()], max_turns=1)
    assert len(requests) == 1
    assert result["claims"]["stage"]["status"] == "UNPROVEN"


def test_correction_turn_adds_no_rereview(monkeypatch):
    result, requests = _run(monkeypatch, [
        _stage("VERIFIED", "REPAY is publicly traded.", "Public"), _stage(),
    ], max_turns=1)
    assert len(requests) == 2
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert "fetched_public_stage_evidence_not_adjudicated" not in str(requests)


def test_reserve_gets_at_most_one_existing_rereview(monkeypatch):
    result, requests = _run(monkeypatch, [_stage()])
    assert len(requests) == 2
    assert result["claims"]["stage"]["status"] == "UNPROVEN"


def test_deadline_crossing_after_queue_retains_valid_sibling(monkeypatch):
    result, requests = _run(monkeypatch, [_stage()], sibling=True,
                            cross_before_next_iteration=True)
    assert len(requests) == 1
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["reasoning_turns"] == 1


def test_stage_only_rereview_cannot_replace_validated_sibling(monkeypatch):
    stage = _stage("VERIFIED", LISTING, "Public")
    stage["observed_industry"] = "Unrelated industry"
    result, requests = _run(monkeypatch, [
        _stage(), stage,
    ], sibling=True)
    assert len(requests) == 2
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_quote"] == ACTIVITY
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert [t["function"]["name"] for t in requests[1]["tools"]] == ["submit_findings"]
    finding_schema = requests[1]["tools"][0]["function"]["parameters"]["properties"]["findings"]["items"]
    assert finding_schema["properties"]["target"]["enum"] == ["stage"]


def test_existing_unproven_sibling_remains_local(monkeypatch):
    result, requests = _run(monkeypatch, [
        _stage(), _stage("VERIFIED", LISTING, "Public"),
    ], sibling=True, sibling_status="UNPROVEN")
    assert len(requests) == 2
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
