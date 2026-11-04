"""Native November 3 target denial keeps existing bounded company finality.

Only HTTP transport is substituted for retrieval. The saved Exa action66/67
response shapes omit request IDs/timing; no private receipt file is required.
"""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest

from lab_arena import scoring, verify
from qualification.scoring import competition
from qualification.scoring import intent_verification_three_stage as intent
from tests.lab_arena.test_lab_arena_scoring import _ICPS, breakdown, scored_company
from tests.test_same_event_source_local_finality import (
    _propagate_to_breakdown, _score_to_finality,
)


SOURCE = "https://mdc.luxurygoodsinsider.com/symu/filson-store-nyc/"


def native_denial():
    return {"results": [], "statuses": [{
        "id": SOURCE, "status": "error", "error": {
            "httpStatusCode": 403, "tag": "CRAWL_HTTP_403",
        },
    }]}


def install_native_transport(monkeypatch, replies):
    calls = []
    replies = iter(replies)

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def get(self, url, **kwargs):
            calls.append(("GET", url))
            if url == "https://api.scrapingdog.com/scrape":
                assert kwargs["params"]["url"] == SOURCE
                return httpx.Response(502, json={"error": {
                    "code": "provider_unavailable",
                }}, request=httpx.Request("GET", url))
            assert url == "https://archive.org/wayback/available?url=" + SOURCE
            return httpx.Response(200, json={
                "archived_snapshots": {}, "url": SOURCE,
            }, request=httpx.Request("GET", url))

        async def post(self, url, **kwargs):
            calls.append(("POST", url))
            assert url == "https://api.exa.ai/contents"
            assert kwargs["json"]["ids"] == [SOURCE]
            status, document = next(replies)
            return httpx.Response(status, json=document,
                                  request=httpx.Request("POST", url))

    async def no_wait(_seconds):
        pass

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setattr(intent.httpx, "AsyncClient", Client)
    monkeypatch.setattr(intent.asyncio, "sleep", no_wait)
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("network forbidden"))
    return calls


def run_native_verifier(monkeypatch, replies):
    calls = install_native_transport(monkeypatch, replies)
    search, judge = AsyncMock(), AsyncMock()
    monkeypatch.setattr(intent, "_find_same_event_source", search)
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    result = asyncio.run(intent.verify_three_stage(
        object(), company_name="Filson", company_website="https://www.filson.com/",
        company_linkedin="", source_url=SOURCE,
        miner_claim="Filson opened a New York store.",
        target_signal_text="Opened a new retail store",
        miner_signal_date="2026-08-01", evidence_type="EXPANSION",
        stage1_soft_reject=True, integrity_policy=True,
    ))
    search.assert_not_awaited()
    judge.assert_not_awaited()
    assert result["decision"] == "unavailable"
    assert result["rejection_reason"] == "evidence_fetch_failed"
    assert "source_resolution" not in result
    return result, calls


def test_saved_native403_reaches_terminal_zero_without_rejudging_completed_siblings(monkeypatch):
    result, calls = run_native_verifier(monkeypatch, [(200, native_denial())] * 2)
    assert [method for method, _ in calls] == ["GET", "GET", "POST", "POST"]
    status = result["scrape"]["statuses"][0]
    assert status["sd_stage"] == "all_tiers_exhausted:http_502"
    receipt = status["exa_target_crawl_failure"]
    assert receipt == {"id_matches_requested_url": True, "confirmed_attempts": 2,
                       "observations": [{"error_tag": "CRAWL_HTTP_403",
                                         "error_http_status": 403}] * 2}
    assert competition._confirmed_exa_target_crawl_failure(receipt)
    row = _propagate_to_breakdown(monkeypatch, result)
    row.update(company_identity_key="domain:scored2.example.com",
               company_identity_alias_keys=["domain:scored2.example.com"])
    assert competition.scorer_breakdown_has_company_local_verification_failure(row)
    good = breakdown(91.0)
    negative = breakdown(0.0, "Company fit mismatch: industry")
    for index, sibling in enumerate((good, negative)):
        sibling.update(company_index=index, company_qualified=index == 0,
                       company_identity_key=f"domain:scored{index}.example.com",
                       company_identity_alias_keys=[f"domain:scored{index}.example.com"])
    batches = []

    def scorer(batch, *_args):
        batches.append([item["company_name"] for item in batch])
        if len(batches) == 1:
            return [deepcopy(good), deepcopy(negative), {**deepcopy(row), "company_index": 2}]
        return [deepcopy(row)]

    scorer.integrity_policy = True
    rows = scoring.score_work_item(
        {"scored_run_id": "offline-native-target-denial"}, icp=_ICPS[0],
        companies=[scored_company(i) for i in range(3)], scorer=scorer,
    )
    assert batches == [["Scored Co 0", "Scored Co 1", "Scored Co 2"],
                       ["Scored Co 2"], ["Scored Co 2"]]
    assert rows[:2] == [good, negative]
    terminal = rows[2]
    assert terminal["final_score"] == 0.0 and terminal["company_qualified"] is False
    public = verify.redact_breakdown(terminal)
    assert competition.scorer_breakdown_is_terminal_company_verification_failure(public)
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(
        public, integrity_policy=True,
    )
    assert competition.count_penalizable_false_positives(
        [public], icp_has_intent_signals=True,
    ) == (0, 0)
    assert terminal["intent_signals_detail"][0]["judge_verdict"]["pipeline_decision"] == "unavailable"


@pytest.mark.parametrize("case", [
    "outer401", "outer402", "outer403", "outer429", "outer500", "outer502", "outer503",
    "wrong_url", "wrong_tag", "wrong_code", "float_code", "pending", "malformed",
    "mixed_targets", "top_error", "top_errors", "one_observation", "mixed_outer",
])
def test_uncertain_native_responses_remain_retryable_infrastructure(monkeypatch, case):
    document = native_denial()
    status = document["statuses"][0]
    if case.startswith("outer"):
        replies = [(int(case[5:]), document)] * 2
    else:
        if case == "wrong_url":
            status["id"] = "https://other.example/page"
        elif case == "wrong_tag":
            status["error"]["tag"] = "UNAUTHORIZED"
        elif case == "wrong_code":
            status["error"]["httpStatusCode"] = 401
        elif case == "float_code":
            status["error"]["httpStatusCode"] = 403.0
        elif case == "pending":
            status["status"] = "pending"
        elif case == "malformed":
            document["statuses"] = [None]
        elif case == "mixed_targets":
            document["statuses"].append({**deepcopy(status), "id": "https://other.example/page"})
        elif case in {"top_error", "top_errors"}:
            document[case[4:]] = {"code": "provider_unavailable"}
        replies = [(200, document)] * 2
        if case == "one_observation":
            replies = [(200, native_denial()), (200, {"results": [], "statuses": []})]
        elif case == "mixed_outer":
            replies = [(200, native_denial()), (503, document)]
    result, _ = run_native_verifier(monkeypatch, replies)
    assert not result["scrape"]["statuses"][0].get("exa_target_crawl_failure")
    row = _propagate_to_breakdown(monkeypatch, result)
    assert not competition.scorer_breakdown_has_company_local_verification_failure(row)
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=True)
    with pytest.raises(scoring.ScoringError, match="judge_exhausted attempts=3"):
        _score_to_finality(row)


@pytest.mark.parametrize("denied_first", [False, True])
def test_successful_exact_source_is_preserved_after_optional_first_denial(monkeypatch, denied_first):
    text = "Filson opened a new retail store in New York. " * 12
    success = {"results": [{"url": SOURCE, "text": text}],
               "statuses": [{"id": SOURCE, "status": "success"}]}
    replies = [(200, native_denial()), (200, success)] if denied_first else [(200, success)]
    calls = install_native_transport(monkeypatch, replies)
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert fetched["results"][0]["text"] == text
    projected = intent._project_contents_for_prompt(fetched)
    assert projected["results"][0]["text"] == text
    assert projected["statuses"][0]["stage"] == "exa_scraped"
    assert not projected["statuses"][0].get("exa_target_crawl_failure")
    assert not competition._intent_attempts_have_exact_target_crawl_exhaustion(projected["statuses"])
    assert sum(method == "POST" for method, _ in calls) == (2 if denied_first else 1)
