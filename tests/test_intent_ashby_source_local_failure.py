"""A missing Ashby listing must not turn thin-source exhaustion systemic."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import httpx
import pytest

from lab_arena import scoring, verify
from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.competition import (
    _intent_detail_has_source_local_failure,
    count_penalizable_false_positives,
    scorer_breakdown_has_company_local_verification_failure,
)


JOB_ID = "9787c187-a03f-491b-a852-30452bbf5a3c"
URL = f"https://jobs.ashbyhq.com/vanta/{JOB_ID}"
OTHER_ID = "12345678-1234-1234-1234-123456789abc"
BUSINESSWIRE_URL = (
    "https://www.businesswire.com/news/home/20260303812541/en/"
    "SentinelOne-Appoints-Sonalee-Parekh-as-Chief-Financial-Officer"
)


def listing(*, job_id=OTHER_ID, tenant="vanta", **extra):
    return {"jobs": [{
        "id": job_id,
        "jobUrl": f"https://jobs.ashbyhq.com/{tenant}/{job_id}",
        "isListed": True,
        "title": "Security Engineer",
        "descriptionPlain": "Build and maintain security engineering systems.",
        **extra,
    }]}


def install_api(monkeypatch, replies):
    calls = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers, params):
            assert url == "https://api.scrapingdog.com/scrape"
            assert params["url"] == (
                "https://api.ashbyhq.com/posting-api/job-board/vanta"
            )
            assert params["dynamic"] == "false"
            reply = replies[len(calls)]
            calls.append(params)
            if isinstance(reply, int):
                return httpx.Response(reply)
            if isinstance(reply, str):
                return httpx.Response(200, text=reply)
            return httpx.Response(200, json=reply)

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-key")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kw: Client())
    return calls


async def thin_contents(monkeypatch):
    async def sd(_url):
        return {"ok": False, "stage": "all_tiers_exhausted:body_too_short",
                "content": "", "error": "body_too_short"}

    async def exa(_url):
        return {"ok": False, "stage": "exa_thin", "content": "",
                "error": "<300 chars"}

    monkeypatch.setattr(intent, "_scrape_sd_hardened", sd)
    monkeypatch.setattr(intent, "_scrape_exa", exa)
    return await intent._fetch_sd_then_exa([URL])


def unavailable_detail(contents):
    return {
        "matched_icp_signal": 0,
        "after_decay": 0.0,
        "judge_verdict": {
            "decision": "rejected_verifier_error",
            "pipeline_decision": "unavailable",
            "rejection_reason": "evidence_fetch_failed",
            "verification_trace": {
                "provider_attempts": intent._project_contents_for_prompt(
                    contents
                )["statuses"],
            },
        },
    }


def company(name):
    return {
        "company_name": name,
        "company_website": f"https://{name.lower()}.example",
        "industry": "Software",
        "employee_count": "51-200",
        "country": "United States",
        "intent_signals": [],
    }


def retryable_breakdown(*, reason, receipt_reason, detail=None):
    return {
        "final_score": 0.0,
        "company_qualified": False,
        "failure_reason": reason,
        "intent_signals_detail": [detail] if detail else [],
        "verifier_gate_receipts": [{
            "gate": "intent_verification",
            "decision": "error",
            "failure_reason_code": receipt_reason,
        }],
    }


def businesswire_local_breakdown():
    detail = unavailable_detail({
        "results": [],
        "failures": [],
        "statuses": [{
            "url": BUSINESSWIRE_URL,
            "source": "none",
            "sd_stage": "all_tiers_exhausted:body_too_short",
            "exa_stage": "exa_no_results",
        }],
    })
    return retryable_breakdown(
        reason="Intent verification unavailable: source blocked",
        receipt_reason="source_blocked",
        detail=detail,
    )


def systemic_breakdown():
    return retryable_breakdown(
        reason="Intent verification unavailable: verifier provider error",
        receipt_reason="provider_error",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [listing(), {"jobs": []}])
async def test_two_valid_listing_misses_survive_projection(monkeypatch, payload):
    calls = install_api(monkeypatch, [payload, payload])
    contents = await thin_contents(monkeypatch)
    assert len(calls) == 2
    assert contents["results"] == []
    detail = unavailable_detail(contents)
    assert detail["judge_verdict"]["verification_trace"]["provider_attempts"][0] == {
        "url": "https://ashbyhq.com",
        "source": "scrapingdog_ashby_api_fallback",
        "stage": "ashby_posting_not_listed",
    }
    assert _intent_detail_has_source_local_failure(detail)
    # An unlisted posting is not proof that the original source is absent.
    assert not intent._confirmed_source_absence(contents, URL)
    assert detail["after_decay"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [
    429, 502, 400, "not JSON", {}, {"jobs": None}, {"jobs": [None]},
    listing(tenant="different"), listing(id="wrong-id"),
    listing(job_id=JOB_ID, isListed=False),
    listing(job_id=JOB_ID, title=""),
])
async def test_mixed_or_invalid_api_observations_stay_systemic(monkeypatch, first):
    calls = install_api(monkeypatch, [first, listing()])
    contents = await thin_contents(monkeypatch)
    assert len(calls) == (1 if first == 400 else 2)
    assert contents["results"] == []
    assert not _intent_detail_has_source_local_failure(unavailable_detail(contents))


@pytest.mark.asyncio
async def test_second_attempt_can_still_find_exact_listed_posting(monkeypatch):
    calls = install_api(monkeypatch, [listing(), listing(job_id=JOB_ID)])
    result = await intent._scrape_ashby_job(URL)
    assert len(calls) == 2
    assert result["ok"] is True
    assert result["stage"] == "sd:ashby_api:2"
    assert "Security Engineer" in result["content"]


@pytest.mark.parametrize("mutation", [
    {"stage": "ashby_api_exhausted"},
    {"source": "unknown"},
    {"url": ""},
    {"url": None},
    {"sd_stage": "http_429"},
    {"exa_stage": "exa_transient_exhausted"},
    {"unexpected": "ignored"},
])
def test_missing_listing_label_cannot_hide_ambiguous_status(monkeypatch, mutation):
    install_api(monkeypatch, [listing(), listing()])
    detail = unavailable_detail(asyncio.run(thin_contents(monkeypatch)))
    attempts = detail["judge_verdict"]["verification_trace"]["provider_attempts"]
    attempts[0].update(mutation)
    assert not _intent_detail_has_source_local_failure(detail)


@pytest.mark.parametrize("systemic", [False, True])
def test_native_retry_retains_sibling_and_only_zeros_local_exhaustion(
    monkeypatch, systemic,
):
    install_api(monkeypatch, [listing(), listing()])
    contents = asyncio.run(thin_contents(monkeypatch))
    detail = unavailable_detail(contents)
    if systemic:
        detail["judge_verdict"]["verification_trace"]["provider_attempts"][-1][
            "sd_stage"
        ] = "all_tiers_exhausted:http_502"
    failed = {
        "final_score": 0.0,
        "company_qualified": False,
        "failure_reason": "Intent verification unavailable: verifier provider error",
        "intent_signals_detail": [detail],
        "verifier_gate_receipts": [],
    }
    assert scorer_breakdown_has_company_local_verification_failure(failed) is (
        not systemic
    )
    companies = [company(name) for name in ("Good", "Missing")]
    calls = []

    def scorer(batch, _icp, _reference):
        calls.append([company["company_name"] for company in batch])
        results = []
        for index, company in enumerate(batch):
            row = deepcopy(failed) if company["company_name"] == "Missing" else {
                "final_score": 60.0, "company_qualified": True,
                "intent_signals_detail": [], "verifier_gate_receipts": [],
            }
            row["company_index"] = index
            row["company_identity_key"] = company["company_website"]
            row["company_identity_alias_keys"] = [company["company_website"]]
            results.append(row)
        return results

    scorer.integrity_policy = True
    kwargs = {
        "icp": {"icp_id": "fixture", "employee_count": ["51-200"],
                "max_companies": 5},
        "companies": companies,
        "scorer": scorer,
        "max_retries": 3,
    }
    if systemic:
        with pytest.raises(scoring.ScoringError, match="retryable verifier failure"):
            scoring.score_work_item({"scored_run_id": "fixture"}, **kwargs)
    else:
        result = scoring.score_work_item({"scored_run_id": "fixture"}, **kwargs)
        assert [row["final_score"] for row in result] == [60.0, 0.0]
        assert [row["company_index"] for row in result] == [0, 1]
        assert result[1]["verifier_gate_receipts"][-1] == {
            "gate": "intent_verification", "decision": "unavailable",
            "failure_class": "company_verification_exhausted",
        }
        public = verify.redact_breakdown(result[1])
        assert count_penalizable_false_positives(
            [public], icp_has_intent_signals=True
        ) == (0, 0)
        artifact = scoring.build_scoring_output("fixture", result)
        assert scoring.validate_scoring_output_document(artifact) == artifact
    assert calls == [["Good", "Missing"], ["Missing"], ["Missing"]]


@pytest.mark.parametrize(
    "failed_names",
    [("BusinessWire Local", "Systemic"), ("Systemic", "BusinessWire Local")],
)
def test_terminal_local_failure_does_not_mask_remaining_systemic_reason(
    failed_names,
):
    companies = [company("Complete"), *(company(name) for name in failed_names)]
    calls = []

    def scorer(batch, _icp, _reference):
        calls.append([item["company_name"] for item in batch])
        results = []
        for index, item in enumerate(batch):
            name = item["company_name"]
            if name == "BusinessWire Local":
                row = businesswire_local_breakdown()
            elif name == "Systemic":
                row = systemic_breakdown()
            else:
                row = {
                    "final_score": 60.0,
                    "company_qualified": True,
                    "intent_signals_detail": [],
                    "verifier_gate_receipts": [],
                }
            row["company_index"] = index
            row["company_identity_key"] = item["company_website"]
            row["company_identity_alias_keys"] = [item["company_website"]]
            results.append(row)
        return results

    scorer.integrity_policy = True
    with pytest.raises(scoring.ScoringError) as raised:
        scoring.score_work_item(
            {"scored_run_id": "businesswire-mixed"},
            icp={"icp_id": "fixture", "employee_count": ["51-200"],
                 "max_companies": 5},
            companies=companies,
            scorer=scorer,
            max_retries=3,
        )
    assert raised.value.failure_reason == "provider_error"
    assert calls == [
        ["Complete", *failed_names],
        list(failed_names),
        list(failed_names),
    ]


def test_terminal_stage_mismatch_does_not_mask_remaining_systemic_reason():
    companies = [company("SolarWinds"), company("Rapid7")]
    calls = []

    def scorer(batch, _icp, _reference):
        calls.append([item["company_name"] for item in batch])
        results = []
        for index, item in enumerate(batch):
            if item["company_name"] == "SolarWinds":
                row = {
                    "final_score": 0.0,
                    "company_qualified": False,
                    "failure_reason": "company stage mismatch: Acquired",
                    "intent_signals_detail": [],
                    "verifier_gate_receipts": [{
                        "gate": "company_fit",
                        "decision": "mismatch",
                    }],
                }
            else:
                row = systemic_breakdown()
            row["company_index"] = index
            row["company_identity_key"] = item["company_website"]
            row["company_identity_alias_keys"] = [item["company_website"]]
            results.append(row)
        return results

    scorer.integrity_policy = True
    with pytest.raises(scoring.ScoringError) as raised:
        scoring.score_work_item(
            {"scored_run_id": "position-20"},
            icp={"icp_id": "fixture", "employee_count": ["51-200"],
                 "max_companies": 5},
            companies=companies,
            scorer=scorer,
            max_retries=3,
        )
    assert raised.value.failure_reason == "provider_error"
    assert calls == [["SolarWinds", "Rapid7"], ["Rapid7"], ["Rapid7"]]
