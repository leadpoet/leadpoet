from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import scoring as arena_scoring
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.evaluation_clock import use_evaluation_date


def _signal(index: int, *, url: str | None = None, date: str = "2026-09-01") -> dict:
    label = "security hiring" if index == 0 else "cloud hiring"
    return {
        "source": "job_board",
        "description": f"Acme posted a {label} role.",
        "url": url or f"https://jobs.example.com/acme/{index}",
        "date": date,
        "snippet": f"Acme is {label}.",
        "why_now": "The active role shows current hiring demand.",
        "matched_icp_signal": index,
    }


def _company(*, intent_details: str | None = None) -> dict:
    signals = [_signal(0), _signal(1)]
    for signal in signals:
        signal.pop("source")
    row = {
        "company_name": "Acme",
        "company_website": "https://acme.example",
        "company_linkedin": "https://linkedin.com/company/acme",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series B",
        "country": "United States",
        "state": "California",
        "fit_summary": "Acme supplies software to business customers.",
        "fit_evidence_urls": ["https://acme.example/about"],
        "intent_signals": signals,
    }
    if intent_details is not None:
        row["intent_details"] = intent_details
        row.pop("fit_summary")
        row.pop("fit_evidence_urls")
        row["intent_signals"] = [{
            key: signal[key]
            for key in ("matched_icp_signal", "description", "date", "url")
        } for signal in signals]
    return row


def _icp() -> dict:
    return {
        "icp_id": "saved-pos16-shaped",
        "prompt": "Find Series B software companies hiring security and cloud staff",
        "industry": "Software",
        "sub_industry": "SaaS",
        "employee_count": ["51-200"],
        "company_stage": "Series B",
        "country": "United States",
        "intent_signals": ["Hiring security staff", "Hiring cloud staff"],
        "intent_signal_evidence_types": ["HIRING", "HIRING"],
        "intent_signal_max_age_days": [90, 90],
        "intent_max_age_days": 90,
        "max_companies": 1,
    }


def _identity(name: str = "Acme") -> dict:
    return {
        "decision": "match",
        "evidence_source": "company_web_reverification",
        "observed_name": name,
        "observed_domain": "acme.example",
        "observed_linkedin_slug": "acme",
    }


def _icp_model() -> ICPPrompt:
    return ICPPrompt(
        icp_id="saved-pos16-shaped",
        prompt="Find software companies hiring security and cloud staff",
        industry="Software",
        sub_industry="SaaS",
        employee_count="51-200",
        company_stage="Series B",
        geography="United States",
        country="United States",
        product_service="Software",
        intent_signals=["Hiring security staff"],
        intent_signal_evidence_types=["HIRING"],
        intent_signal_max_age_days=[90],
        intent_max_age_days=90,
    )


def _company_model() -> CompanyOutput:
    values = _company()
    values.pop("fit_summary")
    values.pop("fit_evidence_urls")
    values["intent_signals"] = [_signal(0)]
    return CompanyOutput.model_validate(values)


async def _matched_fit(*_args, **_kwargs):
    return company_fit_match(details={
        "dimension_evidence": {
            "identity": {"web_identity_receipt": _identity()}
        }
    })


def _record_positive(signal, evidence_signals, verdict_out) -> tuple:
    urls = [item.url for item in evidence_signals]
    quote = f"Acme posted the role described at {signal.url}."
    verdict_out.append({
        "decision": "verified",
        "pipeline_decision": "approve",
        "claim_support_verdict": "supported",
        "client_ready": True,
        "verification_trace": {
            "evidence_url": signal.url,
            "evidence_urls": urls,
            "evidence_source": "job_board",
            "extracted_signal_date": str(signal.date),
            "verified_source_context": [{"url": signal.url, "text": quote}],
            "intent_verdict": {"signal_evaluations": [{
                "signal_status": "supported",
                "same_entity_check": "pass",
                "supporting_quotes": [quote],
                "evidence_urls_used": [signal.url],
            }]},
            "final_disposition": "approve",
        },
    })
    return 54.0, 90, "in_window", str(signal.date), signal.matched_icp_signal


def _record_unavailable(verdict_out) -> tuple:
    verdict_out.append({
        "decision": "rejected_verifier_error",
        "pipeline_decision": "unavailable",
        "error_class": "provider_error",
        "client_ready": False,
    })
    return 0.0, 0, "uncertain", None, -1


def _scorer(*, intent_details: bool = False):
    policy = arena_scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        intent_details=intent_details,
    )
    return arena_scoring.lab_scorer(policy)


def test_native_retry_reuses_positive_sibling_and_retries_only_unavailable(
    monkeypatch,
):
    calls = Counter()

    async def score_one(signal, *_args, verdict_out, evidence_signals, **_kwargs):
        calls[signal.url] += 1
        if signal.matched_icp_signal == 1 and calls[signal.url] == 1:
            return _record_unavailable(verdict_out)
        return _record_positive(signal, evidence_signals, verdict_out)

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", _matched_fit)
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)

    with use_evaluation_date("2026-09-25"):
        result = arena_scoring.score_work_item(
            {"scored_run_id": "saved-pos16-shaped"},
            icp=_icp(),
            companies=[_company()],
            scorer=_scorer(),
            max_retries=2,
        )

    assert result[0]["final_score"] == 80.0
    assert calls == {
        "https://jobs.example.com/acme/0": 1,
        "https://jobs.example.com/acme/1": 2,
    }
    assert [row["counted_in_aggregate"] for row in result[0]["intent_signals_detail"]] == [
        True,
        True,
    ]


def test_native_same_criterion_retry_reuses_only_terminal_sources(monkeypatch):
    calls = Counter()
    urls = [f"https://jobs.example.com/acme/{suffix}" for suffix in "abc"]

    async def score_one(signal, *_args, verdict_out, evidence_signals, **_kwargs):
        calls[signal.url] += 1
        if signal.url == urls[1] and calls[signal.url] == 1:
            return _record_unavailable(verdict_out)
        return _record_positive(signal, evidence_signals, verdict_out)

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", _matched_fit)
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    company = _company()
    company["intent_signals"] = []
    for url in urls:
        signal = _signal(0, url=url)
        signal.pop("source")
        company["intent_signals"].append(signal)
    icp = {
        **_icp(),
        "intent_signals": ["Hiring security staff"],
        "intent_signal_evidence_types": ["HIRING"],
        "intent_signal_max_age_days": [90],
    }

    with use_evaluation_date("2026-09-25"):
        result = arena_scoring.score_work_item(
            {"scored_run_id": "saved-pos16-three-sources"},
            icp=icp,
            companies=[company],
            scorer=_scorer(),
            max_retries=2,
        )

    assert result[0]["final_score"] == 54.0
    assert calls == {urls[0]: 1, urls[1]: 2, urls[2]: 1}
    details = result[0]["intent_signals_detail"]
    assert sum(bool(row["counted_in_aggregate"]) for row in details) == 1
    assert all(row["matched_icp_signal"] == 0 for row in details)


@pytest.mark.parametrize(
    "change",
    ["context", "identity", "signal", "url", "date", "policy"],
)
def test_retry_cache_key_misses_changed_scoring_context(monkeypatch, change):
    calls = 0

    async def score_one(signal, *_args, verdict_out, evidence_signals, **_kwargs):
        nonlocal calls
        calls += 1
        return _record_positive(signal, evidence_signals, verdict_out)

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    cache = {}
    company = _company_model()
    icp = _icp_model()
    second_company = company
    second_icp = icp
    second_identity = _identity()
    second_context = "scope-one"
    second_flags = {}
    if change == "context":
        second_context = "scope-two"
    elif change == "identity":
        second_identity = _identity("Acme Holdings")
    elif change == "signal":
        second_icp = icp.model_copy(update={"intent_signals": ["Hiring security leaders"]})
    elif change == "url":
        second_company = CompanyOutput.model_validate({
            **company.model_dump(),
            "intent_signals": [_signal(
                0, url="https://jobs.example.com/acme/security"
            )],
        })
    elif change == "date":
        second_company = CompanyOutput.model_validate({
            **company.model_dump(),
            "intent_signals": [_signal(0, date="2026-09-02")],
        })
    else:
        second_flags = {"company_quality": True}

    async def run():
        common = {
            "integrity_policy": True,
            "intent_terminal_retry_cache": cache,
        }
        await lead_scorer.score_company_competition_intent_signal(
            company, icp,
            verified_company_identity=_identity(),
            retry_evidence_context_key="scope-one",
            **common,
        )
        await lead_scorer.score_company_competition_intent_signal(
            second_company, second_icp,
            verified_company_identity=second_identity,
            retry_evidence_context_key=second_context,
            **common,
            **second_flags,
        )

    with use_evaluation_date("2026-09-25"):
        asyncio.run(run())
    assert calls == 2


def test_invalid_retry_entry_is_removed_and_replaced(monkeypatch):
    calls = 0

    async def score_one(signal, *_args, verdict_out, evidence_signals, **_kwargs):
        nonlocal calls
        calls += 1
        return _record_positive(signal, evidence_signals, verdict_out)

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    company = _company_model()
    icp = _icp_model()
    cache = {}
    kwargs = {
        "integrity_policy": True,
        "verified_company_identity": _identity(),
        "intent_terminal_retry_cache": cache,
        "retry_evidence_context_key": "scope",
    }
    with use_evaluation_date("2026-09-25"):
        asyncio.run(lead_scorer.score_company_competition_intent_signal(
            company, icp, **kwargs
        ))
        key = next(iter(cache))
        cache[key] = {"raw": 54.0, "after_decay": 54.0}
        asyncio.run(lead_scorer.score_company_competition_intent_signal(
            company, icp, **kwargs
        ))

    assert calls == 2
    assert cache[key]["judge_verdict"]["verification_trace"][
        "final_disposition"
    ] == "approve"


def test_unavailable_result_is_never_cached(monkeypatch):
    calls = 0

    async def score_one(*_args, verdict_out, **_kwargs):
        nonlocal calls
        calls += 1
        return _record_unavailable(verdict_out)

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    cache = {}
    with use_evaluation_date("2026-09-25"):
        asyncio.run(lead_scorer.score_company_competition_intent_signal(
            _company_model(),
            _icp_model(),
            integrity_policy=True,
            verified_company_identity=_identity(),
            intent_terminal_retry_cache=cache,
            retry_evidence_context_key="scope",
        ))

    assert calls == 1
    assert cache == {}


@pytest.mark.parametrize(
    "mutation",
    [
        "zero_rejected",
        "unsupported_evaluation",
        "wrong_entity",
        "missing_source_context",
        "out_of_window",
    ],
)
def test_incomplete_or_rejected_terminal_result_is_not_cached(
    monkeypatch, mutation,
):
    calls = 0

    async def score_one(signal, *_args, verdict_out, evidence_signals, **_kwargs):
        nonlocal calls
        calls += 1
        result = list(_record_positive(signal, evidence_signals, verdict_out))
        verdict = verdict_out[-1]
        evaluation = verdict["verification_trace"]["intent_verdict"][
            "signal_evaluations"
        ][0]
        if mutation == "zero_rejected":
            result[0] = 0.0
            verdict.update(
                decision="rejected_three_stage",
                pipeline_decision="reject",
                client_ready=False,
            )
        elif mutation == "unsupported_evaluation":
            evaluation["signal_status"] = "partially_supported"
        elif mutation == "wrong_entity":
            evaluation.update(signal_status="wrong_entity", same_entity_check="fail")
        elif mutation == "missing_source_context":
            verdict["verification_trace"].pop("verified_source_context")
        else:
            result[2] = "out_of_window"
        return tuple(result)

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    cache = {}
    kwargs = {
        "integrity_policy": True,
        "verified_company_identity": _identity(),
        "intent_terminal_retry_cache": cache,
        "retry_evidence_context_key": "scope",
    }
    with use_evaluation_date("2026-09-25"):
        for _attempt in range(2):
            asyncio.run(lead_scorer.score_company_competition_intent_signal(
                _company_model(), _icp_model(), **kwargs
            ))

    assert calls == 2
    assert cache == {}


def test_cached_row_is_deep_copied_before_aggregation(monkeypatch):
    calls = 0

    async def score_one(signal, *_args, verdict_out, evidence_signals, **_kwargs):
        nonlocal calls
        calls += 1
        return _record_positive(signal, evidence_signals, verdict_out)

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    cache = {}
    kwargs = {
        "integrity_policy": True,
        "verified_company_identity": _identity(),
        "intent_terminal_retry_cache": cache,
        "retry_evidence_context_key": "scope",
    }
    with use_evaluation_date("2026-09-25"):
        first = asyncio.run(lead_scorer.score_company_competition_intent_signal(
            _company_model(), _icp_model(), **kwargs
        ))
        stored = next(iter(cache.values()))
        assert "counted_in_aggregate" not in stored
        first[-1][0]["judge_verdict"]["verification_trace"][
            "verified_source_context"
        ][0]["text"] = "mutated outside cache"
        second = asyncio.run(lead_scorer.score_company_competition_intent_signal(
            _company_model(), _icp_model(), **kwargs
        ))

    assert calls == 1
    assert second[-1][0]["counted_in_aggregate"] is True
    assert second[-1][0]["judge_verdict"]["verification_trace"][
        "verified_source_context"
    ][0]["text"] != "mutated outside cache"
    assert "counted_in_aggregate" not in stored


def test_retry_cache_is_new_for_each_score_work_item(monkeypatch):
    calls = Counter()

    async def score_one(signal, *_args, verdict_out, evidence_signals, **_kwargs):
        calls[signal.url] += 1
        return _record_positive(signal, evidence_signals, verdict_out)

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", _matched_fit)
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    scorer = _scorer()
    with use_evaluation_date("2026-09-25"):
        for run_id in ("first", "second"):
            result = arena_scoring.score_work_item(
                {"scored_run_id": run_id},
                icp=_icp(),
                companies=[_company()],
                scorer=scorer,
            )
            assert result[0]["final_score"] == 80.0

    assert set(calls.values()) == {2}


def test_retry_cache_still_reruns_paragraph_and_aggregation(monkeypatch):
    signal_calls = Counter()
    paragraph_calls = 0
    paragraph_inputs = []

    async def score_one(signal, *_args, verdict_out, evidence_signals, **_kwargs):
        signal_calls[signal.url] += 1
        return _record_positive(signal, evidence_signals, verdict_out)

    async def review(_company, _icp, signal_results, *_args, **_kwargs):
        nonlocal paragraph_calls
        paragraph_calls += 1
        paragraph_inputs.append(signal_results)
        return {
            "gate": "intent_details",
            "decision": "unavailable" if paragraph_calls == 1 else "match",
        }

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", _matched_fit)
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    from qualification.scoring import intent_details
    monkeypatch.setattr(intent_details, "review_intent_details", review)

    with use_evaluation_date("2026-09-25"):
        result = arena_scoring.score_work_item(
            {"scored_run_id": "paragraph-retry"},
            icp=_icp(),
            companies=[_company(intent_details="Acme is hiring security and cloud staff.")],
            scorer=_scorer(intent_details=True),
            max_retries=2,
        )

    assert result[0]["final_score"] == 80.0
    assert paragraph_calls == 2
    assert paragraph_inputs[0] is not paragraph_inputs[1]
    assert all(
        row["counted_in_aggregate"]
        for rows in paragraph_inputs
        for row in rows
    )
    assert set(signal_calls.values()) == {1}
    assert all(
        row["counted_in_aggregate"]
        for row in result[0]["intent_signals_detail"]
    )
