from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import json

import pytest
from unittest.mock import AsyncMock

from gateway.qualification.models import CompanyOutput, ICPPrompt, IntentSignal
from qualification.scoring import lead_scorer, signal_judgment_cache
from qualification.scoring import intent_verification_three_stage as three_stage
from qualification.scoring.evaluation_clock import use_evaluation_date


def _signal(index: int, suffix: str) -> dict:
    return {
        "source": "job_board",
        "description": f"Acme posted a {suffix} role.",
        "url": f"https://jobs.example.com/acme/{suffix}",
        "date": "2026-09-01",
        "snippet": f"Acme is hiring {suffix} staff.",
        "matched_icp_signal": index,
    }


def _company(signals: list[dict]) -> CompanyOutput:
    return CompanyOutput.model_validate({
        "company_name": "Acme",
        "company_website": "https://acme.example",
        "company_linkedin": "https://linkedin.com/company/acme",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series B",
        "country": "United States",
        "state": "California",
        "intent_signals": signals,
    })


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="signal-cache-control",
        prompt="Find software companies hiring security and cloud staff",
        industry="Software", sub_industry="SaaS",
        employee_count="51-200", company_stage="Series B",
        geography="United States", country="United States",
        product_service="Software",
        intent_signals=["Hiring security staff", "Hiring cloud staff"],
        intent_signal_evidence_types=["HIRING", "HIRING"],
        intent_signal_max_age_days=[90, 90], intent_max_age_days=90,
    )


def _identity(name: str = "Acme") -> dict:
    return {
        "decision": "match", "evidence_source": "company_web_reverification",
        "observed_name": name, "observed_domain": "acme.example",
        "observed_linkedin_slug": "acme",
    }


def _verdict(signal, evidence_signals, *, status: str = "supported") -> dict:
    urls = [item.url for item in evidence_signals]
    quote = f"Acme posted the role described at {signal.url}."
    positive = status == "supported"
    return {
        "decision": "verified" if positive else "rejected_three_stage",
        "pipeline_decision": "approve" if positive else "reject",
        "claim_support_verdict": status,
        "client_ready": positive,
        "verification_trace": {
            "evidence_url": signal.url, "evidence_urls": urls,
            "evidence_source": "job_board",
            "extracted_signal_date": str(signal.date),
            "verified_source_context": [{"url": signal.url, "text": quote}],
            "intent_verdict": {"signal_evaluations": [{
                "signal_status": status,
                "same_entity_check": "fail" if status == "wrong_entity" else "pass",
                "verification_mode": "source_grounded",
                "supporting_quotes": [quote],
                "evidence_urls_used": [signal.url],
            }]},
            "final_disposition": "approve" if positive else "reject",
        },
    }


def _exchange_for(company: CompanyOutput, storage: dict, log: list):
    async def exchange(request):
        log.append(deepcopy(request))
        signal = company.intent_signals[request["signal_index"]]
        scope = json.dumps({
            "signal": signal.model_dump(mode="json"),
            "identity": request["verified_company_identity"],
            "funding": request["verified_funding_stage_context"],
        }, sort_keys=True)
        if request["operation"] == "claim":
            if scope in storage:
                return {"status": "hit", "reference": scope,
                        "judgment": deepcopy(storage[scope])}
            return {"status": "miss", "reference": scope}
        assert request["reference"] == scope
        if request["judgment"] is None:
            return {"status": "released", "reference": scope}
        storage.setdefault(scope, deepcopy(request["judgment"]))
        return {"status": "stored", "reference": scope}
    return exchange


def _score(company, storage, log, *, identity=None):
    async def run():
        with signal_judgment_cache.install_exchange(
            _exchange_for(company, storage, log)
        ):
            return await lead_scorer.score_company_competition_intent_signal(
                company, _icp(), integrity_policy=True,
                verified_company_identity=identity or _identity(),
            )
    with use_evaluation_date("2026-09-25"):
        return asyncio.run(run())


def test_reuses_identical_signal_across_different_siblings_and_order(monkeypatch):
    calls = Counter()

    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        calls[signal.url] += 1
        verdict_out.append(_verdict(signal, evidence_signals))
        return 54.0, 90, "in_window", str(signal.date), signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    a, b, c = _signal(0, "security"), _signal(1, "cloud"), _signal(1, "platform")
    storage, log = {}, []
    first = _score(_company([a, b]), storage, log)
    second = _score(_company([a, c]), storage, log)
    reordered = _score(_company([c, a]), storage, log)

    assert calls == Counter({
        a["url"]: 1, b["url"]: 1, c["url"]: 1,
    })
    assert len(storage) == 3
    assert first[1] == second[1] == reordered[1]
    assert all(row["counted_in_aggregate"] for row in reordered[-1])
    assert [request["operation"] for request in log].count("complete") == 3


def test_signal_index_tracks_flattened_bounded_evidence_order(monkeypatch):
    calls = Counter()

    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        calls[signal.url] += 1
        verdict_out.append(_verdict(signal, evidence_signals))
        return 54.0, 90, "in_window", str(signal.date), signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    signals = [_signal(0, suffix) for suffix in ("one", "two", "three", "four")]
    storage, log = {}, []
    first = _score(_company(signals), storage, log)
    second = _score(_company(signals[:3][::-1]), storage, log)

    assert len(first[-1]) == len(second[-1]) == 3
    assert [request["signal_index"] for request in log
            if request["operation"] == "claim"] == [0, 1, 2, 0, 1, 2]
    assert calls == Counter({signal["url"]: 1 for signal in signals[:3]})


def test_gateway_registers_both_equal_company_positions_even_with_local_retry_cache(monkeypatch):
    calls = Counter()

    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        calls[signal.url] += 1
        verdict_out.append(_verdict(signal, evidence_signals))
        return 54.0, 90, "in_window", str(signal.date), signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    signal = _signal(0, "security")
    company = _company([signal])
    storage, log, local = {}, [], {}

    async def score():
        with signal_judgment_cache.install_exchange(_exchange_for(company, storage, log)):
            return await lead_scorer.score_company_competition_intent_signal(
                company, _icp(), integrity_policy=True,
                verified_company_identity=_identity(),
                intent_terminal_retry_cache=local,
                retry_evidence_context_key="same-company-retry",
            )

    with use_evaluation_date("2026-09-25"):
        first = asyncio.run(score())
        second = asyncio.run(score())
    assert [request["signal_index"] for request in log if request["operation"] == "claim"] == [0, 0]
    assert [request["operation"] for request in log].count("complete") == 1
    assert calls[signal["url"]] == 1
    assert all(row["signal_judgment_reference"] for row in first[-1] + second[-1])


@pytest.mark.parametrize("status", ["supported", "contradicted"])
def test_every_evaluation_must_have_its_own_cited_grounded_quote(status):
    signal = IntentSignal.model_validate(_signal(0, "security"))
    verdict = _verdict(signal, [signal], status=status)
    positive = status == "supported"
    row = {
        "raw": 54.0 if positive else 0.0,
        "after_decay": 54.0 if positive else 0.0,
        "decay": 1.0 if positive else 0.0,
        "confidence": 90 if positive else 0,
        "date_status": "in_window" if positive else "verified",
        "date_verdict": "in_window" if positive else "not_evaluated",
        "claim_support_verdict": status, "matched_icp_signal": 0,
        "evidence_type": "HIRING", "evidence_urls": [signal.url],
        "judge_verdict": verdict,
    }
    check = lambda value: signal_judgment_cache.reusable_terminal_judgment(
        value, expected_urls=[signal.url], expected_index=0,
        expected_evidence_type="HIRING",
    )
    assert check(row)
    evaluations = row["judge_verdict"]["verification_trace"]["intent_verdict"]["signal_evaluations"]
    ungrounded = deepcopy(evaluations[0])
    ungrounded["supporting_quotes"] = ["A second claim absent from the cited page"]
    evaluations.append(ungrounded)
    assert not check(row)


@pytest.mark.parametrize("negative_status", ["contradicted", "wrong_entity"])
def test_grounded_negative_reused_but_unavailable_released(
    monkeypatch, negative_status,
):
    calls = Counter()

    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        calls[signal.url] += 1
        if signal.matched_icp_signal == 0:
            verdict_out.append(_verdict(
                signal, evidence_signals, status=negative_status,
            ))
            return 0.0, 0, "verified", None, -1
        verdict_out.append({
            "decision": "rejected_verifier_error",
            "pipeline_decision": "unavailable", "error_class": "provider_error",
        })
        return 0.0, 0, "uncertain", None, -1

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    a, b = _signal(0, "security"), _signal(1, "cloud")
    storage, log = {}, []
    _score(_company([a, b]), storage, log)
    _score(_company([a, b]), storage, log)

    assert calls == Counter({a["url"]: 1, b["url"]: 2})
    assert len(storage) == 1
    assert any(request["operation"] == "complete"
               and request["judgment"] is None for request in log)


def test_changed_verified_identity_cannot_hit(monkeypatch):
    calls = 0

    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        nonlocal calls
        calls += 1
        verdict_out.append(_verdict(signal, evidence_signals))
        return 54.0, 90, "in_window", str(signal.date), signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    company = _company([_signal(0, "security")])
    storage, log = {}, []
    _score(company, storage, log)
    _score(company, storage, log, identity=_identity("Acme Holdings"))
    assert calls == 2


def test_source_grounded_stale_negative_is_reused(monkeypatch):
    calls = 0

    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        nonlocal calls
        calls += 1
        verdict = _verdict(signal, evidence_signals)
        verdict.update(
            decision="rejected_freshness", date_verdict="out_of_window",
            authoritative_date="2026-01-01", client_ready=False,
        )
        verdict_out.append(verdict)
        return 0.0, 0, "out_of_window", "2026-01-01", -1

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    company = _company([_signal(0, "security")])
    storage, log = {}, []
    first = _score(company, storage, log)
    second = _score(company, storage, log)

    assert calls == 1
    assert len(storage) == 1
    assert first[-1][0]["date_status"] == "out_of_window"
    assert second[-1][0]["signal_judgment_reference"]


def test_funding_context_urls_must_match_verified_context(monkeypatch):
    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        verdict_out.append(_verdict(signal, evidence_signals))
        return 54.0, 90, "in_window", str(signal.date), signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    signal = _signal(0, "security")
    storage, log = {}, []
    _score(_company([signal]), storage, log)
    row = next(iter(storage.values()))
    stage_url = "https://acme.example/news/funding"
    row["judge_verdict"]["verification_trace"]["verified_source_context"].append({
        "url": stage_url, "text": "Acme announced a funding round.",
        "source_role": "matched_company_stage",
    })
    expected = {
        "expected_urls": [signal["url"]], "expected_index": 0,
        "expected_evidence_type": "HIRING",
    }
    assert not signal_judgment_cache.reusable_terminal_judgment(row, **expected)
    assert signal_judgment_cache.reusable_terminal_judgment(
        row, **expected, allowed_context_urls=[stage_url],
    )


def test_invalid_hit_fails_closed(monkeypatch):
    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        verdict_out.append(_verdict(signal, evidence_signals))
        return 54.0, 90, "in_window", str(signal.date), signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    company = _company([_signal(0, "security")])
    storage, log = {}, []
    _score(company, storage, log)
    key = next(iter(storage))
    storage[key]["judge_verdict"]["verification_trace"]["verified_source_context"] = []
    with pytest.raises(signal_judgment_cache.SignalJudgmentCacheError):
        _score(company, storage, log)


def test_grounded_wrong_entity_retains_cited_source_without_changing_decision(
    monkeypatch,
):
    url = "https://news.example.com/story/leadership"
    quote = "Beta appointed a new chief executive."
    text = "Acme was mentioned in the report. " + quote
    provider = AsyncMock(return_value={
        "answer": {
            "signal_evaluations": [{
                "signal_status": "wrong_entity", "confidence": "high",
                "same_entity_check": "fail",
                "verification_mode": "source_grounded",
                "evidence_urls_used": [url],
                "supporting_quotes": [quote],
                "unsupported_parts": [], "risk_notes": [],
            }],
            "overall_verdict": "disqualified", "overall_confidence": "high",
            "summary": "The cited event concerns Beta.", "missing_or_risks": [],
        },
        "model": "test-model", "usage": {},
    })
    monkeypatch.setattr(three_stage, "_call_openrouter", provider)
    monkeypatch.setattr(three_stage, "_fetch_sd_then_exa", AsyncMock(
        return_value={"results": [{"url": url, "text": text}], "statuses": []}
    ))

    async def run():
        return await three_stage.verify_three_stage(
            None, company_name="Acme", company_website="https://acme.com",
            company_linkedin="https://linkedin.com/company/acme",
            source_url=url, miner_claim="Acme appointed a new chief executive.",
            target_signal_text="Recent leadership change",
            miner_signal_date="2026-09-14", evidence_type="LEADERSHIP_CHANGE",
            declared_source="news", integrity_policy=True,
            stage1_soft_reject=True,
            verified_company_identity={
                **_identity("acme"), "observed_domain": "acme.com",
            },
            buyer_max_age_days=365,
            evidence_bundle=[{
                "url": url, "description": "Acme appointed a new chief executive.",
                "date": "2026-09-14", "snippet": quote,
            }],
        )

    with use_evaluation_date("2026-09-25"):
        result = asyncio.run(run())
    assert result["decision"] == "reject"
    assert result["client_ready"] is False
    assert result["verdict"]["signal_evaluations"][0]["signal_status"] == "wrong_entity"
    assert result.get("verified_source_context"), {
        "scrape": result.get("scrape"),
        "stage3": result.get("stage3"),
        "evaluation": result["verdict"]["signal_evaluations"][0],
    }
    assert result["verified_source_context"][0]["url"] == url
    assert quote in result["verified_source_context"][0]["text"]


@pytest.mark.usefixtures("scorer_process_environment")
def test_full_scoring_input_reuses_a_across_mixed_company_submissions(
    monkeypatch,
):
    from lab_arena import scorer_entrypoint, scoring, signal_cache_control
    from qualification.scoring.company_fit_decision import company_fit_match
    from tests.lab_arena.legacy_company_judgments_test import _lease
    from tests.test_arena_intent_retry_cache import (
        _company as submission_company, _icp as submission_icp,
    )

    # The real scorer entrypoint binds these exact placeholder credentials.
    for name, value in scorer_entrypoint.PLACEHOLDER_CREDENTIALS.items():
        monkeypatch.setenv(name, value)

    calls = Counter()
    storage = {}
    current_input = None

    async def fit(company, *_args, **_kwargs):
        name = company.company_name.casefold()
        domain = company.company_website.removeprefix("https://")
        return company_fit_match(details={"dimension_evidence": {
            "identity": {"web_identity_receipt": {
                "decision": "match", "evidence_source": "company_web_reverification",
                "observed_name": name, "observed_domain": domain,
                "observed_linkedin_slug": name,
            }}
        }})

    async def score_one(signal, *_args, evidence_signals, verdict_out, **_kwargs):
        calls[signal.url] += 1
        verdict_out.append(_verdict(signal, evidence_signals))
        return 54.0, 90, "in_window", str(signal.date), signal.matched_icp_signal

    async def exchange(request):
        row = current_input["companies"][request["company_index"]]["intent_signals"][request["signal_index"]]
        key = json.dumps({
            "company": current_input["companies"][request["company_index"]]["company_name"],
            "signal": row, "identity": request["verified_company_identity"],
        }, sort_keys=True)
        if request["operation"] == "claim":
            if key in storage:
                return {"status": "hit", "reference": key,
                        "judgment": deepcopy(storage[key])}
            return {"status": "miss", "reference": key}
        assert request["reference"] == key
        if request["judgment"] is None:
            return {"status": "released", "reference": key}
        storage.setdefault(key, deepcopy(request["judgment"]))
        return {"status": "stored", "reference": key}

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    monkeypatch.setattr(signal_cache_control, "exchange", exchange)

    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        company_judgment_cache=True,
    )
    policy["signal_judgment_policy"] = "signal_judgment_v1"
    first = submission_company()
    second = deepcopy(first)
    second.update(
        company_name="Other", company_website="https://other.example",
        company_linkedin="https://linkedin.com/company/other",
    )
    second["intent_signals"][0]["url"] = "https://jobs.example.com/other/0"
    second["intent_signals"][1]["url"] = "https://jobs.example.com/other/1"

    def score(primary):
        nonlocal current_input
        icp = submission_icp()
        icp["max_companies"] = 2
        current_input = scoring.build_scoring_input(
            scored_run_id="full-score-signal-cache", icp=icp,
            companies=[primary, second], policy=policy,
            evaluation_date="2026-09-25",
        )
        current_input["company_judgment_cache"] = _lease(current_input)
        return scorer_entrypoint.score_input(current_input)

    first_output = score(first)
    changed = deepcopy(first)
    changed["intent_signals"][1].update(
        url="https://jobs.example.com/acme/platform",
        description="Acme posted a platform role.",
        snippet="Acme is hiring platform staff.",
    )
    second_output = score(changed)
    assert "failure" not in first_output
    assert "failure" not in second_output
    assert calls[first["intent_signals"][0]["url"]] == 1
    assert calls[first["intent_signals"][1]["url"]] == 1
    assert calls[changed["intent_signals"][1]["url"]] == 1
    assert calls[second["intent_signals"][0]["url"]] == 1
    assert calls[second["intent_signals"][1]["url"]] == 1
    assert first_output["breakdowns"][0]["final_score"] == second_output["breakdowns"][0]["final_score"]
    assert second_output["breakdowns"][0]["intent_signals_detail"][0]["signal_judgment_reference"]
