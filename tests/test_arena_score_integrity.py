from __future__ import annotations

import asyncio
import copy
from datetime import date

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.competition_models import CompetitionCompany
from qualification.scoring import lead_scorer
from qualification.scoring.arena_integrity import (
    canonical_company_identity,
    company_identity_alias_keys,
    mark_duplicate_companies,
    source_dates_from_verdict,
    source_grounded_date_verdict,
)
from qualification.scoring.competition import (
    CompetitionCompanyScorer,
    _normalized_icp,
    scorer_breakdown_has_retryable_infrastructure_failure,
)
from qualification.scoring.evaluation_clock import use_evaluation_date


def _public_company(
    name: str,
    *,
    domain: str = "acme.co.uk",
    linkedin: str = "",
    signal_date: str | None = None,
) -> dict:
    return {
        "company_name": name,
        "company_website": f"https://{domain}",
        "company_linkedin": linkedin,
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "",
        "country": "United States",
        "state": "",
        "fit_summary": "A software company matching the requested market.",
        "fit_evidence_urls": [f"https://{domain}/about"],
        "intent_signals": [{
            "matched_icp_signal": 0,
            "description": "The company announced a product launch.",
            "date": signal_date,
            "why_now": "The launch creates a timely buying trigger.",
            "url": f"https://{domain}/news/launch",
            "snippet": "The company launched the product.",
        }],
    }


def _icp() -> dict:
    return {
        "icp_id": "arena:test:0",
        "prompt": "Find software companies with a product launch",
        "industry": "Software",
        "sub_industry": "SaaS",
        "employee_count": ["51-200"],
        "company_stage": "Any",
        "country": "United States",
        "intent_signals": ["Announced a product launch"],
        "intent_max_age_days": 90,
        "max_companies": 5,
    }


def _fit_receipt(
    observed_name: str, observed_domain: str, observed_linkedin_slug: str = ""
) -> list[dict]:
    return [{
        "gate": "company_fit",
        "decision": "match",
        "dimension_evidence": {
            "identity": {
                "web_identity_receipt": {
                    "decision": "match",
                    "evidence_source": "company_web_reverification",
                    "observed_name": observed_name,
                    "observed_domain": observed_domain,
                    "observed_linkedin_slug": observed_linkedin_slug,
                }
            }
        },
    }]


def _positive_breakdown(
    observed_name: str, observed_domain: str, observed_linkedin_slug: str = ""
) -> dict:
    return {
        "icp_fit": 0.0,
        "decision_maker": 0.0,
        "intent_signal_raw": 60.0,
        "time_decay_multiplier": 1.0,
        "intent_signal_final": 60.0,
        "cost_penalty": 0.0,
        "time_penalty": 0.0,
        "final_score": 60.0,
        "failure_reason": None,
        "verifier_gate_receipts": _fit_receipt(
            observed_name, observed_domain, observed_linkedin_slug
        ),
        "intent_signals_detail": [{
            "raw": 60.0,
            "after_decay": 60.0,
            "matched_icp_signal": 0,
            "judge_verdict": {
                "decision": "verified",
                "pipeline_decision": "approve",
                "verification_trace": {
                    "intent_verdict": {
                        "signal_evaluations": [{"signal_status": "supported"}]
                    }
                },
            },
        }],
    }


def test_public_contract_accepts_null_date_and_round_trips() -> None:
    validated = CompetitionCompany.model_validate(_public_company("Acme"))
    assert validated.intent_signals[0].date is None
    assert CompetitionCompany.model_validate_json(validated.model_dump_json()) == validated


def test_identity_uses_psl_domain_and_ignores_unverified_linkedin() -> None:
    first = _public_company(
        "Acme, Inc.",
        domain="jobs.acme.co.uk",
        linkedin="https://linkedin.com/company/acme-old",
    )
    alias = _public_company(
        "Acme",
        domain="www.acme.co.uk",
        linkedin="https://linkedin.com/company/acme-new",
    )
    distinct = _public_company("Acme Labs", domain="store.acme.co.uk")
    fake_linkedin_other_domain = _public_company(
        "Acme",
        domain="different.example",
        linkedin="https://linkedin.com/company/acme-old",
    )
    decisions = mark_duplicate_companies(
        [first, alias, distinct, fake_linkedin_other_domain], limit=4
    )
    assert decisions[0].identity.registrable_domain == "acme.co.uk"
    assert [row.duplicate_company for row in decisions] == [
        False, True, False, False
    ]
    assert decisions[1].duplicate_of_index == 0


def test_verified_identity_can_canonicalize_a_submitted_alias() -> None:
    company = _public_company("IBM", domain="ibm.com")
    receipt = {
        "decision": "match",
        "evidence_source": "company_homepage",
        "observed_name": "International Business Machines, Inc.",
        "observed_domain": "ibm.com",
    }
    identity = canonical_company_identity(
        company, verified_identity_receipt=receipt
    )
    assert identity.verified is True
    assert identity.key == "domain:ibm.com|name:international business machines"


def test_verified_linkedin_corroborates_distinct_legal_name_aliases() -> None:
    short = canonical_company_identity(
        _public_company("IBM", domain="ibm.com"),
        verified_identity_receipt={
            "decision": "match",
            "evidence_source": "company_web_reverification",
            "observed_name": "IBM",
            "observed_domain": "ibm.com",
            "observed_linkedin_slug": "ibm",
        },
    )
    legal = canonical_company_identity(
        _public_company("International Business Machines", domain="ibm.com"),
        verified_identity_receipt={
            "decision": "match",
            "evidence_source": "company_web_reverification",
            "observed_name": "International Business Machines",
            "observed_domain": "ibm.com",
            "observed_linkedin_slug": "ibm",
        },
    )
    assert short.key == legal.key == "linkedin:ibm"
    assert set(company_identity_alias_keys(short)) == {
        "domain:ibm.com|name:ibm",
        "linkedin:ibm",
    }
    assert set(company_identity_alias_keys(legal)) == {
        "domain:ibm.com|name:international business machines",
        "linkedin:ibm",
    }


def test_adapter_zeros_only_after_first_verified_identity(monkeypatch) -> None:
    calls: list[str] = []

    async def score_company(**kwargs):
        calls.append(kwargs["company"].company_name)
        observed_name = (
            "Acme Labs" if kwargs["company"].company_name == "Acme Labs" else "Acme"
        )
        return _positive_breakdown(observed_name, "acme.co.uk")

    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent", score_company
    )
    rows = asyncio.run(
        CompetitionCompanyScorer(integrity_policy=True).score_with_breakdowns(
            [
                _public_company("Acme, Inc.", linkedin="https://linkedin.com/company/a"),
                _public_company("Acme", linkedin="https://linkedin.com/company/b"),
                _public_company("Acme Labs"),
            ],
            _icp(),
            False,
        )
    )
    assert calls == ["Acme, Inc.", "Acme Labs"]
    assert [row["company_index"] for row in rows] == [0, 1, 2]
    assert [row["duplicate_company"] for row in rows] == [False, True, False]
    assert rows[1]["final_score"] == 0.0
    assert rows[0]["company_qualified"] is True
    assert rows[2]["company_qualified"] is True


def test_verified_entity_deduplicates_across_domains_and_keeps_subsidiaries(monkeypatch):
    observations = {
        "Acme": ("Acme", "acme.com", "acme"),
        "Acme Global": ("Acme Global", "acme.io", "acme"),
        "Acme Labs": ("Acme Labs", "acme.com", "acme-labs"),
        "Acme Medical": ("Acme Medical", "acme.com", "acme-medical"),
        "Unrelated": ("Unrelated", "unrelated.com", ""),
    }
    async def score_company(**kwargs):
        return _positive_breakdown(*observations[kwargs["company"].company_name])
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    companies = [
        _public_company(name, domain=observed[1], linkedin="https://linkedin.com/company/acme")
        for name, observed in observations.items()
    ]
    rows = asyncio.run(CompetitionCompanyScorer(integrity_policy=True).score_with_breakdowns(companies, _icp(), False))
    assert [row["duplicate_company"] for row in rows] == [False, True, False, False, False]
    assert [row["final_score"] for row in rows] == [60, 0, 60, 60, 60]
    assert rows[0]["company_identity_key"] == rows[1]["company_identity_key"] == "linkedin:acme"
    assert len({row["company_identity_key"] for row in rows if row["company_qualified"]}) == 4


@pytest.mark.parametrize("distinct_urls", [False, True])
def test_repeated_primary_evidence_gets_one_decision_not_a_judging_lottery(monkeypatch, distinct_urls):
    calls = []
    async def judge(signal, *args, evidence_signals, **kwargs):
        calls.append(evidence_signals)
        # The old code would reach the fifth, lucky acceptance.
        return (54.0 if len(calls) == 5 else 0.0), 90, "uncertain", None, 0
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", judge)
    evidence = {"source": "news", "description": "Acme launched a product",
                "url": "https://acme.com/launch", "date": None,
                "snippet": "Product launch", "matched_icp_signal": 0}
    signals = [dict(evidence, url=evidence["url"] + (str(i) if distinct_urls else "")) for i in range(5)]
    result = asyncio.run(lead_scorer.score_company_competition_intent_signal(_company_model(signals), _icp_model(), integrity_policy=True))
    assert len(calls) == 1
    assert len(calls[0]) == (3 if distinct_urls else 1)
    assert result[1] == 0
    assert not lead_scorer.required_intent_satisfied(result[-1])


def test_combined_verifier_fetches_all_sources_and_issues_one_final_judgment(monkeypatch):
    import httpx
    from qualification.scoring import intent_verification_three_stage as verifier
    from tests.test_site_verifier_three_stage_source_grounding import supported

    urls = ["https://acme.com/news/a", "https://news.example.com/acme"]
    calls, fetches = [], []
    async def fetch(values):
        fetches.append(values)
        return {"results": [{"url": url, "text": "Acme launched a new product.",
                             "source_publication_date": day} for url, day in zip(urls, ["2026-08-01", "2026-09-01"])], "statuses": []}
    async def judge(client, model, prompt):
        calls.append(prompt)
        return supported(urls[0])
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    monkeypatch.setenv("RESEARCH_LAB_INTENT_CORROBORATION_RESCUE", "1")
    async def run():
        async with httpx.AsyncClient() as client:
            return await verifier.verify_three_stage(client, company_name="Acme",
                company_website="https://acme.com", company_linkedin="", source_url=urls[0],
                miner_claim="Acme launched a product", target_signal_text="Product launch",
                integrity_policy=True, stage1_soft_reject=True,
                evidence_bundle=[{"url": url, "description": "Acme launched a product", "date": None, "snippet": "Product launch"} for url in urls])
    result = asyncio.run(run())
    assert result["client_ready"] is True
    assert fetches == [urls]
    assert len(calls) == 2  # One advisory pass, one source-grounded terminal pass.
    assert "COMBINED CRITERION EVIDENCE" in calls[1]
    assert all(url in calls[1] for url in urls)
    assert result["source_publication_dates"] == ["2026-08-01"]


def test_adapter_merges_only_verifier_corroborated_name_aliases(monkeypatch) -> None:
    async def score_company(**kwargs):
        return _positive_breakdown(
            kwargs["company"].company_name,
            "ibm.com",
            "ibm",
        )

    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent", score_company
    )
    rows = asyncio.run(
        CompetitionCompanyScorer(integrity_policy=True).score_with_breakdowns(
            [
                _public_company("IBM", domain="ibm.com"),
                _public_company("International Business Machines", domain="ibm.com"),
            ],
            _icp(),
            False,
        )
    )
    assert [row["duplicate_company"] for row in rows] == [False, True]
    assert rows[0]["company_identity_key"] == rows[1]["company_identity_key"]
    assert "linkedin:ibm" in rows[0][
        "company_identity_alias_keys"
    ]


def test_source_date_prefers_old_event_over_new_publication() -> None:
    event, publications = source_dates_from_verdict(
        {"risk_notes": ["source_event_date:2025-01-01"]},
        ["2026-09-01"],
    )
    verdict = source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=90,
        evaluated_on=date(2026, 9, 10),
    )
    assert verdict.verdict == "out_of_window"
    assert verdict.basis == "event_date"
    assert verdict.authoritative_date == "2025-01-01"


def test_missing_or_conflicting_source_dates_are_uncertain() -> None:
    assert source_grounded_date_verdict(
        event_date=None,
        publication_dates=[],
        buyer_cap_days=90,
        evaluated_on=date(2026, 9, 10),
    ).verdict == "uncertain"
    event, publications = source_dates_from_verdict(
        {"risk_notes": [
            "source_event_date:2026-08-01",
            "source_event_date:2025-01-01",
            "source_publication_date:2026-09-01",
        ]},
        ["2026-09-01"],
    )
    assert source_grounded_date_verdict(
        event_date=event,
        publication_dates=publications,
        buyer_cap_days=90,
        evaluated_on=date(2026, 9, 10),
    ).verdict == "uncertain"
    assert source_grounded_date_verdict(
        event_date=None,
        publication_dates=["2026-08-01", "2025-01-01"],
        buyer_cap_days=90,
        evaluated_on=date(2026, 9, 10),
    ).verdict == "uncertain"


def _company_model(signals: list[dict]) -> CompanyOutput:
    return CompanyOutput(
        company_name="Acme",
        company_website="https://acme.com",
        company_linkedin="",
        industry="Software",
        employee_count="51-200",
        country="United States",
        intent_signals=signals,
    )


def _icp_model() -> ICPPrompt:
    return ICPPrompt(
        icp_id="arena:test:0",
        prompt="Find software companies",
        industry="Software",
        sub_industry="SaaS",
        target_roles=[],
        target_seniority="",
        employee_count="51-200",
        company_stage="Any",
        geography="United States",
        country="United States",
        product_service="Software",
        required_attribute="",
        excluded_companies=[],
        intent_signals=["Launch", "Hiring"],
        intent_signal_evidence_types=["NEWS", "HIRING"],
        intent_max_age_days=90,
    )


def test_integrity_judges_combined_evidence_once_per_requested_signal(monkeypatch) -> None:
    calls = []
    async def score_one(signal, *args, verdict_out=None, **kwargs):
        calls.append([item.url for item in kwargs["evidence_signals"]])
        score = 60.0 if "primary strong" in signal.description else 54.0
        if verdict_out is not None:
            verdict_out.append({"decision": "verified", "pipeline_decision": "approve"})
        return score, 90, "uncertain", None, signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", score_one)
    company = _company_model([
        {"source": "news", "description": "primary strong evidence", "url": "https://acme.com/a", "date": None, "snippet": "launch", "matched_icp_signal": 0},
        {"source": "news", "description": "primary backup evidence", "url": "https://acme.com/b", "date": None, "snippet": "launch", "matched_icp_signal": 0},
        {"source": "news", "description": "bonus evidence", "url": "https://acme.com/c", "date": None, "snippet": "hiring", "matched_icp_signal": 1},
    ])
    result = asyncio.run(lead_scorer.score_company_competition_intent_signal(
        company, _icp_model(), integrity_policy=True
    ))
    assert result[0] == 80.0
    details = result[-1]
    assert calls == [["https://acme.com/a", "https://acme.com/b"], ["https://acme.com/c"]]
    assert [row["counted_in_aggregate"] for row in details] == [True, True]
    assert all(row["raw"] > 0 for row in details)


def test_integrity_date_policy_accepts_mismatch_in_window_and_rejects_old_event(
    monkeypatch,
) -> None:
    from qualification.scoring import intent_verification_three_stage as verifier

    async def judge(*args, **kwargs):
        assert kwargs["declared_source"] == "news"
        return {
            "client_ready": True,
            "decision": "approve",
            "rejection_reason": "",
            "stage1": {"status": "supported"},
            "stage3": {
                "status": "supported",
                "decision": "approve",
                "claim_matches_miner_date": "contradicted",
            },
            "scrape": {"result_count": 1, "statuses": []},
            "company_check": True,
            "job_publisher_relationship": "not_applicable",
            "source_publication_dates": [judge.publication],
            "verdict": {"signal_evaluations": [{
                "signal_status": "supported",
                "verification_mode": "source_grounded",
                "same_entity_check": "pass",
                "confidence": "high",
                "risk_notes": list(judge.risk_notes),
            }]},
        }

    judge.publication = "2026-08-20"
    judge.risk_notes = ["source_publication_date:2026-08-20"]
    monkeypatch.setattr(verifier, "verify_three_stage", judge)
    signal = _company_model([{
        "source": "news", "description": "Acme launched a product", "url": "https://news.example.com/acme", "date": "2026-01-01", "snippet": "launch", "matched_icp_signal": 0,
    }]).intent_signals[0]
    with use_evaluation_date("2026-09-10"):
        accepted = asyncio.run(lead_scorer._score_single_intent_signal(
            signal, _icp_model(), None, "Acme", "https://acme.com",
            trust_signal_date=True, stage1_soft_reject=True,
            llm_only_intent_gate=True, integrity_policy=True,
        ))
    assert accepted[0] == 54.0
    assert accepted[2] == "in_window"

    judge.risk_notes = ["source_event_date:2025-01-01"]
    with use_evaluation_date("2026-09-10"):
        rejected = asyncio.run(lead_scorer._score_single_intent_signal(
            signal, _icp_model(), None, "Acme", "https://acme.com",
            trust_signal_date=True, stage1_soft_reject=True,
            llm_only_intent_gate=True, integrity_policy=True,
        ))
    assert rejected[0] == 0.0
    assert rejected[2] == "out_of_window"

    judge.publication = ""
    judge.risk_notes = []
    with use_evaluation_date("2026-09-10"):
        uncertain = asyncio.run(lead_scorer._score_single_intent_signal(
            signal, _icp_model(), None, "Acme", "https://acme.com",
            trust_signal_date=True, stage1_soft_reject=True,
            llm_only_intent_gate=True, integrity_policy=True,
        ))
    assert uncertain[0] == 54.0
    assert uncertain[2] == "uncertain"


def test_integrity_uses_tighter_textual_buyer_window(monkeypatch) -> None:
    from qualification.scoring import intent_verification_three_stage as verifier

    async def judge(*args, **kwargs):
        assert kwargs["buyer_max_age_days"] == 45
        return {
            "client_ready": True,
            "decision": "approve",
            "rejection_reason": "",
            "stage1": {"status": "supported"},
            "stage3": {
                "status": "supported",
                "decision": "approve",
                "claim_matches_miner_date": "no_date_in_content",
            },
            "scrape": {"result_count": 1, "statuses": []},
            "company_check": True,
            "job_publisher_relationship": "not_applicable",
            "source_publication_dates": [],
            "verdict": {"signal_evaluations": [{
                "signal_status": "supported",
                "verification_mode": "source_grounded",
                "same_entity_check": "pass",
                "confidence": "high",
                "risk_notes": ["source_event_date:2026-07-01"],
            }]},
        }

    monkeypatch.setattr(verifier, "verify_three_stage", judge)
    icp = _icp_model().model_copy(update={
        "intent_signals": ["Product launch in the last 30 days"],
        "intent_max_age_days": 365,
    })
    signal = _company_model([{
        "source": "news", "description": "Acme launched a product", "url": "https://news.example.com/acme", "date": None, "snippet": "launch", "matched_icp_signal": 0,
    }]).intent_signals[0]
    with use_evaluation_date("2026-09-10"):
        result = asyncio.run(lead_scorer._score_single_intent_signal(
            signal, icp, None, "Acme", "https://acme.com",
            stage1_soft_reject=True, llm_only_intent_gate=True,
            integrity_policy=True,
        ))
    assert result[0] == 0.0
    assert result[2] == "out_of_window"


def test_integrity_uses_bonus_signal_buyer_window(monkeypatch) -> None:
    from qualification.scoring import intent_verification_three_stage as verifier

    seen_caps = []

    async def judge(*args, **kwargs):
        seen_caps.append(kwargs["buyer_max_age_days"])
        return {
            "client_ready": True,
            "decision": "approve",
            "rejection_reason": "",
            "stage1": {"status": "supported"},
            "stage3": {
                "status": "supported",
                "decision": "approve",
                "claim_matches_miner_date": "no_date_in_content",
            },
            "scrape": {"result_count": 1, "statuses": []},
            "company_check": True,
            "job_publisher_relationship": "not_applicable",
            "source_publication_dates": [],
            "verdict": {"signal_evaluations": [{
                "signal_status": "supported",
                "verification_mode": "source_grounded",
                "same_entity_check": "pass",
                "confidence": "high",
                "risk_notes": ["source_event_date:2026-03-01"],
            }]},
        }

    raw_icp = {
        **_icp(),
        "intent_signals": [
            "Recently raised funding",
            "Hiring for senior engineering or sales roles",
        ],
        "intent_signal": "Recently raised funding",
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
        "bonus_intents": [{
            "intent_signal": "Hiring for senior engineering or sales roles",
            "intent_category": "HIRING",
            "intent_max_age_days": 90,
        }],
    }
    normalized = _normalized_icp(raw_icp)
    assert normalized["intent_signal_max_age_days"] == [365, 90]
    icp = ICPPrompt(**normalized)
    assert ICPPrompt.model_validate_json(icp.model_dump_json()) == icp
    signal = _company_model([{
        "source": "news",
        "description": "Acme is hiring engineers",
        "url": "https://news.example.com/acme-hiring",
        "date": None,
        "snippet": "Acme is hiring engineers",
        "matched_icp_signal": 1,
    }]).intent_signals[0]
    monkeypatch.setattr(verifier, "verify_three_stage", judge)
    with use_evaluation_date("2026-09-10"):
        result = asyncio.run(lead_scorer.score_company_competition_intent_signal(
            _company_model([signal.model_dump()]), icp, integrity_policy=True
        ))
    assert seen_caps == [90]
    assert result[0] == 0.0
    assert result[-1][0]["date_verdict"] == "out_of_window"


def test_integrity_job_premium_requires_verified_publisher_relationship(
    monkeypatch,
) -> None:
    from qualification.scoring import intent_verification_three_stage as verifier

    async def judge(*args, **kwargs):
        assert kwargs["declared_source"] == "job_board"
        return {
            "client_ready": True,
            "decision": "approve",
            "rejection_reason": "",
            "stage1": {"status": "supported"},
            "stage3": {
                "status": "supported",
                "decision": "approve",
                "claim_matches_miner_date": "no_date_in_content",
            },
            "scrape": {"result_count": 1, "statuses": []},
            "company_check": True,
            "job_publisher_relationship": judge.relationship,
            "source_publication_dates": [],
            "verdict": {"signal_evaluations": [{
                "signal_status": "supported",
                "verification_mode": "source_grounded",
                "same_entity_check": "pass",
                "confidence": "high",
                "risk_notes": [],
            }]},
        }

    judge.relationship = "unverified"
    monkeypatch.setattr(verifier, "verify_three_stage", judge)
    signal = _company_model([{
        "source": "job_board", "description": "Acme is hiring engineers", "url": "https://unknown.example/jobs/123", "date": None, "snippet": "Apply for the engineer role", "matched_icp_signal": 1,
    }]).intent_signals[0]
    ordinary = asyncio.run(lead_scorer._score_single_intent_signal(
        signal, _icp_model(), None, "Acme", "https://acme.com",
        trust_signal_date=True, stage1_soft_reject=True,
        llm_only_intent_gate=True, integrity_policy=True,
    ))
    assert ordinary[0] == 54.0

    judge.relationship = "verified"
    premium = asyncio.run(lead_scorer._score_single_intent_signal(
        signal, _icp_model(), None, "Acme", "https://acme.com",
        trust_signal_date=True, stage1_soft_reject=True,
        llm_only_intent_gate=True, integrity_policy=True,
    ))
    assert premium[0] == 60.0


def test_integrity_verified_ats_publisher_applies_to_techstack(monkeypatch) -> None:
    from qualification.scoring import intent_verification_three_stage as verifier

    source_url = "https://boards.greenhouse.io/acme/jobs/12345"
    answer = {
        "overall_verdict": "qualified",
        "overall_confidence": "high",
        "signal_evaluations": [{
            "signal_status": "supported",
            "confidence": "high",
            "same_entity_check": "pass",
            "verification_mode": "source_grounded",
            "evidence_urls_used": [source_url],
        }],
    }

    async def call_openrouter(*args, **kwargs):
        return {"answer": answer, "model": "test-model", "usage": {}}

    async def fetch(*args, **kwargs):
        return {
            "results": [{
                "url": source_url,
                "text": (
                    "Acme job description. Responsibilities include Python. "
                    "Apply for this job."
                ),
                "meta": {"kind": "greenhouse_job"},
            }],
            "statuses": [],
        }

    monkeypatch.setattr(verifier, "_call_openrouter", call_openrouter)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    result = asyncio.run(verifier.verify_three_stage(
        object(),
        company_name="Acme",
        company_linkedin="",
        company_website="https://acme.com",
        source_url=source_url,
        miner_claim="Acme uses Python",
        target_signal_text="Uses Python in its tech stack",
        evidence_type="TECHSTACK",
        declared_source="job_board",
        stage1_soft_reject=True,
        integrity_policy=True,
        buyer_max_age_days=180,
    ))
    assert result["job_publisher_relationship"] == "verified"


def test_integrity_retries_unavailable_bonus_after_verified_primary() -> None:
    from lab_arena import scoring

    primary = _positive_breakdown("Acme", "acme.co.uk")
    primary_detail = primary["intent_signals_detail"][0]
    unavailable_bonus = {
        "raw": 0.0,
        "after_decay": 0.0,
        "matched_icp_signal": 1,
        "judge_verdict": {
            "decision": "rejected_verifier_error",
            "pipeline_decision": "unavailable",
            "error_class": "TimeoutError",
        },
    }
    incomplete = {
        **primary,
        "company_index": 0,
        "company_identity_key": "domain:acme.co.uk|name:acme",
        "company_identity_alias_keys": ["domain:acme.co.uk|name:acme"],
        "company_qualified": True,
        "duplicate_company": False,
        "intent_signals_detail": [primary_detail, unavailable_bonus],
    }
    complete = {
        **incomplete,
        "intent_signal_raw": 80.0,
        "intent_signal_final": 80.0,
        "final_score": 80.0,
        "intent_signals_detail": [
            primary_detail,
            {
                **unavailable_bonus,
                "raw": 54.0,
                "after_decay": 54.0,
                "judge_verdict": {
                    "decision": "verified",
                    "pipeline_decision": "approve",
                },
            },
        ],
    }
    assert not scorer_breakdown_has_retryable_infrastructure_failure(incomplete)
    assert scorer_breakdown_has_retryable_infrastructure_failure(
        incomplete, integrity_policy=True
    )
    calls = []

    def judge(companies, icp, is_reference_model):
        calls.append(list(companies))
        return [incomplete if len(calls) == 1 else complete]

    judge.integrity_policy = True
    rows = scoring.score_work_item(
        {"scored_run_id": "run"},
        icp=_icp(),
        companies=[_public_company("Acme")],
        scorer=judge,
    )
    assert len(calls) == 2
    assert rows[0]["final_score"] == 80.0


def test_integrity_linkedin_job_also_requires_verified_employer(monkeypatch) -> None:
    from qualification.scoring import intent_verification_three_stage as verifier

    async def judge(*args, **kwargs):
        assert kwargs["declared_source"] == "linkedin"
        return {
            "client_ready": True,
            "decision": "approve",
            "rejection_reason": "",
            "stage1": {"status": "supported"},
            "stage3": {
                "status": "supported",
                "decision": "approve",
                "claim_matches_miner_date": "no_date_in_content",
            },
            "scrape": {"result_count": 1, "statuses": []},
            "company_check": True,
            "job_publisher_relationship": "unverified",
            "source_publication_dates": [],
            "verdict": {"signal_evaluations": [{
                "signal_status": "supported",
                "verification_mode": "source_grounded",
                "same_entity_check": "pass",
                "confidence": "high",
                "risk_notes": [],
            }]},
        }

    monkeypatch.setattr(verifier, "verify_three_stage", judge)
    signal = _company_model([{
        "source": "linkedin", "description": "Acme is hiring engineers", "url": "https://linkedin.com/jobs/view/123", "date": None, "snippet": "Apply for the engineer role", "matched_icp_signal": 1,
    }]).intent_signals[0]
    result = asyncio.run(lead_scorer._score_single_intent_signal(
        signal, _icp_model(), None, "Acme", "https://acme.com",
        stage1_soft_reject=True, llm_only_intent_gate=True,
        integrity_policy=True,
    ))
    assert result[0] == 54.0


def test_confirmed_staleness_is_not_sent_to_evidence_repair(monkeypatch) -> None:
    from qualification.scoring import deepline_evidence_repair as repair

    calls = {"n": 0}

    async def repair_sources(**kwargs):
        calls["n"] += 1
        return []

    monkeypatch.setattr(repair, "enabled", lambda: True)
    monkeypatch.setattr(repair, "repair_sources", repair_sources)
    result = asyncio.run(lead_scorer._attempt_competition_evidence_repair(
        _company_model([{
            "source": "news", "description": "Acme launched a product", "url": "https://news.example.com/acme", "date": "2026-09-01", "snippet": "launch", "matched_icp_signal": 0,
        }]),
        _icp_model(),
        integrity_policy=True,
        original_signal_results=[{
            "matched_icp_signal": 0,
            "date_verdict": "out_of_window",
        }],
    ))
    assert result is None
    assert calls["n"] == 0


def test_integrity_never_rerolls_terminal_criteria_through_optional_repair(monkeypatch) -> None:
    from qualification.scoring import deepline_evidence_repair as repair

    calls = {"n": 0}

    async def repair_sources(**kwargs):
        calls["n"] += 1
        return []

    monkeypatch.setattr(repair, "enabled", lambda: True)
    monkeypatch.setattr(repair, "repair_sources", repair_sources)
    company = _company_model([
        {
            "source": "news", "description": "Acme launched a product", "url": "https://news.example.com/acme", "date": None, "snippet": "launch", "matched_icp_signal": 0,
        },
        {
            "source": "news", "description": "Acme is hiring", "url": "https://news.example.com/acme-hiring", "date": None, "snippet": "hiring", "matched_icp_signal": 1,
        },
    ])
    result = asyncio.run(lead_scorer._attempt_competition_evidence_repair(
        company,
        _icp_model(),
        integrity_policy=True,
        original_signal_results=[{
            "matched_icp_signal": 1,
            "date_verdict": "out_of_window",
        }],
    ))
    assert result is None
    assert calls["n"] == 0


def test_arena_date_prompt_is_policy_scoped() -> None:
    from qualification.scoring.prompts._common import build_final_judge_prompt

    contents = {"results": [{"url": "https://example.com/a", "text": "body"}]}
    row = {
        "company": "acme.com",
        "website": "https://acme.com",
        "claim": "Acme launched",
        "signal_date": None,
        "claimed_source_urls": ["https://example.com/a"],
        "_target_signal_text": "Product launch",
    }
    legacy = build_final_judge_prompt(row, contents)
    integrity = build_final_judge_prompt(
        {**row, "_integrity_policy": True, "_buyer_max_age_days": 90},
        contents,
    )
    assert "ARENA INTEGRITY DATE POLICY" not in legacy
    assert "ARENA INTEGRITY DATE POLICY" in integrity
    assert "Buyer freshness window: 90 days" in integrity


@pytest.mark.parametrize("source,url,verified,cited,expected", [
    ("job_board", "https://unknown.example/jobs/123", False, "job", 54.0),
    ("job_board", "https://unknown.example/jobs/123", True, "job", 60.0),
    ("job_board", "https://unknown.example/jobs/123", True, "news", 54.0),
    ("linkedin", "https://linkedin.com/jobs/view/123", False, "job", 54.0),
    ("linkedin", "https://linkedin.com/jobs/view/123", True, "job", 60.0),
    ("linkedin", "https://linkedin.com/posts/acme", False, "job", 60.0),
    ("job_board", "https://unknown.example/jobs/123", True, "neither", 0.0),
])
def test_combined_evidence_premium_uses_only_cited_verified_job_sources(
    monkeypatch, source, url, verified, cited, expected
):
    from qualification.scoring import intent_verification_three_stage as verifier
    news = "https://news.example/acme"
    async def judge(*args, **kwargs):
        return {
            "client_ready": True, "decision": "approve", "rejection_reason": "",
            "stage1": {"status": "supported"},
            "stage3": {"status": "supported", "decision": "approve", "claim_matches_miner_date": "no_date_in_content"},
            "scrape": {"result_count": 2, "statuses": []}, "company_check": True,
            "job_publisher_relationship": "verified" if verified else "unverified",
            "verified_job_source_urls": [url] if verified else [],
            "source_publication_dates": [],
            "verdict": {"signal_evaluations": [{
                "signal_status": "supported", "verification_mode": "source_grounded",
                "same_entity_check": "pass", "confidence": "high", "risk_notes": [],
                "evidence_urls_used": [url if cited == "job" else news] if cited != "neither" else [],
            }]},
        }
    monkeypatch.setattr(verifier, "verify_three_stage", judge)
    signals = _company_model([
        {"source": source, "description": "Acme is hiring engineers", "url": url,
         "date": None, "snippet": "Apply for the engineer role", "matched_icp_signal": 1},
        {"source": "news", "description": "Acme is hiring engineers", "url": news,
         "date": None, "snippet": "Hiring engineers", "matched_icp_signal": 1},
    ]).intent_signals
    result = asyncio.run(lead_scorer._score_single_intent_signal(
        signals[0], _icp_model(), None, "Acme", "https://acme.com",
        stage1_soft_reject=True, llm_only_intent_gate=True,
        integrity_policy=True, evidence_signals=signals,
    ))
    assert result[0] == expected


def test_incomplete_bundle_does_not_judge_only_the_available_source(monkeypatch):
    from qualification.scoring import intent_verification_three_stage as verifier
    from tests.test_site_verifier_three_stage_source_grounding import supported
    from unittest.mock import AsyncMock
    urls = ["https://acme.com/news/a", "https://news.example/acme"]
    judge = AsyncMock(return_value=supported(urls[0]))
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(return_value={
        "results": [{"url": urls[0], "text": "Acme launched a product."}],
        "statuses": [{"url": urls[1], "stage": "timeout", "source": "none"}],
    }))
    result = asyncio.run(verifier.verify_three_stage(
        None, company_name="Acme", company_website="https://acme.com", company_linkedin="",
        source_url=urls[0], miner_claim="Acme launched a product", target_signal_text="Product launch",
        integrity_policy=True, stage1_soft_reject=True,
        evidence_bundle=[{"url": url, "description": "Acme launched a product", "date": None,
                          "snippet": "Product launch"} for url in urls],
    ))
    assert result["decision"] == "unavailable"
    assert result["rejection_reason"] == "evidence_fetch_failed"
    assert result["scrape"]["result_count"] == 1
    assert judge.await_count == 1  # Only the advisory pass; no partial terminal verdict.


@pytest.mark.parametrize("cite_closed", [True, False])
def test_combined_hiring_evidence_scopes_the_closed_linkedin_guard(
    monkeypatch, cite_closed,
):
    from qualification.scoring import intent_verification_three_stage as verifier
    from tests.test_site_verifier_three_stage_source_grounding import supported
    from unittest.mock import AsyncMock

    closed = "https://www.linkedin.com/jobs/view/123"
    filler = "https://news.example.com/acme"
    cited = closed if cite_closed else filler
    verdict = supported(cited)
    verdict["answer"].update({
        "overall_verdict": "qualified",
        "overall_confidence": "high",
    })
    monkeypatch.setattr(
        verifier,
        "_call_openrouter",
        AsyncMock(return_value=verdict),
    )
    monkeypatch.setattr(
        verifier,
        "_fetch_sd_then_exa",
        AsyncMock(return_value={
            "results": [
                {
                    "url": closed,
                    "text": (
                        "Acme Senior Engineer job. Responsibilities include "
                        "Python. Applications are closed."
                    ),
                    "meta": {
                        "kind": "linkedin_job",
                        "is_closed": True,
                        "jobs_status": "Applications are closed",
                    },
                },
                {"url": filler, "text": "Acme company profile.", "meta": {}},
            ],
            "statuses": [],
        }),
    )

    result = asyncio.run(verifier.verify_three_stage(
        None,
        company_name="Acme",
        company_website="https://acme.com",
        company_linkedin="",
        source_url=closed,
        miner_claim="Acme is hiring senior engineers",
        target_signal_text="Hiring senior engineers",
        evidence_type="HIRING",
        declared_source="linkedin",
        stage1_soft_reject=True,
        integrity_policy=True,
        buyer_max_age_days=90,
        evidence_bundle=[
            {
                "url": closed,
                "description": "Acme is hiring senior engineers",
                "date": None,
                "snippet": "Senior Engineer job",
            },
            {
                "url": filler,
                "description": "Acme is hiring senior engineers",
                "date": None,
                "snippet": "Acme company profile",
            },
        ],
    ))

    assert result["client_ready"] is not cite_closed
    assert result["decision"] == ("reject" if cite_closed else "approve")
    assert result["rejection_reason"] == (
        "linkedin_job_closed" if cite_closed else ""
    )
    assert result["verdict"]["overall_verdict"] == (
        "disqualified" if cite_closed else "qualified"
    )
    assert result["verdict"]["signal_evaluations"][0][
        "signal_status"
    ] == ("contradicted" if cite_closed else "supported")


@pytest.mark.parametrize("all_absent", [False, True])
def test_combined_evidence_preserves_confirmed_absence_semantics(monkeypatch, all_absent):
    from qualification.scoring import intent_verification_three_stage as verifier
    from tests.test_site_verifier_three_stage_source_grounding import supported
    from unittest.mock import AsyncMock
    urls = ["https://acme.com/news/a", "https://news.example/acme"]
    judge = AsyncMock(return_value=supported(urls[0]))
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    absent = urls if all_absent else urls[1:]
    statuses = [{
        "url": url, "source": "none", "sd_stage": "genuine_404", "sd_error": "http_404",
        "exa_stage": "exa_target_not_found", "exa_error": "target_not_found",
        "exa_target_absence": {"id_matches_requested_url": True, "status": "error",
            "error_tag": "CRAWL_NOT_FOUND", "error_http_status": 404, "confirmed_attempts": 2},
    } for url in absent]
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(return_value={
        "results": [] if all_absent else [{"url": urls[0], "text": "Acme launched a product."}],
        "statuses": statuses,
    }))
    result = asyncio.run(verifier.verify_three_stage(
        None, company_name="Acme", company_website="https://acme.com", company_linkedin="",
        source_url=urls[0], miner_claim="Acme launched a product", target_signal_text="Product launch",
        integrity_policy=True, stage1_soft_reject=True,
        evidence_bundle=[{"url": url, "description": "Acme launched a product", "date": None,
                          "snippet": "Product launch"} for url in urls],
    ))
    assert result["decision"] == ("reject" if all_absent else "approve")
    assert result["rejection_reason"] == ("evidence_not_found" if all_absent else "")
    assert judge.await_count == (1 if all_absent else 2)


def test_integrity_rejects_multiple_verdicts_for_one_criterion(monkeypatch):
    from qualification.scoring import intent_verification_three_stage as verifier
    from tests.test_site_verifier_three_stage_source_grounding import supported
    from unittest.mock import AsyncMock
    url = "https://acme.com/news/a"
    response = supported(url)
    response["answer"]["signal_evaluations"] *= 2
    monkeypatch.setattr(verifier, "_call_openrouter", AsyncMock(return_value=response))
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(return_value={
        "results": [{"url": url, "text": "Acme launched a product."}], "statuses": [],
    }))
    result = asyncio.run(verifier.verify_three_stage(
        None, company_name="Acme", company_website="https://acme.com", company_linkedin="",
        source_url=url, miner_claim="Acme launched a product", target_signal_text="Product launch",
        integrity_policy=True, stage1_soft_reject=True,
        evidence_bundle=[{"url": url, "description": "Acme launched a product", "date": None,
                          "snippet": "Product launch"}],
    ))
    assert result["decision"] == "unavailable"
    assert result["rejection_reason"] == "stage3_llm_error:invalid_criterion_verdict_count"
