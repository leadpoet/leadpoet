from __future__ import annotations

import asyncio
from datetime import date

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
from qualification.scoring.competition import CompetitionCompanyScorer
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
    assert short.key == legal.key == "domain:ibm.com|linkedin:ibm"
    assert set(company_identity_alias_keys(short)) == {
        "domain:ibm.com|name:ibm",
        "domain:ibm.com|linkedin:ibm",
    }
    assert set(company_identity_alias_keys(legal)) == {
        "domain:ibm.com|name:international business machines",
        "domain:ibm.com|linkedin:ibm",
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
    assert "domain:ibm.com|linkedin:ibm" in rows[0][
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


def test_integrity_aggregates_strongest_once_per_requested_signal(monkeypatch) -> None:
    async def score_one(signal, *args, verdict_out=None, **kwargs):
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
    assert [row["counted_in_aggregate"] for row in details] == [True, False, True]
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


def test_stale_bonus_does_not_block_primary_evidence_repair(monkeypatch) -> None:
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
    assert calls["n"] == 1


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
