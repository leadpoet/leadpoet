"""Row-level contact gating in the shared competition scorer."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from qualification.scoring import contact_verification, lead_scorer
from qualification.scoring.competition import (
    CompetitionScorerInputError,
    CompetitionCompanyScorer,
    count_penalizable_false_positives,
    effective_competition_input,
)


def _contact(call_id: str = "broker-1", *, slug: str = "ada", email: str = "ada@acme.com") -> dict:
    return {
        "full_name": "Ada Lovelace",
        "role": "VP Sales",
        "linkedin_url": f"https://www.linkedin.com/in/{slug}/",
        "location": {"country": "US", "region": "California", "city": "San Francisco"},
        "email": email,
        "email_source": {
            "provider": "harvestapi",
            "tool": "harvestapi_get_profile",
            "broker_call_id": call_id,
        },
    }


def _company(name: str = "Acme", domain: str = "acme.com", **updates: object) -> dict:
    value = {
        "company_name": name,
        "company_website": f"https://{domain}",
        "company_linkedin": f"https://linkedin.com/company/{name.casefold().replace(' ', '-')}",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series B",
        "country": "United States",
        "state": "California",
        "fit_summary": "A software business with a timely product launch.",
        "fit_evidence_urls": [f"https://{domain}/about"],
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "The company announced a product launch.",
                "date": "2026-09-01",
                "why_now": "The launch creates a timely buying trigger.",
                "url": f"https://{domain}/news/launch",
                "snippet": "The company launched the product.",
            }
        ],
        "contact": _contact(),
    }
    value.update(updates)
    return value


def _icp() -> dict:
    return {
        "icp_id": "arena:contacts:0",
        "prompt": "Find software companies with a launch and a sales leader",
        "industry": "Software",
        "sub_industry": "SaaS",
        "employee_count": ["51-200"],
        "company_stage": "Any",
        "country": "United States",
        "intent_signals": ["Announced a product launch"],
        "intent_max_age_days": 90,
        "max_companies": 5,
        "target_roles": ["Vice President of Sales"],
        "target_seniority": "VP",
        "contact_geography": {
            "countries": ["United States"],
            "regions": ["California"],
            "cities": ["San Francisco"],
        },
    }


def _positive_breakdown(
    name: str = "Acme", domain: str = "acme.com", linkedin_slug: str = ""
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
        "verifier_gate_receipts": [
            {
                "gate": "company_fit",
                "decision": "match",
                "dimension_evidence": {
                    "identity": {
                        "web_identity_receipt": {
                            "decision": "match",
                            "evidence_source": "company_web_reverification",
                            "observed_name": name,
                            "observed_domain": domain,
                            "observed_linkedin_slug": linkedin_slug,
                        }
                    }
                },
            }
        ],
        "intent_signals_detail": [
            {
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
            }
        ],
    }


def _contact_result(*, qualified: bool, email_status: str = "unknown") -> dict:
    decision = "verified" if qualified else "mismatch"
    reason = "contact_verified" if qualified else "contact_email_mismatch"
    return {
        "contact_qualified": qualified,
        "contact_identity_key": "contact:stable",
        "email_status": email_status,
        "contact_verification": {
            "decision": decision,
            "reason": reason,
            "subchecks": {},
            "evidence_hashes": {"source": "abc"},
            "evidence_timestamps": {},
        },
        "verifier_gate_receipts": [
            {"gate": "contact", "decision": decision, "reason": reason}
        ],
    }


def test_contact_gate_runs_after_company_success_and_accepts_catch_all(monkeypatch) -> None:
    base_calls: list[dict] = []
    contact_calls: list[dict] = []

    async def score_company(**kwargs):
        base_calls.append(kwargs)
        return _positive_breakdown(linkedin_slug="acme-verified")

    async def verify(company, icp, **kwargs):
        contact_calls.append({"company": company, "icp": icp, **kwargs})
        return _contact_result(qualified=True, email_status="catch_all")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    monkeypatch.setattr(contact_verification, "verify_contact", verify)
    source = {"broker-1": {"provider": "harvestapi", "tool": "harvestapi_get_profile"}}

    company = _company(
        company_linkedin="https://linkedin.com/company/forged-submission"
    )
    rows = asyncio.run(
        CompetitionCompanyScorer(
            contacts_required=True, contact_source_evidence=source
        ).score_with_breakdowns([company], _icp(), False)
    )

    assert base_calls[0]["integrity_policy"] is True
    assert contact_calls[0]["source_evidence"] is source["broker-1"]
    assert contact_calls[0]["icp"] == _icp()
    assert contact_calls[0]["company"]["company_name"] == "Acme"
    assert contact_calls[0]["company"]["company_website"] == "https://acme.com"
    assert contact_calls[0]["company"]["company_linkedin"] == (
        "https://www.linkedin.com/company/acme-verified/"
    )
    assert contact_calls[0]["company"]["contact"] == company["contact"]
    assert rows[0]["company_qualified"] is True
    assert rows[0]["contact_qualified"] is True
    assert rows[0]["email_status"] == "catch_all"
    assert rows[0]["final_score"] == 60.0
    assert [item["gate"] for item in rows[0]["verifier_gate_receipts"]] == [
        "company_fit",
        "contact",
    ]


def test_missing_or_malformed_contact_zeros_only_its_row(monkeypatch) -> None:
    async def score_company(**kwargs):
        company = kwargs["company"]
        return _positive_breakdown(
            company.company_name,
            company.company_website.split("//", 1)[-1].rstrip("/"),
        )

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    missing = _company()
    missing.pop("contact")
    malformed = _company("Beta", "beta.com", contact={"full_name": "Ada"})

    rows = asyncio.run(
        CompetitionCompanyScorer(contacts_required=True).score_with_breakdowns(
            [missing, malformed], _icp(), False
        )
    )

    assert len(rows) == 2
    assert [row["final_score"] for row in rows] == [0.0, 0.0]
    assert [row["contact_verification"]["reason"] for row in rows] == [
        "contact_claim_invalid",
        "contact_claim_invalid",
    ]


def test_base_failure_skips_contact_and_emits_complete_not_evaluated_fields(monkeypatch) -> None:
    async def score_company(**_kwargs):
        result = _positive_breakdown()
        result["intent_signals_detail"] = []
        return result

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("contact verification ran after company failure")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    monkeypatch.setattr(contact_verification, "verify_contact", should_not_run)

    row = asyncio.run(
        CompetitionCompanyScorer(contacts_required=True).score_with_breakdowns(
            [_company()], _icp(), False
        )
    )[0]

    assert row["company_qualified"] is False
    assert row["contact_qualified"] is False
    assert row["contact_identity_key"].startswith("contact:")
    assert row["email_status"] == "unknown"
    assert row["contact_verification"]["decision"] == "not_evaluated"
    assert row["contact_verification"]["reason"] == "company_not_qualified"
    assert row["verifier_gate_receipts"][-1] == {
        "gate": "contact",
        "decision": "not_evaluated",
        "reason": "company_not_qualified",
    }


def test_contact_rejection_keeps_base_breakdown_and_false_positive_penalty_logic(monkeypatch) -> None:
    async def score_company(**_kwargs):
        return _positive_breakdown()

    async def verify(*_args, **_kwargs):
        return _contact_result(qualified=False)

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    monkeypatch.setattr(contact_verification, "verify_contact", verify)

    row = asyncio.run(
        CompetitionCompanyScorer(contacts_required=True).score_with_breakdowns(
            [_company()], _icp(), False
        )
    )[0]

    assert row["final_score"] == 0.0
    assert row["intent_signal_final"] == 60.0
    assert row["intent_signals_detail"][0]["after_decay"] == 60.0
    assert count_penalizable_false_positives(
        [row], icp_has_intent_signals=True
    ) == (0, 0)


def test_duplicate_company_does_not_run_a_second_contact_check(monkeypatch) -> None:
    contact_calls = 0

    async def score_company(**_kwargs):
        return _positive_breakdown()

    async def verify(*_args, **_kwargs):
        nonlocal contact_calls
        contact_calls += 1
        return _contact_result(qualified=True, email_status="valid")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    monkeypatch.setattr(contact_verification, "verify_contact", verify)
    first = _company()
    second = deepcopy(first)
    second["contact"] = _contact("broker-2", slug="grace", email="grace@acme.com")

    rows = asyncio.run(
        CompetitionCompanyScorer(contacts_required=True).score_with_breakdowns(
            [first, second], _icp(), False
        )
    )

    assert contact_calls == 1
    assert rows[1]["duplicate_company"] is True
    assert rows[1]["contact_verification"]["decision"] == "not_evaluated"


def test_legacy_effective_input_is_unchanged_when_contact_mode_is_off() -> None:
    legacy = _company()
    legacy.pop("contact")

    implicit = effective_competition_input([legacy], _icp())
    explicit = effective_competition_input(
        [legacy],
        _icp(),
        contacts_required=False,
        contact_source_evidence={"ignored": {"response": "ignored"}},
    )

    assert implicit == explicit
    assert "contact" not in implicit["companies"][0]
    assert implicit["icp"]["target_roles"] == []
    assert implicit["icp"]["target_seniority"] == ""

    with pytest.raises(CompetitionScorerInputError):
        effective_competition_input([_company()], _icp())


def test_contact_effective_input_hashes_semantics_and_excludes_call_metadata() -> None:
    company = _company()
    evidence = {
        "broker-1": {
            "provider": "harvestapi",
            "tool": "harvestapi_get_profile",
            "input": {"url": company["contact"]["linkedin_url"], "findEmail": "true"},
            "response": {
                "status": "completed",
                "job_id": "job-one",
                "billing": {"credits": 1},
                "result": {
                    "data": {
                        "element": {
                            "id": "person-1",
                            "linkedinUrl": company["contact"]["linkedin_url"],
                            "firstName": "Ada",
                            "lastName": "Lovelace",
                            "emails": [{"email": "ada@acme.com"}],
                        }
                    }
                },
            },
            "call_identity": "broker-1",
            "observed_at": "2026-09-11T12:00:00Z",
        }
    }

    first = effective_competition_input(
        [company], _icp(), contacts_required=True, contact_source_evidence=evidence
    )
    metadata_changed = deepcopy(evidence)
    metadata_changed["broker-1"]["call_identity"] = "replacement-id"
    metadata_changed["broker-1"]["observed_at"] = "2030-01-01T00:00:00Z"
    metadata_changed["broker-1"]["response"]["job_id"] = "job-two"
    metadata_changed["broker-1"]["response"]["billing"] = {"credits": 999}
    second = effective_competition_input(
        [company],
        _icp(),
        contacts_required=True,
        contact_source_evidence=metadata_changed,
    )
    response_changed = deepcopy(evidence)
    response_changed["broker-1"]["response"]["result"]["data"]["element"]["id"] = "person-2"
    third = effective_competition_input(
        [company],
        _icp(),
        contacts_required=True,
        contact_source_evidence=response_changed,
    )

    assert first == second
    assert first != third
    row = first["companies"][0]
    assert "broker_call_id" not in row["contact"]["email_source"]
    assert first["icp"]["target_roles"] == ["Vice President of Sales"]
    assert first["icp"]["contact_geography"] == {
        "countries": ["US"],
        "regions": ["california"],
        "cities": ["san francisco"],
    }

    email_changed = deepcopy(evidence)
    email_changed["broker-1"]["response"]["result"]["data"]["element"]["emails"] = [
        {"email": "other@acme.com"}
    ]
    employer_changed = deepcopy(evidence)
    employer_changed["broker-1"]["response"]["result"]["data"]["element"]["currentPosition"] = [
        {"position": "VP Sales", "companyName": "Other Company"}
    ]
    assert first != effective_competition_input(
        [company], _icp(), contacts_required=True, contact_source_evidence=email_changed
    )
    assert first != effective_competition_input(
        [company], _icp(), contacts_required=True, contact_source_evidence=employer_changed
    )


def test_invalid_contact_cache_hash_also_excludes_broker_metadata() -> None:
    first = _company(contact={"email_source": {"broker_call_id": "call-one"}})
    second = _company(contact={"email_source": {"broker_call_id": "call-two"}})

    left = effective_competition_input([first], _icp(), contacts_required=True)
    right = effective_competition_input([second], _icp(), contacts_required=True)

    assert left == right


def test_invalid_source_reason_changes_contact_cache_semantics() -> None:
    company = _company()
    missing = effective_competition_input(
        [company], _icp(), contacts_required=True, contact_source_evidence={}
    )
    invalid = effective_competition_input(
        [company],
        _icp(),
        contacts_required=True,
        contact_source_evidence={
            "broker-1": {
                "invalid": True,
                "reason": "email_source_reference_invalid",
            }
        },
    )

    assert missing != invalid


def test_source_mapping_order_and_contact_positions_do_not_cross_evidence() -> None:
    first = _company()
    second = _company(
        "Beta",
        "beta.com",
        contact=_contact("broker-2", slug="grace", email="grace@beta.com"),
    )
    one = {
        "provider": "harvestapi",
        "tool": "harvestapi_get_profile",
        "input": {"url": first["contact"]["linkedin_url"]},
        "response": {"element": {"id": "person-1"}},
    }
    two = {
        "provider": "harvestapi",
        "tool": "harvestapi_get_profile",
        "input": {"url": second["contact"]["linkedin_url"]},
        "response": {"element": {"id": "person-2"}},
    }

    forward = effective_competition_input(
        [first, second],
        _icp(),
        contacts_required=True,
        contact_source_evidence={"broker-1": one, "broker-2": two},
    )
    reordered = effective_competition_input(
        [first, second],
        _icp(),
        contacts_required=True,
        contact_source_evidence={"broker-2": two, "broker-1": one},
    )

    assert forward == reordered
    assert (
        forward["companies"][0]["contact_source_evidence_hash"]
        != forward["companies"][1]["contact_source_evidence_hash"]
    )
