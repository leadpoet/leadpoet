"""Contact-free intent output and scoring regressions for current Arena rounds."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from urllib.parse import urlsplit

import pytest

from lab_arena import contact_policy, intent_details_policy
from lab_arena.output import OutputInvalid, output_document_from_bytes
from qualification.scoring import contact_verification, lead_scorer
from qualification.scoring.competition import CompetitionCompanyScorer
from tests.test_arena_company_quality_policy import _positive_breakdown
from tests.test_arena_score_integrity import _icp, _public_company


def _company(name: str = "Acme", domain: str = "acme.com") -> dict:
    row = _public_company(
        name,
        domain=domain,
        linkedin=f"https://linkedin.com/company/{name.casefold()}",
        signal_date="2026-09-01",
    )
    row.pop("fit_summary")
    row.pop("fit_evidence_urls")
    row["state"] = "CA"
    row["intent_details"] = (
        f"{name} announced a current product launch that matches the ICP."
    )
    for signal in row["intent_signals"]:
        signal.pop("why_now")
        signal.pop("snippet")
    return row


def test_intent_output_schema_keeps_v5_for_contacts_and_uses_v6_without_them():
    intent = {"intent_details_policy": intent_details_policy.POLICY}

    assert contact_policy.output_schema(intent) == (
        intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
    )
    assert contact_policy.output_schema(
        {**intent, "contact_policy": contact_policy.POLICY}
    ) == intent_details_policy.OUTPUT_SCHEMA


def test_v6_is_exact_v5_company_shape_without_contact():
    company = _company()
    v6 = output_document_from_bytes(
        json.dumps([company]).encode(),
        expected_schema_version=intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
    )
    v5_without_contact = output_document_from_bytes(
        json.dumps([company]).encode(),
        expected_schema_version=intent_details_policy.OUTPUT_SCHEMA,
    )
    expected_company = dict(v5_without_contact["companies"][0])
    expected_company.pop("contact")
    assert v6 == {
        "schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
        "companies": [expected_company],
    }

    with_contact = {**company, "contact": {"full_name": "Ignored Person"}}
    v5 = output_document_from_bytes(
        json.dumps([with_contact]).encode(),
        expected_schema_version=intent_details_policy.OUTPUT_SCHEMA,
    )
    assert v5["companies"][0]["contact"] == with_contact["contact"]
    v6_with_contact = output_document_from_bytes(
        json.dumps([with_contact]).encode(),
        expected_schema_version=intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
    )
    assert v6_with_contact == v6

    with pytest.raises(OutputInvalid, match="public output contract"):
        output_document_from_bytes(
            json.dumps([{**company, "unknown": True}]).encode(),
            expected_schema_version=intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
        )


def test_company_only_scorer_is_contact_blind_and_never_calls_contact_verifier(
    monkeypatch,
):
    async def judge(*, company, **_kwargs):
        domain = urlsplit(company.company_website).hostname
        slug = company.company_linkedin.rstrip("/").rsplit("/", 1)[-1]
        return _positive_breakdown(company.company_name, domain, slug)

    async def forbidden_contact_verifier(*_args, **_kwargs):
        pytest.fail("company-only scoring called the contact verifier")

    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent", judge
    )
    monkeypatch.setattr(
        contact_verification, "verify_contact", forbidden_contact_verifier
    )
    scorer = CompetitionCompanyScorer(
        integrity_policy=True,
        contacts_required=False,
        company_quality=True,
        evidence_investigator=True,
    )
    base = _company()
    variants = [
        base,
        {**base, "contact": None},
        {
            **base,
            "contact": {
                "full_name": "Different Person",
                "role": "Unrelated role",
                "email": "different@example.net",
            },
        },
    ]

    results = [
        asyncio.run(scorer.score_with_breakdowns([row], _icp(), False))[0]
        for row in variants
    ]

    assert results[0] == results[1] == results[2]
    assert results[0]["company_qualified"] is True
    assert results[0]["final_score"] == 60.0
    assert "contact_qualified" not in results[0]
    assert all(
        receipt.get("gate") != "contact"
        for receipt in results[0]["verifier_gate_receipts"]
    )


def test_company_and_intent_failures_still_zero_only_their_company(monkeypatch):
    calls: list[str] = []

    async def judge(*, company, **_kwargs):
        calls.append(company.company_name)
        domain = urlsplit(company.company_website).hostname
        slug = company.company_linkedin.rstrip("/").rsplit("/", 1)[-1]
        result = _positive_breakdown(company.company_name, domain, slug)
        if "unsupported claim" in company.intent_details:
            result["verifier_gate_receipts"].append(
                {
                    "gate": "intent_details",
                    "decision": "mismatch",
                    "reason": "intent_details_not_grounded",
                }
            )
        return result

    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent", judge
    )
    bad_company = {**_company("BadCo", "badco.com"), "company_linkedin": None}
    bad_intent = deepcopy(_company("IntentCo", "intentco.com"))
    bad_intent["intent_details"] = "IntentCo has an unsupported claim for this ICP."
    good = _company("GoodCo", "goodco.com")

    results = asyncio.run(
        CompetitionCompanyScorer(
            integrity_policy=True,
            contacts_required=False,
            company_quality=True,
            evidence_investigator=True,
        ).score_with_breakdowns(
            [bad_company, bad_intent, good], _icp(), False
        )
    )

    assert calls == ["IntentCo", "GoodCo"]
    assert [row["company_qualified"] for row in results] == [False, False, True]
    assert [row["final_score"] for row in results] == [0.0, 0.0, 60.0]
    # Per-company sourcing eligibility uses these independently verified flags.
    assert sum(row["company_qualified"] for row in results) == 1
    assert not any("contact_qualified" in row for row in results)
