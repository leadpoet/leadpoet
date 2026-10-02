"""The paragraph review preserves the frozen primary/optional intent contract."""

import asyncio
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import intent_details, lead_scorer, verification_helpers
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.competition import _normalized_company, _normalized_icp
from qualification.scoring.evaluation_clock import use_evaluation_date
from tests.test_sep24_intent_details_unit_grounding import (
    _nonfactual_unit, _unproven_unit, _verified_unit,
)


FIXTURE = Path(__file__).parent / "lab_arena/fixtures/oct01_koho_optional_bonus.json"


def _inputs():
    saved = json.loads(FIXTURE.read_text())
    company = CompanyOutput(**_normalized_company(
        saved["company"], integrity_policy=True, company_quality=True,
    ))
    icp = ICPPrompt(**_normalized_icp(saved["icp"]))
    return saved, company, icp


def _roles(document, saved):
    assert document["icp"]["prompt"] == saved["icp"]["prompt"]
    assert "intent_signals" not in document["icp"]
    assert document["icp"]["required_primary_intent"] == {
        "matched_icp_signal": 0,
        "criterion": saved["icp"]["intent_signal"],
    }
    assert document["icp"]["optional_bonus_intents"] == [{
        "matched_icp_signal": 1,
        "criterion": saved["icp"]["bonus_intents"][0]["intent_signal"],
    }]
    # Requested criteria remain context, never newly admitted factual proof.
    assert saved["icp"]["intent_signal"] not in json.dumps(document["admitted_evidence"])


def _response(document, saved, *, control="supported"):
    units = [
        _verified_unit(document, 0, saved["funding_quote"]),
        _verified_unit(document, 1, saved["workforce_quote"]),
        _nonfactual_unit(2),
    ]
    if control == "asserted_bonus":
        units.append(_unproven_unit(3))
    return {
        "unit_grounding": units,
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        **{name: True for name in intent_details._CHECKS},
        "facts_supported": control != "asserted_bonus",
        "connects_icp": control != "generic_relevance",
    }


@pytest.mark.parametrize(
    ("control", "expected_score"),
    [("supported", 54), ("asserted_bonus", 0), ("generic_relevance", 0)],
)
def test_saved_koho_primary_bonus_boundary_through_company_scoring(
    monkeypatch, control, expected_score,
):
    saved, company, icp = _inputs()
    if control == "asserted_bonus":
        company.intent_details += " KOHO expanded into the United States this year."
    elif control == "generic_relevance":
        company.intent_details = company.intent_details.rsplit(". ", 1)[0] + ". The company may grow."
    calls = []

    async def judge(prompt, **kwargs):
        document = json.loads(prompt)
        _roles(document, saved)
        system = " ".join(kwargs["system_prompt"].split())
        assert "structured distinction takes precedence" in system
        assert "missing bonus evidence into an additional qualification requirement" in system
        assert "including any asserted bonus activity" in system
        assert "Still cover every distinct activity present in verified_signals" in system
        calls.append(document)
        return json.dumps(_response(document, saved, control=control))

    async def verify_signal(signal, *_args, verdict_out, **_kwargs):
        verdict_out.append(deepcopy(saved["signal_results"][0]["judge_verdict"]))
        return 54.0, 90, "verified", date(2026, 6, 11), signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", AsyncMock(
        return_value=company_fit_match(details=saved["company_fit"]),
    ))
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", verify_signal)
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    with use_evaluation_date("2026-10-01"):
        result = asyncio.run(lead_scorer.score_company_competition_intent(
            company, icp, 0.0, 1.0, set(), integrity_policy=True,
            company_quality=True, evidence_investigator=True,
        ))
    assert len(calls) == 1
    assert result.final_score == expected_score
    assert result.intent_signals_detail[0]["after_decay"] == 54
    receipt = result.verifier_gate_receipts[-1]
    assert receipt["decision"] == ("match" if expected_score else "mismatch")
    if control == "asserted_bonus":
        assert receipt["checks"]["facts_supported"] is False
    if control == "generic_relevance":
        assert receipt["checks"]["connects_icp"] is False


def test_optional_bonus_cannot_replace_missing_required_primary(monkeypatch):
    saved, company, icp = _inputs()
    company.intent_signals[0].matched_icp_signal = 1

    async def verified_bonus(signal, *_args, verdict_out, **_kwargs):
        verdict_out.append(deepcopy(saved["signal_results"][0]["judge_verdict"]))
        return 54.0, 90, "verified", date(2026, 6, 11), signal.matched_icp_signal

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", AsyncMock(
        return_value=company_fit_match(details=saved["company_fit"]),
    ))
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", verified_bonus)
    reviewer = AsyncMock(side_effect=AssertionError("No paragraph can rescue missing primary"))
    monkeypatch.setattr(verification_helpers, "openrouter_chat", reviewer)
    with use_evaluation_date("2026-10-01"):
        result = asyncio.run(lead_scorer.score_company_competition_intent(
            company, icp, 0.0, 1.0, set(), integrity_policy=True,
            company_quality=True, evidence_investigator=True,
        ))
    assert result.final_score == 0
    assert result.intent_signals_detail[0]["after_decay"] == 54
    assert not lead_scorer.required_intent_satisfied(result.intent_signals_detail)
    reviewer.assert_not_called()


@pytest.mark.parametrize("repair_kind", ["citation", "semantic"])
def test_repair_keeps_authoritative_intent_roles_without_forgiving_bonus_claims(repair_kind):
    saved, company, icp = _inputs()
    document = intent_details.review_evidence(
        company, icp, saved["signal_results"], saved["company_fit"],
    )
    held = _response(document, saved)
    issues = {0: {"nonexact_quote"}}
    if repair_kind == "citation":
        prompt = intent_details._citation_repair_user_prompt(document, issues, held)
        system = intent_details._citation_repair_prompt(intent_details._SYSTEM, issues, held)
        assert "including any asserted bonus activity" in system
    else:
        prompt = intent_details._semantic_repair_user_prompt(document, issues, held, {0})
        system = intent_details._semantic_repair_prompt(intent_details._SYSTEM, issues, held, {0})
        assert "any paragraph assertion that it" in system
        assert "still requires the same factual support" in system
    assert json.loads(prompt)["review_document"]["icp"] == document["icp"]
    _roles(document, saved)


def test_normalized_roles_preserve_call_sine_optional_hiring_contract():
    from tests.test_arena_intent_details_grounding import inputs

    company, _icp, results, fit = inputs()
    primary = "Launched a new product or major platform capability in the last 12 months"
    bonus = "Actively hiring for specific product, engineering, or customer-implementation roles, per current job postings or careers page"
    icp = ICPPrompt(**_normalized_icp({
        "icp_id": "callsine-bonus-control", "industry": "Software",
        "employee_count": ["11-50"], "geography": "United States",
        "product_service": "Workflow software",
        "prompt": "hey, gonna need SaaS firms that just launched a new workflow, automation, or AI feature and are hiring hard",
        "intent_signal": primary,
        "bonus_intents": [{"intent_signal": bonus}],
    }))
    document = intent_details.review_evidence(company, icp, results[:1], fit)
    assert document["icp"]["required_primary_intent"]["criterion"] == primary
    assert document["icp"]["optional_bonus_intents"] == [{
        "matched_icp_signal": 1, "criterion": bonus,
    }]
    assert {row["matched_icp_signal"] for row in document["verified_signals"]} == {0}


def test_verified_bonus_still_requires_paragraph_coverage():
    from tests.test_arena_intent_details_grounding import inputs, _review_response

    company, icp, results, fit = inputs()
    document = intent_details.review_evidence(company, icp, results, fit)
    assert document["icp"]["optional_bonus_intents"][0]["matched_icp_signal"] == 1
    response = _review_response(
        {name: name != "verified_signals_covered" for name in intent_details._CHECKS},
        [{"matched_icp_signal": 0, "covered": True}, {"matched_icp_signal": 1, "covered": False}],
        document,
    )
    checks = intent_details._validate_review_response(json.dumps(response), document)
    assert checks["verified_signals_covered"] is False


def test_primary_bonus_context_keeps_existing_document_and_repair_bounds():
    saved, company, icp = _inputs()
    # Large buyer prose and fetched context exercise the real context trimmer.
    # Role descriptions must remain exact and must not increase the 48k cap.
    icp.prompt += " Buyer context." * 2100
    results = deepcopy(saved["signal_results"])
    context = results[0]["judge_verdict"]["verification_trace"]["verified_source_context"][0]
    context["text"] += " Background information." * 1000
    document = intent_details.review_evidence(company, icp, results, saved["company_fit"])
    assert len(json.dumps(document, ensure_ascii=False)) <= intent_details._MAX_REVIEW_DOCUMENT_CHARACTERS
    assert document["icp"]["required_primary_intent"]["criterion"] == saved["icp"]["intent_signal"]
    assert document["icp"]["optional_bonus_intents"][0]["criterion"] == saved["icp"]["bonus_intents"][0]["intent_signal"]
    source_texts = [text for source in document["admitted_evidence"]
                    for text in source.get("admitted_text", [])]
    assert max(map(len, source_texts)) < len(context["text"])
    held = _response(document, saved)
    issues = {0: {"nonexact_quote"}}
    for prompt in (
        intent_details._citation_repair_user_prompt(document, issues, held),
        intent_details._semantic_repair_user_prompt(document, issues, held, {0}),
    ):
        assert len(prompt) <= intent_details._MAX_REVIEW_DOCUMENT_CHARACTERS
        assert json.loads(prompt)["review_document"]["icp"] == document["icp"]
    icp.prompt += " Additional buyer context." * 1000
    with pytest.raises(ValueError, match="exceeds its bound"):
        intent_details.review_evidence(company, icp, results, saved["company_fit"])
