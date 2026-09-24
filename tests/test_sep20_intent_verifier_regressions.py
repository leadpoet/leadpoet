from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_details
from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring import verification_helpers


SOURCE_URL = "https://acme.example/news/acme-control-copilot"
CLAIM = "Acme's Control Copilot connects assistants to live policy data."
TARGET = "Launched a major security capability in the last 12 months."
SOURCE_TEXT = (
    "A security operations breakthrough.\n"
    "Harbor City, CA, March 12, 2026 - Acme today announced the Control "
    "Copilot. Acme's Control Copilot connects assistants to live policy data. "
    "The capability is available now."
)


def _signal_verdict(*, confidence: str, unsupported_parts=None) -> dict:
    return {
        "answer": {
            "overall_verdict": "qualified",
            "overall_confidence": confidence,
            "signal_evaluations": [{
                "signal_status": "supported",
                "verification_mode": "source_grounded",
                "same_entity_check": "pass",
                "confidence": confidence,
                "evidence_urls_used": [SOURCE_URL],
                "claim_matches_miner_date": "consistent",
                "source_accessibility": "accessible",
                "claim": CLAIM,
                "supporting_quotes": [f"“{CLAIM}”"],
                "contradicting_quotes": [],
                "risk_notes": ["source_publication_date:2026-03-12"],
                "unsupported_parts": unsupported_parts or [],
            }],
        },
        "model": "test-stage3",
        "usage": {},
    }




def _mind_style_inputs():
    company = SimpleNamespace(
        company_name="Acme",
        company_website="https://acme.example",
        intent_details=(
            f"The source dated 2026-03-12 reports: “{CLAIM}” These reported "
            "activities connect to the requested security capability."
        ),
        intent_signals=[SimpleNamespace(
            matched_icp_signal=0,
            description=CLAIM,
            url=SOURCE_URL,
        )],
    )
    icp = SimpleNamespace(
        prompt="Find security companies that launched a major capability.",
        product_service="Security operations software",
        intent_signals=[TARGET],
    )
    signal_results = [{
        "after_decay": 51,
        "matched_icp_signal": 0,
        "evidence_urls": [SOURCE_URL],
        "judge_verdict": {
            "decision": "verified",
            "client_ready": True,
            "authoritative_date": None,
            "authoritative_date_basis": "missing_or_conflicting",
            "verification_trace": {
                "intent_verdict": _signal_verdict(confidence="high")["answer"],
                "verified_source_context": [{
                    "url": SOURCE_URL,
                    "text": SOURCE_TEXT,
                    "source_publication_date": "",
                }],
            },
        },
    }]
    return company, icp, signal_results, {}


def _unit_grounding(document, *, facts_supported):
    source_index, values = next(iter(
        intent_details._bound_evidence_sources(document).items()
    ))
    quote = values[0][:intent_details._MAX_UNIT_EVIDENCE_QUOTE_LENGTH]
    return [
        {
            "unit_id": unit["unit_id"],
            "contains_factual_claim": True,
            "status": (
                "UNPROVEN"
                if not facts_supported and unit["unit_id"] == 0
                else "VERIFIED"
            ),
            "evidence": (
                []
                if not facts_supported and unit["unit_id"] == 0
                else [{"source_index": source_index, "quote": quote}]
            ),
        }
        for unit in document["intent_details_units"]
    ]


def test_review_evidence_surfaces_strict_first_party_body_dateline():
    company, icp, signal_results, fit = _mind_style_inputs()

    evidence = intent_details.review_evidence(company, icp, signal_results, fit)

    context = evidence["verified_signals"][0]["source_context"][0]
    assert context["source_body_dateline_dates"] == ["2026-03-12"]
    assert CLAIM in context["text"]
    assert "source dated 2026-03-12" in " ".join(
        unit["text"] for unit in evidence["intent_details_units"]
    )
    assert "concrete factual clause" in intent_details._SYSTEM


@pytest.mark.asyncio
async def test_exact_dated_quote_requires_concrete_diagnostic_when_rejected(
    monkeypatch,
):
    response = {
        **{name: True for name in intent_details._CHECKS},
        "facts_supported": False,
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
    }

    async def judge(prompt, **_kwargs):
        document = json.loads(prompt)
        grounding = _unit_grounding(document, facts_supported=True)
        grounding[1].update({
            "status": "UNPROVEN",
            "evidence": [],
        })
        return json.dumps({
            **response,
            "unit_grounding": grounding,
        })

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*_mind_style_inputs())

    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.asyncio
async def test_exact_dated_quote_preserves_grounded_factual_rejection(
    monkeypatch,
):
    paragraph = _mind_style_inputs()[0].intent_details
    response = {
        **{name: True for name in intent_details._CHECKS},
        "facts_supported": False,
        "unsupported_factual_clause": (
            "These reported activities connect to the requested security capability."
        ),
        "unsupported_factual_reason": "The evidence does not establish this connection.",
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
    }
    assert response["unsupported_factual_clause"] in paragraph

    async def judge(prompt, **_kwargs):
        document = json.loads(prompt)
        grounding = _unit_grounding(document, facts_supported=True)
        grounding[1].update({
            "status": "UNPROVEN",
            "evidence": [],
        })
        return json.dumps({
            **response,
            "unit_grounding": grounding,
        })

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*_mind_style_inputs())

    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False
    assert "unsupported_factual_clause" not in receipt


@pytest.mark.asyncio
async def test_exact_dated_quote_accepts_empty_diagnostic_only_when_supported(
    monkeypatch,
):
    response = {
        **{name: True for name in intent_details._CHECKS},
        "unsupported_factual_clause": "",
        "unsupported_factual_reason": "",
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
    }

    async def judge(prompt, **_kwargs):
        document = json.loads(prompt)
        return json.dumps({
            **response,
            "unit_grounding": _unit_grounding(
                document, facts_supported=True,
            ),
        })

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*_mind_style_inputs())

    assert receipt["decision"] == "match"
    assert receipt["checks"]["facts_supported"] is True


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, True),
        ({"unsupported_parts": ["launch timing"]}, False),
        ({"same_entity_check": "unclear"}, False),
        ({"supporting_quotes": ["Text absent from the source."]}, False),
        # Claim prose need not be copied verbatim to request a clarification.
        ({"claim": "Acme launched a capability absent from the source."}, True),
        ({"signal_status": "partially_supported"}, False),
    ],
)
def test_medium_supported_clarification_requires_coherent_exact_evidence(
    updates, expected
):
    verdict = _signal_verdict(confidence="medium")["answer"]
    item = verdict["signal_evaluations"][0]
    item.update(updates)

    assert verifier._supported_medium_needs_clarification(
        verdict, item, SOURCE_TEXT
    ) is expected


async def _verify_with_verdicts(monkeypatch, *verdicts):
    call = AsyncMock(side_effect=verdicts)
    fetch = AsyncMock(return_value={
        "results": [{
            "url": SOURCE_URL,
            "title": "Acme launches Control Copilot",
            "text": SOURCE_TEXT,
            "source_publication_date": "2026-03-12",
        }],
        "statuses": [{"source": "scrapingdog", "stage": "ok"}],
    })
    monkeypatch.setattr(verifier, "_call_openrouter", call)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    result = await verifier.verify_three_stage(
        object(),
        company_name="Acme",
        company_linkedin="https://www.linkedin.com/company/acme",
        company_website="https://acme.example",
        source_url=SOURCE_URL,
        miner_claim=CLAIM,
        target_signal_text=TARGET,
        miner_signal_date="2026-03-12",
        evidence_type="PRODUCT_LAUNCH",
        stage1_soft_reject=True,
    )
    return result, call


@pytest.mark.asyncio
async def test_exact_supported_medium_gets_one_bounded_clarification(monkeypatch):
    result, call = await _verify_with_verdicts(
        monkeypatch,
        _signal_verdict(confidence="medium"),
        _signal_verdict(confidence="high"),
    )

    assert call.await_count == 2
    assert call.await_args_list[1].kwargs["max_attempts"] == 1
    assert "ONE BOUNDED EVIDENCE-CONFIDENCE CLARIFICATION" in (
        call.await_args_list[1].args[2]
    )
    assert result["decision"] == "approve"
    assert result["client_ready"] is True
    assert result["stage3"]["confidence"] == "high"
    assert result["evidence_clarification"] == {
        "attempted": True,
        "resolved": True,
        "provider_error": False,
    }


@pytest.mark.asyncio
async def test_unresolved_medium_clarification_stays_fail_closed(monkeypatch):
    result, call = await _verify_with_verdicts(
        monkeypatch,
        _signal_verdict(confidence="medium"),
        _signal_verdict(confidence="medium"),
    )

    assert call.await_count == 2
    assert result["decision"] == "review"
    assert result["client_ready"] is False
    assert result["rejection_reason"] == "stage3_review"
    assert result["evidence_clarification"]["resolved"] is False


@pytest.mark.asyncio
async def test_medium_verdict_with_unsupported_part_does_not_get_clarification(
    monkeypatch,
):
    result, call = await _verify_with_verdicts(
        monkeypatch,
        _signal_verdict(
            confidence="medium", unsupported_parts=["launch timing"]
        ),
    )

    assert call.await_count == 1
    assert result["decision"] == "review"
    assert result["client_ready"] is False
    assert "evidence_clarification" not in result


@pytest.mark.asyncio
async def test_supported_medium_clarification_provider_error_is_unavailable(
    monkeypatch,
):
    result, call = await _verify_with_verdicts(
        monkeypatch,
        _signal_verdict(confidence="medium"),
        {"_error": "http_503"},
    )

    assert call.await_count == 2
    assert result["decision"] == "unavailable"
    assert result["client_ready"] is False
    assert result["rejection_reason"] == (
        "stage3_supported_medium_clarification_error:http_503"
    )
    assert result["evidence_clarification"] == {
        "attempted": True,
        "resolved": False,
        "provider_error": True,
    }
