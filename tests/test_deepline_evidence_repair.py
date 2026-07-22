"""Deepline evidence repair: gating, parsing, and the lab scorer hook.

Repair supplies replacement evidence candidates for an all-zero intent
verdict; the scorer re-verifies them itself and remains the only authority.
Everything fails open — a repair problem can never break scoring.
"""

import asyncio
from types import SimpleNamespace

import pytest

import qualification.scoring.deepline_evidence_repair as repair_module
import qualification.scoring.lead_scorer as lead_scorer
from gateway.qualification.models import CompanyOutput, ICPPrompt


def _company() -> CompanyOutput:
    return CompanyOutput.model_validate({
        "company_name": "TestCo",
        "company_website": "https://testco.com",
        "company_linkedin": "https://linkedin.com/company/testco",
        "industry": "Software",
        "sub_industry": "B2B SaaS",
        "employee_count": "51-200",
        "company_stage": "Growth",
        "country": "United States",
        "state": "",
        "description": "TestCo builds software.",
        "intent_signals": [{
            "source": "news",
            "description": "raised a round",
            "url": "https://aggregator.example/x",
            "date": "2026-07-01",
            "snippet": "TestCo raised",
            "matched_icp_signal": 0,
        }],
    })


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="i", prompt="", industry="Software", sub_industry="",
        target_roles=[], target_seniority="", employee_count="51-200",
        company_stage="", geography="United States", country="United States",
        product_service="", intent_signals=["Raised a funding round"],
    )


def test_disabled_without_flag_and_key(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_ENABLED", raising=False)
    monkeypatch.setenv("DEEPLINE_API_KEY", "k")
    assert repair_module.enabled() is False
    monkeypatch.setenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_ENABLED", "true")
    monkeypatch.delenv("DEEPLINE_API_KEY", raising=False)
    assert repair_module.enabled() is False
    monkeypatch.setenv("DEEPLINE_API_KEY", "k")
    assert repair_module.enabled() is True


def test_disabled_repair_returns_empty(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_ENABLED", raising=False)
    out = asyncio.run(repair_module.repair_sources(
        company_name="X", company_domain="x.com", requested_criterion="c"
    ))
    assert out == []


def test_budget_is_bounded(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_TIMEOUT_SECONDS", "5")
    assert repair_module._budget_s() == 10.0
    monkeypatch.setenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_TIMEOUT_SECONDS", "600")
    assert repair_module._budget_s() == 300.0
    monkeypatch.setenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_TIMEOUT_SECONDS", "90")
    assert repair_module._budget_s() == 90.0


def test_sources_found_at_any_nesting_depth():
    flat = {"sources": [{"url": "https://a.com"}]}
    nested = {"output": {"deep": [{"sources": [{"url": "https://b.com", "excerpt": "e"}]}]}}
    assert repair_module._sources(flat)[0]["url"] == "https://a.com"
    assert repair_module._sources(nested)[0]["url"] == "https://b.com"
    assert repair_module._sources({"sources": [{"noturl": 1}]}) == []


def test_run_id_and_status_synonyms():
    assert repair_module._run_id({"workflowId": "w"}) == "w"
    assert repair_module._run_id({"run": {"id": "r"}}) == "r"
    assert repair_module._status({"run": {"status": "Succeeded"}}) == "succeeded"
    assert repair_module._status({"state": "FAILED"}) == "failed"


def test_hook_rescues_all_fabricated(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_ENABLED", "true")
    monkeypatch.setenv("DEEPLINE_API_KEY", "k")

    async def fake_repair(**kwargs):
        assert kwargs["evidence_kind"] == "intent"
        return [{"url": "https://testco.com/blog/round",
                 "excerpt": "TestCo raised a round",
                 "published_date": "2026-07-02"}]

    async def fake_rescore(candidate, icp_arg):
        assert candidate.intent_signals[0].url == "https://testco.com/blog/round"
        return (60.0, 55.0, 0.9, 8, False, [{"raw": 60.0}])

    monkeypatch.setattr(repair_module, "repair_sources", fake_repair)
    monkeypatch.setattr(
        lead_scorer, "score_company_autoresearch_intent_signal", fake_rescore
    )
    out = asyncio.run(
        lead_scorer._attempt_autoresearch_evidence_repair(_company(), _icp())
    )
    assert out is not None
    assert out[4] is False and out[1] == 55.0


def test_hook_fails_open_when_repair_empty_or_unverified(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_ENABLED", "true")
    monkeypatch.setenv("DEEPLINE_API_KEY", "k")

    async def repair_empty(**kwargs):
        return []

    monkeypatch.setattr(repair_module, "repair_sources", repair_empty)
    assert asyncio.run(
        lead_scorer._attempt_autoresearch_evidence_repair(_company(), _icp())
    ) is None

    async def repair_junk(**kwargs):
        return [{"url": "notaurl"}]

    monkeypatch.setattr(repair_module, "repair_sources", repair_junk)
    assert asyncio.run(
        lead_scorer._attempt_autoresearch_evidence_repair(_company(), _icp())
    ) is None

    async def repair_ok(**kwargs):
        return [{"url": "https://testco.com/x"}]

    async def rescore_still_fabricated(candidate, icp_arg):
        return (0.0, 0.0, 0.0, 0, True, [])

    monkeypatch.setattr(repair_module, "repair_sources", repair_ok)
    monkeypatch.setattr(
        lead_scorer, "score_company_autoresearch_intent_signal",
        rescore_still_fabricated,
    )
    assert asyncio.run(
        lead_scorer._attempt_autoresearch_evidence_repair(_company(), _icp())
    ) is None


def test_hook_disabled_is_no_op(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_ENABLED", "false")
    assert asyncio.run(
        lead_scorer._attempt_autoresearch_evidence_repair(_company(), _icp())
    ) is None
