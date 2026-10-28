"""Replay the saved MAA fit gate through JSON and real host recomputation."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import scoring
from qualification.scoring import lead_scorer
from qualification.scoring.arena_integrity import verified_identity_receipt
from qualification.scoring.company_fit_decision import (
    CompanyFitDecisionResult, company_quality_receipt_matches_claim,
    reconcile_company_fit_decisions,
)
from qualification.scoring.competition import (
    _normalized_company, _normalized_icp, apply_company_judgment_context,
)
from qualification.scoring.evaluation_clock import use_evaluation_date


@pytest.mark.parametrize("quality", [False, True])
def test_saved_maa_gate_receipt_survives_json_and_host_recomputation(monkeypatch, quality):
    fixture = json.loads((Path(__file__).parent / "fixtures/verified_homepage_brand_maa_host_roundtrip.json").read_text())
    before = deepcopy(fixture)
    original = fixture["raw_judgment"]
    saved_gate, = original["verifier_gate_receipts"]
    saved = saved_gate["dimension_evidence"]
    company = CompanyOutput(**_normalized_company(fixture["company"], integrity_policy=True, company_quality=quality))
    icp = ICPPrompt(**_normalized_icp(fixture["icp"]))
    homepage = CompanyFitDecisionResult(
        saved["identity"]["homepage_identity_decision"],
        saved["identity"]["homepage_identity_reason"],
        details={"identity": deepcopy(saved["identity"]["identity"]),
                 "actual_final_url": saved["identity"]["actual_final_url"]},
    )
    fetch_homepage = AsyncMock(return_value=homepage)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", fetch_homepage)

    async def saved_web_result(claim, requested, **arguments):
        # Replay the saved post-provider observations/evidence. Only identity
        # is re-evaluated; no new raw page, model finding or intent is invented.
        assert claim == company and requested == icp
        anchor = lead_scorer._verified_homepage_identity_anchor(arguments["verified_homepage_identity"])
        receipt = lead_scorer._web_identity_receipt(
            claim, deepcopy(saved["web_observations"]), verified_homepage_identity=anchor,
            verified_rebrand_identity=deepcopy(saved_gate["supporting_receipts"][0]["claims"]["rebrand"]),
            company_quality=quality,
        )
        decisions = {key: saved[key]["observed_decision"]
                     for key in ("employee_size", "industry", "geography", "stage")}
        grounding = deepcopy(saved_gate["supporting_receipts"][1])
        grounding.pop("gate")
        return CompanyFitDecisionResult(
            reconcile_company_fit_decisions([receipt["decision"], *decisions.values(),
                                             saved_gate["required_attribute_decision"]]),
            details={
                "identity_decision": receipt["decision"], "identity_receipt": receipt,
                "dimension_decisions": decisions,
                "dimension_evidence": {key: deepcopy(saved[key]["web_evidence"])
                                       for key in (*decisions, "required_attribute")},
                "required_attribute_decision": saved_gate["required_attribute_decision"],
                "provider_observations": deepcopy(saved["web_observations"]),
                "investigation_receipt": deepcopy(saved_gate["supporting_receipts"][0]),
                "required_attribute_grounding": grounding,
            },
        )

    web = AsyncMock(side_effect=saved_web_result)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web)
    with use_evaluation_date("2026-10-27"):
        gate = asyncio.run(lead_scorer._verify_company_fit(
            company, icp, 0.0, 0.0, set(), require_https_transport=True, company_quality=quality,
        ))
    assert gate.decision == "match"
    gate_receipt = gate.receipt("company_fit")
    assert gate_receipt["company_fit_dimensions"] == {key: "match" for key in saved_gate["company_fit_dimensions"]}
    assert {key: value for key, value in gate_receipt["dimension_evidence"].items() if key != "identity"} == {
        key: value for key, value in saved.items() if key != "identity"
    }
    assert gate_receipt["supporting_receipts"] == saved_gate["supporting_receipts"]
    # Exercise the production receipt/breakdown assembly without pretending
    # that the previously unevaluated primary intent has passed.
    breakdown = lead_scorer._zero_company_breakdown(
        "Intent not evaluated in this gate-only replay", verifier_gate_receipts=[gate_receipt],
    ).model_dump(mode="json")
    wire = json.loads(json.dumps(breakdown, allow_nan=False))
    receipt = verified_identity_receipt(wire["verifier_gate_receipts"])
    assert receipt["reason_code"] == "verified_homepage_brand_alias"
    assert receipt["verified_homepage_brand_alias"] is True
    assert company_quality_receipt_matches_claim(receipt, fixture["company"])
    recomputed = apply_company_judgment_context([fixture["company"]], [wire], company_quality=quality)
    assert scoring.validate_breakdowns_for_item(
        recomputed, icp=fixture["icp"], companies=[fixture["company"]],
        integrity_policy=True, company_quality=quality,
    ) == recomputed
    assert recomputed[0]["company_identity_key"] == "linkedin:mid-america-apartment-communities"
    assert recomputed[0]["company_qualified"] is False
    for field in ("final_score", "intent_signal_raw", "intent_signal_final", "intent_signals_detail"):
        assert recomputed[0][field] == original[field]
    missing = json.loads(json.dumps(wire))
    missing_receipt = verified_identity_receipt(missing["verifier_gate_receipts"])
    missing_receipt.pop("verified_homepage_brand_alias")
    assert not company_quality_receipt_matches_claim(missing_receipt, fixture["company"])
    assert apply_company_judgment_context([fixture["company"]], [missing], company_quality=quality)[0]["company_qualified"] is False
    fetch_homepage.assert_awaited_once()
    web.assert_awaited_once()
    assert fixture == before
