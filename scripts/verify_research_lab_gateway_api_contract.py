#!/usr/bin/env python3
"""Verify Research Lab gateway API contracts without network or Supabase."""

from __future__ import annotations

import asyncio
from inspect import signature
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import HTTPException

import gateway.research_lab.api as research_lab_api
from gateway.research_lab.api import (
    _OPENROUTER_KEY_REGISTRATION_ATTEMPTS,
    _enforce_openrouter_key_registration_rate_limit,
    _effective_budget_doc,
    _require_default_research_model_tier,
    _validate_allowed_research_island,
    router,
)
from gateway.research_lab.bundles import build_shadow_report_bundle, sha256_json
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.models import (
    ResearchLabCandidateEvaluationResultRequest,
    ResearchLabLoopStartRequest,
    ResearchLabLoopTopUpRequest,
    ResearchLabOpenRouterKeyRegisterRequest,
    ResearchLabReceiptCreateRequest,
    ResearchLabScoreBundleCreateRequest,
    ResearchLabTicketCreateRequest,
)
from gateway.research_lab.key_vault import openrouter_key_ref, validate_openrouter_key_format
from leadpoet_verifier.research_evaluation import build_research_evaluation_score_bundle


def main() -> int:
    now = int(time.time())
    defaults = ResearchLabGatewayConfig.from_env()
    errors = []
    if defaults.api_enabled:
        errors.append("gateway API must default disabled")
    if defaults.production_writes_enabled:
        errors.append("production writes must default disabled")
    if any(defaults.live_mutation_flags().values()):
        errors.append("live mutation flags must default false")
    if defaults.evaluation_bundles_enabled:
        errors.append("evaluation bundle writes must default disabled")
    if defaults.loop_topups_enabled:
        errors.append("Research Lab top-ups must default disabled")
    if defaults.public_status().get("allowed_research_areas") != ["generalist"]:
        errors.append("Research Lab launch must default to generalist-only research area")
    payment_text = (ROOT / "gateway" / "qualification" / "api" / "payment.py").read_text(encoding="utf-8")
    helper_text = (ROOT / "gateway" / "qualification" / "utils" / "helpers.py").read_text(encoding="utf-8")
    for marker in ("fallback TAO", "taostats.io/api/price", "return 500.0", "return 400.0"):
        if marker in payment_text or marker in helper_text:
            errors.append(f"TAO price path still contains non-CoinGecko fallback marker: {marker}")
    try:
        _validate_allowed_research_island(defaults, "healthcare")
        errors.append("non-generalist research area was accepted by default")
    except HTTPException:
        pass
    tier_limited_config = ResearchLabGatewayConfig(
        auto_research_model="openrouter/test-model",
        default_compute_budget_usd=3.0,
        min_compute_budget_usd=1.0,
        max_compute_budget_usd=100.0,
        approved_auto_research_models_json=(
            '{"default":{"model":"openrouter/test-model","max_compute_budget_usd":5.0}}'
        ),
    )
    budget_doc = _effective_budget_doc(
        tier_limited_config,
        ticket={"ticket_doc": {}},
        research_model_tier="default",
        requested_compute_budget_usd=4.0,
        max_compute_budget_usd=99.0,
    )
    if budget_doc.get("max_compute_budget_usd") != 5.0:
        errors.append("tier max_compute_budget_usd was not enforced over miner-supplied max")
    try:
        _require_default_research_model_tier(tier_limited_config, "custom")
        errors.append("non-default research model tier was accepted for launch")
    except HTTPException:
        pass
    try:
        _effective_budget_doc(
            tier_limited_config,
            ticket={"ticket_doc": {}},
            research_model_tier="default",
            requested_compute_budget_usd=6.0,
            max_compute_budget_usd=99.0,
        )
        errors.append("requested compute budget above tier cap was accepted")
    except HTTPException:
        pass

    paths = {route.path for route in router.routes}
    for required in {
        "/research-lab/status",
        "/research-lab/openrouter-keys",
        "/research-lab/tickets",
        "/research-lab/probes",
        "/research-lab/loop-start",
        "/research-lab/loop-topups",
        "/research-lab/receipts",
        "/research-lab/tickets/{ticket_id}",
        "/research-lab/receipts/{receipt_id}",
        "/research-lab/evaluations/score-bundles",
        "/research-lab/evaluations/candidate-results",
        "/research-lab/evaluations/score-bundles/{score_bundle_id}",
        "/research-lab/evaluations/latest/{epoch}",
        "/research-lab/allocations/live/{epoch}",
        "/research-lab/reports/shadow/{epoch}",
    }:
        if required not in paths:
            errors.append(f"missing route: {required}")
    endpoints = {route.path: route.endpoint for route in router.routes}
    for protected_path in {
        "/research-lab/tickets/{ticket_id}",
        "/research-lab/receipts/{receipt_id}",
        "/research-lab/evaluations/score-bundles/{score_bundle_id}",
        "/research-lab/evaluations/latest/{epoch}",
    }:
        endpoint = endpoints.get(protected_path)
        if endpoint is None or "x_leadpoet_internal_key" not in signature(endpoint).parameters:
            errors.append(f"raw read route is missing internal-key guard parameter: {protected_path}")

    ticket = ResearchLabTicketCreateRequest(
        miner_hotkey="5FminerHotkey11111111111111111111111111111111",
        signature="0x" + "11" * 64,
        timestamp=now,
        idempotency_key="ticket-idempotency-001",
        island="generalist",
        brief_sanitized_ref="brief_sanitized:sha256:abc123",
        brief_public_summary="Improve evidence freshness scoring and reduce overbroad company matches.",
    )
    reparsed_ticket = ResearchLabTicketCreateRequest.model_validate(ticket.model_dump(mode="json"))
    if reparsed_ticket != ticket:
        errors.append("ticket request failed json round-trip")

    key_registration = ResearchLabOpenRouterKeyRegisterRequest(
        miner_hotkey=ticket.miner_hotkey,
        signature=ticket.signature,
        timestamp=now,
        idempotency_key="openrouter-key-idempotency-001",
        openrouter_api_key="sk-or-v1-" + "a" * 48,
        key_label="research-lab-test-key",
    )
    reparsed_key_registration = ResearchLabOpenRouterKeyRegisterRequest.model_validate(
        key_registration.model_dump(mode="json")
    )
    if reparsed_key_registration != key_registration:
        errors.append("OpenRouter key registration request failed json round-trip")
    try:
        validate_openrouter_key_format(key_registration.openrouter_api_key)
    except ValueError:
        errors.append("valid OpenRouter key format was rejected")
    ref = openrouter_key_ref(miner_hotkey=ticket.miner_hotkey, key_hash="a" * 64)
    if not ref.startswith("encrypted_ref:openrouter:") or len(ref.rsplit(":", 1)[-1]) != 32:
        errors.append("OpenRouter key ref shape is invalid")
    errors.extend(asyncio.run(_verify_openrouter_key_ref_validation(ticket.miner_hotkey, ref)))
    try:
        _OPENROUTER_KEY_REGISTRATION_ATTEMPTS.clear()
        _enforce_openrouter_key_registration_rate_limit(ticket.miner_hotkey)
    except NameError:
        errors.append("OpenRouter key registration rate limiter helper is missing")
    except HTTPException:
        errors.append("first OpenRouter key registration attempt was rate limited")

    loop_start = ResearchLabLoopStartRequest(
        miner_hotkey=ticket.miner_hotkey,
        signature=ticket.signature,
        timestamp=now,
        idempotency_key="loop-start-idempotency-001",
        ticket_id="11111111-1111-4111-8111-111111111111",
        payment_block_hash="0x" + "22" * 32,
        payment_extrinsic_index=4,
        miner_openrouter_key_ref="encrypted_ref:vault:miner-openrouter-key-001",
        miner_openrouter_key_handling="encrypted_ref",
        miner_openrouter_preflight_status="passed",
        research_model_tier="default",
        requested_compute_budget_usd=5.0,
        max_compute_budget_usd=25.0,
    )
    reparsed_loop_start = ResearchLabLoopStartRequest.model_validate(loop_start.model_dump(mode="json"))
    if reparsed_loop_start != loop_start:
        errors.append("loop-start request failed json round-trip")
    loop_signed_payload = loop_start.signed_payload()
    if "credit_id" in loop_signed_payload:
        errors.append("loop-start signed payload included omitted credit_id default")
    credit_loop_start = ResearchLabLoopStartRequest(
        miner_hotkey=ticket.miner_hotkey,
        signature=ticket.signature,
        timestamp=now,
        idempotency_key="loop-start-credit-idempotency-001",
        ticket_id="11111111-1111-4111-8111-111111111111",
        credit_id="loop_start_credit:" + "c" * 32,
        miner_openrouter_key_ref="encrypted_ref:vault:miner-openrouter-key-001",
        miner_openrouter_key_handling="encrypted_ref",
        miner_openrouter_preflight_status="passed",
        research_model_tier="default",
        requested_compute_budget_usd=5.0,
        max_compute_budget_usd=25.0,
    )
    reparsed_credit_loop_start = ResearchLabLoopStartRequest.model_validate(credit_loop_start.model_dump(mode="json"))
    if reparsed_credit_loop_start != credit_loop_start:
        errors.append("credit loop-start request failed json round-trip")
    try:
        ResearchLabLoopStartRequest(
            **{
                **loop_start.model_dump(),
                "credit_id": "loop_start_credit:" + "d" * 32,
            }
        )
        errors.append("loop-start request accepted both payment fields and credit_id")
    except ValueError:
        pass

    topup = ResearchLabLoopTopUpRequest(
        miner_hotkey=ticket.miner_hotkey,
        signature=ticket.signature,
        timestamp=now,
        idempotency_key="loop-topup-idempotency-001",
        ticket_id="11111111-1111-4111-8111-111111111111",
        continue_from_run_id="22222222-2222-4222-8222-222222222222",
        payment_block_hash="0x" + "33" * 32,
        payment_extrinsic_index=5,
        additional_compute_budget_usd=10.0,
        research_model_tier="default",
        miner_openrouter_key_ref="encrypted_ref:vault:miner-openrouter-key-001",
        miner_openrouter_key_handling="encrypted_ref",
        miner_openrouter_preflight_status="passed",
    )
    reparsed_topup = ResearchLabLoopTopUpRequest.model_validate(topup.model_dump(mode="json"))
    if reparsed_topup != topup:
        errors.append("loop-topup request failed json round-trip")

    receipt = ResearchLabReceiptCreateRequest(
        internal_run_ref="runner:research-loop:001",
        ticket_id="11111111-1111-4111-8111-111111111111",
        miner_hotkey=ticket.miner_hotkey,
        island="generalist",
        receipt_status="completed",
        provider_usage=[
            {"provider": "openrouter", "key_source": "miner_key_ref", "key_ref": "encrypted_ref:vault:miner-openrouter-key-001"},
            {"provider": "exa", "key_source": "leadpoet_server_side"},
        ],
        cost_ledger={"total_usd": 1.25},
        receipt_doc={"summary": "sanitized receipt"},
    )
    reparsed_receipt = ResearchLabReceiptCreateRequest.model_validate(receipt.model_dump(mode="json"))
    if reparsed_receipt != receipt:
        errors.append("receipt request failed json round-trip")

    try:
        ResearchLabTicketCreateRequest(
            **{
                **ticket.model_dump(),
                "brief_public_summary": "raw_secret_should_fail",
            }
        )
        errors.append("raw secret marker in ticket public summary was accepted")
    except ValueError:
        pass

    try:
        ResearchLabLoopStartRequest(
            **{
                **loop_start.model_dump(),
                "miner_openrouter_key_ref": "raw_secret_should_fail",
            }
        )
        errors.append("raw OpenRouter key was accepted")
    except ValueError:
        pass

    bundle = build_shadow_report_bundle(
        epoch=123,
        weight_input_snapshots=[
            {
                "weight_input_snapshot_id": "11111111-1111-4111-8111-111111111111",
                "epoch": 123,
                "snapshot_status": "shadow",
                "snapshot_doc": {},
            }
        ],
        ticket_rows=[],
        queue_rows=[],
        receipt_rows=[],
        reimbursement_rows=[],
    )
    if not bundle.get("shadow_only") or not bundle.get("read_only"):
        errors.append("shadow report bundle must be read-only")
    if bundle.get("submission_allowed") or bundle.get("on_chain_submission_allowed"):
        errors.append("shadow report bundle must not allow submission")
    if sha256_json(bundle["source_state"]) != bundle["source_state_hash"]:
        errors.append("shadow report source_state_hash mismatch")
    if sha256_json(bundle["weight_vector"]) != bundle["weight_vector_hash"]:
        errors.append("shadow report weight_vector_hash mismatch")
    try:
        build_shadow_report_bundle(
            epoch=123,
            weight_input_snapshots=[],
            ticket_rows=[{"ticket_doc": {"raw_secret": "should-fail"}}],
            queue_rows=[],
            receipt_rows=[],
            reimbursement_rows=[],
        )
        errors.append("shadow report accepted raw secret source state")
    except ValueError:
        pass

    score_bundle = build_research_evaluation_score_bundle(
        run_id="11111111-1111-4111-8111-111111111111",
        ticket_id="11111111-1111-4111-8111-111111111111",
        miner_hotkey=ticket.miner_hotkey,
        island="generalist",
        evaluation_epoch=123,
        parent_artifact_hash="sha256:" + "1" * 64,
        candidate_artifact_hash="sha256:" + "2" * 64,
        private_model_manifest_hash="sha256:" + "3" * 64,
        candidate_patch_hash="sha256:" + "4" * 64,
        icp_set_hash="sha256:" + "5" * 64,
        scoring_version="qualification-company-scorer:v1",
        evaluator_version="research-lab-private-evaluator:v1",
        per_icp_results=[
            {
                "icp_ref": "icp:a",
                "icp_hash": "sha256:" + "a" * 64,
                "base_company_scores": [80, 60],
                "candidate_company_scores": [90, 70],
            }
        ],
        evidence_bundle_refs=["evidence_bundle:sha256:" + "6" * 64],
        execution_trace_ref="execution_trace:11111111-1111-4111-8111-111111111111",
        cost_ledger_ref="cost_ledger:sha256:" + "7" * 64,
        benchmark_split_ref="sealed_benchmark:qualification:intent:v1",
        policy={
            "min_delta": 2.0,
            "min_delta_lcb": 2.0,
            "min_successful_icps": 1,
            "min_candidate_score": 15.0,
            "observed_cost_usd": 1.25,
        },
        signature_ref="kms-signature:research-lab-eval:test",
    )
    score_request = ResearchLabScoreBundleCreateRequest(score_bundle=score_bundle)
    reparsed_score_request = ResearchLabScoreBundleCreateRequest.model_validate(score_request.model_dump(mode="json"))
    if reparsed_score_request != score_request:
        errors.append("score-bundle request failed json round-trip")
    try:
        ResearchLabScoreBundleCreateRequest(score_bundle={**score_bundle, "signature_ref": ""})
        errors.append("score-bundle request accepted missing signature_ref")
    except ValueError:
        pass

    candidate_result = ResearchLabCandidateEvaluationResultRequest(
        candidate_id="candidate:" + "8" * 64,
        candidate_status="scored",
        evaluator_ref="validator:test",
        score_bundle=score_bundle,
        result_doc={"summary": "validator completed paired scoring"},
    )
    reparsed_candidate_result = ResearchLabCandidateEvaluationResultRequest.model_validate(
        candidate_result.model_dump(mode="json")
    )
    if reparsed_candidate_result != candidate_result:
        errors.append("candidate evaluation result request failed json round-trip")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Research Lab gateway API contract verified: routes mounted, flags default false, models round-trip, raw key rejected.")
    return 0


async def _verify_openrouter_key_ref_validation(miner_hotkey: str, key_ref: str) -> list[str]:
    errors: list[str] = []
    original_select_one = research_lab_api.select_one
    expected_row = {
        "key_ref": key_ref,
        "miner_hotkey": miner_hotkey,
        "preflight_status": "passed",
    }

    async def fake_select_one(table: str, **kwargs):
        if table != "research_lab_openrouter_key_refs":
            raise AssertionError(f"unexpected select_one table: {table}")
        filters = dict(kwargs.get("filters") or ())
        if filters.get("key_ref") == key_ref and filters.get("miner_hotkey") == miner_hotkey:
            return dict(expected_row)
        return None

    research_lab_api.select_one = fake_select_one
    try:
        config = ResearchLabGatewayConfig(miner_openrouter_key_required=True)
        try:
            await research_lab_api._validate_miner_openrouter_key_ref(
                config,
                miner_hotkey=miner_hotkey,
                key_ref=key_ref,
                key_handling="encrypted_ref",
            )
        except HTTPException as exc:
            errors.append(f"valid encrypted OpenRouter key ref was rejected: {exc.detail}")
        try:
            await research_lab_api._validate_miner_openrouter_key_ref(
                config,
                miner_hotkey="5FotherHotkey11111111111111111111111111111",
                key_ref=key_ref,
                key_handling="encrypted_ref",
            )
            errors.append("encrypted OpenRouter key ref ownership mismatch was accepted")
        except HTTPException:
            pass
        try:
            await research_lab_api._validate_miner_openrouter_key_ref(
                config,
                miner_hotkey=miner_hotkey,
                key_ref="ephemeral_ref:client-claimed",
                key_handling="encrypted_ref",
            )
            errors.append("encrypted OpenRouter key handling accepted a non-encrypted ref")
        except HTTPException:
            pass
    finally:
        research_lab_api.select_one = original_select_one
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
