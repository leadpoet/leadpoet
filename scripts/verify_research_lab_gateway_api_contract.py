#!/usr/bin/env python3
"""Verify Research Lab gateway API contracts without network or Supabase."""

from __future__ import annotations

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.api import router
from gateway.research_lab.bundles import build_shadow_report_bundle, sha256_json
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.models import (
    ResearchLabLoopStartRequest,
    ResearchLabReceiptCreateRequest,
    ResearchLabTicketCreateRequest,
)


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

    paths = {route.path for route in router.routes}
    for required in {
        "/research-lab/status",
        "/research-lab/tickets",
        "/research-lab/probes",
        "/research-lab/loop-start",
        "/research-lab/receipts",
        "/research-lab/tickets/{ticket_id}",
        "/research-lab/receipts/{receipt_id}",
        "/research-lab/reports/shadow/{epoch}",
    }:
        if required not in paths:
            errors.append(f"missing route: {required}")

    ticket = ResearchLabTicketCreateRequest(
        miner_hotkey="5FminerHotkey11111111111111111111111111111111",
        signature="0x" + "11" * 64,
        timestamp=now,
        idempotency_key="ticket-idempotency-001",
        island="generalist",
        brief_sanitized_ref="brief_sanitized:sha256:abc123",
    )
    reparsed_ticket = ResearchLabTicketCreateRequest.model_validate(ticket.model_dump(mode="json"))
    if reparsed_ticket != ticket:
        errors.append("ticket request failed json round-trip")

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
    )
    reparsed_loop_start = ResearchLabLoopStartRequest.model_validate(loop_start.model_dump(mode="json"))
    if reparsed_loop_start != loop_start:
        errors.append("loop-start request failed json round-trip")

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

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Research Lab gateway API contract verified: routes mounted, flags default false, models round-trip, raw key rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
