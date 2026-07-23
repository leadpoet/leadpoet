"""Allocation ancestry graphs are pruned of transport detail before the enclave.

Historical champion/reimbursement parent graphs accumulate ~2,600 transport
attempts per epoch (~5MB each). Shipping the full reimbursement-window history
every epoch pushed the coordinator scoring payload past the enclave's 64MB
MAX_INPUT_BYTES limit (epoch 24116, whole-subnet weight miss). The coordinator
enclave consumes these graphs by root receipt authority only, so the per-attempt
transport/host detail is dropped before send while the receipts, boots, and root
the enclave verifies are preserved.
"""

from __future__ import annotations

import json

from gateway.research_lab.v2_authority import _prune_ancestry_transport_detail


def _sample_graph() -> dict:
    root = "sha256:" + "a" * 64
    return {
        "schema_version": "leadpoet.attested_execution_receipt_graph.v2",
        "root_receipt_hash": root,
        "receipts": [
            {
                "receipt_hash": root,
                "role": "gateway_coordinator",
                "purpose": "research_lab.promotion_decision.v2",
                "transport_root": "sha256:" + "b" * 64,
            },
            {
                "receipt_hash": "sha256:" + "c" * 64,
                "role": "gateway_scoring",
                "purpose": "research_lab.candidate_evaluation.v2",
                "transport_root": "sha256:" + "d" * 64,
            },
        ],
        "boot_identities": [
            {"boot_identity_hash": "sha256:" + "e" * 64, "physical_role": "gateway_coordinator"},
        ],
        "transport_attempts": [
            {"attempt_hash": "sha256:%064x" % i, "job_id": "j", "purpose": "p"}
            for i in range(2600)
        ],
        "host_operations": [
            {"request": {"request_hash": "sha256:%064x" % i}} for i in range(40)
        ],
    }


def test_prune_empties_transport_and_host_only():
    graph = _sample_graph()
    pruned = _prune_ancestry_transport_detail(graph)

    assert pruned["transport_attempts"] == []
    assert pruned["host_operations"] == []
    # Everything the enclave verifies is preserved byte-for-byte.
    assert pruned["root_receipt_hash"] == graph["root_receipt_hash"]
    assert pruned["receipts"] == graph["receipts"]
    assert pruned["boot_identities"] == graph["boot_identities"]
    assert pruned["schema_version"] == graph["schema_version"]


def test_prune_does_not_mutate_the_validated_original():
    graph = _sample_graph()
    _prune_ancestry_transport_detail(graph)
    # The gateway still holds the full, already-validated graph for its own
    # binding extraction; pruning must not reach back into it.
    assert len(graph["transport_attempts"]) == 2600
    assert len(graph["host_operations"]) == 40


def test_prune_root_receipt_authority_survives():
    graph = _sample_graph()
    pruned = _prune_ancestry_transport_detail(graph)
    root = {r["receipt_hash"]: r for r in pruned["receipts"]}[
        pruned["root_receipt_hash"]
    ]
    # The reward-ancestry check reads root purpose/role/hash — all intact.
    assert root["purpose"] == "research_lab.promotion_decision.v2"
    assert root["role"] == "gateway_coordinator"


def test_prune_shrinks_payload_below_enclave_pressure():
    graph = _sample_graph()
    full = len(json.dumps(graph))
    pruned = len(json.dumps(_prune_ancestry_transport_detail(graph)))
    # The transport/host detail is the dominant, unbounded-growth component.
    assert pruned < full * 0.25
