from __future__ import annotations

import copy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.coordinator_weight_source_v2 import (
    CoordinatorWeightSourceV2,
    CoordinatorWeightSourceV2Error,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_execution_receipt_body,
    create_signed_execution_receipt,
    sha256_json,
)
from leadpoet_canonical.sourcing_history_v2 import (
    build_sourcing_decision_v2,
    build_sourcing_epoch_v2,
)
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    weight_config_hash,
)
from leadpoet_canonical.weight_authority_v2 import (
    WEIGHT_INPUT_PURPOSES,
    gateway_weight_input_value_documents_v2,
)


HASH = "sha256:" + "a" * 64
FULFILLMENT_POOL = 1.0 - 0.2 - 0.0 - 0.095


class FakeReader:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        return [dict(row) for row in self.rows.get(policy_id, [])]


def _snapshot(**overrides):
    value = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "commit_sha": "a" * 40,
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": "burn",
        "metagraph_hotkeys": ["burn", "miner"],
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": True,
        "base_burn_share": 0.0,
        "champion_share": 0.0,
        "champion_uid": None,
        "effective_champion_share": 0.0,
        "research_lab_fallback_share": 0.2,
        "research_lab_allocation_doc": {
            "allocation_hash": HASH,
            "lab_cap_percent": 20.0,
            "unallocated_percent": 20.0,
            "source_add_allocations": [],
            "reimbursement_allocations": [],
            "champion_allocations": [],
            "queued_champion_allocations": [],
        },
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": FULFILLMENT_POOL,
        "fulfillment_rows": [
            {"hotkey": "miner", "share": FULFILLMENT_POOL}
        ],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125000,
        "min_total_rep_for_distribution": 100,
    }
    value.update(overrides)
    value["config_hash"] = weight_config_hash(value)
    return value


def _payload(category, snapshot=None, **overrides):
    value = {
        "category": category,
        "calculation_snapshot": snapshot or _snapshot(),
        "gateway_authority_event_hash": "sha256:" + "2" * 64,
        "allocation_receipt": None,
        "leaderboard_window_start": "2026-07-03T20:00:00Z",
        "leaderboard_window_end": "2026-07-10T20:00:00Z",
    }
    value.update(overrides)
    return value


def _context(purpose, parents=()):
    return ExecutionContextV2(
        job_id="weight-input:test:100",
        purpose=purpose,
        epoch_id=100,
        parent_receipt_hashes=tuple(parents),
    )


def _sourcing_epoch_receipt(source_doc):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_execution_receipt_body(
        role="gateway_scoring",
        purpose="qualification.sourcing_epoch.v2",
        job_id="qualification-sourcing-epoch:%d" % source_doc["epoch_id"],
        epoch_id=source_doc["epoch_id"],
        sequence=0,
        commit_sha="b" * 40,
        pcr0="c" * 96,
        build_manifest_hash="sha256:" + "d" * 64,
        dependency_lock_hash="sha256:" + "e" * 64,
        config_hash="sha256:" + "f" * 64,
        boot_identity_hash="sha256:" + "1" * 64,
        input_root=source_doc["decision_root"],
        output_root=sha256_json(source_doc),
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=EMPTY_ARTIFACT_ROOT,
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at="2026-07-10T20:00:00Z",
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )


def test_bans_are_reconstructed_from_authenticated_rows_not_host_snapshot():
    reader = FakeReader({"banned_hotkeys": [{"hotkey": "5B"}, {"hotkey": "5A"}]})
    source = CoordinatorWeightSourceV2(reader)
    snapshot = _snapshot(
        banned_hotkeys=["5A", "5B"],
        rolling_scores=[],
    )
    document = source.resolve(
        payload=_payload("bans", snapshot=snapshot),
        context=_context("research_lab.ban_input.v2"),
    )
    assert document["value"] == {
        "banned_hotkeys": ["5A", "5B"],
        "banned_lookup_ok": True,
    }

    forged = copy.deepcopy(snapshot)
    forged["banned_hotkeys"] = []
    with pytest.raises(CoordinatorWeightSourceV2Error, match="differs"):
        source.resolve(
            payload=_payload("bans", snapshot=forged),
            context=_context("research_lab.ban_input.v2"),
        )


def test_fulfillment_rows_and_pool_cap_match_existing_formula_exactly():
    reader = FakeReader(
        {
            "fulfillment_active_rewards": [
                {
                    "miner_hotkey": "miner",
                    "reward_pct": 0.8,
                    "reward_expires_epoch": 101,
                }
            ]
        }
    )
    source = CoordinatorWeightSourceV2(reader)
    document = source.resolve(
        payload=_payload("fulfillment_rewards"),
        context=_context("research_lab.fulfillment_input.v2"),
    )
    assert document["value"]["fulfillment_share"] == FULFILLMENT_POOL
    assert document["value"]["fulfillment_rows"] == [
        {"hotkey": "miner", "share": FULFILLMENT_POOL}
    ]


def test_leaderboard_reconstructs_wins_tiebreak_and_ban_filter():
    reader = FakeReader(
        {
            "fulfillment_leaderboard_winners": [
                {"miner_hotkey": "miner", "reward_pct": 0.5},
                {"miner_hotkey": "miner", "reward_pct": 0.2},
                {"miner_hotkey": "banned", "reward_pct": 1.0},
            ],
            "banned_hotkeys": [{"hotkey": "banned"}],
        }
    )
    snapshot = _snapshot(
        leaderboard_entries=[{"miner_hotkey": "miner", "wins": 2}]
    )
    source = CoordinatorWeightSourceV2(reader)
    document = source.resolve(
        payload=_payload("leaderboard", snapshot=snapshot),
        context=_context("research_lab.leaderboard_input.v2"),
    )
    assert document["value"]["leaderboard_entries"] == [
        {"miner_hotkey": "miner", "wins": 2}
    ]


def test_disabled_fulfillment_commits_empty_leaderboard_without_paying_rows():
    reader = FakeReader(
        {
            "fulfillment_leaderboard_winners": [
                {"miner_hotkey": "miner", "reward_pct": 0.5},
            ],
            "banned_hotkeys": [],
        }
    )
    snapshot = _snapshot(
        ff_enabled=False,
        fulfillment_share=0.0,
        fulfillment_rows=[],
        leaderboard_entries=[],
    )
    source = CoordinatorWeightSourceV2(reader)
    document = source.resolve(
        payload=_payload("leaderboard", snapshot=snapshot),
        context=_context("research_lab.leaderboard_input.v2"),
    )
    assert document["value"]["leaderboard_entries"] == []
    assert reader.calls[0] == (
        "fulfillment_leaderboard_winners",
        {
            "window_start": "2026-07-03T20:00:00Z",
            "window_end": "2026-07-10T20:00:00Z",
        },
    )


def test_sourcing_history_is_rebuilt_only_from_signed_epoch_receipts():
    decisions = [
        build_sourcing_decision_v2(
            epoch_id=99,
            sequence=sequence,
            lead_id_hash=sha256_json({"lead": sequence}),
            miner_hotkey="miner",
            decision="approve",
            rep_score=score,
            is_icp_multiplier=0,
        )
        for sequence, score in ((0, 4), (1, 6))
    ]
    source_doc = build_sourcing_epoch_v2(epoch_id=99, decisions=decisions)
    receipt = _sourcing_epoch_receipt(source_doc)
    reader = FakeReader(
        {
            "sourcing_epoch_inputs": [
                {
                    "epoch_id": 99,
                    "epoch_hash": source_doc["epoch_hash"],
                    "receipt_hash": receipt["receipt_hash"],
                    "source_doc": source_doc,
                    "receipt_doc": receipt,
                },
            ],
            "attested_receipt_by_hash": [{"receipt_doc": receipt}],
        }
    )
    snapshot = _snapshot(
        rolling_lead_count=2,
        rolling_scores=[{"hotkey": "miner", "score": 10}],
    )
    document = CoordinatorWeightSourceV2(reader).resolve(
        payload=_payload("sourcing_history", snapshot=snapshot),
        context=_context(
            "research_lab.sourcing_input.v2",
            parents=(receipt["receipt_hash"],),
        ),
    )
    assert document["value"] == {
        "rolling_lead_count": 2,
        "rolling_scores": [{"hotkey": "miner", "score": 10}],
    }


def test_sourcing_history_rejects_undeclared_or_modified_epoch_receipt():
    source_doc = build_sourcing_epoch_v2(epoch_id=99, decisions=[])
    receipt = _sourcing_epoch_receipt(source_doc)
    row = {
        "epoch_id": 99,
        "epoch_hash": source_doc["epoch_hash"],
        "receipt_hash": receipt["receipt_hash"],
        "source_doc": source_doc,
        "receipt_doc": receipt,
    }
    source = CoordinatorWeightSourceV2(
        FakeReader(
            {
                "sourcing_epoch_inputs": [row],
                "attested_receipt_by_hash": [{"receipt_doc": receipt}],
            }
        )
    )
    with pytest.raises(CoordinatorWeightSourceV2Error, match="declared source"):
        source.resolve(
            payload=_payload("sourcing_history"),
            context=_context("research_lab.sourcing_input.v2"),
        )

    tampered = copy.deepcopy(row)
    tampered["source_doc"]["approved_lead_count"] = 1
    with pytest.raises(CoordinatorWeightSourceV2Error, match="document is invalid"):
        CoordinatorWeightSourceV2(
            FakeReader(
                {
                    "sourcing_epoch_inputs": [tampered],
                    "attested_receipt_by_hash": [{"receipt_doc": receipt}],
                }
            )
        ).resolve(
            payload=_payload("sourcing_history"),
            context=_context(
                "research_lab.sourcing_input.v2",
                parents=(receipt["receipt_hash"],),
            ),
        )


def test_allocation_projection_requires_a_signed_declared_parent():
    source = CoordinatorWeightSourceV2(FakeReader({}))
    with pytest.raises(CoordinatorWeightSourceV2Error, match="receipt is missing"):
        source.resolve(
            payload=_payload("research_lab_allocation"),
            context=_context("research_lab.allocation.v2"),
        )


def test_anomaly_hash_is_derived_only_from_signed_upstream_documents():
    snapshot = _snapshot()
    documents = gateway_weight_input_value_documents_v2(
        calculation_snapshot=snapshot,
        gateway_authority_event_hash="sha256:" + "2" * 64,
    )
    categories = (
        "research_lab_allocation",
        "fulfillment_rewards",
        "leaderboard",
        "bans",
        "sourcing_history",
    )
    upstream = {category: documents[category] for category in categories}
    graphs = []
    parent_hashes = []
    for index, category in enumerate(categories, start=1):
        receipt_hash = "sha256:" + ("%x" % index) * 64
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        graphs.append(
            {
                "root_receipt_hash": receipt_hash,
                "receipts": [
                    {
                        "receipt_hash": receipt_hash,
                        "role": role,
                        "purpose": purpose,
                        "output_root": sha256_json(upstream[category]),
                    }
                ],
            }
        )
        parent_hashes.append(receipt_hash)
    context = _context(
        "research_lab.anomaly_adjustment_input.v2",
        parents=parent_hashes,
    )
    context.external_receipt_graphs = graphs
    source = CoordinatorWeightSourceV2(FakeReader({}))

    result = source.resolve(
        payload=_payload(
            "anomaly_adjustments",
            snapshot=snapshot,
            upstream_documents=upstream,
        ),
        context=context,
    )
    assert result == documents["anomaly_adjustments"]

    tampered = copy.deepcopy(upstream)
    tampered["bans"]["value"]["banned_hotkeys"] = ["forged"]
    with pytest.raises(CoordinatorWeightSourceV2Error, match="differs from its receipt"):
        source.resolve(
            payload=_payload(
                "anomaly_adjustments",
                snapshot=snapshot,
                upstream_documents=tampered,
            ),
            context=context,
        )
