from __future__ import annotations

import pytest

from gateway.research_lab import attested_weight_inputs_v2
from gateway.research_lab.attested_weight_inputs_v2 import (
    AttestedWeightInputsV2Error,
    build_gateway_weight_inputs_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    gateway_weight_input_value_documents_v2,
)
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    weight_config_hash,
)


def _snapshot():
    value = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 100,
        "block": 36001,
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
            "lab_cap_percent": 20.0,
            "unallocated_percent": 20.0,
            "champion_allocations": [],
            "queued_champion_allocations": [],
            "reimbursement_allocations": [],
            "source_add_allocations": [],
        },
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.705,
        "fulfillment_rows": [{"hotkey": "miner", "share": 0.705}],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125000,
        "min_total_rep_for_distribution": 100,
    }
    value["config_hash"] = weight_config_hash(value)
    return value


def _allocation_graph():
    receipt_hash = sha256_json({"allocation": 100})
    receipt = {
        "receipt_hash": receipt_hash,
        "role": "gateway_coordinator",
        "purpose": "research_lab.allocation.v2",
        "epoch_id": 100,
    }
    return {
        "root_receipt_hash": receipt_hash,
        "boot_identities": [],
        "receipts": [receipt],
        "transport_attempts": [],
        "host_operations": [],
    }


@pytest.mark.asyncio
async def test_gateway_weight_input_builder_attests_every_category_without_host_block(
    monkeypatch,
):
    monkeypatch.setattr(
        attested_weight_inputs_v2,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: (),
    )
    snapshot = _snapshot()
    allocation = _allocation_graph()
    event_hash = allocation["root_receipt_hash"]
    expected = gateway_weight_input_value_documents_v2(
        calculation_snapshot=snapshot,
        gateway_authority_event_hash=event_hash,
    )
    calls = []

    async def execute(**kwargs):
        calls.append(kwargs)
        category = kwargs["payload"]["category"]
        receipt_hash = sha256_json({"category": category})
        receipt = {
            "receipt_hash": receipt_hash,
            "purpose": WEIGHT_INPUT_PURPOSES[category][1],
        }
        graph = {
            "root_receipt_hash": receipt_hash,
            "boot_identities": [],
            "receipts": [receipt],
            "transport_attempts": [],
            "host_operations": [],
        }
        return {
            "status": "succeeded",
            "result": expected[category],
            "receipt": receipt,
            "receipt_graph": graph,
        }

    result = await build_gateway_weight_inputs_v2(
        calculation_snapshot=snapshot,
        allocation_graph=allocation,
        leaderboard_window_start="2026-07-03T00:00:00Z",
        leaderboard_window_end="2026-07-10T00:00:00Z",
        execute=execute,
        load_sourcing_graphs=lambda **_kwargs: _async_value([]),
    )

    assert set(result["input_receipt_hashes"]) == set(
        GATEWAY_WEIGHT_INPUT_CATEGORIES
    )
    assert len(calls) == len(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    assert all(
        "finalized_chain_state_root" not in call["payload"] for call in calls
    )
    assert result["gateway_authority_event_hash"] == event_hash


@pytest.mark.asyncio
async def test_gateway_weight_input_builder_rejects_measured_value_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        attested_weight_inputs_v2,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: (),
    )

    async def execute(**kwargs):
        category = kwargs["payload"]["category"]
        receipt_hash = sha256_json({"category": category})
        receipt = {
            "receipt_hash": receipt_hash,
            "purpose": WEIGHT_INPUT_PURPOSES[category][1],
        }
        return {
            "status": "succeeded",
            "result": {"forged": True},
            "receipt": receipt,
            "receipt_graph": {
                "root_receipt_hash": receipt_hash,
                "boot_identities": [],
                "receipts": [receipt],
                "transport_attempts": [],
                "host_operations": [],
            },
        }

    with pytest.raises(AttestedWeightInputsV2Error, match="differs"):
        await build_gateway_weight_inputs_v2(
            calculation_snapshot=_snapshot(),
            allocation_graph=_allocation_graph(),
            leaderboard_window_start="2026-07-03T00:00:00Z",
            leaderboard_window_end="2026-07-10T00:00:00Z",
            execute=execute,
            load_sourcing_graphs=lambda **_kwargs: _async_value([]),
        )


async def _async_value(value):
    return value
