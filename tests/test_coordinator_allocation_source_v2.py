from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.tee.coordinator_allocation_source_v2 import (
    CoordinatorAllocationSourceV2,
    CoordinatorAllocationSourceV2Error,
)
from gateway.tee import coordinator_allocation_source_v2 as allocation_source
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_verifier.economics import allocate_research_lab_epoch


class FakeReader:
    def __init__(self, rows=None):
        self.rows = dict(rows or {})
        self.calls = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        return [dict(item) for item in self.rows.get(policy_id, [])]


class FakeChainSource:
    def read_finalized_metagraph(self, *, netuid, context, attempt_number=0):
        assert netuid == 71
        assert context.purpose == "research_lab.allocation.v2"
        assert attempt_number == 0
        return {
            "finalized_block_hash": "a" * 64,
            "header": {"block": 100 * 360 + 10},
            "metagraph": {
                "netuid": 71,
                "block": 100 * 360 + 10,
                "owner_hotkey": "burn",
                "hotkeys": ["burn", "miner"],
            },
        }

    def resolve_live_prices(self, **_kwargs):
        raise AssertionError("dynamic pricing is disabled in this fixture")


def _policy():
    return {
        "policy_id": "policy:v2-test",
        "enabled": True,
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
        "reimbursement_allow_overpay_without_champions": True,
        "reimbursement_max_cost_multiplier_with_champions": 1.0,
        "champion_placeholder_alpha_percent": 0.0001,
        "champion_queue_trigger_ratio": 0.5,
        "usd_per_0_1_percent_epoch": 0.666667,
    }


def _config():
    return SimpleNamespace(
        reimbursement_dynamic_alpha_price_enabled=False,
        reimbursement_require_live_alpha_price=False,
        reimbursement_miner_alpha_per_epoch=100.0,
        reimbursement_usd_per_0_1_percent_epoch=0.666667,
        reimbursement_policy_doc=lambda enabled: {**_policy(), "enabled": bool(enabled)},
    )


def _context(parents=()):
    return ExecutionContextV2(
        job_id="allocation-v2:test",
        purpose="research_lab.allocation.v2",
        epoch_id=100,
        parent_receipt_hashes=tuple(parents),
    )


def test_allocation_is_built_from_measured_empty_sources():
    reader = FakeReader()
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )

    result = resolver.resolve(
        payload={"epoch": 100, "netuid": 71},
        context=_context(),
    )

    inputs = result["allocation_inputs"]
    expected = allocate_research_lab_epoch(
        100,
        inputs["policy"],
        [],
        [],
        active_source_add_obligations=[],
    )
    assert result["allocation"] == expected
    assert result["source_state"]["reimbursement_obligations"] == []
    assert result["source_state"]["champion_obligations"] == []
    assert result["source_state_hash"] == sha256_json(result["source_state"])
    assert ("allocation_champion_rewards", {"epoch_id": 100}) in reader.calls
    assert ("allocation_source_add_rewards", {"epoch_id": 100}) in reader.calls


def test_unreceipted_source_add_reward_fails_closed():
    resolver = CoordinatorAllocationSourceV2(
        reader=FakeReader(
            {
                "allocation_source_add_rewards": [
                    {
                        "reward_ref": "source_add_reward:" + "1" * 16,
                        "adapter_id": "adapter:test",
                        "miner_hotkey": "miner",
                        "leg": 1,
                        "reward_kind": "source_acceptance",
                        "alpha_percent": 1.0,
                        "reward_epochs": 20,
                        "start_epoch": 100,
                        "current_reward_status": "active",
                        "desired_alpha_percent": 1.0,
                        "epoch_count": 20,
                    }
                ]
            }
        ),
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="receipt link is missing or ambiguous",
    ):
        resolver.resolve(
            payload={"epoch": 100, "netuid": 71},
            context=_context(),
        )


def test_extra_parent_receipt_is_rejected_even_with_no_rewards():
    resolver = CoordinatorAllocationSourceV2(
        reader=FakeReader(),
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )
    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="parent receipt set differs",
    ):
        resolver.resolve(
            payload={"epoch": 100, "netuid": 71},
            context=_context(("sha256:" + "f" * 64,)),
        )


def test_finalized_champion_history_requires_declared_chain_roots(monkeypatch):
    finalization_root = "sha256:" + "1" * 64
    allocation_receipt = "sha256:" + "2" * 64
    allocation = {
        "allocation_hash": "sha256:" + "3" * 64,
        "champion_allocations": [],
        "queued_champion_allocations": [],
    }
    reader = FakeReader(
        {
            "finalized_allocation_authorities": [
                {"finalization_receipt_hash": finalization_root}
            ]
        }
    )
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_finalized_allocation_authorities_v2",
        lambda rows, *, finalization_graphs: [
                {
                    "epoch": 99,
                    "netuid": 71,
                    "allocation_hash": allocation["allocation_hash"],
                "allocation_doc": allocation,
                "allocation_receipt_hash": allocation_receipt,
            }
        ],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_legacy_settlement_migrations_v2",
        lambda rows, *, receipt_graphs: [],
    )
    monkeypatch.setattr(
        resolver,
        "_require_allocation_receipt",
        lambda **_kwargs: allocation_receipt,
    )
    context = _context((allocation_receipt, finalization_root))
    context.external_receipt_graphs = [
        {
            "root_receipt_hash": finalization_root,
            "receipts": [{"receipt_hash": finalization_root}],
        }
    ]
    required = set()

    history = resolver._finalized_champion_history(
        epoch=100,
        netuid=71,
        champion_rows=[{"start_epoch": 99}],
        context=context,
        required_parents=required,
    )

    assert history[0]["allocation_doc"] == allocation
    assert required == {finalization_root}
    assert reader.calls == [
        (
            "finalized_allocation_authorities",
            {"netuid": 71, "start_epoch": 99, "end_epoch": 99},
        ),
        (
            "legacy_finalized_allocation_migrations",
            {"netuid": 71, "start_epoch": 99, "end_epoch": 99},
        ),
    ]

    context.external_receipt_graphs = []
    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="finalized allocation graph is not a declared source",
    ):
        resolver._finalized_champion_history(
            epoch=100,
            netuid=71,
            champion_rows=[{"start_epoch": 99}],
            context=context,
            required_parents=set(),
        )


def test_legacy_finalized_champion_history_requires_migration_receipt(
    monkeypatch,
):
    settlement_receipt = "sha256:" + "4" * 64
    allocation = {
        "allocation_hash": "sha256:" + "5" * 64,
        "champion_allocations": [],
        "queued_champion_allocations": [],
    }
    reader = FakeReader(
        {
            "legacy_finalized_allocation_migrations": [
                {"settlement_receipt_hash": settlement_receipt}
            ]
        }
    )
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_finalized_allocation_authorities_v2",
        lambda rows, *, finalization_graphs: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_legacy_settlement_migrations_v2",
        lambda rows, *, receipt_graphs: [
            {
                "epoch": 99,
                "netuid": 71,
                "allocation_hash": allocation["allocation_hash"],
                "allocation_doc": allocation,
                "allocation_receipt_hash": settlement_receipt,
                "legacy_settlement_receipt_hash": settlement_receipt,
                "authority_types": ["legacy_finalized_chain_migration_v2"],
                "finalized_authority_count": 1,
                "finalized_bundle_hashes": [],
                "finalization_receipt_hashes": [],
            }
        ],
    )
    context = _context((settlement_receipt,))
    context.external_receipt_graphs = [
        {
            "root_receipt_hash": settlement_receipt,
            "receipts": [{"receipt_hash": settlement_receipt}],
        }
    ]
    required = set()

    history = resolver._finalized_champion_history(
        epoch=100,
        netuid=71,
        champion_rows=[{"start_epoch": 99}],
        context=context,
        required_parents=required,
    )

    assert history[0]["allocation_doc"] == allocation
    assert required == {settlement_receipt}

    context.parent_receipt_hashes = ()
    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="legacy finalized allocation receipt is not a declared source",
    ):
        resolver._finalized_champion_history(
            epoch=100,
            netuid=71,
            champion_rows=[{"start_epoch": 99}],
            context=context,
            required_parents=set(),
        )
