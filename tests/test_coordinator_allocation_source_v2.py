from __future__ import annotations

from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from gateway.tee.coordinator_allocation_source_v2 import (
    CoordinatorAllocationSourceV2,
    CoordinatorAllocationSourceV2Error,
)
from gateway.tee import coordinator_allocation_source_v2 as allocation_source
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_execution_receipt_body,
    create_signed_execution_receipt,
    merkle_root,
    sha256_json,
)
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    build_allocation_settlement_frontier_bootstrap_v2,
    frontier_bootstrap_artifact_hashes_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
    frontier_artifact_hashes_v2,
)
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
            "workflow_epoch_id": 100,
            "metagraph": {
                "netuid": 71,
                "block": 100 * 360 + 10,
                "owner_hotkey": "burn",
                "hotkeys": ["burn", "miner"],
            },
        }

    def resolve_live_prices(self, **_kwargs):
        raise AssertionError("dynamic pricing is disabled in this fixture")


class MissingWorkflowEpochChainSource(FakeChainSource):
    def read_finalized_metagraph(self, **kwargs):
        result = super().read_finalized_metagraph(**kwargs)
        result.pop("workflow_epoch_id")
        return result


def _policy():
    return {
        "policy_id": "policy:v2-test",
        "enabled": True,
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
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


def _no_burn_config():
    return SimpleNamespace(
        reimbursement_dynamic_alpha_price_enabled=False,
        reimbursement_require_live_alpha_price=False,
        reimbursement_miner_alpha_per_epoch=100.0,
        reimbursement_usd_per_0_1_percent_epoch=0.666667,
        reimbursement_policy_doc=lambda enabled: {
            **_policy(),
            "enabled": bool(enabled),
            "enable_conservative": False,
            "enable_champ_cap": False,
            "reimbursement_max_cost_multiplier_with_champions": 2.0,
        },
    )


def _context(parents=()):
    return ExecutionContextV2(
        job_id="allocation-v2:test",
        purpose="research_lab.allocation.v2",
        epoch_id=100,
        parent_receipt_hashes=tuple(parents),
    )


def _signed_coordinator_receipt(
    *,
    purpose,
    job_id,
    epoch_id,
    input_root,
    output_root,
    artifact_root,
    parents=(),
):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_execution_receipt_body(
        role="gateway_coordinator",
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch_id,
        sequence=0,
        commit_sha="a" * 40,
        pcr0="b" * 96,
        build_manifest_hash="sha256:" + "c" * 64,
        dependency_lock_hash="sha256:" + "d" * 64,
        config_hash="sha256:" + "e" * 64,
        boot_identity_hash="sha256:" + "f" * 64,
        input_root=input_root,
        output_root=output_root,
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=artifact_root,
        parent_receipt_hashes=parents,
        status="succeeded",
        failure_code=None,
        issued_at="2026-08-02T18:00:00Z",
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )


def _chain_realized_activation(*, first_epoch_id=100):
    return {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        ),
        "netuid": 71,
        "first_epoch_id": first_epoch_id,
        "source_bundle_epoch_id": first_epoch_id,
        "source_bundle_hash": "sha256:" + "a" * 64,
        "source_finalized_block": first_epoch_id * 360,
    }


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
    assert (
        "allocation_champion_rewards",
        {"epoch_id": 100, "include_paid": False},
    ) in reader.calls
    assert ("allocation_source_add_rewards", {"epoch_id": 100}) in reader.calls


def test_allocation_never_falls_back_to_finalized_block_modulo():
    resolver = CoordinatorAllocationSourceV2(
        reader=FakeReader(),
        chain_source=MissingWorkflowEpochChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="finalized chain state differs",
    ):
        resolver.resolve(
            payload={"epoch": 100, "netuid": 71},
            context=_context(),
        )


def test_measured_no_burn_allocation_uses_prior_compute_snapshot(monkeypatch):
    settlement_receipt = "sha256:" + "9" * 64
    allocation_payload = {
        "epoch": 99,
        "reimbursement_allocations": [
            {
                "uid": 77,
                "miner_hotkey": "miner",
                "source_id": "reimbursement_schedule:source",
                "spend_microusd": 1_000_000,
                "eligible_compute_microusd": 3_000_000,
                "reason": "full_reimbursement",
            }
        ],
    }
    allocation_hash = sha256_json(allocation_payload)
    source_row = {
        "epoch": 99,
        "netuid": 71,
        "allocation_hash": allocation_hash,
        "allocation_doc": {
            **allocation_payload,
            "allocation_hash": allocation_hash,
        },
    }
    reader = FakeReader(
        {
            "latest_legacy_compute_allocation_authority": [
                {
                    **source_row,
                    "epoch_id": source_row["epoch"],
                }
            ]
        }
    )
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_no_burn_config,
        network_supplier=lambda: "finney",
    )

    def load_authority(*, required_parents, **_kwargs):
        required_parents.add(settlement_receipt)
        return source_row

    monkeypatch.setattr(
        resolver,
        "_load_historical_compute_authority",
        load_authority,
    )

    result = resolver.resolve(
        payload={"epoch": 100, "netuid": 71},
        context=_context((settlement_receipt,)),
    )

    assert (
        "latest_native_compute_allocation_authority",
        {"epoch_id": 100, "netuid": 71},
    ) in reader.calls
    assert (
        "latest_legacy_compute_allocation_authority",
        {"epoch_id": 100, "netuid": 71},
    ) in reader.calls
    assert (
        "allocation_champion_rewards",
        {"epoch_id": 100, "include_paid": True},
    ) in reader.calls
    assert result["allocation"]["unallocated_percent"] == pytest.approx(0.0)
    assert result["allocation"]["reimbursement_allocations"][0][
        "uid"
    ] == 1
    assert result["allocation"]["reimbursement_allocations"][0][
        "paid_alpha_percent"
    ] == pytest.approx(20.0)
    assert result["source_state"][
        "historical_compute_fallback_source"
    ]["source_allocation_epoch"] == 99


def test_measured_no_burn_source_requires_finalized_authority(monkeypatch):
    settlement_receipt = "sha256:" + "8" * 64
    allocation_payload = {
        "epoch": 99,
        "reimbursement_allocations": [
            {
                "uid": 77,
                "miner_hotkey": "miner",
                "source_id": "reimbursement_schedule:source",
                "spend_microusd": 1_000_000,
                "eligible_compute_microusd": 3_000_000,
                "reason": "full_reimbursement",
            }
        ],
    }
    allocation_hash = sha256_json(allocation_payload)
    source_row = {
        "epoch": 99,
        "netuid": 71,
        "allocation_hash": allocation_hash,
        "allocation_doc": {
            **allocation_payload,
            "allocation_hash": allocation_hash,
        },
    }
    authority = {
        **source_row,
        "authority_types": ["legacy_finalized_chain_migration_v2"],
        "legacy_settlement_receipt_hash": settlement_receipt,
    }
    reader = FakeReader(
        {
            "latest_legacy_compute_allocation_authority": [
                {
                    **source_row,
                    "epoch_id": 99,
                }
            ],
            "legacy_finalized_allocation_migrations": [
                {"settlement_receipt_hash": settlement_receipt}
            ],
        }
    )
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_no_burn_config,
        network_supplier=lambda: "finney",
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_finalized_allocation_authorities_v2",
        lambda _rows, *, finalization_graphs: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_legacy_settlement_migrations_v2",
        lambda _rows, *, receipt_graphs: [authority],
    )
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, _parents: {
            settlement_receipt: {
                "root_receipt_hash": settlement_receipt,
                "receipts": [{"receipt_hash": settlement_receipt}],
            }
        },
    )
    context = _context((settlement_receipt,))

    result = resolver.resolve(
        payload={"epoch": 100, "netuid": 71},
        context=context,
    )

    assert result["source_state"][
        "historical_compute_fallback_source"
    ]["source_allocation_epoch"] == 99
    assert (
        "latest_legacy_compute_allocation_authority",
        {"epoch_id": 100, "netuid": 71},
    ) in reader.calls
    assert (
        "legacy_finalized_allocation_migrations",
        {"netuid": 71, "start_epoch": 99, "end_epoch": 99},
    ) in reader.calls


def test_unreceipted_source_add_reward_fails_closed():
    resolver = CoordinatorAllocationSourceV2(
        reader=FakeReader(
            {
                "chain_realized_settlement_activation": [
                    _chain_realized_activation()
                ],
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


def test_business_receipt_lookup_binds_the_expected_artifact_hash(monkeypatch):
    artifact_hash = "sha256:" + "3" * 64
    receipt_hash = "sha256:" + "4" * 64
    receipt = {
        "receipt_hash": receipt_hash,
        "role": "gateway_coordinator",
        "purpose": "research_lab.allocation.v2",
        "epoch_id": 99,
        "output_root": "sha256:" + "5" * 64,
        "boot_identity_hash": "sha256:" + "6" * 64,
    }
    reader = FakeReader(
        {
            "attested_business_artifact_by_ref": [
                {
                    "receipt_hash": receipt_hash,
                    "artifact_kind": "allocation",
                    "artifact_ref": "epoch:99",
                    "artifact_hash": artifact_hash,
                }
            ],
            "attested_receipt_by_hash": [
                {
                    **receipt,
                    "receipt_doc": receipt,
                }
            ],
        }
    )
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )
    context = _context((receipt_hash,))
    monkeypatch.setattr(
        allocation_source,
        "validate_signed_execution_receipt",
        lambda _value: None,
    )

    link, observed = resolver._business_receipt(
        artifact_kind="allocation",
        artifact_ref="epoch:99",
        artifact_hash=artifact_hash,
        context=context,
    )

    assert link["artifact_hash"] == artifact_hash
    assert observed == receipt
    assert reader.calls == [
        (
            "attested_business_artifact_by_ref",
            {
                "artifact_kind": "allocation",
                "artifact_ref": "epoch:99",
                "artifact_hash": artifact_hash,
            },
        ),
        ("attested_receipt_by_hash", {"receipt_hash": receipt_hash}),
    ]


def test_business_receipt_rejects_a_different_artifact_hash(monkeypatch):
    artifact_hash = "sha256:" + "3" * 64
    receipt_hash = "sha256:" + "4" * 64
    receipt = {
        "receipt_hash": receipt_hash,
        "role": "gateway_coordinator",
        "purpose": "research_lab.allocation.v2",
        "epoch_id": 99,
        "output_root": "sha256:" + "5" * 64,
        "boot_identity_hash": "sha256:" + "6" * 64,
    }
    reader = FakeReader(
        {
            "attested_business_artifact_by_ref": [
                {
                    "receipt_hash": receipt_hash,
                    "artifact_kind": "allocation",
                    "artifact_ref": "epoch:99",
                    "artifact_hash": "sha256:" + "7" * 64,
                }
            ],
            "attested_receipt_by_hash": [{**receipt, "receipt_doc": receipt}],
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
        "validate_signed_execution_receipt",
        lambda _value: None,
    )

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="not a declared source",
    ):
        resolver._business_receipt(
            artifact_kind="allocation",
            artifact_ref="epoch:99",
            artifact_hash=artifact_hash,
            context=_context((receipt_hash,)),
        )


def test_prior_settlement_frontier_reconstructs_exact_signed_authority(
    monkeypatch,
):
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=99,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    receipt_hash = "sha256:" + "4" * 64
    source_state = {"settlement_frontier": frontier}
    source_state_hash = sha256_json(source_state)
    allocation = {"allocation_hash": "sha256:" + "5" * 64}
    result = {
        "allocation": allocation,
        "source_state": source_state,
        "source_state_hash": source_state_hash,
    }
    artifact_hashes = sorted(
        set(frontier_artifact_hashes_v2(frontier)) | {source_state_hash}
    )
    artifact_root = merkle_root(
        artifact_hashes,
        domain="leadpoet-artifact-v2",
    )
    input_root = "sha256:" + "6" * 64
    output_root = sha256_json({"allocation": allocation})
    receipt = {
        "receipt_hash": receipt_hash,
        "role": "gateway_coordinator",
        "purpose": "research_lab.allocation.v2",
        "job_id": "allocation-v2:frontier-authority:99",
        "sequence": 0,
        "status": "succeeded",
        "epoch_id": 99,
        "input_root": input_root,
        "output_root": output_root,
        "artifact_root": artifact_root,
    }
    reader = FakeReader(
        {
            "allocation_settlement_frontier_activation": [
                {
                    "schema_version": (
                        "leadpoet.research_lab_allocation_"
                        "settlement_frontier_activation.v2"
                    ),
                    "netuid": 71,
                    "first_allocation_epoch": 99,
                    "first_frontier_hash": frontier["frontier_hash"],
                    "source_receipt_hash": receipt_hash,
                }
            ],
            "allocation_settlement_frontiers": [
                {
                    "schema_version": frontier["schema_version"],
                    "netuid": 71,
                    "allocation_epoch": 99,
                    "settled_through_epoch": 98,
                    "frontier_hash": frontier["frontier_hash"],
                    "predecessor_frontier_hash": None,
                    "source_receipt_hash": receipt_hash,
                    "source_state_hash": source_state_hash,
                    "frontier_doc": frontier,
                }
            ],
            "allocation_settlement_frontier_by_epoch": [
                {
                    "schema_version": frontier["schema_version"],
                    "netuid": 71,
                    "allocation_epoch": 99,
                    "settled_through_epoch": 98,
                    "frontier_hash": frontier["frontier_hash"],
                    "predecessor_frontier_hash": None,
                    "source_receipt_hash": receipt_hash,
                    "source_state_hash": source_state_hash,
                    "frontier_doc": frontier,
                }
            ],
            "attested_execution_result_by_receipt": [
                {
                    "schema_version": "leadpoet.attested_execution_result.v2",
                    "receipt_hash": receipt_hash,
                    "role": "gateway_coordinator",
                    "operation": "research_lab_allocation",
                    "purpose": "research_lab.allocation.v2",
                    "job_id": "allocation-v2:frontier-authority:99",
                    "sequence": 0,
                    "epoch_id": 99,
                    "release_hash": "sha256:" + "7" * 64,
                    "result_doc": result,
                    "result_hash": sha256_json(result),
                    "artifact_hashes": artifact_hashes,
                    "artifact_root": artifact_root,
                    "input_root": input_root,
                    "output_root": output_root,
                }
            ],
            "attested_receipt_by_hash": [
                {"receipt_hash": receipt_hash, "receipt_doc": receipt}
            ],
        }
    )
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )
    context = _context((receipt_hash,))
    context.external_receipt_graphs = [
        {
            "root_receipt_hash": receipt_hash,
            "receipts": [{"receipt_hash": receipt_hash}],
        }
    ]
    monkeypatch.setattr(
        allocation_source,
        "validate_signed_execution_receipt",
        lambda _value: None,
    )
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, declared_roots: {
            root: {"root_receipt_hash": root}
            for root in declared_roots
        },
    )
    required = set()

    observed = resolver._load_prior_settlement_frontier(
        epoch=100,
        netuid=71,
        context=context,
        required_parents=required,
    )

    assert observed == {"frontier": frontier, "receipt_hash": receipt_hash}
    assert required == {receipt_hash}
    assert reader.calls == [
        ("allocation_settlement_frontier_activation", {"netuid": 71}),
        (
            "allocation_settlement_frontiers",
            {"netuid": 71, "before_epoch": 100},
        ),
        (
            "allocation_settlement_frontier_by_epoch",
            {"netuid": 71, "allocation_epoch": 99},
        ),
        (
            "attested_execution_result_by_receipt",
            {"receipt_hash": receipt_hash},
        ),
        ("attested_receipt_by_hash", {"receipt_hash": receipt_hash}),
    ]


def test_first_frontier_same_epoch_replay_has_no_prior_frontier():
    activation = {
        "schema_version": (
            "leadpoet.research_lab_allocation_settlement_frontier_activation.v2"
        ),
        "netuid": 71,
        "first_allocation_epoch": 100,
        "first_frontier_hash": "sha256:" + "1" * 64,
        "source_receipt_hash": "sha256:" + "2" * 64,
    }
    reader = FakeReader(
        {
            "allocation_settlement_frontier_activation": [activation],
            "allocation_settlement_frontiers": [],
        }
    )
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )

    assert resolver._load_prior_settlement_frontier(
        epoch=100,
        netuid=71,
        context=_context(()),
        required_parents=set(),
    ) is None

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="unavailable or ambiguous",
    ):
        resolver._load_prior_settlement_frontier(
            epoch=101,
            netuid=71,
            context=_context(()),
            required_parents=set(),
        )


def test_successor_frontier_requires_signed_bootstrap_and_latest_authorities(
    monkeypatch,
):
    bootstrap = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=98,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    successor = build_allocation_settlement_frontier_v2(
        mode="bounded_delta_v1",
        netuid=71,
        allocation_epoch=99,
        predecessor_frontier_hash=bootstrap["frontier_hash"],
        reward_checkpoints=(),
    )

    def authority(frontier, receipt_digit, allocation_digit):
        receipt_hash = "sha256:" + receipt_digit * 64
        source_state = {"settlement_frontier": frontier}
        source_state_hash = sha256_json(source_state)
        allocation = {
            "allocation_hash": "sha256:" + allocation_digit * 64
        }
        result = {
            "allocation": allocation,
            "source_state": source_state,
            "source_state_hash": source_state_hash,
        }
        artifact_hashes = sorted(
            set(frontier_artifact_hashes_v2(frontier))
            | {source_state_hash}
        )
        artifact_root = merkle_root(
            artifact_hashes,
            domain="leadpoet-artifact-v2",
        )
        input_root = "sha256:" + allocation_digit * 64
        output_root = sha256_json({"allocation": allocation})
        receipt = {
            "receipt_hash": receipt_hash,
            "role": "gateway_coordinator",
            "purpose": "research_lab.allocation.v2",
            "job_id": (
                "allocation-v2:frontier-authority:"
                f"{frontier['allocation_epoch']}"
            ),
            "sequence": 0,
            "status": "succeeded",
            "epoch_id": frontier["allocation_epoch"],
            "input_root": input_root,
            "output_root": output_root,
            "artifact_root": artifact_root,
        }
        row = {
            "schema_version": frontier["schema_version"],
            "netuid": 71,
            "allocation_epoch": frontier["allocation_epoch"],
            "settled_through_epoch": frontier["settled_through_epoch"],
            "frontier_hash": frontier["frontier_hash"],
            "predecessor_frontier_hash": frontier[
                "predecessor_frontier_hash"
            ],
            "source_receipt_hash": receipt_hash,
            "source_state_hash": source_state_hash,
            "frontier_doc": frontier,
        }
        execution = {
            "schema_version": "leadpoet.attested_execution_result.v2",
            "receipt_hash": receipt_hash,
            "role": "gateway_coordinator",
            "operation": "research_lab_allocation",
            "purpose": "research_lab.allocation.v2",
            "job_id": receipt["job_id"],
            "sequence": 0,
            "epoch_id": frontier["allocation_epoch"],
            "release_hash": "sha256:" + "7" * 64,
            "result_doc": result,
            "result_hash": sha256_json(result),
            "artifact_hashes": artifact_hashes,
            "artifact_root": artifact_root,
            "input_root": input_root,
            "output_root": output_root,
        }
        return row, execution, receipt

    first_row, first_execution, first_receipt = authority(
        bootstrap,
        "1",
        "3",
    )
    latest_row, latest_execution, latest_receipt = authority(
        successor,
        "2",
        "4",
    )
    executions = {
        first_receipt["receipt_hash"]: first_execution,
        latest_receipt["receipt_hash"]: latest_execution,
    }
    receipts = {
        first_receipt["receipt_hash"]: first_receipt,
        latest_receipt["receipt_hash"]: latest_receipt,
    }

    class FrontierReader(FakeReader):
        def read(self, *, policy_id, parameters, **_kwargs):
            self.calls.append((policy_id, dict(parameters)))
            if policy_id == "allocation_settlement_frontier_activation":
                return [
                    {
                        "schema_version": (
                            "leadpoet.research_lab_allocation_"
                            "settlement_frontier_activation.v2"
                        ),
                        "netuid": 71,
                        "first_allocation_epoch": 98,
                        "first_frontier_hash": bootstrap["frontier_hash"],
                        "source_receipt_hash": first_receipt["receipt_hash"],
                    }
                ]
            if policy_id == "allocation_settlement_frontiers":
                return [dict(latest_row)]
            if policy_id == "allocation_settlement_frontier_by_epoch":
                return [dict(first_row)]
            if policy_id == "attested_execution_result_by_receipt":
                return [dict(executions[parameters["receipt_hash"]])]
            if policy_id == "attested_receipt_by_hash":
                receipt = receipts[parameters["receipt_hash"]]
                return [
                    {
                        "receipt_hash": receipt["receipt_hash"],
                        "receipt_doc": dict(receipt),
                    }
                ]
            return []

    reader = FrontierReader()
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )
    parent_hashes = (
        first_receipt["receipt_hash"],
        latest_receipt["receipt_hash"],
    )
    context = _context(parent_hashes)
    context.external_receipt_graphs = [
        {
            "root_receipt_hash": receipt_hash,
            "receipts": [{"receipt_hash": receipt_hash}],
        }
        for receipt_hash in parent_hashes
    ]
    monkeypatch.setattr(
        allocation_source,
        "validate_signed_execution_receipt",
        lambda _value: None,
    )
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, declared_roots: {
            root: {"root_receipt_hash": root} for root in declared_roots
        },
    )
    required = set()

    observed = resolver._load_prior_settlement_frontier(
        epoch=100,
        netuid=71,
        context=context,
        required_parents=required,
    )

    assert observed == {
        "frontier": successor,
        "receipt_hash": latest_receipt["receipt_hash"],
    }
    assert required == set(parent_hashes)

    context.parent_receipt_hashes = (latest_receipt["receipt_hash"],)
    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="not a declared source",
    ):
        resolver._load_prior_settlement_frontier(
            epoch=100,
            netuid=71,
            context=context,
            required_parents=set(),
        )


def test_bootstrap_frontier_accepts_canonical_receipts_without_release_hash(
    monkeypatch,
):
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=99,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    allocation_state = {"netuid": 71, "epoch": 99, "rewards": []}
    source_state_hash = sha256_json(allocation_state)
    allocation = {"allocation_hash": "sha256:" + "1" * 64}
    allocation_result = {
        "allocation": allocation,
        "source_state": allocation_state,
        "source_state_hash": source_state_hash,
    }
    allocation_artifacts = [source_state_hash]
    allocation_artifact_root = merkle_root(
        allocation_artifacts,
        domain="leadpoet-artifact-v2",
    )
    allocation_output_root = sha256_json({"allocation": allocation})
    allocation_receipt = _signed_coordinator_receipt(
        purpose="research_lab.allocation.v2",
        job_id="allocation-v2:bootstrap-source:99",
        epoch_id=99,
        input_root="sha256:" + "2" * 64,
        output_root=allocation_output_root,
        artifact_root=allocation_artifact_root,
    )
    bootstrap = build_allocation_settlement_frontier_bootstrap_v2(
        netuid=71,
        bootstrap_epoch=100,
        allocation_source_receipt_hash=allocation_receipt["receipt_hash"],
        source_state_hash=source_state_hash,
        frontier=frontier,
    )
    bootstrap_artifacts = sorted(frontier_bootstrap_artifact_hashes_v2(bootstrap))
    bootstrap_artifact_root = merkle_root(
        bootstrap_artifacts,
        domain="leadpoet-artifact-v2",
    )
    bootstrap_receipt = _signed_coordinator_receipt(
        purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
        job_id="allocation-frontier-bootstrap-v2:100",
        epoch_id=100,
        input_root="sha256:" + "3" * 64,
        output_root=sha256_json(bootstrap),
        artifact_root=bootstrap_artifact_root,
        parents=(allocation_receipt["receipt_hash"],),
    )
    release_hash = "sha256:" + "4" * 64
    executions = {
        allocation_receipt["receipt_hash"]: {
            "schema_version": "leadpoet.attested_execution_result.v2",
            "receipt_hash": allocation_receipt["receipt_hash"],
            "role": "gateway_coordinator",
            "operation": "research_lab_allocation",
            "purpose": "research_lab.allocation.v2",
            "job_id": allocation_receipt["job_id"],
            "epoch_id": 99,
            "sequence": 0,
            "release_hash": release_hash,
            "input_root": allocation_receipt["input_root"],
            "output_root": allocation_output_root,
            "artifact_root": allocation_artifact_root,
            "result_hash": sha256_json(allocation_result),
            "artifact_hashes": allocation_artifacts,
            "result_doc": allocation_result,
        },
        bootstrap_receipt["receipt_hash"]: {
            "schema_version": "leadpoet.attested_execution_result.v2",
            "receipt_hash": bootstrap_receipt["receipt_hash"],
            "role": "gateway_coordinator",
            "operation": ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
            "purpose": ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
            "job_id": bootstrap_receipt["job_id"],
            "epoch_id": 100,
            "sequence": 0,
            "release_hash": release_hash,
            "input_root": bootstrap_receipt["input_root"],
            "output_root": bootstrap_receipt["output_root"],
            "artifact_root": bootstrap_artifact_root,
            "result_hash": sha256_json(bootstrap),
            "artifact_hashes": bootstrap_artifacts,
            "result_doc": bootstrap,
        },
    }
    receipts = {
        allocation_receipt["receipt_hash"]: allocation_receipt,
        bootstrap_receipt["receipt_hash"]: bootstrap_receipt,
    }

    class BootstrapReader(FakeReader):
        def read(self, *, policy_id, parameters, **_kwargs):
            self.calls.append((policy_id, dict(parameters)))
            if policy_id == "allocation_settlement_frontier_activation":
                return [
                    {
                        "schema_version": (
                            "leadpoet.research_lab_allocation_"
                            "settlement_frontier_activation.v2"
                        ),
                        "netuid": 71,
                        "first_allocation_epoch": 99,
                        "first_frontier_hash": frontier["frontier_hash"],
                        "source_receipt_hash": bootstrap_receipt["receipt_hash"],
                    }
                ]
            if policy_id in {
                "allocation_settlement_frontiers",
                "allocation_settlement_frontier_by_epoch",
            }:
                return [
                    {
                        "schema_version": frontier["schema_version"],
                        "netuid": 71,
                        "allocation_epoch": 99,
                        "settled_through_epoch": 98,
                        "frontier_hash": frontier["frontier_hash"],
                        "predecessor_frontier_hash": None,
                        "source_receipt_hash": bootstrap_receipt["receipt_hash"],
                        "source_state_hash": source_state_hash,
                        "frontier_doc": frontier,
                    }
                ]
            receipt_hash = parameters.get("receipt_hash")
            if policy_id == "attested_execution_result_by_receipt":
                return [dict(executions[receipt_hash])]
            if policy_id == "attested_receipt_by_hash":
                return [
                    {
                        "receipt_hash": receipt_hash,
                        "receipt_doc": dict(receipts[receipt_hash]),
                    }
                ]
            return []

    reader = BootstrapReader()
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )
    context = _context((bootstrap_receipt["receipt_hash"],))
    context.external_receipt_graphs = [
        {
            "root_receipt_hash": bootstrap_receipt["receipt_hash"],
            "receipts": [bootstrap_receipt, allocation_receipt],
        }
    ]
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, declared_roots: {
            root: {"root_receipt_hash": root} for root in declared_roots
        },
    )
    assert "release_hash" not in allocation_receipt
    assert "release_hash" not in bootstrap_receipt

    required = set()
    observed = resolver._load_prior_settlement_frontier(
        epoch=100,
        netuid=71,
        context=context,
        required_parents=required,
    )

    assert observed == {
        "frontier": frontier,
        "receipt_hash": bootstrap_receipt["receipt_hash"],
    }
    assert required == {bootstrap_receipt["receipt_hash"]}

    executions[bootstrap_receipt["receipt_hash"]]["release_hash"] = "invalid"
    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="execution authority differs",
    ):
        resolver._load_prior_settlement_frontier(
            epoch=100,
            netuid=71,
            context=context,
            required_parents=set(),
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


def test_declared_roots_are_reconstructed_from_one_compact_graph(monkeypatch):
    parent_hash = "sha256:" + "a" * 64
    child_hash = "sha256:" + "b" * 64
    boot_hash = "sha256:" + "c" * 64
    parent = {
        "receipt_hash": parent_hash,
        "parent_receipt_hashes": [],
        "boot_identity_hash": boot_hash,
        "job_id": "parent-job",
        "purpose": "research_lab.allocation.v2",
    }
    child = {
        "receipt_hash": child_hash,
        "parent_receipt_hashes": [parent_hash],
        "boot_identity_hash": boot_hash,
        "job_id": "child-job",
        "purpose": "validator.weight_finalization.v2",
    }
    compact_graph = {
        "schema_version": "leadpoet.receipt_graph.v2",
        "root_receipt_hash": child_hash,
        "boot_identities": [{"boot_identity_hash": boot_hash}],
        "receipts": [parent, child],
        "transport_attempts": [],
        "host_operations": [],
    }
    monkeypatch.setattr(
        allocation_source,
        "validate_receipt_graphs",
        lambda _values: None,
    )

    graphs = allocation_source._receipt_graphs_by_declared_root(
        [compact_graph],
        [parent_hash, child_hash],
    )

    assert set(graphs) == {parent_hash, child_hash}
    assert graphs[parent_hash]["root_receipt_hash"] == parent_hash
    assert graphs[parent_hash]["receipts"] == [parent]
    assert graphs[child_hash] == compact_graph


def test_checkpointed_declared_root_preserves_certificate_bound_graph(monkeypatch):
    root = "sha256:" + "a" * 64
    external_parent = "sha256:" + "b" * 64
    graph = {
        "schema_version": CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
        "root_receipt_hash": root,
        "boot_identities": [],
        "receipts": [
            {
                "receipt_hash": root,
                "parent_receipt_hashes": [external_parent],
            }
        ],
        "transport_attempts": [],
        "host_operations": [],
        "ancestry_lineage_id": "sha256:" + "c" * 64,
        "ancestry_proof": {"certificate": "preserved"},
    }
    monkeypatch.setattr(
        allocation_source,
        "validate_receipt_graphs",
        lambda _values: None,
    )

    derived = allocation_source._receipt_graphs_by_declared_root(
        [graph],
        [root],
    )

    assert derived == {root: graph}
    assert derived[root]["ancestry_proof"] == graph["ancestry_proof"]

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="differs from certified root",
    ):
        allocation_source._receipt_subgraph_from_validated(
            graph,
            root_receipt_hash=external_parent,
        )


def test_declared_root_lookup_indexes_each_graph_once(monkeypatch):
    class CountingGraph(dict):
        receipt_lookups = 0

        def get(self, key, default=None):
            if key == "receipts":
                self.receipt_lookups += 1
            return super().get(key, default)

    roots = ["sha256:" + f"{index:064x}" for index in range(128)]
    compact_graphs = [
        CountingGraph({"receipts": [{"receipt_hash": root}]})
        for root in roots
    ]
    monkeypatch.setattr(
        allocation_source,
        "_receipt_subgraph_from_validated",
        lambda _graph, *, root_receipt_hash: {
            "root_receipt_hash": root_receipt_hash
        },
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_receipt_graphs",
        lambda _values: None,
    )

    graphs = allocation_source._receipt_graphs_by_declared_root(
        compact_graphs,
        roots,
    )

    assert set(graphs) == set(roots)
    assert sum(graph.receipt_lookups for graph in compact_graphs) == len(
        compact_graphs
    )


def test_declared_root_lookup_still_rejects_conflicting_graphs(monkeypatch):
    root = "sha256:" + "a" * 64
    monkeypatch.setattr(
        allocation_source,
        "_receipt_subgraph_from_validated",
        lambda graph, *, root_receipt_hash: {
            "root_receipt_hash": root_receipt_hash,
            "marker": graph["marker"],
        },
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_receipt_graphs",
        lambda _values: None,
    )

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="declared allocation parent graphs conflict",
    ):
        allocation_source._receipt_graphs_by_declared_root(
            [
                {"marker": "first", "receipts": [{"receipt_hash": root}]},
                {"marker": "second", "receipts": [{"receipt_hash": root}]},
            ],
            [root],
        )


def test_finalized_champion_history_requires_declared_chain_roots(monkeypatch):
    finalization_root = "sha256:" + "1" * 64
    allocation_receipt = "sha256:" + "2" * 64
    allocation_input_receipt = "sha256:" + "3" * 64
    allocation = {
        "allocation_hash": "sha256:" + "4" * 64,
        "champion_allocations": [],
        "queued_champion_allocations": [],
    }
    reader = FakeReader(
        {
            "finalized_allocation_authorities": [
                {"finalization_receipt_hash": finalization_root}
            ],
            "chain_realized_settlement_activation": [
                {
                    "netuid": 71,
                    "schema_version": (
                        "leadpoet.research_lab_chain_realized_"
                        "settlement_activation.v1"
                    ),
                    "first_epoch_id": 100,
                    "source_bundle_hash": "sha256:" + "8" * 64,
                    "source_bundle_epoch_id": 100,
                    "source_finalized_block": 1000,
                }
            ],
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
                "allocation_receipt_hash": allocation_input_receipt,
                "allocation_authority_receipt_hash": allocation_receipt,
            }
        ],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_legacy_settlement_migrations_v2",
        lambda rows, *, receipt_graphs: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda graphs, declared_roots: {
            root: {
                "root_receipt_hash": root,
                "receipts": [{"receipt_hash": root}],
            }
            for root in declared_roots
        }
        if graphs
        else {},
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
            "chain_realized_settlement_activation",
            {"netuid": 71},
        ),
        (
            "finalized_allocation_authorities",
            {"netuid": 71, "start_epoch": 99, "end_epoch": 99},
        ),
        (
            "legacy_finalized_allocation_migrations",
            {"netuid": 71, "start_epoch": 99, "end_epoch": 99},
        ),
    ]

    monkeypatch.setattr(
        resolver,
        "_require_allocation_receipt",
        lambda **_kwargs: allocation_input_receipt,
    )
    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="used another allocation receipt",
    ):
        resolver._finalized_champion_history(
            epoch=100,
            netuid=71,
            champion_rows=[{"start_epoch": 99}],
            context=context,
            required_parents=set(),
        )

    monkeypatch.setattr(
        resolver,
        "_require_allocation_receipt",
        lambda **_kwargs: allocation_receipt,
    )
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


def test_finalized_history_ignores_superseded_native_authority_graphs(
    monkeypatch,
):
    chain_root = "sha256:" + "9" * 64

    class CutoverReader(FakeReader):
        def read(self, *, policy_id, parameters, **_kwargs):
            self.calls.append((policy_id, dict(parameters)))
            if policy_id == "chain_realized_settlement_activation":
                return [_chain_realized_activation(first_epoch_id=99)]
            if policy_id == "chain_realized_epoch_settlements":
                return [
                    {
                        "epoch_id": 99,
                        "settlement_receipt_hash": chain_root,
                    }
                ]
            if (
                policy_id == "finalized_allocation_authorities"
                and int(parameters["end_epoch"]) >= 99
            ):
                return [
                    {
                        "epoch_id": 99,
                        "finalization_receipt_hash": "sha256:" + "8" * 64,
                    }
                ]
            return []

    reader = CutoverReader()
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=FakeChainSource(),
        config_supplier=_config,
        network_supplier=lambda: "finney",
    )

    def validate_native(rows, *, finalization_graphs):
        if rows:
            raise allocation_source.ChampionSettlementV2Error(
                "finalized allocation receipt graph is missing"
            )
        return []

    monkeypatch.setattr(
        allocation_source,
        "validate_finalized_allocation_authorities_v2",
        validate_native,
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_legacy_settlement_migrations_v2",
        lambda rows, *, receipt_graphs: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_chain_realized_epoch_settlements_v1",
        lambda rows, *, receipt_graphs, _receipt_graphs_prevalidated=False: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_chain_realized_obligation_credits_v1",
        lambda rows, *, settlement_rows, receipt_graphs, _receipt_graphs_prevalidated=False: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, declared_roots: {
            root: {
                "root_receipt_hash": root,
                "receipts": [{"receipt_hash": root}],
            }
            for root in declared_roots
        },
    )
    context = _context((chain_root,))
    context.external_receipt_graphs = [
        {
            "root_receipt_hash": chain_root,
            "receipts": [{"receipt_hash": chain_root}],
        }
    ]
    required = set()

    history = resolver._finalized_champion_history(
        epoch=100,
        netuid=71,
        champion_rows=[{"start_epoch": 98}],
        context=context,
        required_parents=required,
    )

    assert history == []
    assert required == {chain_root}
    assert reader.calls == [
        (
            "chain_realized_settlement_activation",
            {"netuid": 71},
        ),
        (
            "finalized_allocation_authorities",
            {"netuid": 71, "start_epoch": 98, "end_epoch": 98},
        ),
        (
            "legacy_finalized_allocation_migrations",
            {"netuid": 71, "start_epoch": 98, "end_epoch": 98},
        ),
        (
            "chain_realized_epoch_settlements",
            {"netuid": 71, "start_epoch": 99, "end_epoch": 99},
        ),
        (
            "chain_realized_obligation_credits",
            {"netuid": 71, "start_epoch": 99, "end_epoch": 99},
        ),
    ]


def test_finalized_history_requires_every_raw_authority_graph(monkeypatch):
    native_root = "sha256:" + "1" * 64
    legacy_root = "sha256:" + "2" * 64
    settlement_root = "sha256:" + "3" * 64
    credit_root = "sha256:" + "4" * 64
    reader = FakeReader(
        {
            "finalized_allocation_authorities": [
                {"finalization_receipt_hash": native_root}
            ],
            "legacy_finalized_allocation_migrations": [
                {"settlement_receipt_hash": legacy_root}
            ],
            "chain_realized_epoch_settlements": [
                {
                    "epoch_id": 99,
                    "settlement_receipt_hash": settlement_root,
                }
            ],
            "chain_realized_obligation_credits": [
                {"credit_receipt_hash": credit_root}
            ],
            "chain_realized_settlement_activation": [
                _chain_realized_activation(first_epoch_id=99)
            ],
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
        lambda _rows, *, finalization_graphs: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_legacy_settlement_migrations_v2",
        lambda _rows, *, receipt_graphs: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_chain_realized_epoch_settlements_v1",
        lambda _rows, *, receipt_graphs, _receipt_graphs_prevalidated=False: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "validate_chain_realized_obligation_credits_v1",
        lambda _rows, *, settlement_rows, receipt_graphs, _receipt_graphs_prevalidated=False: [],
    )
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, declared_roots: {
            root: {
                "root_receipt_hash": root,
                "receipts": [{"receipt_hash": root}],
            }
            for root in declared_roots
        },
    )

    roots = (native_root, legacy_root, settlement_root, credit_root)
    context = _context(roots)
    context.external_receipt_graphs = [
        {
            "root_receipt_hash": root,
            "receipts": [{"receipt_hash": root}],
        }
        for root in roots
    ]
    required = set()

    assert (
        resolver._finalized_champion_history(
            epoch=100,
            netuid=71,
            champion_rows=[{"start_epoch": 98}],
            context=context,
            required_parents=required,
        )
        == []
    )
    assert required == set(roots)

    context.parent_receipt_hashes = (native_root, legacy_root, credit_root)
    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="settlement authority graph is not a declared source",
    ):
        resolver._finalized_champion_history(
            epoch=100,
            netuid=71,
            champion_rows=[{"start_epoch": 98}],
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
            "chain_realized_settlement_activation": [
                _chain_realized_activation()
            ],
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
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda graphs, declared_roots: {
            root: {
                "root_receipt_hash": root,
                "receipts": [{"receipt_hash": root}],
            }
            for root in declared_roots
        }
        if graphs
        else {},
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
