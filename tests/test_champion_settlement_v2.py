from __future__ import annotations

import asyncio
import copy
from decimal import Decimal
import threading

import pytest

from gateway.research_lab import champion_settlement_v2 as settlement
from gateway.research_lab import attested_v2_store, store
from leadpoet_canonical.attested_v2 import sha256_json


def _allocation(*, paid: float = 5.0) -> dict:
    payload = {
        "schema_version": "leadpoet.research_lab_allocation.v2",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": "champion_reward:test",
                "paid_alpha_percent": paid,
                "base_desired_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
    }
    return {**payload, "allocation_hash": sha256_json(payload)}


def _minimal_receipt_graph(
    receipt_hash: str,
    *,
    purpose: str,
    output_root: str,
) -> dict:
    return {
        "root_receipt_hash": receipt_hash,
        "receipts": [
            {
                "receipt_hash": receipt_hash,
                "role": "gateway_coordinator",
                "purpose": purpose,
                "status": "succeeded",
                "output_root": output_root,
            }
        ],
    }


def _chain_realized_fixture(
    *,
    epoch: int = 100,
    reward_id: str = "champion_reward:test",
    observed: str = "30.000000",
    attributed: str = "5.000000",
    scheduled: str = "5.000000",
    credited: str = "5.000000",
    kind: str = "champion",
) -> tuple[dict, dict, dict, dict[str, dict]]:
    credit_doc = {
        "schema_version": settlement.CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1,
        "netuid": 71,
        "epoch_id": epoch,
        "obligation_kind": kind,
        "obligation_source_id": reward_id,
        "miner_hotkey": "miner-hotkey",
        "miner_uid": 7,
        "observed_chain_alpha_percent": observed,
        "lab_attributed_alpha_percent": attributed,
        "scheduled_alpha_percent": scheduled,
        "credited_alpha_percent": credited,
        "attribution_doc": {
            "schema_version": "leadpoet.chain_realized_lab_attribution.v1",
            "source_bundle_hash": "sha256:" + "1" * 64,
            "source_bundle_epoch_id": epoch,
            "source_allocation_hash": "sha256:" + "2" * 64,
            "source_allocation_receipt_hash": "sha256:" + "3" * 64,
            "allocation_section": {
                "champion": "champion_allocations",
                "queued_champion": "queued_champion_allocations",
                "source_add": "source_add_allocations",
                "reimbursement": "reimbursement_allocations",
            }[kind],
        },
        "observation_doc": {
            "schema_version": (
                "leadpoet.chain_realized_weight_observation_ref.v1"
            ),
            "observation_hash": "sha256:" + "4" * 64,
            "close_block": 1000,
            "close_block_hash": "a" * 64,
            "weights_vector_hash": "sha256:" + "5" * 64,
        },
    }
    credit_hash = sha256_json(credit_doc)
    settlement_doc = {
        "schema_version": settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1,
        "netuid": 71,
        "epoch_id": epoch,
        "credit_hashes": [credit_hash],
        "observation_summary": {
            "schema_version": "leadpoet.chain_realized_observation_summary.v1",
            "observation_hash": "sha256:" + "4" * 64,
            "weights_vector_hash": "sha256:" + "5" * 64,
            "close_block": 1000,
            "close_block_hash": "a" * 64,
            "official_subnet_epoch_id": epoch,
            "validator_hotkey": "validator-hotkey",
            "validator_uid": 0,
            "source_bundle_hash": "sha256:" + "1" * 64,
            "source_bundle_epoch_id": epoch,
            "source_bundle_finalized_block": 990,
            "source_bundle_finalized_block_hash": "b" * 64,
            "last_update_block": 990,
            "last_update_block_hash": "b" * 64,
            "active_source_epoch_id": epoch,
            "complete": True,
        },
    }
    settlement_hash = sha256_json(settlement_doc)
    settlement_receipt = "sha256:" + "7" * 64
    settlement_row = {
        "netuid": 71,
        "epoch_id": epoch,
        "schema_version": settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1,
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": settlement_receipt,
        "settlement_doc": settlement_doc,
    }
    credit_row = {
        "netuid": 71,
        "epoch_id": epoch,
        "settlement_hash": settlement_hash,
        "schema_version": settlement.CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1,
        "obligation_kind": kind,
        "obligation_source_id": reward_id,
        "miner_hotkey": "miner-hotkey",
        "miner_uid": 7,
        "observed_chain_alpha_percent": Decimal(observed),
        "lab_attributed_alpha_percent": Decimal(attributed),
        "scheduled_alpha_percent": Decimal(scheduled),
        "credited_alpha_percent": Decimal(credited),
        "credit_hash": credit_hash,
        "credit_receipt_hash": settlement_receipt,
        "credit_doc": credit_doc,
    }
    graphs = {
        settlement_receipt: _minimal_receipt_graph(
            settlement_receipt,
            purpose=settlement.CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
            output_root=settlement_hash,
        ),
    }
    return settlement_row, credit_row, settlement_doc, graphs


@pytest.mark.asyncio
async def test_finalized_history_validation_does_not_block_event_loop(
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()

    async def select_all(*_args, **_kwargs):
        return []

    async def load_graphs(_roots):
        return {}

    def validate_native(_rows, *, finalization_graphs):
        assert finalization_graphs == {}
        started.set()
        assert release.wait(timeout=2)
        return []

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graphs_v2",
        load_graphs,
    )
    monkeypatch.setattr(
        settlement,
        "validate_finalized_allocation_authorities_v2",
        validate_native,
    )
    monkeypatch.setattr(
        settlement,
        "validate_legacy_settlement_migrations_v2",
        lambda _rows, *, receipt_graphs: [],
    )

    task = asyncio.create_task(
        settlement.load_finalized_allocation_history_v2(
            netuid=71,
            start_epoch=1,
            end_epoch=2,
        )
    )
    assert await asyncio.to_thread(started.wait, 1)
    try:
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.2)
        assert task.done() is False
    finally:
        release.set()

    assert await asyncio.wait_for(task, timeout=1) == []


def test_legacy_classification_detects_source_add_payments():
    payload = {
        "schema_version": "leadpoet.research_lab_allocation.v2",
        "epoch": 100,
        "champion_allocations": [],
        "queued_champion_allocations": [],
        "source_add_allocations": [
            {
                "source_add_reward_id": "source_add_reward:test",
                "paid_alpha_percent": 1.0,
            }
        ],
    }
    allocation = {**payload, "allocation_hash": sha256_json(payload)}

    epoch, allocation_hash, pays_active = (
        settlement._legacy_allocation_active_champion_payment_v2(
            {
                "epoch": 100,
                "netuid": 71,
                "allocation_hash": allocation["allocation_hash"],
                "allocation_doc": allocation,
            },
            netuid=71,
            active_reward_ids=set(),
            active_source_reward_ids={"source_add_reward:test"},
        )
    )

    assert epoch == 100
    assert allocation_hash == allocation["allocation_hash"]
    assert pays_active is True


def _authority_row(
    marker: str,
    *,
    allocation: dict,
) -> tuple[dict, dict, dict]:
    bundle_hash = "sha256:" + marker * 64
    root_hash = "sha256:" + chr(ord(marker) + 1) * 64
    finalization_root = "sha256:" + chr(ord(marker) + 2) * 64
    publication_root = "sha256:" + chr(ord(marker) + 3) * 64
    allocation_receipt = "sha256:" + "a" * 64
    allocation_authority_receipt = "sha256:" + "9" * 64
    verified_bundle = {
        "bundle_hash": bundle_hash,
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "validator_hotkey": f"validator-{marker}",
        "root_receipt_hash": root_hash,
        "weights_hash": marker * 64,
        "snapshot_hash": "sha256:" + "b" * 64,
        "weight_receipt_hash": "sha256:" + "c" * 64,
    }
    bundle_doc = {
        "schema_version": "leadpoet.published_weight_bundle.v2",
        "fixture_marker": marker,
        "weight_snapshot": {
            "calculation_snapshot": {
                "research_lab_allocation_doc": allocation,
            },
            "input_receipt_hashes": {
                "research_lab_allocation": allocation_receipt,
            },
        },
        "receipt_graph": {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": allocation_receipt,
                    "parent_receipt_hashes": [
                        allocation_authority_receipt
                    ],
                },
                {
                    "receipt_hash": allocation_authority_receipt,
                    "role": "gateway_coordinator",
                    "purpose": "research_lab.allocation.v2",
                    "epoch_id": 100,
                    "status": "succeeded",
                    "output_root": sha256_json({"allocation": allocation}),
                },
            ],
        },
    }
    expected_bundle_row = {
        "bundle_hash": bundle_hash,
        "schema_version": bundle_doc["schema_version"],
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "validator_hotkey": f"validator-{marker}",
        "root_receipt_hash": root_hash,
        "weights_hash": marker * 64,
        "snapshot_hash": "sha256:" + "b" * 64,
        "bundle_doc": bundle_doc,
    }
    durable_hash = sha256_json(expected_bundle_row)
    transparency_hash = "sha256:" + "d" * 64
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": bundle_hash,
        "root_receipt_hash": root_hash,
        "durable_readback_hash": durable_hash,
        "transparency_event_hash": transparency_hash,
    }
    submission_event = sha256_json(
        {
            "bundle_hash": bundle_hash,
            "publication_receipt_hash": publication_root,
            "transparency_event_hash": transparency_hash,
            "durable_readback_hash": durable_hash,
        }
    )
    verified_finalization = {
        "validator_hotkey": f"validator-{marker}",
        "netuid": 71,
        "epoch_id": 100,
        "weights_hash": marker * 64,
        "weight_receipt_hash": "sha256:" + "c" * 64,
        "extrinsic_authorization_hash": "sha256:" + "e" * 64,
        "extrinsic_hash": "0x" + marker * 64,
        "finalized_block": 36105,
        "finalized_block_hash": marker * 64,
        "state_transition_hash": "sha256:" + "f" * 64,
    }
    finalization_doc = {"fixture_marker": marker}
    finalization_event = sha256_json(
        {
            "weight_submission_event_hash": submission_event,
            "bundle_hash": bundle_hash,
            "finalization_receipt_hash": finalization_root,
            "extrinsic_authorization_hash": verified_finalization[
                "extrinsic_authorization_hash"
            ],
            "extrinsic_hash": verified_finalization["extrinsic_hash"],
            "finalized_block": verified_finalization["finalized_block"],
            "finalized_block_hash": verified_finalization[
                "finalized_block_hash"
            ],
            "state_transition_hash": verified_finalization[
                "state_transition_hash"
            ],
        }
    )
    row = {
        **expected_bundle_row,
        "weight_submission_event_hash": submission_event,
        "publication_receipt_hash": publication_root,
        "transparency_event_hash": transparency_hash,
        "durable_readback_hash": durable_hash,
        "publication_doc": publication_doc,
        "weight_finalization_event_hash": finalization_event,
        "finalization_receipt_hash": finalization_root,
        "extrinsic_authorization_hash": verified_finalization[
            "extrinsic_authorization_hash"
        ],
        "extrinsic_hash": verified_finalization["extrinsic_hash"],
        "finalized_block": verified_finalization["finalized_block"],
        "finalized_block_hash": verified_finalization["finalized_block_hash"],
        "state_transition_hash": verified_finalization[
            "state_transition_hash"
        ],
        "finalization_doc": finalization_doc,
    }
    graph = {
        "root_receipt_hash": finalization_root,
        "receipts": [{"receipt_hash": finalization_root}],
    }
    return row, graph, {
        "bundle": verified_bundle,
        "finalization": verified_finalization,
    }


def _install_validators(monkeypatch, validations: dict[str, dict]) -> None:
    monkeypatch.setattr(
        settlement,
        "validate_published_weight_bundle_v2",
        lambda document: validations[str(document["fixture_marker"])]["bundle"],
    )
    monkeypatch.setattr(
        settlement,
        "validate_weight_finalization_submission_v2",
        lambda submission: validations[
            str(submission["finalization"]["fixture_marker"])
        ]["finalization"],
    )


def test_preliminary_chain_authority_accepts_exact_view_shape(monkeypatch):
    row, _graph, verified = _authority_row("1", allocation=_allocation())
    _install_validators(monkeypatch, {"1": verified})
    row["finalization_doc"].update(verified["finalization"])

    authority = settlement._preliminary_finalized_bundle_authority_v1(row)

    assert "weight_receipt_hash" not in row
    assert authority["weight_receipt_hash"] == (
        verified["bundle"]["weight_receipt_hash"]
    )


def test_preliminary_chain_authority_rejects_tampered_weight_receipt(
    monkeypatch,
):
    row, _graph, verified = _authority_row("1", allocation=_allocation())
    _install_validators(monkeypatch, {"1": verified})
    row["finalization_doc"].update(verified["finalization"])
    row["finalization_doc"]["weight_receipt_hash"] = "sha256:" + "0" * 64

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="finalization differs at weight_receipt_hash",
    ):
        settlement._preliminary_finalized_bundle_authority_v1(row)


def _chain_observation(
    *,
    epoch_id: int,
    source_epoch_id: int = 100,
    miner_hotkey: str = "miner-hotkey",
) -> dict:
    close_block = 1_000 + ((epoch_id - source_epoch_id + 1) * 360) - 1
    weights = [[0, 65535], [7, 3449]]
    hotkeys = ["validator-hotkey"] + [
        "unused-%d" % uid for uid in range(1, 7)
    ] + [miner_hotkey]
    return {
        "schema_version": settlement.CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
        "netuid": 71,
        "epoch_id": epoch_id,
        "official_subnet_epoch_id": epoch_id,
        "cutover_mapping_hash": "sha256:" + "a" * 64,
        "close_block": close_block,
        "close_block_hash": ("%064x" % close_block),
        "close_state_root": "b" * 64,
        "next_epoch_block": close_block + 1,
        "next_epoch_block_hash": ("%064x" % (close_block + 1)),
        "validator_hotkey": "validator-hotkey",
        "validator_uid": 0,
        "metagraph_hotkeys": hotkeys,
        "weights": weights,
        "weights_storage_key": "0x1234",
        "last_update_storage_key": "0x5678",
        "last_update_block": 1_000,
        "last_update_block_hash": "c" * 64,
        "last_update_official_subnet_epoch_id": source_epoch_id,
        "active_source_epoch_id": source_epoch_id,
        "weights_vector_hash": sha256_json(
            {
                "uids": [item[0] for item in weights],
                "weights_u16": [item[1] for item in weights],
            }
        ),
    }


def _chain_package_authority(
    *,
    source_epoch_id: int = 100,
    allocations: list[dict] | None = None,
    queued_allocations: list[dict] | None = None,
) -> tuple[dict, dict]:
    allocation_body = {
        "schema_version": "leadpoet.research_lab_allocation.v2",
        "epoch": source_epoch_id,
        "champion_allocations": list(
            allocations
            if allocations is not None
            else [
                {
                    "source_id": "champion_reward:test",
                    "miner_hotkey": "miner-hotkey",
                    "uid": 7,
                    "paid_alpha_percent": 5.0,
                    "base_desired_alpha_percent": 5.0,
                }
            ]
        ),
        "queued_champion_allocations": list(queued_allocations or []),
        "source_add_allocations": [],
        "reimbursement_allocations": [],
    }
    allocation = {
        **allocation_body,
        "allocation_hash": sha256_json(allocation_body),
    }
    bundle_hash = "sha256:" + "1" * 64
    bundle_doc = {
        "weight_result": {
            "uids": [0, 7],
            "weights": [0.95, 0.05],
        },
        "weight_snapshot": {
            "calculation_snapshot": {
                "research_lab_allocation_doc": allocation,
            },
            "input_receipt_hashes": {
                "research_lab_allocation": "sha256:" + "2" * 64,
            },
        },
    }
    authority = {
        "bundle_hash": bundle_hash,
        "bundle_doc": bundle_doc,
        "finalized_block": 1_000,
        "finalized_block_hash": "c" * 64,
        "finalization_receipt_hash": "sha256:" + "3" * 64,
    }
    verified_bundle = {
        "bundle_hash": bundle_hash,
        "epoch_id": source_epoch_id,
        "netuid": 71,
        "validator_hotkey": "validator-hotkey",
        "uids": [0, 7],
        "weights_u16": [65535, 3449],
    }
    return authority, verified_bundle


def test_chain_realized_bundle_selection_requires_exact_finalized_source(
    monkeypatch,
):
    observation = _chain_observation(epoch_id=101)
    authority, _verified = _chain_package_authority()
    exact = {
        **authority,
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": "validator-hotkey",
        "uids": [0, 7],
        "weights_u16": [65535, 3449],
    }
    monkeypatch.setattr(
        settlement,
        "_preliminary_finalized_bundle_authority_v1",
        lambda row: dict(row),
    )

    selected = settlement.select_chain_realized_bundle_candidate_v1(
        [
            {**exact, "epoch_id": 99},
            {**exact, "finalized_block_hash": "d" * 64},
            exact,
        ],
        observation=observation,
    )

    assert selected["bundle_hash"] == exact["bundle_hash"]
    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="no finalized canonical bundle",
    ):
        settlement.select_chain_realized_bundle_candidate_v1(
            [{**exact, "finalized_block": 999}],
            observation=observation,
        )


def test_chain_realized_stale_vector_credits_each_epoch_exactly(
    monkeypatch,
):
    authority, verified = _chain_package_authority()
    monkeypatch.setattr(
        settlement,
        "validate_published_weight_bundle_v2",
        lambda _document: verified,
    )

    first = settlement.build_chain_realized_settlement_package_v1(
        observation=_chain_observation(epoch_id=101),
        authority=authority,
    )
    second = settlement.build_chain_realized_settlement_package_v1(
        observation=_chain_observation(epoch_id=102),
        authority=authority,
    )

    assert first["settlement_hash"] != second["settlement_hash"]
    assert first["credits"][0]["credit_hash"] != second["credits"][0]["credit_hash"]
    for epoch_id, package in ((101, first), (102, second)):
        credit = package["credits"][0]["credit_doc"]
        assert package["settlement_doc"]["epoch_id"] == epoch_id
        assert credit["epoch_id"] == epoch_id
        assert credit["credited_alpha_percent"] == "4.999710077699"
        assert credit["attribution_doc"]["source_bundle_epoch_id"] == 100


def test_chain_realized_u16_rounding_is_distributed_without_overcredit(
    monkeypatch,
):
    authority, verified = _chain_package_authority(
        allocations=[
            {
                "source_id": "champion_reward:first",
                "miner_hotkey": "miner-hotkey",
                "uid": 7,
                "paid_alpha_percent": 3.0,
                "base_desired_alpha_percent": 3.0,
            },
            {
                "source_id": "champion_reward:second",
                "miner_hotkey": "miner-hotkey",
                "uid": 7,
                "paid_alpha_percent": 2.0,
                "base_desired_alpha_percent": 2.0,
            },
        ]
    )
    monkeypatch.setattr(
        settlement,
        "validate_published_weight_bundle_v2",
        lambda _document: verified,
    )

    package = settlement.build_chain_realized_settlement_package_v1(
        observation=_chain_observation(epoch_id=101),
        authority=authority,
    )

    credits = [
        Decimal(item["credit_doc"]["credited_alpha_percent"])
        for item in package["credits"]
    ]
    assert sum(credits) == Decimal("4.999710077699")
    assert sorted(credits) == [
        Decimal("1.999884031080"),
        Decimal("2.999826046619"),
    ]


def test_chain_realized_uid_reassignment_does_not_credit_old_hotkey(
    monkeypatch,
):
    authority, verified = _chain_package_authority()
    monkeypatch.setattr(
        settlement,
        "validate_published_weight_bundle_v2",
        lambda _document: verified,
    )

    package = settlement.build_chain_realized_settlement_package_v1(
        observation=_chain_observation(
            epoch_id=101,
            miner_hotkey="replacement-hotkey",
        ),
        authority=authority,
    )

    assert package["credits"] == []
    assert package["settlement_doc"]["credit_hashes"] == []


def test_chain_realized_allocation_cannot_exceed_canonical_uid_weight(
    monkeypatch,
):
    authority, verified = _chain_package_authority(
        allocations=[
            {
                "source_id": "champion_reward:first",
                "miner_hotkey": "miner-hotkey",
                "uid": 7,
                "paid_alpha_percent": 4.0,
                "base_desired_alpha_percent": 4.0,
            },
            {
                "source_id": "champion_reward:second",
                "miner_hotkey": "miner-hotkey",
                "uid": 7,
                "paid_alpha_percent": 4.0,
                "base_desired_alpha_percent": 4.0,
            },
        ]
    )
    monkeypatch.setattr(
        settlement,
        "validate_published_weight_bundle_v2",
        lambda _document: verified,
    )

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="exceeds canonical weight",
    ):
        settlement.build_chain_realized_settlement_package_v1(
            observation=_chain_observation(epoch_id=101),
            authority=authority,
        )


def test_chain_realized_allocation_rejects_active_and_queued_duplicate(
    monkeypatch,
):
    duplicate = {
        "source_id": "champion_reward:test",
        "miner_hotkey": "miner-hotkey",
        "uid": 7,
        "paid_alpha_percent": 2.5,
        "base_desired_alpha_percent": 2.5,
    }
    authority, verified = _chain_package_authority(
        allocations=[duplicate],
        queued_allocations=[duplicate],
    )
    monkeypatch.setattr(
        settlement,
        "validate_published_weight_bundle_v2",
        lambda _document: verified,
    )

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="allocation identity is invalid",
    ):
        settlement.build_chain_realized_settlement_package_v1(
            observation=_chain_observation(epoch_id=101),
            authority=authority,
        )


def test_finalized_allocation_authority_collapses_validator_duplicates(monkeypatch):
    allocation = _allocation()
    first, first_graph, first_verified = _authority_row(
        "1", allocation=allocation
    )
    second, second_graph, second_verified = _authority_row(
        "2", allocation=allocation
    )
    _install_validators(
        monkeypatch,
        {"1": first_verified, "2": second_verified},
    )

    result = settlement.validate_finalized_allocation_authorities_v2(
        [first, second],
        finalization_graphs={
            first_graph["root_receipt_hash"]: first_graph,
            second_graph["root_receipt_hash"]: second_graph,
        },
    )

    assert len(result) == 1
    assert result[0]["allocation_doc"] == allocation
    assert result[0]["finalized_authority_count"] == 2
    assert result[0]["allocation_receipt_hash"] == "sha256:" + "a" * 64
    assert result[0]["allocation_authority_receipt_hash"] == (
        "sha256:" + "9" * 64
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda bundle: bundle["receipt_graph"]["receipts"][0].update(
                {"parent_receipt_hashes": []}
            ),
            "allocation input ancestry is invalid",
        ),
        (
            lambda bundle: bundle["receipt_graph"]["receipts"][1].update(
                {"output_root": "sha256:" + "0" * 64}
            ),
            "allocation authority receipt is invalid",
        ),
    ),
)
def test_finalized_allocation_authority_rejects_invalid_input_ancestry(
    mutate,
    message,
):
    allocation = _allocation()
    row, _graph, _verified = _authority_row("1", allocation=allocation)
    mutate(row["bundle_doc"])

    with pytest.raises(settlement.ChampionSettlementV2Error, match=message):
        settlement._allocation_authority_receipt_hash_v2(
            bundle_doc=row["bundle_doc"],
            allocation_input_receipt_hash="sha256:" + "a" * 64,
            allocation=allocation,
            epoch_id=100,
        )


def test_finalized_allocation_authority_fails_on_missing_or_tampered_evidence(
    monkeypatch,
):
    row, graph, verified = _authority_row("1", allocation=_allocation())
    _install_validators(monkeypatch, {"1": verified})

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="receipt graph is missing",
    ):
        settlement.validate_finalized_allocation_authorities_v2(
            [row], finalization_graphs={}
        )

    tampered = copy.deepcopy(row)
    tampered["bundle_doc"]["weight_snapshot"]["calculation_snapshot"][
        "research_lab_allocation_doc"
    ]["champion_allocations"][0]["paid_alpha_percent"] = 99.0
    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="publication differs from its bundle",
    ):
        settlement.validate_finalized_allocation_authorities_v2(
            [tampered],
            finalization_graphs={graph["root_receipt_hash"]: graph},
        )


@pytest.mark.asyncio
async def test_chain_realized_pristine_bootstrap_validates_activation_source(
    monkeypatch,
):
    from gateway.research_lab import store

    source_hash = "sha256:" + "a" * 64
    activation = {
        "netuid": 71,
        "schema_version": (
            "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        ),
        "first_epoch_id": 100,
        "source_bundle_hash": source_hash,
        "source_bundle_epoch_id": 100,
        "source_finalized_block": 1234,
    }

    async def select_many(table, *, filters=(), **_kwargs):
        if table == settlement.CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1:
            return [activation]
        if table == settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1:
            return []
        if table == settlement.FINALIZED_ALLOCATION_VIEW_V2:
            assert ("bundle_hash", source_hash) in filters
            assert ("finalized_block", 1234) in filters
            return [
                {
                    "bundle_hash": source_hash,
                    "netuid": 71,
                    "epoch_id": 100,
                    "finalized_block": 1234,
                    "finalization_receipt_hash": "sha256:" + "b" * 64,
                }
            ]
        raise AssertionError(table)

    async def load_finalized(**kwargs):
        assert kwargs == {"netuid": 71, "start_epoch": 100, "end_epoch": 103}
        return [
            {
                "epoch": 100,
                "netuid": 71,
                "finalized_bundle_hashes": [source_hash],
            },
            {
                "epoch": 103,
                "netuid": 71,
                "finalized_bundle_hashes": ["sha256:" + "c" * 64],
            },
        ]

    monkeypatch.setattr(store, "select_many", select_many)
    monkeypatch.setattr(
        settlement,
        "load_finalized_allocation_history_v2",
        load_finalized,
    )

    result = await settlement.validate_chain_realized_settlement_bootstrap_v1(
        netuid=71,
        target_epoch=103,
    )

    assert result["status"] == "pristine_bootstrap_pending"
    assert result["activation_epoch"] == 100
    assert result["target_epoch"] == 103
    assert result["backlog_epoch_count"] == 4
    assert result["validated_finalized_candidate_epochs"] == [100, 103]


@pytest.mark.asyncio
async def test_chain_realized_bootstrap_rejects_any_partial_history(
    monkeypatch,
):
    from gateway.research_lab import store

    async def select_many(table, **_kwargs):
        if table == settlement.CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1:
            return [
                {
                    "netuid": 71,
                    "schema_version": (
                        "leadpoet.research_lab_chain_realized_"
                        "settlement_activation.v1"
                    ),
                    "first_epoch_id": 100,
                    "source_bundle_hash": "sha256:" + "a" * 64,
                    "source_bundle_epoch_id": 100,
                    "source_finalized_block": 1234,
                }
            ]
        if table == settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1:
            return [
                {
                    "netuid": 71,
                    "epoch_id": 100,
                    "settlement_hash": "sha256:" + "b" * 64,
                }
            ]
        raise AssertionError(table)

    monkeypatch.setattr(store, "select_many", select_many)

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="history is incomplete",
    ):
        await settlement.validate_chain_realized_settlement_bootstrap_v1(
            netuid=71,
            target_epoch=103,
        )


@pytest.mark.asyncio
async def test_chain_realized_bootstrap_rejects_unbounded_or_missing_source(
    monkeypatch,
):
    from gateway.research_lab import store

    activation = {
        "netuid": 71,
        "schema_version": (
            "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        ),
        "first_epoch_id": 100,
        "source_bundle_hash": "sha256:" + "a" * 64,
        "source_bundle_epoch_id": 100,
        "source_finalized_block": 1234,
    }

    async def select_many(table, **_kwargs):
        if table == settlement.CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1:
            return [activation]
        if table == settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1:
            return []
        if table == settlement.FINALIZED_ALLOCATION_VIEW_V2:
            return []
        raise AssertionError(table)

    monkeypatch.setattr(store, "select_many", select_many)

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="backlog exceeds policy",
    ):
        await settlement.validate_chain_realized_settlement_bootstrap_v1(
            netuid=71,
            target_epoch=200,
        )
    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="activation source is unavailable",
    ):
        await settlement.validate_chain_realized_settlement_bootstrap_v1(
            netuid=71,
            target_epoch=103,
        )


def test_chain_realized_history_replaces_finalized_weight_intent(monkeypatch):
    monkeypatch.setattr(settlement, "validate_receipt_graph", lambda _graph: ())
    settlement_row, credit_row, _doc, graphs = _chain_realized_fixture(
        observed="30.000000",
        attributed="5.000000",
        scheduled="5.000000",
        credited="5.000000",
    )

    chain_epochs = settlement.validate_chain_realized_epoch_settlements_v1(
        [settlement_row],
        receipt_graphs=graphs,
    )
    chain_history = settlement.validate_chain_realized_obligation_credits_v1(
        [credit_row],
        settlement_rows=chain_epochs,
        receipt_graphs=graphs,
    )
    finalized_history = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": "sha256:" + "1" * 64,
            "allocation_doc": {
                "champion_allocations": [
                    {
                        "source_id": "champion_reward:test",
                        "paid_alpha_percent": 30.0,
                        "base_desired_alpha_percent": 5.0,
                    }
                ],
                "queued_champion_allocations": [],
            },
            "authority_types": ["native_v2_finalization"],
        }
    ]

    merged = settlement.merge_settled_allocation_histories_v2(
        finalized_history,
        chain_history,
    )

    assert len(merged) == 1
    assert merged[0]["authority_types"] == [
        settlement.CHAIN_REALIZED_AUTHORITY_TYPE_V1
    ]
    assert merged[0]["replaced_authority_types"] == [
        "native_v2_finalization"
    ]
    allocation = merged[0]["allocation_doc"]["champion_allocations"][0]
    assert allocation["paid_alpha_percent"] == pytest.approx(5.0)
    assert allocation["observed_chain_alpha_percent"] == pytest.approx(30.0)
    assert allocation["lab_attributed_alpha_percent"] == pytest.approx(5.0)


def test_unattributed_chain_history_remains_zero_credit_v2_authority(
    monkeypatch,
):
    monkeypatch.setattr(settlement, "validate_receipt_graph", lambda _graph: ())
    settlement_row, _credit_row, settlement_doc, _graphs = (
        _chain_realized_fixture()
    )
    summary = dict(settlement_doc["observation_summary"])
    for field in (
        "source_bundle_hash",
        "source_bundle_epoch_id",
        "source_bundle_finalized_block",
        "source_bundle_finalized_block_hash",
    ):
        summary.pop(field)
    summary["schema_version"] = (
        "leadpoet.chain_realized_unattributed_observation_summary.v1"
    )
    summary["authority_mode"] = "unattributed_chain_observation"
    unattributed_doc = {
        **settlement_doc,
        "schema_version": (
            settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2
        ),
        "credit_hashes": [],
        "observation_summary": summary,
    }
    settlement_hash = sha256_json(unattributed_doc)
    receipt_hash = settlement_row["settlement_receipt_hash"]
    unattributed_row = {
        **settlement_row,
        "schema_version": (
            settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2
        ),
        "settlement_hash": settlement_hash,
        "settlement_doc": unattributed_doc,
    }
    graphs = {
        receipt_hash: _minimal_receipt_graph(
            receipt_hash,
            purpose=settlement.CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
            output_root=settlement_hash,
        )
    }

    chain_epochs = settlement.validate_chain_realized_epoch_settlements_v1(
        [unattributed_row],
        receipt_graphs=graphs,
    )
    history = settlement.validate_chain_realized_obligation_credits_v1(
        [],
        settlement_rows=chain_epochs,
        receipt_graphs=graphs,
    )

    assert history[0]["authority_types"] == [
        settlement.CHAIN_REALIZED_UNATTRIBUTED_AUTHORITY_TYPE_V1
    ]
    assert history[0]["allocation_doc"]["schema_version"] == (
        settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2
    )
    assert history[0]["allocation_doc"]["authority_type"] == (
        settlement.CHAIN_REALIZED_UNATTRIBUTED_AUTHORITY_TYPE_V1
    )
    assert history[0]["allocation_doc"]["champion_allocations"] == []
    assert history[0]["chain_realized_credit_hashes"] == []


@pytest.mark.parametrize(
    ("kind", "source_id", "section", "identity_field"),
    (
        (
            "queued_champion",
            "champion_reward:queued",
            "queued_champion_allocations",
            "champion_reward_id",
        ),
        (
            "source_add",
            "source_add_reward:abcd1234",
            "source_add_allocations",
            "source_add_reward_id",
        ),
        (
            "reimbursement",
            "reimbursement_schedule:test",
            "reimbursement_allocations",
            "schedule_id",
        ),
    ),
)
def test_chain_realized_credit_kinds_map_to_allocation_sections(
    monkeypatch,
    kind,
    source_id,
    section,
    identity_field,
):
    monkeypatch.setattr(settlement, "validate_receipt_graph", lambda _graph: ())
    settlement_row, credit_row, _doc, graphs = _chain_realized_fixture(
        reward_id=source_id,
        kind=kind,
    )
    chain_epochs = settlement.validate_chain_realized_epoch_settlements_v1(
        [settlement_row],
        receipt_graphs=graphs,
    )

    history = settlement.validate_chain_realized_obligation_credits_v1(
        [credit_row],
        settlement_rows=chain_epochs,
        receipt_graphs=graphs,
    )

    allocation = history[0]["allocation_doc"]
    assert allocation[section][0]["source_id"] == source_id
    assert allocation[section][0][identity_field] == source_id
    for empty_section in {
        "source_add_allocations",
        "reimbursement_allocations",
        "champion_allocations",
        "queued_champion_allocations",
    } - {section}:
        assert allocation[empty_section] == []


def test_chain_realized_credit_sets_must_be_epoch_complete(monkeypatch):
    monkeypatch.setattr(settlement, "validate_receipt_graph", lambda _graph: ())
    settlement_row, credit_row, _doc, graphs = _chain_realized_fixture()
    missing_hash_doc = dict(settlement_row["settlement_doc"])
    missing_hash_doc["credit_hashes"] = [
        credit_row["credit_hash"],
        "sha256:" + "9" * 64,
    ]
    settlement_row = {
        **settlement_row,
        "settlement_doc": missing_hash_doc,
        "settlement_hash": sha256_json(missing_hash_doc),
    }
    credit_row = {
        **credit_row,
        "settlement_hash": settlement_row["settlement_hash"],
    }
    settlement_receipt = settlement_row["settlement_receipt_hash"]
    graphs[settlement_receipt] = _minimal_receipt_graph(
        settlement_receipt,
        purpose=settlement.CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
        output_root=settlement_row["settlement_hash"],
    )

    chain_epochs = settlement.validate_chain_realized_epoch_settlements_v1(
        [settlement_row],
        receipt_graphs=graphs,
    )

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="credit set is incomplete",
    ):
        settlement.validate_chain_realized_obligation_credits_v1(
            [credit_row],
            settlement_rows=chain_epochs,
            receipt_graphs=graphs,
        )


@pytest.mark.parametrize(
    ("observed", "attributed", "scheduled", "credited", "message"),
    (
        (
            "5.000000",
            "6.000000",
            "5.000000",
            "5.000000",
            "exceeds observed attribution",
        ),
        (
            "6.000000",
            "6.000000",
            "5.000000",
            "6.000000",
            "exceeds scheduled epoch amount",
        ),
    ),
)
def test_chain_realized_credits_cannot_overcredit_obligations(
    monkeypatch,
    observed,
    attributed,
    scheduled,
    credited,
    message,
):
    monkeypatch.setattr(settlement, "validate_receipt_graph", lambda _graph: ())
    settlement_row, credit_row, _doc, graphs = _chain_realized_fixture(
        observed=observed,
        attributed=attributed,
        scheduled=scheduled,
        credited=credited,
    )
    chain_epochs = settlement.validate_chain_realized_epoch_settlements_v1(
        [settlement_row],
        receipt_graphs=graphs,
    )

    with pytest.raises(settlement.ChampionSettlementV2Error, match=message):
        settlement.validate_chain_realized_obligation_credits_v1(
            [credit_row],
            settlement_rows=chain_epochs,
            receipt_graphs=graphs,
        )


def test_chain_realized_credit_set_cannot_overattribute_one_uid(monkeypatch):
    monkeypatch.setattr(settlement, "validate_receipt_graph", lambda _graph: ())
    settlement_row, first, _doc, graphs = _chain_realized_fixture(
        reward_id="champion_reward:first",
        observed="5.000000",
        attributed="3.000000",
        scheduled="3.000000",
        credited="3.000000",
    )
    _second_settlement, second, _doc, _graphs = _chain_realized_fixture(
        reward_id="champion_reward:second",
        observed="5.000000",
        attributed="3.000000",
        scheduled="3.000000",
        credited="3.000000",
    )
    credit_hashes = sorted([first["credit_hash"], second["credit_hash"]])
    settlement_doc = {
        **settlement_row["settlement_doc"],
        "credit_hashes": credit_hashes,
    }
    settlement_hash = sha256_json(settlement_doc)
    settlement_row = {
        **settlement_row,
        "settlement_doc": settlement_doc,
        "settlement_hash": settlement_hash,
    }
    first = {**first, "settlement_hash": settlement_hash}
    second = {
        **second,
        "settlement_hash": settlement_hash,
        "credit_receipt_hash": first["credit_receipt_hash"],
    }
    receipt_hash = settlement_row["settlement_receipt_hash"]
    graphs[receipt_hash] = _minimal_receipt_graph(
        receipt_hash,
        purpose=settlement.CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
        output_root=settlement_hash,
    )
    chain_epochs = settlement.validate_chain_realized_epoch_settlements_v1(
        [settlement_row],
        receipt_graphs=graphs,
    )

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="UID attribution exceeds observed weight",
    ):
        settlement.validate_chain_realized_obligation_credits_v1(
            [first, second],
            settlement_rows=chain_epochs,
            receipt_graphs=graphs,
        )


def test_chain_realized_credit_rejects_active_and_queued_duplicate(monkeypatch):
    monkeypatch.setattr(settlement, "validate_receipt_graph", lambda _graph: ())
    settlement_row, first, _doc, graphs = _chain_realized_fixture(
        reward_id="champion_reward:duplicate",
        observed="5.000000",
        attributed="2.500000",
        scheduled="2.500000",
        credited="2.500000",
    )
    _second_settlement, second, _doc, _graphs = _chain_realized_fixture(
        reward_id="champion_reward:duplicate",
        observed="5.000000",
        attributed="2.500000",
        scheduled="2.500000",
        credited="2.500000",
        kind="queued_champion",
    )
    credit_hashes = sorted([first["credit_hash"], second["credit_hash"]])
    settlement_doc = {
        **settlement_row["settlement_doc"],
        "credit_hashes": credit_hashes,
    }
    settlement_hash = sha256_json(settlement_doc)
    settlement_row = {
        **settlement_row,
        "settlement_doc": settlement_doc,
        "settlement_hash": settlement_hash,
    }
    first = {**first, "settlement_hash": settlement_hash}
    second = {
        **second,
        "settlement_hash": settlement_hash,
        "credit_receipt_hash": first["credit_receipt_hash"],
    }
    receipt_hash = settlement_row["settlement_receipt_hash"]
    graphs[receipt_hash] = _minimal_receipt_graph(
        receipt_hash,
        purpose=settlement.CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
        output_root=settlement_hash,
    )
    chain_epochs = settlement.validate_chain_realized_epoch_settlements_v1(
        [settlement_row],
        receipt_graphs=graphs,
    )

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="obligation credit is duplicated",
    ):
        settlement.validate_chain_realized_obligation_credits_v1(
            [first, second],
            settlement_rows=chain_epochs,
            receipt_graphs=graphs,
        )


@pytest.mark.asyncio
async def test_cutover_requires_receipts_for_every_positive_balance(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    settled_id = "champion_reward:sha256:" + "1" * 64
    positive_id = "champion_reward:sha256:" + "2" * 64

    def reward_row(reward_id: str, uid: int) -> dict:
        return {
            "champion_reward_id": reward_id,
            "score_bundle_id": f"score-{uid}",
            "candidate_id": f"candidate-{uid}",
            "run_id": f"run-{uid}",
            "miner_hotkey": f"miner-{uid}",
            "miner_uid": uid,
            "island": "generalist",
            "evaluation_epoch": 99,
            "start_epoch": 100,
            "epoch_count": 2,
            "improvement_points": 2.0,
            "threshold_points": 1.0,
            "desired_alpha_percent": 5.0,
            "input_hash": "sha256:" + "3" * 64,
            "anchored_hash": "sha256:" + str(uid + 3) * 64,
            "current_reward_status": "active",
        }

    rows = [reward_row(settled_id, 1), reward_row(positive_id, 2)]

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_emission_allocation_current":
            return []
        requested_status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return rows if requested_status == "active" else []

    finalized_allocation_body = {
        "schema_version": "1.0",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": settled_id,
                "paid_alpha_percent": 10.0,
            },
            {
                "source_id": positive_id,
                "paid_alpha_percent": 5.0,
            },
        ],
        "queued_champion_allocations": [],
    }
    finalized_allocation = {
        **finalized_allocation_body,
        "allocation_hash": sha256_json(finalized_allocation_body),
    }
    finalized = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": finalized_allocation["allocation_hash"],
            "allocation_doc": finalized_allocation,
        }
    ]

    async def load_finalized(**_kwargs):
        return finalized

    positive_projection = champion_reward_row_projection_v2(rows[1])
    root_hash = "sha256:" + "9" * 64

    async def load_graph(**_kwargs):
        return {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "role": "gateway_coordinator",
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": sha256_json(positive_projection),
                }
            ],
        }

    async def load_graphs(artifacts):
        return {
            key: await load_graph(
                artifact_kind=key[0],
                artifact_ref=key[1],
            )
            for key in artifacts
        }

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_settled_allocation_history_v2",
        load_finalized,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_graphs,
    )

    ready = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert ready["ready"] is True
    assert ready["receipt_coverage"] == 1.0
    assert ready["covered_positive_balance_count"] == 1
    assert [
        item["champion_reward_id"] for item in ready["zero_balance_active_rows"]
    ] == [settled_id]

    async def missing_graph(**_kwargs):
        raise RuntimeError("not migrated")

    async def missing_graphs(_artifacts):
        raise RuntimeError("not migrated")

    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        missing_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        missing_graphs,
    )
    blocked = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert blocked["ready"] is False
    assert blocked["receipt_coverage"] == 0.0
    assert blocked["missing"] == [
        {
            "champion_reward_id": positive_id,
            "remaining_alpha_percent": 5.0,
            "reason": "missing_or_invalid_v2_reward_receipt",
        }
    ]


@pytest.mark.asyncio
async def test_cutover_does_not_trust_paid_status_when_chain_balance_remains(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store

    reward_id = "champion_reward:sha256:" + "4" * 64
    reward = {
        "champion_reward_id": reward_id,
        "start_epoch": 100,
        "epoch_count": 2,
        "desired_alpha_percent": 5.0,
        "current_reward_status": "paid",
    }

    async def select_all(table, *, filters=(), **_kwargs):
        if table in {
            "research_lab_emission_allocation_current",
            "research_lab_emission_allocation_snapshots",
            "research_lab_arweave_epoch_audit_anchor_current",
            "published_weight_bundles",
        }:
            return []
        status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return [reward] if status == "paid" else []

    async def no_finalized_payments(**_kwargs):
        return []

    async def no_receipt(**_kwargs):
        raise RuntimeError("not migrated")

    async def no_receipts(_artifacts):
        raise RuntimeError("not migrated")

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_settled_allocation_history_v2",
        no_finalized_payments,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        no_receipt,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        no_receipts,
    )

    readiness = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )

    assert readiness["ready"] is False
    assert readiness["required_positive_balance_count"] == 1
    assert readiness["missing"][0]["champion_reward_id"] == reward_id
    assert readiness["missing"][0]["remaining_alpha_percent"] == 10.0


@pytest.mark.asyncio
async def test_cutover_blocks_until_every_historical_payment_epoch_is_attested(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    reward_id = "champion_reward:sha256:" + "6" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score-1",
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "miner_hotkey": "miner-1",
        "miner_uid": 1,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 2,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
        "input_hash": "sha256:" + "7" * 64,
        "anchored_hash": "sha256:" + "8" * 64,
        "current_reward_status": "active",
    }
    allocation_body = {
        "schema_version": "1.0",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": reward_id,
                "paid_alpha_percent": 5.0,
                "base_desired_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
    }
    allocation = {
        **allocation_body,
        "allocation_hash": sha256_json(allocation_body),
    }

    async def select_all(table, *, filters=(), **_kwargs):
        if table in {
            "research_lab_emission_allocation_current",
            "research_lab_emission_allocation_snapshots",
        }:
            return [
                {
                    "epoch": 100,
                    "netuid": 71,
                    "allocation_hash": allocation["allocation_hash"],
                    "allocation_doc": allocation,
                }
            ]
        if table == "research_lab_arweave_epoch_audit_anchor_current":
            return [
                {
                    "epoch": 100,
                    "allocation_hash": allocation["allocation_hash"],
                    "weights_hash": "sha256:" + "a" * 64,
                    "current_arweave_tx_id": "A" * 43,
                    "current_transparency_event_hash": "b" * 64,
                }
            ]
        if table == "published_weight_bundles":
            return [
                {
                    "epoch_id": 100,
                    "weights_hash": "a" * 64,
                }
            ]
        status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return [reward] if status == "active" else []

    state = {"finalized": [], "nonfinalized": []}

    async def load_finalized(**_kwargs):
        return list(state["finalized"])

    async def load_nonfinalized(**_kwargs):
        return list(state["nonfinalized"])

    root_hash = "sha256:" + "9" * 64

    async def load_graph(**_kwargs):
        return {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "role": "gateway_coordinator",
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": sha256_json(
                        champion_reward_row_projection_v2(reward)
                    ),
                }
            ],
        }

    async def load_graphs(artifacts):
        return {
            key: await load_graph(
                artifact_kind=key[0],
                artifact_ref=key[1],
            )
            for key in artifacts
        }

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_settled_allocation_history_v2",
        load_finalized,
    )
    monkeypatch.setattr(
        settlement,
        "load_legacy_allocation_nonfinalizations_v2",
        load_nonfinalized,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_graphs,
    )

    blocked = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert blocked["ready"] is False
    assert blocked["receipt_coverage"] == 1.0
    assert blocked["historical_settlement_coverage"] == 0.0
    assert blocked["missing_historical_settlements"] == [
        {
            "epoch": 100,
            "allocation_hash": allocation["allocation_hash"],
                "reason": "missing_finalized_chain_classification_authority",
        }
    ]

    state["nonfinalized"] = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": allocation["allocation_hash"],
            "allocation_doc": allocation,
            "finding_hash": "sha256:" + "f" * 64,
        }
    ]
    classified_unpaid = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert classified_unpaid["ready"] is True
    assert classified_unpaid[
        "covered_historical_nonfinalization_epochs"
    ] == [100]
    assert classified_unpaid["migrated_finalized_allocation_epoch_count"] == 0
    assert classified_unpaid["unproven_historical_allocations"] == [
        {
            "epoch": 100,
            "allocation_hash": allocation["allocation_hash"],
            "reason": "finalized_chain_vector_mismatch",
        }
    ]

    state["nonfinalized"] = []
    state["finalized"] = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": allocation["allocation_hash"],
            "allocation_doc": allocation,
            "authority_types": ["legacy_finalized_chain_migration_v2"],
        }
    ]
    ready = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert ready["ready"] is True
    assert ready["historical_settlement_coverage"] == 1.0

    mismatched_body = {
        key: value
        for key, value in allocation.items()
        if key != "allocation_hash"
    }
    mismatched_body["netuid"] = 72
    allocation.clear()
    allocation.update(
        {
            **mismatched_body,
            "allocation_hash": sha256_json(mismatched_body),
        }
    )
    blocked_scope = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert blocked_scope["ready"] is False
    assert blocked_scope["missing_historical_settlements"][0]["reason"] == (
        "invalid_historical_allocation"
    )


@pytest.mark.asyncio
async def test_cutover_uses_anchor_bound_snapshot_not_later_current_view(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    reward_id = "champion_reward:sha256:" + "1" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score-anchor",
        "candidate_id": "candidate-anchor",
        "run_id": "run-anchor",
        "miner_hotkey": "miner-anchor",
        "miner_uid": 1,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 2,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
        "input_hash": "sha256:" + "2" * 64,
        "anchored_hash": "sha256:" + "3" * 64,
        "current_reward_status": "active",
    }
    anchored_body = {
        "schema_version": "1.0",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": reward_id,
                "paid_alpha_percent": 5.0,
                "base_desired_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
        "snapshot_generation": "submitted",
    }
    anchored = {
        **anchored_body,
        "allocation_hash": sha256_json(anchored_body),
    }
    current_body = {
        **anchored_body,
        "snapshot_generation": "recomputed",
    }
    current = {
        **current_body,
        "allocation_hash": sha256_json(current_body),
    }

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_emission_allocation_current":
            documents = [current]
        elif table == "research_lab_emission_allocation_snapshots":
            documents = [anchored, current]
        else:
            documents = []
        if documents:
            return [
                {
                    "epoch": 100,
                    "netuid": 71,
                    "allocation_hash": document["allocation_hash"],
                    "allocation_doc": document,
                }
                for document in documents
            ]
        if table == "research_lab_arweave_epoch_audit_anchor_current":
            return [
                {
                    "epoch": 100,
                    "allocation_hash": anchored["allocation_hash"],
                    "weights_hash": "sha256:" + "4" * 64,
                    "current_arweave_tx_id": "A" * 43,
                    "current_transparency_event_hash": "5" * 64,
                }
            ]
        if table == "published_weight_bundles":
            return [{"epoch_id": 100, "weights_hash": "4" * 64}]
        status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return [reward] if status == "active" else []

    state = {"finalized": []}

    async def load_finalized(**_kwargs):
        return list(state["finalized"])

    root_hash = "sha256:" + "6" * 64
    receipt_graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "role": "gateway_coordinator",
                "purpose": "research_lab.reward_decision.v2",
                "output_root": sha256_json(
                    champion_reward_row_projection_v2(reward)
                ),
            }
        ],
    }

    async def load_graph(**_kwargs):
        return receipt_graph

    async def load_graphs(artifacts):
        return {key: receipt_graph for key in artifacts}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_settled_allocation_history_v2",
        load_finalized,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_graphs,
    )

    blocked = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )

    assert blocked["missing_historical_settlements"] == [
        {
            "epoch": 100,
            "allocation_hash": anchored["allocation_hash"],
                "reason": "missing_finalized_chain_classification_authority",
        }
    ]
    assert blocked["unproven_historical_allocations"] == [
        {
            "epoch": 100,
            "allocation_hash": current["allocation_hash"],
            "reason": "current_allocation_not_checkpointed",
        }
    ]

    state["finalized"] = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": anchored["allocation_hash"],
            "allocation_doc": anchored,
            "authority_types": ["legacy_finalized_chain_migration_v2"],
        }
    ]
    ready = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert ready["ready"] is True
    assert ready["covered_historical_settlement_epochs"] == [100]


@pytest.mark.asyncio
async def test_cutover_does_not_credit_unsubmitted_historical_allocation(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    reward_id = "champion_reward:sha256:" + "3" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score-3",
        "candidate_id": "candidate-3",
        "run_id": "run-3",
        "miner_hotkey": "miner-3",
        "miner_uid": 3,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 2,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
        "input_hash": "sha256:" + "4" * 64,
        "anchored_hash": "sha256:" + "5" * 64,
        "current_reward_status": "active",
    }
    allocation_body = {
        "schema_version": "1.0",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": reward_id,
                "paid_alpha_percent": 5.0,
                "base_desired_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
    }
    allocation = {
        **allocation_body,
        "allocation_hash": sha256_json(allocation_body),
    }

    async def select_all(table, *, filters=(), **_kwargs):
        if table in {
            "research_lab_emission_allocation_current",
            "research_lab_emission_allocation_snapshots",
        }:
            return [
                {
                    "epoch": 100,
                    "netuid": 71,
                    "allocation_hash": allocation["allocation_hash"],
                    "allocation_doc": allocation,
                }
            ]
        if table in {
            "research_lab_arweave_epoch_audit_anchor_current",
            "published_weight_bundles",
        }:
            return []
        status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return [reward] if status == "active" else []

    async def no_finalized_payments(**_kwargs):
        return []

    root_hash = "sha256:" + "6" * 64
    receipt_graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "role": "gateway_coordinator",
                "purpose": "research_lab.reward_decision.v2",
                "output_root": sha256_json(
                    champion_reward_row_projection_v2(reward)
                ),
            }
        ],
    }

    async def load_graph(**_kwargs):
        return receipt_graph

    async def load_graphs(artifacts):
        return {key: receipt_graph for key in artifacts}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_settled_allocation_history_v2",
        no_finalized_payments,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_graphs,
    )

    readiness = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )

    assert readiness["ready"] is True
    assert readiness["required_positive_balance_count"] == 1
    assert readiness["covered_positive_balance_count"] == 1
    assert readiness["required_historical_settlement_count"] == 0
    assert readiness["unproven_historical_allocation_count"] == 1
    assert readiness["unproven_historical_allocations"] == [
        {
            "epoch": 100,
            "allocation_hash": allocation["allocation_hash"],
            "reason": "no_checkpointed_audit_anchor",
        }
    ]


@pytest.mark.asyncio
async def test_cutover_does_not_reclassify_complete_chain_realized_epoch(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store

    reward_id = "champion_reward:sha256:" + "d" * 64
    reward = {
        "champion_reward_id": reward_id,
        "start_epoch": 100,
        "epoch_count": 1,
        "desired_alpha_percent": 5.0,
        "current_reward_status": "active",
    }
    current_allocation = _allocation(paid=5.0)
    current_allocation["champion_allocations"][0]["source_id"] = reward_id
    current_body = {
        key: value
        for key, value in current_allocation.items()
        if key != "allocation_hash"
    }
    current_allocation["allocation_hash"] = sha256_json(current_body)
    settlement_hash = "sha256:" + "e" * 64
    chain_history = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": settlement_hash,
            "allocation_doc": {
                "schema_version": (
                    settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1
                ),
                "epoch": 100,
                "netuid": 71,
                "settlement_hash": settlement_hash,
                "authority_type": (
                    settlement.CHAIN_REALIZED_AUTHORITY_TYPE_V1
                ),
                "source": "chain_realized_obligation_credits",
                "champion_allocations": [
                    {
                        "source_id": reward_id,
                        "champion_reward_id": reward_id,
                        "paid_alpha_percent": 5.0,
                        "base_desired_alpha_percent": 5.0,
                    }
                ],
                "queued_champion_allocations": [],
                "source_add_allocations": [],
                "reimbursement_allocations": [],
            },
            "authority_types": [
                settlement.CHAIN_REALIZED_AUTHORITY_TYPE_V1
            ],
        }
    ]

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_champion_reward_current":
            status = next(
                (
                    value
                    for field, value in filters
                    if field == "current_reward_status"
                ),
                "",
            )
            return [reward] if status == "active" else []
        if table == "research_lab_source_add_reward_current":
            return []
        if table == "research_lab_emission_allocation_current":
            return [
                {
                    "epoch": 100,
                    "netuid": 71,
                    "allocation_hash": current_allocation["allocation_hash"],
                    "allocation_doc": current_allocation,
                }
            ]
        if table in {
            "research_lab_arweave_epoch_audit_anchor_current",
            "published_weight_bundles",
            "research_lab_emission_allocation_snapshots",
        }:
            return []
        raise AssertionError(table)

    async def load_history(**_kwargs):
        return chain_history

    async def no_nonfinalizations(**_kwargs):
        return []

    async def unexpected_graph(**_kwargs):
        raise AssertionError("settled reward must not require another graph")

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_settled_allocation_history_v2",
        load_history,
    )
    monkeypatch.setattr(
        settlement,
        "load_legacy_allocation_nonfinalizations_v2",
        no_nonfinalizations,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        unexpected_graph,
    )

    readiness = await settlement.champion_v2_cutover_readiness(
        epoch=101,
        netuid=71,
    )

    assert readiness["ready"] is True
    assert readiness["zero_balance_active_rows"][0][
        "champion_reward_id"
    ] == reward_id
    assert readiness["missing_historical_settlements"] == []
    assert readiness["unproven_historical_allocations"] == []


@pytest.mark.asyncio
async def test_status_reconciler_only_closes_finalized_chain_balances(monkeypatch):
    from gateway.research_lab import maintenance
    from gateway.research_lab import store
    from gateway.research_lab import allocations

    reward_id = "champion_reward:sha256:" + "7" * 64
    reward = {
        "champion_reward_id": reward_id,
        "miner_uid": 7,
        "desired_alpha_percent": 5.0,
        "epoch_count": 2,
        "current_reward_status": "active",
    }

    async def select_all(_table, *, filters=(), **_kwargs):
        status = next(
            (value for field, value in filters if field == "current_reward_status"),
            "",
        )
        return [reward] if status == "active" else []

    async def fully_settled(**_kwargs):
        return {reward_id: 99.0}

    writes = []

    async def create_event(**kwargs):
        writes.append(kwargs)
        return {"seq": 2, "anchored_hash": "sha256:" + "8" * 64}

    monkeypatch.setattr(maintenance, "select_all", select_all)
    monkeypatch.setattr(
        allocations,
        "_champion_finalized_paid_alpha_to_date",
        fully_settled,
    )
    monkeypatch.setattr(store, "create_champion_reward_event", create_event)

    result = await maintenance.reconcile_champion_reward_statuses(
        epoch=102,
        netuid=71,
        actor_ref="test:reconciler",
        dry_run=False,
    )

    assert result["ok"] is True
    assert result["repaired_count"] == 1
    assert result["repaired"][0]["paid_alpha_percent_to_date"] == 10.0
    assert len(writes) == 1
    assert writes[0]["event_type"] == "paid"
    assert writes[0]["event_doc"]["settlement_authority"] == (
        "finalized_v2_weight_extrinsics"
    )

    async def partially_settled(**_kwargs):
        return {reward_id: 9.999}

    monkeypatch.setattr(
        allocations,
        "_champion_finalized_paid_alpha_to_date",
        partially_settled,
    )
    writes.clear()
    held = await maintenance.reconcile_champion_reward_statuses(
        epoch=102,
        netuid=71,
        dry_run=False,
    )
    assert held["planned_count"] == 0
    assert writes == []


@pytest.mark.asyncio
async def test_default_v2_allocation_path_blocks_incomplete_champion_coverage(
    monkeypatch,
):
    from gateway.research_lab import v2_authority

    async def not_ready(**_kwargs):
        return {
            "ready": False,
            "receipt_coverage": 0.5,
            "missing": [{"champion_reward_id": "champion:missing"}],
        }

    monkeypatch.setattr(
        settlement,
        "champion_v2_cutover_readiness",
        not_ready,
    )
    async def settle_history(**_kwargs):
        return []

    monkeypatch.setattr(
        v2_authority,
        "ensure_chain_realized_settlements_v1",
        settle_history,
    )

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="champion V2 cutover blocked",
    ):
        await v2_authority.build_allocation_v2(
            epoch_id=102,
            netuid=71,
            policy={},
        )


@pytest.mark.asyncio
async def test_source_add_receipt_backfill_is_idempotent_and_measured(monkeypatch):
    from gateway.research_lab import attested_v2_store, maintenance, v2_authority
    from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2

    reward_ref = "source_add_reward:201a08f0d2b503bf"
    reward = {
        "reward_ref": reward_ref,
        "adapter_id": "adapter:uspto-patents-center-api-86bb73c0149e",
        "miner_hotkey": "miner-1",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": 1.0,
        "reward_epochs": 20,
        "start_epoch": 23870,
        "trigger_evidence_doc": {
            "submission_id": "source_add_submission:a3d8f3e562dca636",
            "precheck_status": "provenance_precheck_passed",
            "reward_trigger": "provenance_precheck_passed",
        },
        "public_label": "Source acceptance reward",
        "current_reward_status": "active",
        "created_at": "2026-07-10T00:00:00Z",
    }

    async def select_all(_table, *, filters=(), **_kwargs):
        status = next(
            (value for field, value in filters if field == "current_reward_status"),
            "",
        )
        return [reward] if status == "active" else []

    migrated = []
    state = {"covered": False}
    root_hash = "sha256:" + "e" * 64

    async def load_graph(**_kwargs):
        if not state["covered"]:
            raise RuntimeError("not migrated")
        return {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": sha256_json(
                        source_add_reward_row_projection_v2(
                            "source_add_leg1",
                            {**reward, "initial_reward_status": "active"},
                        )
                    ),
                }
            ],
        }

    async def attest(**kwargs):
        migrated.append(kwargs)
        state["covered"] = True
        return {"execution_receipt": {"receipt_hash": root_hash}}

    monkeypatch.setattr(maintenance, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        v2_authority,
        "attest_historical_source_add_reward_v2",
        attest,
    )

    first = await maintenance.backfill_source_add_reward_v2_authority(
        epoch=24038,
        dry_run=False,
    )
    second = await maintenance.backfill_source_add_reward_v2_authority(
        epoch=24038,
        dry_run=False,
    )

    assert first["migrated_count"] == 1
    assert second["already_covered_count"] == 1
    assert second["migrated_count"] == 0
    assert migrated == [{"epoch_id": 24038, "reward_ref": reward_ref}]


@pytest.mark.asyncio
async def test_champion_receipt_backfill_is_idempotent_and_measured(monkeypatch):
    from gateway.research_lab import attested_v2_store, maintenance, v2_authority
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    reward_id = "champion_reward:sha256:" + "a" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score_bundle:" + "b" * 64,
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "miner_hotkey": "miner-1",
        "miner_uid": 1,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 20,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
        "input_hash": "sha256:" + "c" * 64,
        "anchored_hash": "sha256:" + "d" * 64,
        "current_reward_status": "active",
        "created_at": "2026-07-10T00:00:00Z",
    }

    async def select_all(_table, *, filters=(), **_kwargs):
        status = next(
            (value for field, value in filters if field == "current_reward_status"),
            "",
        )
        return [reward] if status == "active" else []

    migrated = []
    state = {"covered": False}
    root_hash = "sha256:" + "e" * 64

    async def load_graph(**_kwargs):
        if not state["covered"]:
            raise RuntimeError("not migrated")
        return {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": sha256_json(
                        champion_reward_row_projection_v2(reward)
                    ),
                }
            ],
        }

    async def attest(**kwargs):
        migrated.append(kwargs)
        state["covered"] = True
        return {"execution_receipt": {"receipt_hash": root_hash}}

    monkeypatch.setattr(maintenance, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        v2_authority,
        "attest_historical_champion_reward_v2",
        attest,
    )

    first = await maintenance.backfill_champion_reward_v2_authority(
        epoch=102,
        dry_run=False,
    )
    second = await maintenance.backfill_champion_reward_v2_authority(
        epoch=102,
        dry_run=False,
    )

    assert first["migrated_count"] == 1
    assert second["already_covered_count"] == 1
    assert second["migrated_count"] == 0
    assert migrated == [
        {"epoch_id": 102, "champion_reward_id": reward_id}
    ]


@pytest.mark.asyncio
async def test_champion_settlement_backfill_is_dry_run_safe_and_resumable(
    monkeypatch,
):
    from gateway.research_lab import maintenance, v2_authority

    missing = {
        "epoch": 100,
        "allocation_hash": "sha256:" + "1" * 64,
        "reason": "missing_finalized_chain_classification_authority",
    }
    state = {"covered": False}
    calls = []

    async def readiness(**kwargs):
        assert kwargs == {"epoch": 102, "netuid": 71}
        return {
            "ready": state["covered"],
                "missing_historical_classifications": (
                    [] if state["covered"] else [missing]
                ),
        }

    async def classify(**kwargs):
        calls.append(kwargs)
        state["covered"] = True
        return {
            "status": "finalized",
            "result": {"settlement_hash": "sha256:" + "2" * 64},
            "execution_receipt": {"receipt_hash": "sha256:" + "3" * 64},
        }

    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        readiness,
    )
    monkeypatch.setattr(
        v2_authority,
        "classify_historical_champion_allocation_v2",
        classify,
    )
    dry = await maintenance.backfill_champion_settlement_v2_authority(
        epoch=102,
        netuid=71,
        dry_run=True,
    )
    assert dry["planned"] == [
        {"epoch": 100, "allocation_hash": "sha256:" + "1" * 64}
    ]
    assert calls == []

    written = await maintenance.backfill_champion_settlement_v2_authority(
        epoch=102,
        netuid=71,
        dry_run=False,
    )
    assert written["ok"] is True
    assert written["migrated_count"] == 1
    assert written["readiness_after"]["ready"] is True
    assert calls == [
        {"epoch_id": 102, "netuid": 71, "settlement_epoch_id": 100}
    ]

    repeated = await maintenance.backfill_champion_settlement_v2_authority(
        epoch=102,
        netuid=71,
        dry_run=False,
    )
    assert repeated["planned_count"] == 0
    assert repeated["migrated_count"] == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_champion_settlement_backfill_never_migrates_invalid_evidence(
    monkeypatch,
):
    from gateway.research_lab import maintenance

    async def readiness(**_kwargs):
        return {
            "ready": False,
            "missing_historical_settlements": [
                {
                    "epoch": 100,
                    "reason": "finalized_chain_allocation_hash_mismatch",
                }
            ],
        }

    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        readiness,
    )
    result = await maintenance.backfill_champion_settlement_v2_authority(
        epoch=102,
        netuid=71,
        dry_run=False,
    )
    assert result["ok"] is False
    assert result["planned_count"] == 0
    assert result["blocked"][0]["reason"] == (
        "finalized_chain_allocation_hash_mismatch"
    )


@pytest.mark.asyncio
async def test_legacy_v1_paid_helper_keeps_snapshot_accounting(monkeypatch):
    from gateway.research_lab import allocations

    reward_id = "champion_reward:sha256:" + "9" * 64
    calls = []

    async def select_all(table, **kwargs):
        calls.append((table, kwargs))
        return [
            {
                "epoch": 100,
                "allocation_doc": {
                    "champion_allocations": [
                        {
                            "source_id": reward_id,
                            "paid_alpha_percent": 5.0,
                            "base_desired_alpha_percent": 5.0,
                        }
                    ]
                },
            }
        ]

    monkeypatch.setattr(allocations, "select_all", select_all)
    paid = await allocations._champion_paid_alpha_to_date(
        epoch=102,
        netuid=71,
        champion_rows=[
            {
                "champion_reward_id": reward_id,
                "start_epoch": 100,
                "epoch_count": 20,
                "desired_alpha_percent": 5.0,
            }
        ],
    )

    assert paid == {reward_id: 5.0}
    assert calls[0][0] == "research_lab_emission_allocation_current"
    assert calls[0][1]["allow_partial"] is True


@pytest.mark.asyncio
async def test_finalized_paid_helper_uses_settled_chain_realized_history(
    monkeypatch,
):
    from gateway.research_lab import allocations

    reward_id = "champion_reward:sha256:" + "a" * 64

    async def load_settled(**kwargs):
        assert kwargs == {
            "netuid": 71,
            "start_epoch": 100,
            "end_epoch": 100,
        }
        return [
            {
                "epoch": 100,
                "netuid": 71,
                "allocation_doc": {
                    "champion_allocations": [
                        {
                            "source_id": reward_id,
                            "paid_alpha_percent": 5.0,
                            "base_desired_alpha_percent": 5.0,
                            "reason": settlement.CHAIN_REALIZED_AUTHORITY_TYPE_V1,
                        }
                    ],
                    "queued_champion_allocations": [],
                },
                "authority_types": [
                    settlement.CHAIN_REALIZED_AUTHORITY_TYPE_V1
                ],
            }
        ]

    monkeypatch.setattr(
        settlement,
        "load_settled_allocation_history_v2",
        load_settled,
    )

    paid = await allocations._champion_finalized_paid_alpha_to_date(
        epoch=101,
        netuid=71,
        champion_rows=[
            {
                "champion_reward_id": reward_id,
                "start_epoch": 100,
                "epoch_count": 20,
                "desired_alpha_percent": 5.0,
            }
        ],
    )

    assert paid == {reward_id: 5.0}
