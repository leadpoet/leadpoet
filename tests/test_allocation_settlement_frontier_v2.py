from __future__ import annotations

import copy
from decimal import Decimal

import pytest

from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    AllocationSettlementFrontierV2Error,
    MAX_REWARD_CHECKPOINTS,
    build_allocation_settlement_frontier_v2,
    build_reward_settlement_checkpoint_v2,
    frontier_artifact_hashes_v2,
    frontier_paid_maps_v2,
    validate_allocation_settlement_frontier_v2,
    validate_frontier_successor_v2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.reward_executor_v2 import (
    champion_reward_row_projection_v2,
    source_add_reward_row_projection_v2,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_verifier.economics import (
    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1,
)


def _checkpoint(*, applied="1.000000", realized="1.000000"):
    applied_amount = Decimal(str(applied))
    realized_amount = Decimal(str(realized))
    return build_reward_settlement_checkpoint_v2(
        reward_kind="champion",
        source_id="champion_reward:" + "a" * 64,
        obligation_hash="sha256:" + "b" * 64,
        start_epoch=100,
        epoch_count=20,
        desired_alpha_percent="7.300000",
        applied_alpha_percent=applied,
        realized_alpha_percent=realized,
        excess_alpha_percent=realized_amount - applied_amount,
    )


def _champion_row(*, status="paid", input_hash=None):
    return {
        "champion_reward_id": "champion_reward:sha256:" + "a" * 64,
        "score_bundle_id": "score-bundle-1",
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "miner_hotkey": "5Champion",
        "miner_uid": 10,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 20,
        "improvement_points": 1.0,
        "threshold_points": 0.0,
        "desired_alpha_percent": 7.3,
        "input_hash": input_hash or "sha256:" + "b" * 64,
        "anchored_hash": "sha256:" + "c" * 64,
        "current_reward_status": status,
    }


def _source_add_row(*, status="stopped_forward", alpha_percent=1.0):
    return {
        "reward_ref": "source_add_reward:" + "d" * 16,
        "adapter_id": "adapter-1",
        "miner_hotkey": "5SourceAdd",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": alpha_percent,
        "reward_epochs": 20,
        "start_epoch": 100,
        "current_reward_status": status,
        "trigger_evidence_doc": {
            "submission_id": "source_add_submission:abcd1234abcd1234"
        },
        "public_label": "Source acceptance",
        "desired_alpha_percent": alpha_percent,
        "epoch_count": 20,
    }


def _checkpoint_for_row(*, reward_kind, row, applied="30", realized="30"):
    if reward_kind == "champion":
        source_id = row["champion_reward_id"]
        projection = champion_reward_row_projection_v2(row)
        desired = row["desired_alpha_percent"]
        epoch_count = row["epoch_count"]
    else:
        source_id = row["reward_ref"]
        projection = source_add_reward_row_projection_v2(
            "source_add_leg%d" % row["leg"],
            {**row, "initial_reward_status": "active"},
        )
        desired = row["alpha_percent"]
        epoch_count = row["reward_epochs"]
    return build_reward_settlement_checkpoint_v2(
        reward_kind=reward_kind,
        source_id=source_id,
        obligation_hash=sha256_json(projection),
        start_epoch=row["start_epoch"],
        epoch_count=epoch_count,
        desired_alpha_percent=desired,
        applied_alpha_percent=applied,
        realized_alpha_percent=realized,
        excess_alpha_percent=Decimal(realized) - Decimal(applied),
    )


def _execution_context():
    return ExecutionContextV2(
        job_id="allocation-v2:retirement-test",
        purpose="research_lab.allocation.v2",
        epoch_id=121,
        parent_receipt_hashes=(),
    )


def test_frontier_is_canonical_and_binds_every_checkpoint():
    checkpoint = _checkpoint()
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(checkpoint,),
    )

    assert validate_allocation_settlement_frontier_v2(frontier) == frontier
    assert frontier_paid_maps_v2(frontier) == {
        "champion": {checkpoint["source_id"]: 1.0},
        "source_add": {},
    }
    assert frontier_artifact_hashes_v2(frontier) == (
        frontier["frontier_hash"],
        checkpoint["checkpoint_hash"],
    )


def test_frontier_successor_rejects_rewind_fork_and_tampering():
    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(_checkpoint(),),
    )
    successor = build_allocation_settlement_frontier_v2(
        mode="bounded_delta_v1",
        netuid=71,
        allocation_epoch=123,
        predecessor_frontier_hash=predecessor["frontier_hash"],
        reward_checkpoints=(_checkpoint(applied="4", realized="4"),),
    )
    assert validate_frontier_successor_v2(predecessor, successor) == (
        predecessor,
        successor,
    )

    fork = copy.deepcopy(successor)
    fork["predecessor_frontier_hash"] = "sha256:" + "c" * 64
    with pytest.raises(AllocationSettlementFrontierV2Error):
        validate_allocation_settlement_frontier_v2(fork)

    tampered = copy.deepcopy(successor)
    tampered["reward_checkpoints"][0]["applied_alpha_percent"] = "5.000000"
    with pytest.raises(AllocationSettlementFrontierV2Error):
        validate_allocation_settlement_frontier_v2(tampered)


def test_frontier_size_remains_bounded_across_one_hundred_epochs():
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(_checkpoint(applied="0", realized="0"),),
    )
    sizes = []
    for epoch in range(101, 201):
        amount = min(146, epoch - 100)
        checkpoint = _checkpoint(applied=str(amount), realized=str(amount))
        frontier = build_allocation_settlement_frontier_v2(
            mode="bounded_delta_v1",
            netuid=71,
            allocation_epoch=epoch,
            predecessor_frontier_hash=frontier["frontier_hash"],
            reward_checkpoints=(checkpoint,),
        )
        sizes.append(len(canonical_json(frontier)))

    assert frontier["reward_checkpoint_count"] == 1
    assert max(sizes) - min(sizes) < 8
    assert len(frontier_artifact_hashes_v2(frontier)) == 2


def test_long_gap_delta_collapses_to_one_cumulative_checkpoint():
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
        CoordinatorAllocationSourceV2Error,
    )

    reward_id = "champion_reward:" + "a" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score-bundle-1",
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "miner_hotkey": "5Champion",
        "miner_uid": 10,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 200,
        "improvement_points": 1.0,
        "threshold_points": 0.0,
        "desired_alpha_percent": 1.0,
        "input_hash": "sha256:" + "b" * 64,
        "anchored_hash": "sha256:" + "c" * 64,
    }
    resolver = object.__new__(CoordinatorAllocationSourceV2)
    predecessor = resolver._build_settlement_frontier(
        epoch=100,
        netuid=71,
        champion_rows=[reward],
        source_add_rows=[],
        history=[],
        predecessor=None,
    )
    history = [
        {
            "epoch": settled_epoch,
            "allocation_doc": {
                "epoch": settled_epoch,
                "champion_credit_policy": (
                    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
                ),
                "champion_allocations": [
                    {
                        "source_id": reward_id,
                        "paid_alpha_percent": 1.0,
                        "champion_credit_policy": (
                            CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
                        ),
                    }
                ],
                "queued_champion_allocations": [],
            },
        }
        for settled_epoch in range(100, 200)
    ]
    frontier = resolver._build_settlement_frontier(
        epoch=200,
        netuid=71,
        champion_rows=[reward],
        source_add_rows=[],
        history=history,
        predecessor=predecessor,
    )

    assert frontier["settled_through_epoch"] == 199
    assert frontier["reward_checkpoint_count"] == 1
    assert frontier["reward_checkpoints"][0]["applied_alpha_percent"] == (
        "100.000000"
    )
    assert len(canonical_json(frontier)) < 2_000


def test_maximum_active_frontier_has_fixed_artifact_and_wire_bounds():
    checkpoints = [
        build_reward_settlement_checkpoint_v2(
            reward_kind="champion",
            source_id="champion_reward:%064x" % index,
            obligation_hash="sha256:%064x" % index,
            start_epoch=100,
            epoch_count=20,
            desired_alpha_percent=1,
            applied_alpha_percent=0,
            realized_alpha_percent=0,
            excess_alpha_percent=0,
        )
        for index in range(MAX_REWARD_CHECKPOINTS)
    ]
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=checkpoints,
    )

    assert frontier["reward_checkpoint_count"] == MAX_REWARD_CHECKPOINTS
    assert len(frontier_artifact_hashes_v2(frontier)) == MAX_REWARD_CHECKPOINTS + 1
    assert len(canonical_json(frontier).encode("utf-8")) < 1024 * 1024


def test_frontier_enforces_active_obligation_bound():
    checkpoints = []
    for index in range(MAX_REWARD_CHECKPOINTS + 1):
        checkpoints.append(
            build_reward_settlement_checkpoint_v2(
                reward_kind="champion",
                source_id="champion_reward:%064x" % index,
                obligation_hash="sha256:%064x" % index,
                start_epoch=100,
                epoch_count=1,
                desired_alpha_percent=1,
                applied_alpha_percent=0,
                realized_alpha_percent=0,
                excess_alpha_percent=0,
            )
        )

    with pytest.raises(
        AllocationSettlementFrontierV2Error,
        match="active-obligation bound",
    ):
        build_allocation_settlement_frontier_v2(
            mode="legacy_full_history_bootstrap",
            netuid=71,
            allocation_epoch=100,
            predecessor_frontier_hash=None,
            reward_checkpoints=checkpoints,
        )


def test_checkpoint_rejects_understated_applied_lifetime_credit():
    with pytest.raises(
        AllocationSettlementFrontierV2Error,
        match="applied reward differs from realized lifetime credit",
    ):
        build_reward_settlement_checkpoint_v2(
            reward_kind="champion",
            source_id="champion_reward:" + "a" * 64,
            obligation_hash="sha256:" + "b" * 64,
            start_epoch=100,
            epoch_count=20,
            desired_alpha_percent="7.300000",
            applied_alpha_percent="30.000000",
            realized_alpha_percent="31.000000",
            excess_alpha_percent="1.000000",
        )


def test_unsettled_reward_cannot_disappear_from_successor_frontier():
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
        CoordinatorAllocationSourceV2Error,
    )

    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(_checkpoint(applied="30", realized="30"),),
    )
    resolver = object.__new__(CoordinatorAllocationSourceV2)

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="unsettled reward disappeared",
    ):
        resolver._build_settlement_frontier(
            epoch=121,
            netuid=71,
            champion_rows=[],
            source_add_rows=[],
            history=[],
            predecessor=predecessor,
        )


def test_terminal_paid_champion_retires_with_hash_bound_evidence(monkeypatch):
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
    )

    row = _champion_row()
    checkpoint = _checkpoint_for_row(reward_kind="champion", row=row)
    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(checkpoint,),
    )
    resolver = object.__new__(CoordinatorAllocationSourceV2)
    calls = []

    def read(policy_id, parameters, _context):
        calls.append((policy_id, parameters))
        return [row]

    monkeypatch.setattr(resolver, "_read", read)
    retirements = resolver._resolve_settlement_frontier_retirements(
        predecessor=predecessor,
        champion_rows=[],
        source_add_rows=[],
        context=_execution_context(),
    )
    successor = resolver._build_settlement_frontier(
        epoch=121,
        netuid=71,
        champion_rows=[],
        source_add_rows=[],
        history=[],
        predecessor=predecessor,
        terminal_retirements=retirements,
    )

    assert calls == [
        ("champion_reward_by_id", {"champion_reward_id": row["champion_reward_id"]})
    ]
    assert retirements[0]["terminal_status"] == "paid"
    assert retirements[0]["obligation_hash"] == checkpoint["obligation_hash"]
    assert retirements[0]["predecessor_checkpoint_hash"] == checkpoint[
        "checkpoint_hash"
    ]
    assert successor["reward_checkpoint_count"] == 0


def test_terminal_source_add_retires_with_hash_bound_evidence(monkeypatch):
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
    )

    row = _source_add_row()
    checkpoint = _checkpoint_for_row(
        reward_kind="source_add",
        row=row,
        applied="10",
        realized="10",
    )
    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(checkpoint,),
    )
    resolver = object.__new__(CoordinatorAllocationSourceV2)

    monkeypatch.setattr(
        resolver,
        "_read",
        lambda policy_id, parameters, _context: [row],
    )
    retirements = resolver._resolve_settlement_frontier_retirements(
        predecessor=predecessor,
        champion_rows=[],
        source_add_rows=[],
        context=_execution_context(),
    )
    successor = resolver._build_settlement_frontier(
        epoch=121,
        netuid=71,
        champion_rows=[],
        source_add_rows=[],
        history=[],
        predecessor=predecessor,
        terminal_retirements=retirements,
    )

    assert retirements[0]["terminal_status"] == "stopped_forward"
    assert successor["reward_checkpoint_count"] == 0


@pytest.mark.parametrize(
    ("row", "error"),
    [
        (_champion_row(status="active"), "reward is not terminal"),
        (
            _champion_row(input_hash="sha256:" + "e" * 64),
            "terminal reward identity changed",
        ),
    ],
)
def test_terminal_retirement_fails_closed_on_status_or_identity(
    monkeypatch,
    row,
    error,
):
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
        CoordinatorAllocationSourceV2Error,
    )

    original = _champion_row()
    checkpoint = _checkpoint_for_row(reward_kind="champion", row=original)
    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(checkpoint,),
    )
    resolver = object.__new__(CoordinatorAllocationSourceV2)
    monkeypatch.setattr(
        resolver,
        "_read",
        lambda policy_id, parameters, _context: [row],
    )

    with pytest.raises(CoordinatorAllocationSourceV2Error, match=error):
        resolver._resolve_settlement_frontier_retirements(
            predecessor=predecessor,
            champion_rows=[],
            source_add_rows=[],
            context=_execution_context(),
        )


def test_terminal_retirement_cannot_be_replayed_for_an_active_reward(monkeypatch):
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
        CoordinatorAllocationSourceV2Error,
    )

    terminal_row = _champion_row()
    active_row = _champion_row(status="active")
    checkpoint = _checkpoint_for_row(reward_kind="champion", row=terminal_row)
    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(checkpoint,),
    )
    resolver = object.__new__(CoordinatorAllocationSourceV2)
    monkeypatch.setattr(
        resolver,
        "_read",
        lambda policy_id, parameters, _context: [terminal_row],
    )
    retirements = resolver._resolve_settlement_frontier_retirements(
        predecessor=predecessor,
        champion_rows=[],
        source_add_rows=[],
        context=_execution_context(),
    )

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="active reward has terminal settlement evidence",
    ):
        resolver._build_settlement_frontier(
            epoch=121,
            netuid=71,
            champion_rows=[active_row],
            source_add_rows=[],
            history=[],
            predecessor=predecessor,
            terminal_retirements=retirements,
        )


def test_fully_settled_reward_retires_from_successor_frontier():
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
    )

    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(_checkpoint(applied="146", realized="146"),),
    )
    resolver = object.__new__(CoordinatorAllocationSourceV2)

    successor = resolver._build_settlement_frontier(
        epoch=121,
        netuid=71,
        champion_rows=[],
        source_add_rows=[],
        history=[],
        predecessor=predecessor,
    )

    assert successor["mode"] == "bounded_delta_v1"
    assert successor["predecessor_frontier_hash"] == predecessor["frontier_hash"]
    assert successor["reward_checkpoint_count"] == 0


def test_coordinator_frontier_stays_bounded_and_retires_across_one_hundred_epochs():
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
    )

    reward_id = "champion_reward:" + "a" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score-bundle-1",
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "miner_hotkey": "5Champion",
        "miner_uid": 10,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 20,
        "improvement_points": 1.0,
        "threshold_points": 0.0,
        "desired_alpha_percent": 7.3,
        "input_hash": "sha256:" + "b" * 64,
        "anchored_hash": "sha256:" + "c" * 64,
    }
    payments = {100: 30.0, 101: 30.0, 102: 30.0, 103: 30.0, 104: 26.0}
    resolver = object.__new__(CoordinatorAllocationSourceV2)
    frontier = None
    encoded_sizes = []

    for epoch in range(100, 200):
        previous_epoch = epoch - 1
        paid = payments.get(previous_epoch)
        history = []
        if paid is not None:
            history = [
                {
                    "epoch": previous_epoch,
                    "allocation_doc": {
                        "epoch": previous_epoch,
                        "champion_credit_policy": (
                            CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
                        ),
                        "champion_allocations": [
                            {
                                "source_id": reward_id,
                                "paid_alpha_percent": paid,
                                "champion_credit_policy": (
                                    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
                                ),
                            }
                        ],
                        "queued_champion_allocations": [],
                    },
                }
            ]
        frontier = resolver._build_settlement_frontier(
            epoch=epoch,
            netuid=71,
            champion_rows=[reward] if epoch <= 105 else [],
            source_add_rows=[],
            history=history,
            predecessor=frontier,
        )
        encoded_sizes.append(len(canonical_json(frontier)))
        assert frontier["reward_checkpoint_count"] <= 1

    assert frontier is not None
    assert frontier["allocation_epoch"] == 199
    assert frontier["reward_checkpoint_count"] == 0
    assert max(encoded_sizes) < 2_000
    assert len(set(encoded_sizes[6:])) == 1
