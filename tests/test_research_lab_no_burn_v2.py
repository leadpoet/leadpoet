from __future__ import annotations

from decimal import Decimal

import pytest

from gateway.research_lab.allocations import (
    _active_champion_obligations,
    _champion_paid_alpha_to_date_from_snapshots,
    _champion_replay_obligation,
    _historical_compute_fallback_from_snapshot,
)
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab import maintenance
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_verifier.economics import allocate_research_lab_epoch
from research_lab import validator_integration


def _policy(
    *,
    lab_cap: float = 30.0,
    conservative: bool = False,
    champ_cap: bool = False,
) -> dict[str, object]:
    return {
        "policy_id": "no-burn-v2-test",
        "enabled": True,
        "research_lab_emission_percent": lab_cap,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
        "reimbursement_max_cost_multiplier_with_champions": 2.0,
        "champion_threshold_points": 1.0,
        "champion_min_alpha_percent": 7.0,
        "champion_extra_alpha_percent_per_point": 0.3,
        "champion_max_alpha_percent": 30.0,
        "champion_placeholder_alpha_percent": 0.0001,
        "champion_queue_trigger_ratio": 0.5,
        "usd_per_0_1_percent_epoch": 1.0,
        "enable_conservative": conservative,
        "enable_champ_cap": champ_cap,
    }


def _champion(
    uid: int,
    *,
    points: float,
    start_epoch: int = 100,
) -> dict[str, object]:
    return {
        "uid": uid,
        "miner_hotkey": "champion-%d" % uid,
        "source_id": "champion_reward:%d" % uid,
        "champion_reward_id": "champion_reward:%d" % uid,
        "island": "generalist",
        "status": "active",
        "start_epoch": start_epoch,
        "epoch_count": 20,
        "improvement_points": points,
    }


def _fallback(
    uid: int,
    compute_microusd: int,
    *,
    source_epoch: int = 99,
) -> dict[str, object]:
    return {
        "uid": uid,
        "miner_hotkey": "compute-%d" % uid,
        "source_id": "historical_compute_fallback:%064d" % uid,
        "island": "historical_compute",
        "status": "active",
        "target_reimbursement_microusd": compute_microusd,
        "fallback_window_start_epoch": source_epoch - 19,
        "fallback_window_end_epoch": source_epoch,
        "source_allocation_epoch": source_epoch,
        "source_allocation_hash": "sha256:" + "a" * 64,
        "contribution_count": 1,
        "contribution_hash": "sha256:" + ("%064x" % uid),
    }


def _champion_rows(allocation: dict[str, object]) -> list[dict[str, object]]:
    return [
        *allocation["champion_allocations"],
        *allocation["queued_champion_allocations"],
    ]


def test_new_config_defaults_are_explicitly_conservative_with_champion_cap(
    monkeypatch,
):
    monkeypatch.delenv("ENABLE_CONSERVATIVE", raising=False)
    monkeypatch.delenv("ENABLE_CHAMP_CAP", raising=False)
    config = ResearchLabGatewayConfig.from_env()

    assert config.enable_conservative is True
    assert config.enable_champ_cap is True
    assert config.lab_reimbursement_max_cost_multiplier_with_champions == 2.0
    policy = config.reimbursement_policy_doc(enabled=True)
    assert policy["enable_conservative"] is True
    assert policy["enable_champ_cap"] is True


def test_historical_policy_without_new_flags_preserves_old_burn_modes():
    policy = _policy()
    policy.pop("enable_conservative")
    policy.pop("enable_champ_cap")

    allocation = allocate_research_lab_epoch(100, policy, [], [])

    assert allocation["unallocated_percent"] == pytest.approx(30.0)
    assert "conservative_mode" not in allocation
    assert "champion_cap_enabled" not in allocation


def test_no_champion_fallback_allocates_full_lab_cap_by_compute():
    allocation = allocate_research_lab_epoch(
        100,
        _policy(),
        [],
        [],
        fallback_reimbursement_obligations=[
            _fallback(1, 1_000_000),
            _fallback(2, 3_000_000),
        ],
    )

    paid = {
        int(item["uid"]): Decimal(str(item["paid_alpha_percent"]))
        for item in allocation["reimbursement_allocations"]
    }
    assert paid == {1: Decimal("7.5"), 2: Decimal("22.5")}
    assert allocation["reimbursement_alpha_percent"] == pytest.approx(30.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)
    assert allocation["historical_compute_fallback_source_epoch"] == 99


def test_non_conservative_mode_fails_closed_without_any_compute_authority():
    with pytest.raises(
        ValueError,
        match="requires historical compute fallback",
    ):
        allocate_research_lab_epoch(100, _policy(), [], [])


def test_current_reimbursement_takes_priority_over_historical_fallback():
    allocation = allocate_research_lab_epoch(
        100,
        _policy(),
        [
            {
                "uid": 3,
                "miner_hotkey": "current-3",
                "source_id": "reimbursement_schedule:current",
                "island": "generalist",
                "status": "active",
                "start_epoch": 100,
                "epoch_count": 20,
                "target_reimbursement_microusd": 20_000_000,
                "eligible_compute_microusd": 20_000_000,
            }
        ],
        [],
        fallback_reimbursement_obligations=[_fallback(1, 1_000_000)],
    )

    assert [row["uid"] for row in allocation["reimbursement_allocations"]] == [3]
    assert allocation["reimbursement_alpha_percent"] == pytest.approx(30.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)
    assert "historical_compute_fallback_source_epoch" not in allocation


def test_champions_split_remaining_pool_by_configured_desired_reward():
    allocation = allocate_research_lab_epoch(
        100,
        _policy(lab_cap=29.0),
        [],
        [
            _champion(7, points=1.0),
            _champion(8, points=5.0),
        ],
        fallback_reimbursement_obligations=[_fallback(1, 1_000_000)],
    )
    paid = {
        int(item["uid"]): Decimal(str(item["paid_alpha_percent"]))
        for item in _champion_rows(allocation)
    }

    assert paid == {
        7: Decimal("13.355263"),
        8: Decimal("15.644737"),
    }
    assert sum(paid.values()) == Decimal("29.000000")
    assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_uncapped_champion_receives_full_pool_for_all_twenty_epochs():
    policy = _policy()
    reward = _champion(7, points=1.0)
    reward["desired_alpha_percent"] = 7.0
    snapshots: list[dict[str, object]] = []

    for epoch in range(100, 120):
        paid_by_reward = _champion_paid_alpha_to_date_from_snapshots(
            snapshots,
            obligation_caps={"champion_reward:7": Decimal("140")},
        )
        replay = _champion_replay_obligation(
            reward,
            paid_by_reward=paid_by_reward,
            epoch=epoch,
            enable_champ_cap=False,
        )
        assert replay is not None
        allocation = allocate_research_lab_epoch(
            epoch,
            policy,
            [],
            [{**reward, **replay}],
            fallback_reimbursement_obligations=[_fallback(1, 1_000_000)],
        )
        rows = _champion_rows(allocation)
        assert sum(float(row["paid_alpha_percent"]) for row in rows) == pytest.approx(
            30.0
        )
        snapshots.append({"epoch": epoch, "allocation_doc": allocation})

    paid_by_reward = _champion_paid_alpha_to_date_from_snapshots(
        snapshots,
        obligation_caps={"champion_reward:7": Decimal("140")},
    )
    assert paid_by_reward["champion_reward:7"] == pytest.approx(140.0)
    assert (
        _champion_replay_obligation(
            reward,
            paid_by_reward=paid_by_reward,
            epoch=120,
            enable_champ_cap=False,
        )
        is None
    )


def test_uncapped_champion_extends_past_twenty_epochs_until_floor_is_paid():
    reward = _champion(7, points=1.0)
    reward["desired_alpha_percent"] = 7.0
    replay = _champion_replay_obligation(
        reward,
        paid_by_reward={"champion_reward:7": 100.0},
        epoch=120,
        enable_champ_cap=False,
    )

    assert replay is not None
    assert replay["replay_status"] == "extended_replay"
    assert replay["remaining_alpha_percent"] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_uncapped_source_reactivates_paid_projection_inside_nominal_window(
    monkeypatch,
):
    from gateway.research_lab import allocations as allocation_source

    reward_row = {
        **_champion(7, points=1.0),
        "current_reward_status": "paid",
        "desired_alpha_percent": 7.0,
    }
    requested_statuses: list[str] = []

    async def select_rows(_table, *, filters, **_kwargs):
        status = str(filters[0][1])
        requested_statuses.append(status)
        return [reward_row] if status == "paid" else []

    async def paid_to_date(**_kwargs):
        return {"champion_reward:7": 140.0}

    async def hotkey_uids(_hotkeys):
        return {"champion-7": 17}

    monkeypatch.setattr(allocation_source, "select_all", select_rows)
    monkeypatch.setattr(
        allocation_source,
        "_champion_paid_alpha_to_date",
        paid_to_date,
    )
    monkeypatch.setattr(allocation_source, "resolve_hotkey_uids", hotkey_uids)

    obligations, skipped = await _active_champion_obligations(
        110,
        netuid=71,
        enable_champ_cap=False,
    )

    assert "paid" in requested_statuses
    assert skipped == []
    assert len(obligations) == 1
    assert obligations[0]["uid"] == 17
    assert obligations[0]["champ_cap_enabled"] is False
    assert obligations[0]["remaining_alpha_percent"] == pytest.approx(0.0)


def test_reimbursement_intended_amount_doubles_with_champion_and_caps_at_2x():
    reimbursement = {
        "uid": 3,
        "miner_hotkey": "current-3",
        "source_id": "reimbursement_schedule:current",
        "island": "generalist",
        "status": "active",
        "start_epoch": 100,
        "epoch_count": 20,
        # Deliberately above eligible compute: the cap is tied to compute,
        # rather than blindly doubling the target reimbursement field.
        "target_reimbursement_microusd": 40_000_000,
        "eligible_compute_microusd": 20_000_000,
    }
    policy = _policy()
    policy["reimbursement_max_cost_multiplier_with_champions"] = 9.0
    with_champion = allocate_research_lab_epoch(
        100,
        policy,
        [reimbursement],
        [_champion(7, points=1.0)],
        fallback_reimbursement_obligations=[_fallback(1, 1_000_000)],
    )
    without_champion = allocate_research_lab_epoch(
        100,
        policy,
        [reimbursement],
        [],
        fallback_reimbursement_obligations=[_fallback(1, 1_000_000)],
    )

    assert with_champion["reimbursement_allocations"][0][
        "intended_alpha_percent"
    ] == pytest.approx(0.2)
    assert without_champion["reimbursement_allocations"][0][
        "intended_alpha_percent"
    ] == pytest.approx(0.1)


def test_champion_cap_surplus_does_not_exceed_current_compute_2x_ceiling():
    policy = _policy(champ_cap=True)
    reimbursement = {
        "uid": 3,
        "miner_hotkey": "current-3",
        "source_id": "reimbursement_schedule:current",
        "island": "generalist",
        "status": "active",
        "start_epoch": 100,
        "epoch_count": 20,
        "target_reimbursement_microusd": 20_000_000,
        "eligible_compute_microusd": 20_000_000,
    }
    champion = {
        **_champion(7, points=1.0),
        "total_due_alpha_percent": 140.0,
        "paid_alpha_percent_to_date": 139.0,
        "remaining_alpha_percent": 1.0,
        "replay_status": "nominal_window",
    }

    allocation = allocate_research_lab_epoch(
        100,
        policy,
        [reimbursement],
        [champion],
        fallback_reimbursement_obligations=[_fallback(1, 1_000_000)],
    )
    by_source = {
        row["source_id"]: row
        for row in allocation["reimbursement_allocations"]
    }

    assert by_source["reimbursement_schedule:current"][
        "paid_alpha_percent"
    ] == pytest.approx(0.2)
    assert by_source[_fallback(1, 1_000_000)["source_id"]][
        "paid_alpha_percent"
    ] == pytest.approx(28.8)
    assert allocation["champion_alpha_percent"] == pytest.approx(1.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_non_conservative_allocation_conserves_every_pool_across_100_epochs():
    caps = (0.000001, 1.0, 30.0, 100.0)
    for offset in range(100):
        cap = caps[offset % len(caps)]
        policy = _policy(lab_cap=cap)
        allocation = allocate_research_lab_epoch(
            1000 + offset,
            policy,
            [],
            [],
            fallback_reimbursement_obligations=[
                _fallback(1, 1_000_000, source_epoch=999),
                _fallback(2, 2_000_000, source_epoch=999),
                _fallback(3, 3_000_000, source_epoch=999),
            ],
        )
        paid = sum(
            Decimal(str(row["paid_alpha_percent"]))
            for row in allocation["reimbursement_allocations"]
        )
        assert paid == Decimal(str(cap)).quantize(Decimal("0.000001"))
        assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_validator_independently_recomputes_historical_fallback_allocation():
    policy = _policy()
    fallback = [_fallback(1, 1_000_000), _fallback(2, 3_000_000)]
    allocation = allocate_research_lab_epoch(
        100,
        policy,
        [],
        [],
        fallback_reimbursement_obligations=fallback,
    )
    source_state = {
        "epoch": 100,
        "netuid": 71,
        "policy_id": policy["policy_id"],
        "policy": policy,
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "fallback_reimbursement_obligations": fallback,
    }
    bundle = {
        "bundle_type": "research_lab_live_allocation_bundle",
        "bundle_id": "bundle:no-burn",
        "epoch": 100,
        "netuid": 71,
        "submission_allowed": True,
        "on_chain_submission_allowed": True,
        "source_state": source_state,
        "source_state_hash": sha256_json(source_state),
        "allocation_doc": allocation,
        "allocation_hash": allocation["allocation_hash"],
    }

    verification = validator_integration.verify_research_lab_allocation_bundle(
        bundle,
        flags=validator_integration.ResearchLabValidatorFlags(
            fetch_enabled=True
        ),
    )

    assert verification["passed"] is True
    assert verification["recomputed_allocation_hash"] == allocation[
        "allocation_hash"
    ]


def _snapshot_row() -> dict[str, object]:
    allocation = {
        "epoch": 100,
        "lab_cap_percent": 30.0,
        "reimbursement_allocations": [
            {
                "uid": 4,
                "miner_hotkey": "hotkey-a",
                "source_id": "reimbursement_schedule:a",
                "spend_microusd": 5_000_000,
                "eligible_compute_microusd": 7_000_000,
                "reason": "surplus_reimbursement_no_burn",
            },
            {
                "uid": 5,
                "miner_hotkey": "hotkey-b",
                "source_id": "reimbursement_schedule:b",
                "spend_microusd": 3_000_000,
                "reason": "full_reimbursement",
            },
        ],
    }
    allocation_hash = sha256_json(allocation)
    allocation["allocation_hash"] = allocation_hash
    return {
        "epoch": 100,
        "netuid": 71,
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation,
    }


@pytest.mark.asyncio
async def test_latest_compute_fallback_is_selected_from_finalized_authority(
    monkeypatch,
):
    from gateway.research_lab import allocations, champion_settlement_v2

    row = _snapshot_row()
    authority = {
        **row,
        "authority_types": ["legacy_finalized_chain_migration_v2"],
        "legacy_settlement_receipt_hash": "sha256:" + "9" * 64,
    }
    queried_tables = []

    async def select_authority(table, **_kwargs):
        queried_tables.append(table)
        if table == allocations.LATEST_LEGACY_COMPUTE_AUTHORITY_TABLE:
            return [{**row, "epoch_id": row["epoch"]}]
        return []

    async def load_history(**kwargs):
        assert kwargs == {
            "netuid": 71,
            "start_epoch": 100,
            "end_epoch": 100,
        }
        return [authority]

    monkeypatch.setattr(allocations, "select_all", select_authority)
    monkeypatch.setattr(
        champion_settlement_v2,
        "load_finalized_allocation_history_v2",
        load_history,
    )

    resolved = await allocations._load_latest_finalized_compute_snapshot_v2(
        epoch=101,
        netuid=71,
    )

    assert resolved == (row, authority)
    assert "research_lab_emission_allocation_current" not in queried_tables


@pytest.mark.asyncio
async def test_readiness_uses_existing_compute_authority_without_classifying(
    monkeypatch,
):
    from gateway.research_lab import allocations, v2_authority

    row = _snapshot_row()
    authority = {
        **row,
        "authority_types": ["legacy_finalized_chain_migration_v2"],
    }

    async def resolve_epoch(_epoch):
        return 101

    async def load_finalized(**_kwargs):
        return row, authority

    async def classify(**_kwargs):
        raise AssertionError("existing authority must not be reclassified")

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve_epoch)
    monkeypatch.setattr(
        ResearchLabGatewayConfig,
        "from_env",
        classmethod(
            lambda cls: ResearchLabGatewayConfig(enable_conservative=False)
        ),
    )
    monkeypatch.setattr(
        allocations,
        "_load_latest_finalized_compute_snapshot_v2",
        load_finalized,
    )
    monkeypatch.setattr(
        v2_authority,
        "classify_historical_champion_allocation_v2",
        classify,
    )

    result = (
        await maintenance.backfill_historical_compute_fallback_v2_authority(
            epoch=101,
            netuid=71,
            dry_run=False,
        )
    )

    assert result["ok"] is True
    assert result["status"] == "already_classified"
    assert result["source_allocation_epoch"] == 100
    assert result["classified_count"] == 0


def test_historical_snapshot_uses_compute_amount_and_current_uid_mapping():
    obligations, skipped, source = _historical_compute_fallback_from_snapshot(
        _snapshot_row(),
        hotkey_uids={"hotkey-a": 14},
        reward_epochs=20,
        expected_netuid=71,
    )

    assert len(obligations) == 1
    assert obligations[0]["uid"] == 14
    assert obligations[0]["target_reimbursement_microusd"] == 7_000_000
    assert source["window_start_epoch"] == 81
    assert source["window_end_epoch"] == 100
    assert skipped == [
        {
            "source_id": "reimbursement_schedule:b",
            "miner_hotkey": "hotkey-b",
            "reason": "miner_hotkey_not_registered",
        }
    ]


def test_historical_snapshot_rejects_recursive_fallback():
    row = _snapshot_row()
    row["allocation_doc"]["historical_compute_fallback_source_epoch"] = 99
    body = {
        key: value
        for key, value in row["allocation_doc"].items()
        if key != "allocation_hash"
    }
    row["allocation_hash"] = sha256_json(body)
    row["allocation_doc"]["allocation_hash"] = row["allocation_hash"]

    with pytest.raises(
        ValueError,
        match="historical compute allocation authority differs",
    ):
        _historical_compute_fallback_from_snapshot(
            row,
            hotkey_uids={"hotkey-a": 14, "hotkey-b": 15},
            reward_epochs=20,
            expected_netuid=71,
        )


@pytest.mark.asyncio
async def test_readiness_backfill_classifies_and_reads_back_exact_source(
    monkeypatch,
):
    from gateway.research_lab import (
        allocations,
        champion_settlement_v2,
        v2_authority,
    )

    row = _snapshot_row()
    observed_filters = []
    authorities: list[dict[str, object]] = []

    async def resolve_epoch(epoch):
        assert epoch == 101
        return 101

    async def select_source(_table, *, filters, **_kwargs):
        observed_filters.extend(filters)
        return [row]

    async def load_history(**_kwargs):
        return list(authorities)

    async def load_finalized_source(**_kwargs):
        if not authorities:
            return None
        return row, authorities[0]

    async def classify(**kwargs):
        assert kwargs == {
            "epoch_id": 101,
            "netuid": 71,
            "settlement_epoch_id": 100,
        }
        authorities.append(
            {
                "epoch": 100,
                "netuid": 71,
                "allocation_hash": row["allocation_hash"],
                "allocation_doc": row["allocation_doc"],
                "authority_types": [
                    "legacy_finalized_chain_migration_v2"
                ],
            }
        )
        return {"status": "finalized"}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve_epoch)
    monkeypatch.setattr(maintenance, "select_all", select_source)
    monkeypatch.setattr(
        ResearchLabGatewayConfig,
        "from_env",
        classmethod(
            lambda cls: ResearchLabGatewayConfig(enable_conservative=False)
        ),
    )
    monkeypatch.setattr(
        champion_settlement_v2,
        "load_finalized_allocation_history_v2",
        load_history,
    )
    monkeypatch.setattr(
        allocations,
        "_load_latest_finalized_compute_snapshot_v2",
        load_finalized_source,
    )
    monkeypatch.setattr(
        v2_authority,
        "classify_historical_champion_allocation_v2",
        classify,
    )

    dry = await maintenance.backfill_historical_compute_fallback_v2_authority(
        epoch=101,
        netuid=71,
        dry_run=True,
    )
    written = (
        await maintenance.backfill_historical_compute_fallback_v2_authority(
            epoch=101,
            netuid=71,
            dry_run=False,
        )
    )
    repeated = (
        await maintenance.backfill_historical_compute_fallback_v2_authority(
            epoch=101,
            netuid=71,
            dry_run=False,
        )
    )

    assert ("epoch", "lt", 101) in observed_filters
    assert dry["status"] == "classification_required"
    assert written["status"] == "finalized"
    assert written["classified_count"] == 1
    assert repeated["status"] == "already_classified"
    assert repeated["classified_count"] == 0


@pytest.mark.asyncio
async def test_readiness_backfill_rejects_nonfinalized_source(monkeypatch):
    from gateway.research_lab import (
        allocations,
        champion_settlement_v2,
        v2_authority,
    )

    async def resolve_epoch(_epoch):
        return 101

    async def select_source(_table, **_kwargs):
        return [_snapshot_row()]

    async def no_history(**_kwargs):
        return []

    async def no_finalized_source(**_kwargs):
        return None

    async def not_finalized(**_kwargs):
        return {"status": "not_finalized"}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve_epoch)
    monkeypatch.setattr(maintenance, "select_all", select_source)
    monkeypatch.setattr(
        ResearchLabGatewayConfig,
        "from_env",
        classmethod(
            lambda cls: ResearchLabGatewayConfig(enable_conservative=False)
        ),
    )
    monkeypatch.setattr(
        champion_settlement_v2,
        "load_finalized_allocation_history_v2",
        no_history,
    )
    monkeypatch.setattr(
        allocations,
        "_load_latest_finalized_compute_snapshot_v2",
        no_finalized_source,
    )
    monkeypatch.setattr(
        v2_authority,
        "classify_historical_champion_allocation_v2",
        not_finalized,
    )

    result = (
        await maintenance.backfill_historical_compute_fallback_v2_authority(
            epoch=101,
            netuid=71,
            dry_run=False,
        )
    )

    assert result["ok"] is False
    assert result["status"] == "not_finalized"
