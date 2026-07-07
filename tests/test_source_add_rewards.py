"""W6 SOURCE_ADD reward legs: creation, leg-2 trigger gates, rails integration."""

from __future__ import annotations

import pytest

from leadpoet_verifier.economics import allocate_research_lab_epoch
from research_lab.source_add_rewards import (
    REWARD_KIND_SOURCE_ACCEPTANCE,
    REWARD_KIND_SOURCE_IMPLEMENTATION,
    SourceAddRewardState,
    create_leg1_reward,
    create_leg2_reward,
    evaluate_leg2_trigger,
    leg2_expired,
    stop_reward_forward,
    validate_source_add_reward_record,
)


class TestLeg1:
    def test_creation_defaults_match_spec(self):
        record = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:1", start_epoch=100)
        assert record.alpha_percent == 1.0
        assert record.reward_epochs == 20
        assert record.leg == 1
        assert record.reward_kind == REWARD_KIND_SOURCE_ACCEPTANCE
        assert record.public_label == "Source acceptance reward"
        assert validate_source_add_reward_record(record) == []

    def test_one_leg1_per_adapter_ever(self):
        first = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:1", start_epoch=100)
        repeat = create_leg1_reward(
            adapter_id="adapter:a", miner_ref="miner:1", start_epoch=200, existing_rewards=[first.to_dict()]
        )
        assert repeat is None

    def test_queued_when_cap_cannot_pay(self):
        record = create_leg1_reward(
            adapter_id="adapter:b", miner_ref="miner:1", start_epoch=100, state="queued"
        )
        assert record.state == SourceAddRewardState.QUEUED.value
        assert validate_source_add_reward_record(record) == []


def _trigger_kwargs(**overrides):
    kwargs = dict(
        adapter_id="adapter:a",
        catalog_registry_ids=("intentfeed",),
        merged=True,
        merged_diff_routed_registry_ids=("intentfeed",),
        merge_cleared_score_bar=True,
        shadow_monitor_live=True,
        shadow_window_days_elapsed=7.5,
        shadow_window_survived=True,
        ablation_adapter_on_score=6.2,
        ablation_adapter_off_score=5.4,
        now="2026-08-01T00:00:00Z",
        accepted_at="2026-07-06T00:00:00Z",
        market_open_at="2026-07-20T00:00:00Z",
    )
    kwargs.update(overrides)
    return kwargs


class TestLeg2Trigger:
    def test_all_four_conditions_arm_the_trigger(self):
        armed, blockers, evidence = evaluate_leg2_trigger(**_trigger_kwargs())
        assert armed, blockers
        assert evidence["shadow_window_passed"] is True
        assert evidence["ablation_passed"] is True
        assert evidence["ablation_delta_points"] == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "override,expected_blocker",
        [
            ({"merged": False}, "candidate_not_merged"),
            ({"merged_diff_routed_registry_ids": ("other",)}, "diff_does_not_route_to_adapter"),
            ({"merge_cleared_score_bar": False}, "merge_below_score_bar"),
            ({"shadow_monitor_live": False}, "shadow_monitor_not_live"),
            ({"shadow_window_days_elapsed": 3.0}, "shadow_window_not_elapsed"),
            ({"shadow_window_survived": False}, "shadow_window_not_survived"),
            ({"ablation_adapter_on_score": None}, "ablation_not_run"),
            (
                {"ablation_adapter_on_score": 5.7, "ablation_adapter_off_score": 5.4},
                "ablation_attribution_below_threshold",
            ),
        ],
    )
    def test_each_gate_blocks_alone(self, override, expected_blocker):
        armed, blockers, _ = evaluate_leg2_trigger(**_trigger_kwargs(**override))
        assert not armed
        assert expected_blocker in blockers

    def test_leg2_never_created_before_shadow_window_elapses(self):
        armed, blockers, _ = evaluate_leg2_trigger(
            **_trigger_kwargs(shadow_window_days_elapsed=6.9, shadow_window_survived=True)
        )
        assert not armed
        assert "shadow_window_not_elapsed" in blockers

    def test_idempotent_when_leg2_exists(self):
        existing = [{"adapter_id": "adapter:a", "leg": 2}]
        armed, blockers, _ = evaluate_leg2_trigger(**_trigger_kwargs(existing_rewards=existing))
        assert not armed
        assert "leg2_already_created" in blockers

    def test_expiry_clock_anchors_at_market_open_not_acceptance(self):
        # Accepted 2026-01-01, market opened 2026-07-20 → at 2026-08-01 the
        # 6-month clock from market open has NOT elapsed.
        armed, blockers, _ = evaluate_leg2_trigger(
            **_trigger_kwargs(accepted_at="2026-01-01T00:00:00Z", market_open_at="2026-07-20T00:00:00Z")
        )
        assert armed, blockers
        # Without a market open, acceptance anchors the clock → expired.
        armed, blockers, _ = evaluate_leg2_trigger(
            **_trigger_kwargs(accepted_at="2026-01-01T00:00:00Z", market_open_at="")
        )
        assert not armed
        assert "leg2_expired" in blockers


class TestLeg2Expiry:
    def test_expiry_math(self):
        assert not leg2_expired(
            now="2026-12-01T00:00:00Z", accepted_at="2026-07-06T00:00:00Z", expiry_months=6
        )
        assert leg2_expired(
            now="2027-01-07T00:00:00Z", accepted_at="2026-07-06T00:00:00Z", expiry_months=6
        )

    def test_unknown_clocks_never_expire_silently(self):
        assert not leg2_expired(now="", accepted_at="2026-07-06T00:00:00Z")
        assert not leg2_expired(now="2027-01-01T00:00:00Z", accepted_at="")


class TestLeg2Creation:
    def _evidence(self):
        _, _, evidence = evaluate_leg2_trigger(**_trigger_kwargs())
        return evidence

    def test_created_for_adapter_owner_with_spec_defaults(self):
        record = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=500,
            trigger_evidence=self._evidence(),
        )
        assert record.alpha_percent == 5.0
        assert record.reward_epochs == 20
        assert record.miner_ref == "miner:owner"  # owner paid even for house wiring
        assert record.reward_kind == REWARD_KIND_SOURCE_IMPLEMENTATION
        assert validate_source_add_reward_record(record) == []

    def test_one_time_per_adapter(self):
        first = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=500,
            trigger_evidence=self._evidence(),
        )
        repeat = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=501,
            trigger_evidence=self._evidence(),
            existing_rewards=[first.to_dict()],
        )
        assert repeat is None

    def test_creation_without_gate_evidence_raises(self):
        with pytest.raises(ValueError, match="shadow_window_passed"):
            create_leg2_reward(
                adapter_id="adapter:a",
                adapter_owner_miner_ref="miner:owner",
                start_epoch=500,
                trigger_evidence={"ablation_passed": True},
            )

    def test_revert_stops_stream_forward_only(self):
        record = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=500,
            trigger_evidence=self._evidence(),
        )
        stopped = stop_reward_forward(record, reason="implementing patch auto-reverted")
        assert stopped.state == SourceAddRewardState.STOPPED_FORWARD.value
        assert stopped.stopped_reason.startswith("implementing patch")


class TestAllocationRails:
    POLICY = {
        "policy_id": "test-policy",
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
        "champion_min_alpha_percent": 4.0,
        "champion_extra_alpha_percent_per_point": 0.2,
        "champion_max_alpha_percent": 10.0,
        "champion_threshold_points": 1.0,
    }

    def _source_obligation(self, record):
        # The replay-tracking keys mirror what allocations.py builds for
        # production champion obligations (no legacy full-slice overpay).
        row = record.champion_reward_row()
        desired = float(row["desired_alpha_percent"])
        total_due = desired * int(row["epoch_count"])
        return {
            "uid": 7,
            "miner_uid": 7,
            "miner_hotkey": row["miner_hotkey"],
            "source_id": row["champion_reward_id"],
            "champion_reward_id": row["champion_reward_id"],
            "island": row["island"],
            "status": "active",
            "start_epoch": row["start_epoch"],
            "epoch_count": row["epoch_count"],
            "improvement_points": row["improvement_points"],
            "desired_alpha_percent": desired,
            "total_due_alpha_percent": total_due,
            "paid_alpha_percent_to_date": 0.0,
            "remaining_alpha_percent": total_due,
            "reward_kind": row["reward_kind"],
        }

    def test_source_rewards_allocate_with_zero_validator_change(self):
        leg1 = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:owner", start_epoch=100)
        _, _, evidence = evaluate_leg2_trigger(**_trigger_kwargs())
        leg2 = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=100,
            trigger_evidence=evidence,
        )
        allocation = allocate_research_lab_epoch(
            105,
            self.POLICY,
            [],
            [self._source_obligation(leg1), self._source_obligation(leg2)],
        )
        entries = allocation["champion_allocations"]
        assert len(entries) == 2
        by_kind = {entry.get("reward_kind"): entry for entry in entries}
        assert by_kind[REWARD_KIND_SOURCE_ACCEPTANCE]["paid_alpha_percent"] == pytest.approx(1.0)
        assert by_kind[REWARD_KIND_SOURCE_IMPLEMENTATION]["paid_alpha_percent"] == pytest.approx(5.0)

    def test_source_rewards_and_champion_grant_fit_the_cap_concurrently(self):
        # §3.3 cap fit: champion 10% + leg2 5% + leg1 1% = 16% inside 20%.
        leg1 = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:owner", start_epoch=100)
        _, _, evidence = evaluate_leg2_trigger(**_trigger_kwargs())
        leg2 = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=100,
            trigger_evidence=evidence,
        )
        champion = {
            "uid": 9,
            "miner_hotkey": "miner:wirer",
            "source_id": "champion:1",
            "champion_reward_id": "champion:1",
            "island": "generalist",
            "status": "active",
            "start_epoch": 100,
            "epoch_count": 20,
            "improvement_points": 31.0,  # maxes out at the 10% champion cap
            "desired_alpha_percent": 10.0,
            "total_due_alpha_percent": 200.0,
            "paid_alpha_percent_to_date": 0.0,
            "remaining_alpha_percent": 200.0,
        }
        allocation = allocate_research_lab_epoch(
            105,
            self.POLICY,
            [],
            [champion, self._source_obligation(leg1), self._source_obligation(leg2)],
        )
        total = allocation["champion_alpha_percent"]
        assert total == pytest.approx(16.0)
        assert allocation["queued_champion_allocations"] == []
        # Classic champion entries carry no reward_kind — prior shape preserved.
        classic = [e for e in allocation["champion_allocations"] if e["source_id"] == "champion:1"]
        assert classic and "reward_kind" not in classic[0]
