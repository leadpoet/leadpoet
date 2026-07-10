"""SOURCE_ADD reward legs: LLM-only Leg 2 and allocation-rail integration."""

from __future__ import annotations

import pytest

from leadpoet_verifier.economics import allocate_research_lab_epoch
from research_lab.source_add_rewards import (
    REWARD_KIND_SOURCE_ACCEPTANCE,
    REWARD_KIND_SOURCE_IMPLEMENTATION,
    SourceAddRewardState,
    create_leg1_reward,
    create_leg2_reward,
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


def _llm_evidence(**overrides):
    evidence = {
        "llm_judge_passed": True,
        "llm_verdict": "helped",
        "source_used": True,
        "adapter_id": "adapter:a",
        "registry_provider_id": "intentfeed",
        "judge_doc_hash": "sha256:" + "a" * 64,
    }
    evidence.update(overrides)
    return evidence


class TestLeg2Creation:
    def _evidence(self):
        return _llm_evidence()

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

    @pytest.mark.parametrize(
        "evidence",
        [
            {},
            {"llm_judge_passed": False},
            {"llm_judge_passed": 1},
            {"shadow_window_passed": True, "ablation_passed": True},
        ],
    )
    def test_creation_without_exact_llm_pass_evidence_raises(self, evidence):
        with pytest.raises(ValueError, match="llm_judge_passed=true"):
            create_leg2_reward(
                adapter_id="adapter:a",
                adapter_owner_miner_ref="miner:owner",
                start_epoch=500,
                trigger_evidence=evidence,
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
        "research_lab_emission_percent": 30.0,
        "reward_epochs": 20,
        "champion_min_alpha_percent": 7.0,
        "champion_extra_alpha_percent_per_point": 0.3,
        "champion_max_alpha_percent": 15.0,
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
        leg2 = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=100,
            trigger_evidence=_llm_evidence(),
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
        # Cap fit: champion 15% + leg2 5% + leg1 1% = 21% inside 30%.
        leg1 = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:owner", start_epoch=100)
        leg2 = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=100,
            trigger_evidence=_llm_evidence(),
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
            "improvement_points": 31.0,  # maxes out at the 15% champion cap
            "desired_alpha_percent": 15.0,
            "total_due_alpha_percent": 300.0,
            "paid_alpha_percent_to_date": 0.0,
            "remaining_alpha_percent": 300.0,
        }
        allocation = allocate_research_lab_epoch(
            105,
            self.POLICY,
            [],
            [champion, self._source_obligation(leg1), self._source_obligation(leg2)],
        )
        total = allocation["champion_alpha_percent"]
        # Dues fit inside the cap (champion 15 + leg2 5 + leg1 1 = 21); the 9%
        # surplus flows to the genuine champion instead of burning, while the
        # fixed-size SOURCE_ADD legs stay at their owner-set amounts.
        assert total == pytest.approx(30.0)
        assert allocation["unallocated_percent"] == pytest.approx(0.0)
        by_source = {e["source_id"]: e for e in allocation["champion_allocations"]}
        assert by_source["champion:1"]["paid_alpha_percent"] == pytest.approx(24.0)
        assert allocation["queued_champion_allocations"] == []
        # Classic champion entries carry no reward_kind — prior shape preserved.
        classic = [e for e in allocation["champion_allocations"] if e["source_id"] == "champion:1"]
        assert classic and "reward_kind" not in classic[0]
