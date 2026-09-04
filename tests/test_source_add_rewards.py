"""SOURCE_ADD reward legs: LLM-only Leg 2 and allocation-rail integration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from leadpoet_verifier.economics import (
    _canonical_source_add_created_at,
    allocate_research_lab_epoch,
)
from research_lab.canonical import sha256_json
from research_lab.source_add_rewards import (
    REWARD_KIND_SOURCE_ACCEPTANCE,
    REWARD_KIND_SOURCE_IMPLEMENTATION,
    SourceAddRewardState,
    create_leg1_reward,
    create_leg2_reward,
    stop_reward_forward,
    validate_source_add_reward_record,
)
from research_lab.validator_integration import verify_research_lab_allocation_bundle


def _allocation_authority_outcome(
    *,
    epoch: int,
    netuid: int,
    policy: dict,
    source_obligations: list[dict],
    include_source: bool,
) -> dict:
    allocation_inputs = {
        "epoch": int(epoch),
        "policy": dict(policy),
        "active_reimbursement_obligations": [],
        "active_champion_obligations": [],
    }
    if include_source:
        allocation_inputs["active_source_add_obligations"] = list(source_obligations)
    allocation = allocate_research_lab_epoch(
        int(epoch),
        policy,
        [],
        [],
        active_source_add_obligations=(source_obligations if include_source else []),
    )
    source_state = {
        "epoch": int(epoch),
        "netuid": int(netuid),
        "policy_id": str(policy["policy_id"]),
        "policy": dict(policy),
        "reimbursement_obligation_count": 0,
        "champion_obligation_count": 0,
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "skipped": {"reimbursements": [], "champions": []},
    }
    if include_source:
        source_state.update(
            {
                "source_add_obligation_count": len(source_obligations),
                "source_add_obligations": list(source_obligations),
            }
        )
        source_state["skipped"]["source_add"] = []
    return {
        "status": "matched",
        "result": {
            "allocation": allocation,
            "allocation_inputs": allocation_inputs,
            "source_state": source_state,
            "source_state_hash": sha256_json(source_state),
        },
    }


class TestLeg1:
    def test_creation_defaults_match_spec(self):
        record = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:1", start_epoch=100)
        assert record.alpha_percent == 0.2
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
        # The replay-tracking keys mirror the first-class SOURCE_ADD obligation
        # produced by allocations.py.
        row = record.champion_reward_row()
        desired = float(row["desired_alpha_percent"])
        total_due = desired * int(row["epoch_count"])
        return {
            "uid": 7,
            "miner_uid": 7,
            "miner_hotkey": row["miner_hotkey"],
            "source_id": row["champion_reward_id"],
            "source_add_reward_id": row["source_add_reward_id"],
            "adapter_id": row["adapter_id"],
            "leg": row["leg"],
            "created_at": "2026-08-01T00:00:00Z",
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

    @pytest.mark.parametrize(
        ("timestamp", "expected"),
        [
            ("2026-09-02T22:57:50.9+00:00", "2026-09-02T22:57:50.900000Z"),
            ("2026-09-02T22:57:50.92+00:00", "2026-09-02T22:57:50.920000Z"),
            ("2026-09-02T22:57:50.926+00:00", "2026-09-02T22:57:50.926000Z"),
            ("2026-09-02T22:57:50.9266+00:00", "2026-09-02T22:57:50.926600Z"),
            ("2026-09-02T22:57:50.92667+00:00", "2026-09-02T22:57:50.926670Z"),
            ("2026-09-02T22:57:50.926670Z", "2026-09-02T22:57:50.926670Z"),
            ("2026-09-02T22:57:50.926670999Z", "2026-09-02T22:57:50.926670Z"),
            ("2026-09-03T04:27:50.92667+05:30", "2026-09-02T22:57:50.926670Z"),
        ],
    )
    def test_source_add_fifo_timestamp_is_python39_safe(self, timestamp, expected):
        assert _canonical_source_add_created_at(timestamp) == expected

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2026-09-02 22:57:50.92667+00:00",
            "2026-09-02T22:57:50.92667",
            "2026-09-02T22:57:50.92667z",
            "2026-09-02T22:57:50,92667Z",
            "2026-09-02T22:57:50.92667+0000",
            "2026-09-02T22:57:50.92667-00:00",
            "2026-09-02T22:57:50.1234567890Z",
            "2026-02-30T22:57:50.92667Z",
            "2026-09-02T24:00:00Z",
            "2026-09-02T22:57:50.92667+24:00",
        ],
    )
    def test_source_add_fifo_timestamp_rejects_malformed_values(self, timestamp):
        with pytest.raises(ValueError, match="source add created_at is invalid"):
            _canonical_source_add_created_at(timestamp)

    def test_first_live_source_add_allocation_accepts_postgrest_fraction_width(self):
        source = create_leg1_reward(
            adapter_id="adapter:production-timestamp",
            miner_ref="miner:owner",
            start_epoch=100,
        )
        obligation = self._source_obligation(source)
        obligation["created_at"] = "2026-09-02T22:57:50.92667+00:00"

        allocation = allocate_research_lab_epoch(
            100,
            self.POLICY,
            [],
            [],
            active_source_add_obligations=[obligation],
        )

        assert allocation["source_add_alpha_percent"] == pytest.approx(0.2)

    def test_source_rewards_are_first_class_and_fixed_size(self):
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
            [],
            active_source_add_obligations=[
                self._source_obligation(leg1),
                self._source_obligation(leg2),
            ],
        )
        entries = allocation["source_add_allocations"]
        assert len(entries) == 2
        by_kind = {entry.get("reward_kind"): entry for entry in entries}
        assert by_kind[REWARD_KIND_SOURCE_ACCEPTANCE]["paid_alpha_percent"] == pytest.approx(0.2)
        assert by_kind[REWARD_KIND_SOURCE_IMPLEMENTATION]["paid_alpha_percent"] == pytest.approx(5.0)
        assert all(
            entry["source_add_reward_id"] == entry["source_id"]
            for entry in entries
        )
        assert allocation["source_add_alpha_percent"] == pytest.approx(5.2)
        assert allocation["champion_reimbursement_cap_percent"] == pytest.approx(24.8)
        assert allocation["champion_allocations"] == []
        assert allocation["unallocated_percent"] == pytest.approx(24.8)

    def test_source_add_exhausts_cap_before_existing_allocator(self):
        leg2 = create_leg2_reward(
            adapter_id="adapter:a",
            adapter_owner_miner_ref="miner:owner",
            start_epoch=100,
            trigger_evidence=_llm_evidence(),
        )
        champion = {
            "uid": 9,
            "miner_hotkey": "miner:champion",
            "source_id": "champion:no-remaining-cap",
            "status": "active",
            "start_epoch": 100,
            "epoch_count": 20,
            "improvement_points": 5.0,
            "desired_alpha_percent": 8.0,
        }
        policy = {**self.POLICY, "research_lab_emission_percent": 3.0}

        allocation = allocate_research_lab_epoch(
            105,
            policy,
            [],
            [champion],
            active_source_add_obligations=[self._source_obligation(leg2)],
        )
        expected_existing = allocate_research_lab_epoch(
            105,
            {**policy, "research_lab_emission_percent": 0.0},
            [],
            [champion],
        )

        assert allocation["source_add_alpha_percent"] == pytest.approx(3.0)
        assert allocation["source_add_deferred_alpha_percent"] == pytest.approx(2.0)
        assert allocation["champion_reimbursement_cap_percent"] == pytest.approx(0.0)
        assert allocation["champion_allocations"] == expected_existing["champion_allocations"]
        assert allocation["queued_champion_allocations"] == expected_existing["queued_champion_allocations"]

    def test_source_rewards_are_deducted_before_unchanged_champion_logic(self):
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
            [champion],
            active_source_add_obligations=[
                self._source_obligation(leg1),
                self._source_obligation(leg2),
            ],
        )
        expected_existing = allocate_research_lab_epoch(
            105,
            {**self.POLICY, "research_lab_emission_percent": 24.8},
            [],
            [champion],
        )
        assert allocation["source_add_alpha_percent"] == pytest.approx(5.2)
        assert allocation["champion_reimbursement_cap_percent"] == pytest.approx(24.8)
        for key in (
            "reimbursement_allocations",
            "champion_allocations",
            "queued_champion_allocations",
            "reimbursement_alpha_percent",
            "champion_alpha_percent",
            "queued_champion_alpha_percent",
            "unallocated_percent",
        ):
            assert allocation[key] == expected_existing[key]
        assert allocation["champion_alpha_percent"] == pytest.approx(24.8)
        assert allocation["unallocated_percent"] == pytest.approx(0.0)
        by_source = {e["source_id"]: e for e in allocation["champion_allocations"]}
        assert by_source["champion:1"]["paid_alpha_percent"] == pytest.approx(24.8)
        assert allocation["queued_champion_allocations"] == []
        classic = [e for e in allocation["champion_allocations"] if e["source_id"] == "champion:1"]
        assert classic and "reward_kind" not in classic[0]

    @pytest.mark.parametrize(
        ("configured_cap", "expected_remaining"),
        [(30.0, 29.6), (41.0, 40.6)],
    )
    def test_two_source_acceptance_rewards_reduce_configured_cap_by_point_four(
        self,
        configured_cap,
        expected_remaining,
    ):
        first = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:a", start_epoch=100)
        second = create_leg1_reward(adapter_id="adapter:b", miner_ref="miner:b", start_epoch=100)
        first_obligation = {**self._source_obligation(first), "uid": 7, "miner_uid": 7}
        second_obligation = {**self._source_obligation(second), "uid": 8, "miner_uid": 8}
        reimbursement = {
            "uid": 9,
            "miner_hotkey": "miner:reimbursed",
            "source_id": "reimbursement:1",
            "island": "generalist",
            "start_epoch": 100,
            "epoch_count": 20,
            "target_reimbursement_microusd": 1_000_000,
        }

        allocation = allocate_research_lab_epoch(
            105,
            {**self.POLICY, "research_lab_emission_percent": configured_cap},
            [reimbursement],
            [],
            active_source_add_obligations=[first_obligation, second_obligation],
        )
        expected_existing = allocate_research_lab_epoch(
            105,
            {**self.POLICY, "research_lab_emission_percent": expected_remaining},
            [reimbursement],
            [],
        )

        assert allocation["lab_cap_percent"] == pytest.approx(configured_cap)
        assert allocation["source_add_alpha_percent"] == pytest.approx(0.4)
        assert allocation["champion_reimbursement_cap_percent"] == pytest.approx(expected_remaining)
        assert allocation["reimbursement_allocations"] == expected_existing["reimbursement_allocations"]
        assert allocation["reimbursement_alpha_percent"] == expected_existing["reimbursement_alpha_percent"]

    def test_same_start_source_rewards_use_creation_time_before_reward_hash(self):
        older = self._source_obligation(
            create_leg1_reward(
                adapter_id="adapter:older",
                miner_ref="miner:older",
                start_epoch=100,
            )
        )
        newer = self._source_obligation(
            create_leg1_reward(
                adapter_id="adapter:newer",
                miner_ref="miner:newer",
                start_epoch=100,
            )
        )
        older.update(
            {
                "uid": 7,
                "miner_uid": 7,
                "source_id": "source_add_reward:ffffffffffffffff",
                "source_add_reward_id": "source_add_reward:ffffffffffffffff",
                "created_at": "2026-08-01T00:00:00Z",
            }
        )
        newer.update(
            {
                "uid": 8,
                "miner_uid": 8,
                "source_id": "source_add_reward:0000000000000000",
                "source_add_reward_id": "source_add_reward:0000000000000000",
                "created_at": "2026-08-01T00:00:01Z",
            }
        )

        allocation = allocate_research_lab_epoch(
            105,
            {**self.POLICY, "research_lab_emission_percent": 0.2},
            [],
            [],
            active_source_add_obligations=[newer, older],
        )

        assert [
            row["source_add_reward_id"]
            for row in allocation["source_add_allocations"]
        ] == [
            "source_add_reward:ffffffffffffffff",
            "source_add_reward:0000000000000000",
        ]
        assert [
            row["paid_alpha_percent"]
            for row in allocation["source_add_allocations"]
        ] == pytest.approx([0.2, 0.0])

    def test_combined_champion_and_reimbursement_sections_match_legacy_reduced_cap(self):
        source = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:a", start_epoch=100)
        reimbursement = {
            "uid": 8,
            "miner_hotkey": "miner:reimbursed",
            "source_id": "reimbursement:combined",
            "island": "generalist",
            "start_epoch": 100,
            "epoch_count": 20,
            "target_reimbursement_microusd": 50_000_000,
        }
        champion = {
            "uid": 9,
            "miner_hotkey": "miner:champion",
            "source_id": "champion:combined",
            "status": "active",
            "start_epoch": 100,
            "epoch_count": 20,
            "improvement_points": 5.0,
            "desired_alpha_percent": 8.0,
        }

        allocation = allocate_research_lab_epoch(
            105,
            self.POLICY,
            [reimbursement],
            [champion],
            active_source_add_obligations=[self._source_obligation(source)],
        )
        expected_existing = allocate_research_lab_epoch(
            105,
            {**self.POLICY, "research_lab_emission_percent": 29.8},
            [reimbursement],
            [champion],
        )

        for key in (
            "reimbursement_allocations",
            "champion_allocations",
            "queued_champion_allocations",
            "reimbursement_alpha_percent",
            "champion_alpha_percent",
            "queued_champion_alpha_percent",
            "unallocated_percent",
        ):
            assert allocation[key] == expected_existing[key]

    def test_no_source_rewards_preserve_legacy_output_shape_and_hash(self):
        champion = {
            "uid": 9,
            "miner_hotkey": "miner:champion",
            "source_id": "champion:legacy",
            "status": "active",
            "start_epoch": 100,
            "epoch_count": 20,
            "improvement_points": 31.0,
            "desired_alpha_percent": 15.0,
        }
        implicit = allocate_research_lab_epoch(105, self.POLICY, [], [champion])
        explicit = allocate_research_lab_epoch(
            105,
            self.POLICY,
            [],
            [champion],
            active_source_add_obligations=[],
        )

        assert implicit == explicit
        assert "source_add_allocations" not in implicit

    def test_validator_recomputes_separate_source_add_allocation(self):
        source = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:a", start_epoch=100)
        source_obligation = self._source_obligation(source)
        allocation = allocate_research_lab_epoch(
            105,
            self.POLICY,
            [],
            [],
            active_source_add_obligations=[source_obligation],
        )
        source_state = {
            "epoch": 105,
            "netuid": 71,
            "policy_id": self.POLICY["policy_id"],
            "policy": self.POLICY,
            "reimbursement_obligations": [],
            "champion_obligations": [],
            "source_add_obligations": [source_obligation],
        }
        bundle = {
            "bundle_type": "research_lab_live_allocation_bundle",
            "bundle_id": "research_lab_allocation_bundle:test",
            "epoch": 105,
            "netuid": 71,
            "submission_allowed": True,
            "on_chain_submission_allowed": True,
            "source_state": source_state,
            "source_state_hash": sha256_json(source_state),
            "allocation_doc": allocation,
            "allocation_hash": allocation["allocation_hash"],
        }

        verification = verify_research_lab_allocation_bundle(
            bundle,
            flags={"fetch_enabled": True, "reimbursements_enabled": True},
        )
        assert verification["passed"], verification["errors"]

    @pytest.mark.asyncio
    async def test_attested_allocator_uses_the_same_source_add_contract(self):
        from gateway.tee.scoring_executor import execute_scoring_operation

        source = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:a", start_epoch=100)
        source_obligation = self._source_obligation(source)
        outcome = await execute_scoring_operation(
            "research_lab_allocation",
            {
                "epoch": 105,
                "policy": self.POLICY,
                "active_reimbursement_obligations": [],
                "active_champion_obligations": [],
                "active_source_add_obligations": [source_obligation],
            },
        )

        allocation = outcome.result["allocation"]
        assert allocation["source_add_alpha_percent"] == pytest.approx(0.2)
        assert allocation["champion_reimbursement_cap_percent"] == pytest.approx(29.8)
        assert outcome.evidence_roots == {"allocation": allocation["allocation_hash"]}

    @pytest.mark.asyncio
    async def test_gateway_bundle_keeps_source_obligations_out_of_champion_inputs(self, monkeypatch):
        from gateway.research_lab import allocations as gateway_allocations

        _disable_durable_allocation_retry(monkeypatch, gateway_allocations)
        source = create_leg1_reward(adapter_id="adapter:a", miner_ref="miner:a", start_epoch=100)
        source_obligation = self._source_obligation(source)
        monkeypatch.setattr(
            gateway_allocations,
            "build_allocation_v2",
            lambda **_kwargs: _async_value(
                _allocation_authority_outcome(
                    epoch=105,
                    netuid=71,
                    policy=dict(self.POLICY),
                    source_obligations=[source_obligation],
                    include_source=True,
                )
            ),
        )
        config = SimpleNamespace(
            reimbursement_policy_doc=lambda enabled: dict(self.POLICY),
            reimbursement_dynamic_alpha_price_enabled=False,
            reimbursement_require_live_alpha_price=False,
            reimbursement_miner_alpha_per_epoch=0.0,
            reimbursement_usd_per_0_1_percent_epoch=1.0,
            reimbursements_enabled=True,
            weight_mutation_enabled=True,
            production_writes_enabled=False,
        )

        bundle = await gateway_allocations.build_research_lab_allocation_bundle(
            config=config,
            epoch=105,
            netuid=71,
        )

        assert bundle["source_state"]["source_add_obligations"] == [source_obligation]
        assert bundle["source_state"]["champion_obligations"] == []
        assert bundle["allocation_doc"]["source_add_alpha_percent"] == pytest.approx(0.2)
        assert bundle["allocation_doc"]["champion_allocations"] == []

    @pytest.mark.asyncio
    async def test_gateway_bundle_without_source_preserves_legacy_input_shape(self, monkeypatch):
        from gateway.research_lab import allocations as gateway_allocations

        _disable_durable_allocation_retry(monkeypatch, gateway_allocations)
        authority = _allocation_authority_outcome(
            epoch=105,
            netuid=71,
            policy=dict(self.POLICY),
            source_obligations=[],
            include_source=False,
        )
        monkeypatch.setattr(
            gateway_allocations,
            "build_allocation_v2",
            lambda **_kwargs: _async_value(authority),
        )
        config = SimpleNamespace(
            reimbursement_policy_doc=lambda enabled: dict(self.POLICY),
            reimbursement_dynamic_alpha_price_enabled=False,
            reimbursement_require_live_alpha_price=False,
            reimbursement_miner_alpha_per_epoch=0.0,
            reimbursement_usd_per_0_1_percent_epoch=1.0,
            reimbursements_enabled=True,
            weight_mutation_enabled=True,
            production_writes_enabled=False,
        )

        bundle = await gateway_allocations.build_research_lab_allocation_bundle(
            config=config,
            epoch=105,
            netuid=71,
        )

        assert "active_source_add_obligations" not in authority["result"]["allocation_inputs"]
        assert "source_add_obligations" not in bundle["source_state"]
        assert "source_add_allocations" not in bundle["allocation_doc"]
        assert "source_add_alpha_percent" not in bundle["observability"]


@pytest.mark.asyncio
async def test_gateway_allocation_coalesces_only_overlapping_authority_builds(
    monkeypatch,
):
    from gateway.research_lab import allocations as gateway_allocations

    _disable_durable_allocation_retry(monkeypatch, gateway_allocations)
    policy = {
        "policy_id": "research_lab_reimbursement_policy_v1",
        "lab_cap_alpha_percent": 30.0,
        "reimbursement_epochs": 20,
    }
    authority = _allocation_authority_outcome(
        epoch=105,
        netuid=71,
        policy=policy,
        source_obligations=[],
        include_source=False,
    )
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def build(**_kwargs):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return authority

    monkeypatch.setattr(gateway_allocations, "build_allocation_v2", build)
    config = SimpleNamespace(
        reimbursement_policy_doc=lambda enabled: dict(policy),
        reimbursements_enabled=True,
        weight_mutation_enabled=True,
        production_writes_enabled=False,
    )

    first = asyncio.create_task(
        gateway_allocations.build_research_lab_allocation_bundle(
            config=config,
            epoch=105,
            netuid=71,
        )
    )
    await started.wait()
    second = asyncio.create_task(
        gateway_allocations.build_research_lab_allocation_bundle(
            config=config,
            epoch=105,
            netuid=71,
        )
    )
    await asyncio.sleep(0)
    release.set()
    first_bundle, second_bundle = await asyncio.gather(first, second)

    assert calls == 1
    assert first_bundle["allocation_hash"] == second_bundle["allocation_hash"]


@pytest.mark.asyncio
async def test_gateway_allocation_retries_after_shared_build_failure(monkeypatch):
    from gateway.research_lab import allocations as gateway_allocations

    _disable_durable_allocation_retry(monkeypatch, gateway_allocations)
    policy = {
        "policy_id": "research_lab_reimbursement_policy_v1",
        "lab_cap_alpha_percent": 30.0,
        "reimbursement_epochs": 20,
    }
    authority = _allocation_authority_outcome(
        epoch=106,
        netuid=71,
        policy=policy,
        source_obligations=[],
        include_source=False,
    )
    calls = 0

    async def build(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient build failure")
        return authority

    monkeypatch.setattr(gateway_allocations, "build_allocation_v2", build)
    config = SimpleNamespace(
        reimbursement_policy_doc=lambda enabled: dict(policy),
        reimbursements_enabled=True,
        weight_mutation_enabled=True,
        production_writes_enabled=False,
    )

    with pytest.raises(RuntimeError, match="transient build failure"):
        await gateway_allocations.build_research_lab_allocation_bundle(
            config=config,
            epoch=106,
            netuid=71,
        )
    bundle = await gateway_allocations.build_research_lab_allocation_bundle(
        config=config,
        epoch=106,
        netuid=71,
    )

    assert calls == 2
    assert bundle["epoch"] == 106


@pytest.mark.asyncio
async def test_gateway_allocation_resumes_after_durable_failed_generation(
    monkeypatch,
):
    from gateway.research_lab import allocations as gateway_allocations

    selected_sequences = []

    async def durable_generation(**_kwargs):
        return 1

    async def build(**kwargs):
        selected_sequences.append(kwargs["allocation_sequence"])
        return {"status": "matched"}

    monkeypatch.setattr(
        gateway_allocations,
        "_load_durable_allocation_retry_generation",
        durable_generation,
    )
    monkeypatch.setattr(gateway_allocations, "build_allocation_v2", build)

    result = await gateway_allocations._build_allocation_v2_singleflight(
        epoch_id=24940,
        netuid=71,
        policy={"policy_id": "durable-retry"},
    )

    assert result == {"status": "matched"}
    assert selected_sequences == [1]


@pytest.mark.asyncio
async def test_durable_allocation_retry_uses_failed_receipt_sequence(monkeypatch):
    from gateway.research_lab import allocations as gateway_allocations

    commit = "a" * 40
    observed = {}

    async def load(table, **kwargs):
        observed["table"] = table
        observed.update(kwargs)
        return [
            {
                "sequence": 0,
                "receipt_status": "failed",
                "failure_code": "execution_valueerror",
                "job_id": "scoring-v2:research-lab-allocation:job-0",
                "commit_sha": commit,
                "epoch_id": 24940,
                "purpose": "research_lab.allocation.v2",
                "role": "gateway_coordinator",
                "issued_at": _ago_iso(minutes=10),
            }
        ]

    monkeypatch.setattr(gateway_allocations, "_active_gateway_commit", lambda: commit)
    monkeypatch.setattr(gateway_allocations, "select_many", load)

    generation = await gateway_allocations._load_durable_allocation_retry_generation(
        epoch_id=24940,
    )

    assert generation == 1
    assert observed["table"] == "research_lab_attested_execution_receipts_v2"
    assert ("receipt_status", "failed") in observed["filters"]
    assert ("commit_sha", commit) in observed["filters"]
    assert "issued_at" in observed["columns"]
    assert "receipt_doc" not in observed["columns"]


@pytest.mark.asyncio
async def test_durable_allocation_retry_exhaustion_fails_closed(monkeypatch):
    from gateway.research_lab import allocations as gateway_allocations

    commit = "b" * 40

    async def load(_table, **_kwargs):
        return [
            _failed_receipt(
                sequence=sequence,
                commit=commit,
                epoch_id=24941,
                issued_at=_ago_iso(minutes=10 + sequence),
            )
            for sequence in range(8)
        ]

    monkeypatch.setattr(gateway_allocations, "_active_gateway_commit", lambda: commit)
    monkeypatch.setattr(gateway_allocations, "select_many", load)

    with pytest.raises(RuntimeError, match="retry generations are exhausted"):
        await gateway_allocations._load_durable_allocation_retry_generation(
            epoch_id=24941,
        )


@pytest.mark.asyncio
async def test_durable_allocation_retry_allowance_ages_out(monkeypatch):
    """A spent allowance must not refuse an epoch for the life of the process."""

    from gateway.research_lab import allocations as gateway_allocations

    commit = "c" * 40

    async def load(_table, **_kwargs):
        return [
            _failed_receipt(
                sequence=sequence,
                commit=commit,
                epoch_id=24942,
                issued_at=_ago_iso(minutes=120 + sequence),
            )
            for sequence in range(8)
        ]

    monkeypatch.setattr(gateway_allocations, "_active_gateway_commit", lambda: commit)
    monkeypatch.setattr(gateway_allocations, "select_many", load)

    generation = await gateway_allocations._load_durable_allocation_retry_generation(
        epoch_id=24942,
    )

    assert generation == 8


@pytest.mark.asyncio
async def test_durable_allocation_retry_spaces_out_attempts(monkeypatch):
    """A caller retrying immediately must not spend another generation."""

    from gateway.research_lab import allocations as gateway_allocations

    commit = "d" * 40

    async def load(_table, **_kwargs):
        return [
            _failed_receipt(
                sequence=0,
                commit=commit,
                epoch_id=24943,
                issued_at=_ago_iso(seconds=20),
            )
        ]

    monkeypatch.setattr(gateway_allocations, "_active_gateway_commit", lambda: commit)
    monkeypatch.setattr(gateway_allocations, "select_many", load)

    with pytest.raises(RuntimeError, match="cooling down"):
        await gateway_allocations._load_durable_allocation_retry_generation(
            epoch_id=24943,
        )


def _ago_iso(**delta) -> str:
    moment = datetime.now(timezone.utc) - timedelta(**delta)
    return moment.isoformat().replace("+00:00", "Z")


def _failed_receipt(*, sequence: int, commit: str, epoch_id: int, issued_at: str):
    return {
        "sequence": sequence,
        "receipt_status": "failed",
        "failure_code": "execution_valueerror",
        "job_id": "scoring-v2:research-lab-allocation:job-%d" % sequence,
        "commit_sha": commit,
        "epoch_id": epoch_id,
        "purpose": "research_lab.allocation.v2",
        "role": "gateway_coordinator",
        "issued_at": issued_at,
    }


def _disable_durable_allocation_retry(monkeypatch, gateway_allocations):
    monkeypatch.setattr(
        gateway_allocations,
        "_load_durable_allocation_retry_generation",
        lambda **_kwargs: _async_value(0),
    )


async def _async_value(value):
    return value
