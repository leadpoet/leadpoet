from __future__ import annotations

import pytest

from gateway.tee.reward_executor_v2 import (
    RewardExecutorV2Error,
    champion_reward_row_projection_v2,
    execute_reward_decision_v2,
    reimbursement_reward_row_projection_v2,
    source_add_reward_row_projection_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json


def test_leg2_reward_rejects_nonapproving_signed_judge():
    payload = {
        "decision_kind": "source_add_leg2",
        "decision_payload": {
            "adapter_id": "adapter:test",
            "miner_ref": "miner",
            "start_epoch": 101,
            "existing_rewards": [],
            "alpha_percent": 5.0,
            "reward_epochs": 20,
            "trigger_evidence": {"llm_judge_passed": True},
            "judge_result": {
                "verdict": {"verdict": "not_helped", "source_used": False}
            },
        },
    }
    with pytest.raises(RewardExecutorV2Error, match="did not approve"):
        execute_reward_decision_v2(payload)


def test_reward_row_projection_hashes_change_for_payout_field_mutation():
    champion = {
        "champion_reward_id": "champion:1",
        "score_bundle_id": "bundle:1",
        "candidate_id": "candidate:1",
        "run_id": "run:1",
        "miner_hotkey": "hotkey:1",
        "miner_uid": 7,
        "island": "generalist",
        "evaluation_epoch": 100,
        "start_epoch": 101,
        "epoch_count": 20,
        "improvement_points": 2.5,
        "threshold_points": 1.0,
        "desired_alpha_percent": 7.45,
        "input_hash": "sha256:" + "1" * 64,
        "anchored_hash": "sha256:" + "2" * 64,
    }
    before = sha256_json(champion_reward_row_projection_v2(champion))
    after = sha256_json(
        champion_reward_row_projection_v2(
            {**champion, "desired_alpha_percent": 8.45}
        )
    )
    assert after != before

    source_add = {
        "reward_ref": "source_add_reward:1234567890abcdef",
        "adapter_id": "adapter:test",
        "miner_hotkey": "hotkey:1",
        "leg": 2,
        "reward_kind": "source_implementation",
        "alpha_percent": 5.0,
        "reward_epochs": 20,
        "start_epoch": 101,
        "initial_reward_status": "active",
        "trigger_evidence_doc": {"llm_judge_passed": True},
        "public_label": "Source implementation reward",
    }
    before = sha256_json(
        source_add_reward_row_projection_v2("source_add_leg2", source_add)
    )
    after = sha256_json(
        source_add_reward_row_projection_v2(
            "source_add_leg2",
            {**source_add, "alpha_percent": 6.0},
        )
    )
    assert after != before

    award = {
        "award_id": "award:1",
        "run_id": "run:1",
        "miner_hotkey": "hotkey:1",
        "island": "generalist",
        "run_day": "2026-07-10",
        "award_status": "awarded",
        "participation_score": 3.0,
        "participation_fraction": 0.3,
        "rebate_rate": 0.5,
        "eligible_cost_microusd": 10_000_000,
        "target_reimbursement_microusd": 5_000_000,
        "reimbursement_epochs": 20,
        "loop_start_fee_included": False,
        "input_hash": "sha256:" + "3" * 64,
    }
    schedule = {
        "schedule_id": "schedule:1",
        "award_id": "award:1",
        "schedule_status": "scheduled",
        "start_epoch": 101,
        "epoch_count": 20,
        "total_microusd": 5_000_000,
        "entries": [],
    }
    before = sha256_json(reimbursement_reward_row_projection_v2(award, schedule))
    after = sha256_json(
        reimbursement_reward_row_projection_v2(
            {**award, "target_reimbursement_microusd": 5_000_001},
            schedule,
        )
    )
    assert after != before
