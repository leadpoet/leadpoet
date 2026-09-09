from __future__ import annotations

import pytest

from gateway.tee.reward_executor_v2 import (
    RewardExecutorV2Error,
    champion_reward_row_projection_v2,
    execute_reward_decision_v2,
    reimbursement_reward_row_projection_v2,
    reward_receipt_projection_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json














def test_historical_champion_migration_preserves_exact_stored_obligation():
    bundle_doc = {
        "score_bundle_hash": "sha256:" + "5" * 64,
        "aggregates": {
            "per_icp_results": [
                {
                    "icp_ref": "qualification_private_icp_sets:100:icp-a",
                },
                {
                    "icp_ref": "qualification_private_icp_sets:100:icp-b",
                },
                {
                    "icp_ref": "qualification_private_icp_sets:101:icp-c",
                },
            ]
        },
    }
    anchored_payload = {
        "champion_reward_id": "",
        "status": "active",
        "reasons": [],
        "uid": 7,
        "miner_hotkey": "miner-hotkey",
        "island": "generalist",
        "source_id": "candidate-1",
        "score_bundle_id": "score-bundle-1",
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 20,
        "improvement_points": 2.5,
        "threshold_points": 1.0,
        "desired_alpha_percent": 7.45,
        "daily_icp_counts": {"100": 2, "101": 1},
        "required_icp_count": 3,
        "input_hash": "sha256:" + "6" * 64,
    }
    anchored_hash = sha256_json(anchored_payload)
    reward_id = "champion_reward:" + anchored_hash
    reward_row = {
        **anchored_payload,
        "champion_reward_id": reward_id,
        "anchored_hash": anchored_hash,
        "miner_uid": 7,
        "source_score_bundle_hash": bundle_doc["score_bundle_hash"],
    }
    result = execute_reward_decision_v2(
        {
            "decision_kind": "champion_migration",
            "decision_payload": {
                "reward_row": reward_row,
                "score_bundle": {
                    "score_bundle_id": "score-bundle-1",
                    "score_bundle_hash": bundle_doc["score_bundle_hash"],
                    "score_bundle_doc": bundle_doc,
                },
            },
        }
    )
    assert result["decision_kind"] == "champion"
    assert result["reward"]["champion_reward_id"] == reward_id
    assert result["reward"]["anchored_hash"] == anchored_hash
    assert result["reward"]["desired_alpha_percent"] == 7.45

    with pytest.raises(RewardExecutorV2Error, match="anchored payload differs"):
        execute_reward_decision_v2(
            {
                "decision_kind": "champion_migration",
                "decision_payload": {
                    "reward_row": {
                        **reward_row,
                        "desired_alpha_percent": 8.45,
                    },
                    "score_bundle": {
                        "score_bundle_id": "score-bundle-1",
                        "score_bundle_hash": bundle_doc["score_bundle_hash"],
                        "score_bundle_doc": bundle_doc,
                    },
                },
            }
        )




@pytest.mark.parametrize("kind", ("champion", "reimbursement"))
def test_retired_loop_rewards_cannot_issue_new_obligations(kind):
    with pytest.raises(RewardExecutorV2Error, match="kind is unsupported"):
        execute_reward_decision_v2({
            "decision_kind": kind,
            "decision_payload": {},
        })
