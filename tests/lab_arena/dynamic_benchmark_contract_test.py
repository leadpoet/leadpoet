"""Frozen benchmark count and promotion margin contracts."""

from __future__ import annotations

import copy

import pytest

from lab_arena import contracts, scoring, verify
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration


@pytest.mark.parametrize("count", [10, 15, 20, 30])
def test_frozen_count_drives_positions_and_scoring_plan_without_new_json_fields(count):
    configuration = base_round_configuration()
    configuration["stage_1_icp_count"] = (count + 1) // 2
    configuration["stage_2_icp_count"] = count // 2
    configuration["promotion_margin"] = 0.5
    configuration["network_name"] = "finney"
    configuration["netuid"] = 71
    assert contracts.validate_round_configuration(configuration) == configuration
    assert contracts.benchmark_icp_count(configuration) == count
    assert contracts.stage_positions(1, configuration) == tuple(range((count + 1) // 2))
    assert contracts.stage_positions(2, configuration) == tuple(range((count + 1) // 2, count))
    assert contracts.execution_positions(
        1, contracts.BASELINE_SCORED_FIRST_POLICY, configuration
    ) == tuple(range(count))
    assert contracts.execution_positions(
        2, contracts.BASELINE_SCORED_FIRST_POLICY, configuration
    ) == tuple(range(count))
    plan = scoring.build_scoring_plan(
        round_id=configuration["round_id"],
        stage=1,
        runs=[{
            "stage": 1, "submission_id": "baseline", "icp_position": position,
            "status": "failed", "attempt": 1, "terminal_cause": "model_error",
        } for position in range(count)],
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        configuration=configuration,
    )
    assert "benchmark_icp_count" not in plan
    assert len(plan["zero_rows"]) == count
    assert contracts.validate_scoring_plan(plan, configuration) == plan
    assert verify.stage_score([50.0] * count, count) == 50.0


def test_historical_defaults_and_hashes_stay_stable():
    old = base_round_configuration()
    original_hash = contracts.document_hash(old)
    assert contracts.benchmark_icp_count({}) == 20
    assert contracts.promotion_margin(old) == 1.0
    assert contracts.execution_positions(1, contracts.BASELINE_SCORED_FIRST_POLICY) == tuple(range(20))
    validated = contracts.validate_round_configuration(copy.deepcopy(old))
    assert validated == {**old, "network_name": "finney", "netuid": 71}
    assert contracts.document_hash(old) == original_hash
    with pytest.raises(contracts.ArenaContractError, match="together"):
        contracts.benchmark_icp_count({"stage_1_icp_count": 5})
    invalid = dict(old, promotion_margin=None)
    with pytest.raises(contracts.ArenaContractError, match="cannot be null"):
        contracts.validate_round_configuration(invalid)


@pytest.mark.parametrize("margin,expected", [(0.5, "crowned"), (1.0, "no_king")])
def test_exact_frozen_promotion_boundary(margin, expected):
    configuration = {"promotion_margin": margin}
    baseline = {"submission_id": "baseline", "hotkey": "baseline-key", "is_king": True, "final_score": 70.0}
    miner = {"submission_id": "miner", "hotkey": "miner-key", "is_king": False, "final_score": 70.5}
    assert verify.king_decision([miner], baseline, configuration)["outcome"] == expected
    assert verify.king_decision([miner], baseline)["outcome"] == "no_king"
