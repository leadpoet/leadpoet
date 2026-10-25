"""The paid E2E succeeds only with complete strict exports for both roles."""

import copy

import pytest

from scripts import verify_arena_full_thinking_e2e as e2e


def participant():
    executions = [{"run_id": f"run-{i}", "icp_position": i} for i in range(10)]
    scores = [{"run_id": f"score-{i}", "scored_run_id": f"run-{i}"} for i in range(10)]
    per_icp = [{"icp_position": i, "records": 1, "complete": True} for i in range(10)]
    splits = {"train": [{} for _ in range(10)], "eval": []}
    return executions, scores, per_icp, splits


@pytest.mark.parametrize("damage,reason", [
    (lambda part: part[0].pop(), "accepted executions and linked real scores"),
    (lambda part: part[1][-1].update(scored_run_id="unlinked"), "accepted executions and linked real scores"),
    (lambda part: part[2][-1].update(icp_position=8), "ICP positions are incomplete"),
    (lambda part: part[2][-1].update(records=0), "complete strict training record"),
    (lambda part: part[2][-1].update(complete=False), "complete strict training record"),
    (lambda part: part[3]["train"].pop(), "fewer than ten records"),
])
def test_valid_candidate_cannot_hide_incomplete_baseline(damage, reason):
    candidate = participant()
    baseline = copy.deepcopy(participant())
    damage(baseline)
    with pytest.raises(AssertionError, match="baseline: .*" + reason):
        e2e._require_strict_exports(candidate, baseline)


def test_both_complete_participants_pass_and_candidate_is_checked_too():
    candidate = participant()
    baseline = participant()
    e2e._require_strict_exports(candidate, baseline)
    candidate[2][0]["records"] = 0
    with pytest.raises(AssertionError, match="candidate: .*complete strict training record"):
        e2e._require_strict_exports(candidate, baseline)
