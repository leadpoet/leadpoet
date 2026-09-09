"""Public/private ICP selection is based only on a complete baseline."""

import json
from copy import deepcopy

import pytest

from lab_arena.icp_disclosure import baseline_disclosure
from lab_arena.service import ArenaService, ServiceError


def round_and_runs():
    row = {"round_id": "round", "status": "published", "participants": [
        {"submission_id": "base", "is_king": True},
        {"submission_id": "miner", "is_king": False},
    ]}
    runs = [{"submission_id": "base", "kind": "execute", "icp_position": i,
             "attempt": 1, "per_icp_score": float(i), "terminal_cause": "accepted"}
            for i in range(20)]
    return row, runs


def test_seven_weak_three_strong_and_complement():
    row, runs = round_and_runs()
    result = baseline_disclosure(row, runs)
    assert result["public_positions"] == [0, 1, 2, 3, 4, 5, 6, 17, 18, 19]
    assert result["private_positions"] == list(range(7, 17))
    assert len(set(result["public_positions"]) & set(range(10))) == 7
    assert len(set(result["private_positions"]) & set(range(10, 20))) == 7
    # Miner scores never influence disclosure. Restart/order do not change it.
    miner = [dict(run, submission_id="miner", per_icp_score=100 - run["per_icp_score"]) for run in runs]
    assert baseline_disclosure(deepcopy(row), list(reversed(runs)) + miner) == result


@pytest.mark.parametrize("missing", range(20))
def test_every_baseline_score_must_be_complete(missing):
    row, runs = round_and_runs()
    runs[missing]["per_icp_score"] = None
    assert baseline_disclosure(row, runs) is None


@pytest.mark.parametrize("bad", [True, "bad", float("nan"), float("inf"), -1, 101])
def test_invalid_score_does_not_mean_zero(bad):
    row, runs = round_and_runs()
    runs[0]["per_icp_score"] = bad
    assert baseline_disclosure(row, runs) is None


def test_ties_zero_and_all_failed():
    row, runs = round_and_runs()
    for run in runs:
        run["per_icp_score"] = 0
    assert baseline_disclosure(row, runs)["public_positions"] == [0, 1, 2, 3, 4, 5, 6, 17, 18, 19]
    for run in runs:
        run["terminal_cause"] = "model_error"
    assert baseline_disclosure(row, runs) is None


def test_public_benchmark_returns_only_selected_prompts_with_original_positions():
    row, runs = round_and_runs()
    service = ArenaService.__new__(ArenaService)
    service._round = lambda _: row
    service._public_icp_disclosure = lambda _: baseline_disclosure(row, runs)
    service.benchmark_icps = lambda _: [{"icp_id": f"icp-{i}", "prompt": f"PROMPT-{i:02d}!"} for i in range(20)]
    result = service.public_benchmark("round")
    assert len(result["icps"]) == 10
    assert [i["icp_position"] for i in result["icps"]] == [0, 1, 2, 3, 4, 5, 6, 17, 18, 19]
    for i in range(7, 17):
        assert f"PROMPT-{i:02d}!" not in json.dumps(result)
    assert "private_positions" not in result and "baseline_scores" not in result
    runs[0]["per_icp_score"] = None
    with pytest.raises(ServiceError, match="benchmark_not_public"):
        service.public_benchmark("round")
