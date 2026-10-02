"""Public model completion must not expose partial or retryable results."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest

from lab_arena import contracts
from lab_arena.service import ArenaService


@pytest.fixture()
def completed():
    service = ArenaService.__new__(ArenaService)
    service._completed_scores_lock = threading.Lock()
    service._completed_scores_cache = {}
    row = {
        "round_id": "arena-test", "status": "stage2_scoring", "status_generation": 1,
        "configuration_doc": {
            "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
            "sourcing_cost_eligibility_policy": contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
            "stage_1_icp_count": 1, "stage_2_icp_count": 1,
        },
        "participants": [{"submission_id": "miner", "is_king": False}],
        "stage2_scoring_plan_doc": {"work_items": [], "zero_rows": []},
    }
    executions = [{
        "run_id": "execute-%s" % position, "round_id": row["round_id"],
        "submission_id": "miner", "stage": 2, "kind": "execute",
        "icp_position": position, "attempt": 1, "status": "accepted",
        "terminal_cause": "accepted", "per_icp_score": score,
    } for position, score in enumerate([40.0, 80.0])]
    judges = [{**run, "kind": "score", "run_id": "judge-%s" % run["icp_position"],
               "scored_run_id": run["run_id"]} for run in executions]
    row["stage2_scoring_plan_doc"]["work_items"] = [{
        "scored_run_id": run["run_id"], "submission_id": "miner", "icp_position": run["icp_position"],
    } for run in executions]
    service._store = SimpleNamespace(list_runs=lambda _, kind: executions if kind == "execute" else judges)
    eligibility = {
        "eligible": True, "eligibility_reason": "eligible",
        "cost_summary": {"per_icp": [{"icp_position": p, "eligible": True} for p in range(2)]},
    }
    service._submission_cost_eligibility = Mock(return_value=eligibility)
    return service, row, executions, judges, eligibility


def test_cost_disqualified_icp_is_zeroed_without_changing_stored_score(completed):
    service, row, executions, _, eligibility = completed
    before = deepcopy(executions)
    eligibility["cost_summary"]["per_icp"][1]["eligible"] = False
    assert service.completed_submission_scores(row)["miner"]["final_score"] == 20.0
    assert executions == before


def test_completed_zero_is_a_real_score(completed):
    service, row, executions, _, _ = completed
    for run in executions:
        run["per_icp_score"] = 0
    assert service.completed_submission_scores(row)["miner"]["final_score"] == 0


@pytest.mark.parametrize("reason", ["provider_calls_inflight", "provider_cost_uncertain"])
def test_unsettled_costs_hide_score(completed, reason):
    service, row, _, _, eligibility = completed
    eligibility.update(eligible=False, eligibility_reason=reason)
    assert service.completed_submission_scores(row) == {}


@pytest.mark.parametrize("status", ["queued", "leased", "failed"])
def test_old_stored_scores_cannot_bypass_unfinished_judging(completed, status):
    service, row, _, judges, _ = completed
    judges[0].update(status=status, terminal_cause="invalid_output")
    assert service.completed_submission_scores(row) == {}
    service._submission_cost_eligibility.assert_not_called()


def test_missing_judge_cannot_bypass_stored_scores(completed):
    service, row, _, judges, _ = completed
    judges.pop()
    row["status"] = "stage2_judged"
    assert service.completed_submission_scores(row) == {}


def test_judgment_membership_is_checked_even_with_stored_scores(completed):
    service, row, _, judges, _ = completed
    judges[0]["submission_id"] = "other-model"
    assert service.completed_submission_scores(row) == {}


def test_incomplete_execution_set_cannot_publish_partial_average(completed):
    service, row, executions, _, _ = completed
    executions.pop()
    assert service.completed_submission_scores(row) == {}


def test_evidence_read_failure_does_not_hide_other_completed_models(completed):
    service, row, executions, judges, eligibility = completed
    row["participants"].append({"submission_id": "other", "is_king": False})
    copies = deepcopy(executions)
    for run in copies:
        run.update(submission_id="other", run_id="other-" + run["run_id"])
    executions.extend(copies)
    judges.extend([{**run, "kind": "score", "run_id": "judge-" + run["run_id"],
                    "scored_run_id": run["run_id"]} for run in copies])
    row["stage2_scoring_plan_doc"]["work_items"].extend([
        {"scored_run_id": run["run_id"], "submission_id": "other", "icp_position": run["icp_position"]}
        for run in copies
    ])

    def lookup(_row, submission_id, _runs, **_kwargs):
        if submission_id == "miner":
            raise OSError("temporary evidence read failure")
        return eligibility

    service._submission_cost_eligibility.side_effect = lookup
    assert set(service.completed_submission_scores(row)) == {"other"}


@pytest.mark.parametrize("status", ["open", "cancelled", "published"])
def test_non_evaluating_rounds_do_not_use_completed_projection(completed, status):
    service, row, _, _, _ = completed
    row["status"] = status
    assert service.completed_submission_scores(row) == {}


def test_historical_policy_keeps_original_publication_gate(completed):
    service, row, _, _, _ = completed
    row["configuration_doc"].pop("execution_sequence_policy")
    assert service.completed_submission_scores(row) == {}


def test_polling_cache_expires_and_round_transition_invalidates_it(completed, monkeypatch):
    service, row, _, _, _ = completed
    now = [100.0]
    monkeypatch.setattr("lab_arena.service.time.monotonic", lambda: now[0])
    read = Mock(wraps=service._read_completed_submission_scores)
    service._read_completed_submission_scores = read
    service.completed_submission_scores(row)
    service.completed_submission_scores(row)
    assert read.call_count == 1
    now[0] += 16
    service.completed_submission_scores(row)
    assert read.call_count == 2
    row["status_generation"] += 1
    service.completed_submission_scores(row)
    assert read.call_count == 3
