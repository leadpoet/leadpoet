"""Final authority needs measured baseline work, including legitimate zeros."""
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lab_arena import contracts, service


def fixture():
    row = {"round_id": "arena-test", "status": "scored", "participants": [
        {"submission_id": "baseline", "miner_hotkey": "baseline-hotkey", "is_king": True},
        {"submission_id": "miner", "miner_hotkey": "miner-hotkey", "is_king": False},
    ], "configuration_doc": {
        "mode": "live", "rewards_enabled": True,
        "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
        "stage_1_icp_count": 1, "stage_2_icp_count": 1,
        "schedule": {"publication_deadline": "2026-10-13T20:30:03Z"},
    }}
    executions = [{"run_id": "exec-%s" % i, "round_id": row["round_id"],
        "submission_id": "baseline", "miner_hotkey": "baseline-hotkey", "kind": "execute",
        "stage": 1, "icp_position": i, "attempt": 1, "status": "accepted",
        "terminal_cause": "accepted", "per_icp_score": 0.0,
        "output_ref": "output-%s" % i, "result_doc": {"terminal_status": "accepted"}}
        for i in range(2)]
    judgments = [{**r, "run_id": "judge-%s" % i, "kind": "score", "scored_run_id": r["run_id"]}
                 for i, r in enumerate(executions)]
    writes = []
    clock = [datetime(2026, 10, 13, 19, tzinfo=timezone.utc)]
    costs = {"providers": [{"inflight_calls": 0, "uncertain_calls": 99}]}
    instance = object.__new__(service.ArenaService)
    instance._store = SimpleNamespace(
        list_runs=lambda _rid, kind=None: executions if kind == "execute" else judgments if kind == "score" else executions + judgments,
        submission_costs=lambda _sid: costs,
        cancel_round=lambda rid, reason: writes.append((rid, reason)) or {"status": "cancelled"},
    )
    instance._round = lambda _rid: row
    instance.now = lambda: clock[0]
    instance._submission_cost_eligibility = lambda *a, **k: {"eligibility_reason": "eligible"}
    return instance, row, executions, judgments, writes, clock, costs


@pytest.mark.parametrize("mutation", ["failed_execution", "credential_zero", "judge_error", "pending_judge",
    "missing_icp", "duplicate", "stale_judge", "wrong_round", "wrong_stage", "missing_output", "missing_score"])
def test_infrastructure_and_unbound_rows_do_not_authorize_baseline(mutation):
    obj, row, executions, judges, *_ = fixture()
    if mutation in ("failed_execution", "credential_zero"):
        executions[0].update(status="failed", terminal_cause="provider_error" if mutation == "failed_execution" else "credential_error")
    elif mutation == "judge_error": judges[0].update(status="failed", terminal_cause="judge_error")
    elif mutation == "pending_judge": judges[0].update(status="pending", terminal_cause=None)
    elif mutation == "missing_icp": executions.pop()
    elif mutation == "duplicate": executions.append({**executions[0], "run_id": "duplicate", "attempt": 2})
    elif mutation == "stale_judge": judges[0]["scored_run_id"] = "old-execution"
    elif mutation == "wrong_round": judges[0]["round_id"] = "archived-round"
    elif mutation == "wrong_stage": judges[0]["stage"] = 2
    elif mutation == "missing_output": executions[0]["output_ref"] = None
    elif mutation == "missing_score": executions[0]["per_icp_score"] = None
    assert not obj._baseline_fully_succeeded(row)


def test_accepted_zero_and_normal_retry_cache_alias_are_valid():
    obj, row, executions, judges, *_ = fixture()
    assert obj._baseline_fully_succeeded(row)
    executions[0]["attempt"] = 2
    executions.append({**executions[0], "run_id": "failed-first", "attempt": 1,
                       "status": "failed", "terminal_cause": "provider_error"})
    judges[0]["result_doc"] = {"schema_version": "leadpoet.lab_arena.cached_run_result.v1", "terminal_status": "accepted"}
    judges[0]["judgment_cache_source_run_id"] = "original-judge-in-another-round"
    assert obj._baseline_fully_succeeded(row)


@pytest.mark.parametrize("reason,valid", [("eligible", True), ("cost_per_company_exceeded", True),
    ("execution_cap_exceeded", True), ("provider_cost_uncertain", False), ("provider_calls_inflight", False)])
def test_final_readiness_keeps_existing_cost_consequences(reason, valid):
    obj, row, *_ = fixture()
    obj._submission_cost_eligibility = lambda *a, **k: {"eligibility_reason": reason}
    assert obj._baseline_final_ready(row) is valid


def test_failed_baseline_holds_until_deadline_then_preserves_completed_data():
    obj, row, executions, judges, writes, clock, costs = fixture()
    executions[0].update(status="failed", terminal_cause="provider_error")
    before = copy.deepcopy((executions, judges, costs))
    assert obj.publish(row["round_id"])["reason"] == "baseline_incomplete"
    assert not writes
    clock[0] = datetime(2026, 10, 13, 20, 30, 3, tzinfo=timezone.utc)
    assert obj.publish(row["round_id"]) == {"status": "cancelled"}
    assert writes == [(row["round_id"], "scoring_incomplete")]
    assert (executions, judges, costs) == before


@pytest.mark.parametrize("active", ["leased", "submitted", "provider"])
def test_final_failure_does_not_interrupt_active_work(active):
    obj, row, executions, _, writes, clock, costs = fixture()
    clock[0] = datetime(2026, 10, 14, tzinfo=timezone.utc)
    if active == "provider": costs["providers"][0]["inflight_calls"] = 1
    else: executions[0]["status"] = active
    assert obj._hold_incomplete_baseline(row)["status"] == "retry"
    assert not writes


def test_already_activated_or_promoted_history_needs_no_new_baseline_read():
    obj, row, *_ = fixture()
    row.update(reward_activated_at="old", effective_reward_epoch=123, baseline_promoted_at="old")
    obj._baseline_final_ready = lambda _row: pytest.fail("must preserve existing authority")
    assert obj.activate_reward(row["round_id"]) == {"status": "existing", "effective_reward_epoch": 123}
    assert obj.promote_baseline(row["round_id"]) == {"status": "existing"}


def test_ineligible_legacy_reward_and_promotion_queues_continue():
    obj, row, *_ = fixture()
    legacy = {**row, "round_id": "old-incomplete", "status": "published"}
    complete = {**row, "round_id": "new-complete", "status": "published"}
    obj._config = SimpleNamespace(mode="live", pinned_round_id=None)
    obj._pinned_round_id = lambda: None
    obj._chain_scope = lambda: ("finney", 71)
    obj._store.list_rounds = lambda **kw: [complete, legacy]
    called = []
    obj.activate_reward = lambda rid: called.append(rid) or {"status": "waiting_for_baseline" if rid == legacy["round_id"] else "activated"}
    assert obj.activate_pending_rewards() == {"status": "ok", "activated": 1}
    assert called == [legacy["round_id"], complete["round_id"]]
    obj._latest_published_day = lambda: None
    obj._pending_promotion_rows = lambda **kw: [legacy]
    obj._store.get_round = lambda rid: legacy
    obj._baseline_final_ready = lambda r: False
    assert obj._pending_promotion_blocks() is False
    obj.promote_baseline = lambda rid: {"status": "waiting_for_baseline"}
    assert obj.promote_pending_baselines() == {"status": "ok", "promoted": 0}
