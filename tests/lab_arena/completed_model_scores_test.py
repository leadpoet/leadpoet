"""Public model completion must not expose partial or retryable results."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import time

import pytest

from lab_arena import contracts
from lab_arena.service import ArenaService
from lab_arena.store import PUBLIC_CONFIGURATION_FIELDS


@pytest.fixture()
def completed():
    service = ArenaService.__new__(ArenaService)
    service._completed_scores_lock = threading.Lock()
    service._completed_scores_cache = {}
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(2)
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
    assert service._read_completed_submission_scores(row)["miner"]["final_score"] == 20.0
    assert executions == before


def test_completed_zero_is_a_real_score(completed):
    service, row, executions, _, _ = completed
    for run in executions:
        run["per_icp_score"] = 0
    assert service._read_completed_submission_scores(row)["miner"]["final_score"] == 0


@pytest.mark.parametrize("reason", ["provider_calls_inflight", "provider_cost_uncertain"])
def test_unsettled_costs_hide_score(completed, reason):
    service, row, _, _, eligibility = completed
    eligibility.update(eligible=False, eligibility_reason=reason)
    assert service._read_completed_submission_scores(row) == {}


@pytest.mark.parametrize("status", ["queued", "leased", "failed"])
def test_old_stored_scores_cannot_bypass_unfinished_judging(completed, status):
    service, row, _, judges, _ = completed
    judges[0].update(status=status, terminal_cause="invalid_output")
    assert service._read_completed_submission_scores(row) == {}
    service._submission_cost_eligibility.assert_not_called()


def test_missing_judge_cannot_bypass_stored_scores(completed):
    service, row, _, judges, _ = completed
    judges.pop()
    row["status"] = "stage2_judged"
    assert service._read_completed_submission_scores(row) == {}


def test_judgment_membership_is_checked_even_with_stored_scores(completed):
    service, row, _, judges, _ = completed
    judges[0]["submission_id"] = "other-model"
    assert service._read_completed_submission_scores(row) == {}


def test_incomplete_execution_set_cannot_publish_partial_average(completed):
    service, row, executions, _, _ = completed
    executions.pop()
    assert service._read_completed_submission_scores(row) == {}


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
    assert set(service._read_completed_submission_scores(row)) == {"other"}


@pytest.mark.parametrize("status", ["open", "published"])
def test_non_evaluating_rounds_do_not_use_completed_projection(completed, status):
    service, row, _, _, _ = completed
    row["status"] = status
    assert service.completed_submission_scores(row) == {}


def test_historical_policy_keeps_original_publication_gate(completed):
    service, row, _, _, _ = completed
    row["configuration_doc"].pop("execution_sequence_policy")
    assert service.completed_submission_scores(row) == {}


def _wait_for_scores(service, row, predicate, *, attempts=100):
    for _ in range(attempts):
        result = service.completed_submission_scores(row)
        if predicate(result):
            return result
        time.sleep(0.01)
    pytest.fail("completed-score refresh did not finish")


def test_polling_cache_expires_and_round_transition_invalidates_it(completed, monkeypatch):
    service, row, _, _, _ = completed
    now = [100.0]
    monkeypatch.setattr("lab_arena.service.time.monotonic", lambda: now[0])
    read = Mock(wraps=service._read_completed_submission_scores)
    service._read_completed_submission_scores = read
    _wait_for_scores(service, row, lambda scores: "miner" in scores)
    service.completed_submission_scores(row)
    assert read.call_count == 1
    now[0] += 16
    _wait_for_scores(service, row, lambda _: read.call_count == 2 and not service._completed_scores_cache[row["round_id"]]["refreshing"])
    assert read.call_count == 2
    row["status_generation"] += 1
    _wait_for_scores(service, row, lambda _: read.call_count == 3)
    assert read.call_count == 3


def test_compact_refresh_rejects_full_round_generation_mismatch(completed):
    service, row, _, _, _ = completed
    compact = {key: value for key, value in row.items() if not key.endswith("_scoring_plan_doc")}
    newer = {**row, "status_generation": row["status_generation"] + 1}
    service._round = Mock(return_value=newer)
    service._read_completed_submission_scores = Mock(return_value={"miner": {"final_score": 60}})
    assert service.completed_submission_scores(compact) == {}
    for _ in range(100):
        if (not service._completed_scores_cache[row["round_id"]]["refreshing"]
                and service._completed_scores_refresh_slots._value == 2):
            break
        time.sleep(0.01)
    entry = service._completed_scores_cache[row["round_id"]]
    assert entry["completed"] == {}
    assert entry["refreshed_at"] == 0.0
    service._read_completed_submission_scores.assert_not_called()
    compact["status_generation"] = newer["status_generation"]
    assert _wait_for_scores(service, compact, lambda scores: scores.get("miner") == {"final_score": 60})
    assert service._round.call_count == 2


def test_compact_full_round_fetch_failure_releases_slot_and_retries(completed):
    service, row, _, _, _ = completed
    compact = {key: value for key, value in row.items() if not key.endswith("_scoring_plan_doc")}
    service._round = Mock(side_effect=[OSError("temporary round read failure"), row])
    service._read_completed_submission_scores = Mock(return_value={"miner": {"final_score": 60}})
    assert service.completed_submission_scores(compact) == {}
    for _ in range(100):
        if (not service._completed_scores_cache[row["round_id"]]["refreshing"]
                and service._completed_scores_refresh_slots._value == 2):
            break
        time.sleep(0.01)
    entry = service._completed_scores_cache[row["round_id"]]
    assert entry["refreshing"] is False
    assert service._completed_scores_refresh_slots._value == 2
    service._read_completed_submission_scores.assert_not_called()
    entry["refreshed_at"] -= 16
    assert _wait_for_scores(service, compact, lambda scores: scores.get("miner") == {"final_score": 60})
    assert service._round.call_count == 2
    for _ in range(100):
        if service._completed_scores_refresh_slots._value == 2:
            break
        time.sleep(0.01)
    assert service._completed_scores_refresh_slots._value == 2


def test_old_compact_refresh_cannot_write_after_new_generation(completed):
    service, row, _, _, _ = completed
    compact = {key: value for key, value in row.items() if not key.endswith("_scoring_plan_doc")}
    newer = {**row, "status_generation": row["status_generation"] + 1}
    service._round = Mock(side_effect=[row, newer])
    first_started, release_first, old_finished = threading.Event(), threading.Event(), threading.Event()
    original_refresh = service._refresh_completed_submission_scores

    def observed_refresh(refresh_row, entry):
        try:
            return original_refresh(refresh_row, entry)
        finally:
            if entry["generation"] == row["status_generation"]:
                old_finished.set()

    service._refresh_completed_submission_scores = observed_refresh

    def read(full_row, *, on_completed, should_continue):
        if full_row["status_generation"] == row["status_generation"]:
            first_started.set()
            assert release_first.wait(2)
            on_completed("miner", {"final_score": 10})
            return {"miner": {"final_score": 10}}
        on_completed("miner", {"final_score": 60})
        return {"miner": {"final_score": 60}}

    service._read_completed_submission_scores = Mock(side_effect=read)
    try:
        assert service.completed_submission_scores(compact) == {}
        assert first_started.wait(2)
        newer_compact = {**compact, "status_generation": newer["status_generation"]}
        assert _wait_for_scores(service, newer_compact, lambda scores: scores.get("miner") == {"final_score": 60})
    finally:
        release_first.set()
    assert old_finished.wait(2)
    assert service._read_completed_submission_scores.call_count == 2
    assert service.completed_submission_scores(newer_compact) == {"miner": {"final_score": 60}}


def test_slow_refresh_does_not_block_public_reads_and_releases_each_model(completed):
    service, row, _, _, _ = completed
    started = threading.Event()
    release_first = threading.Event()
    first_ready = threading.Event()
    finish = threading.Event()

    def slow_read(_row, *, on_completed, should_continue):
        started.set()
        assert release_first.wait(2)
        score = {"final_score": 60.0}
        on_completed("miner", score)
        first_ready.set()
        assert finish.wait(2)
        return {"miner": score}

    service._read_completed_submission_scores = Mock(side_effect=slow_read)
    try:
        assert service.completed_submission_scores(row) == {}
        assert started.wait(2)
        assert service.completed_submission_scores(row) == {}
        assert service._read_completed_submission_scores.call_count == 1
        release_first.set()
        assert first_ready.wait(2)
        assert service.completed_submission_scores(row)["miner"]["final_score"] == 60.0
        assert service._read_completed_submission_scores.call_count == 1
    finally:
        release_first.set()
        finish.set()
    _wait_for_scores(service, row, lambda _: not service._completed_scores_cache[row["round_id"]]["refreshing"])


def test_refresh_removes_model_when_costs_become_ineligible(completed):
    service, row, _, _, eligibility = completed
    assert _wait_for_scores(service, row, lambda scores: "miner" in scores)
    eligibility.update(eligible=False, eligibility_reason="provider_cost_uncertain")
    service._completed_scores_cache[row["round_id"]]["refreshed_at"] -= 16
    _wait_for_scores(
        service, row,
        lambda scores: not scores and not service._completed_scores_cache[row["round_id"]]["refreshing"],
    )
    assert service.completed_submission_scores(row) == {}


def test_published_transition_rejects_inflight_refresh(completed):
    service, row, _, _, _ = completed
    started = threading.Event()
    finish = threading.Event()

    def slow_read(_row, *, on_completed, should_continue):
        started.set()
        assert finish.wait(2)
        on_completed("miner", {"final_score": 60.0})
        return {"miner": {"final_score": 60.0}}

    service._read_completed_submission_scores = slow_read
    try:
        assert service.completed_submission_scores(row) == {}
        assert started.wait(2)
        row["status"] = "published"
        row["status_generation"] += 1
        assert service.completed_submission_scores(row) == {}
    finally:
        finish.set()
    for _ in range(100):
        if not service._completed_scores_cache:
            break
        time.sleep(0.01)
    assert service._completed_scores_cache == {}


def test_at_most_two_refresh_workers_run_and_failed_worker_releases_slot(completed):
    service, row, _, _, _ = completed
    started = threading.Event()
    release = threading.Event()
    calls = []
    calls_lock = threading.Lock()

    def slow_read(current_row, *, on_completed, should_continue):
        with calls_lock:
            calls.append(current_row["round_id"])
            if len(calls) == 2:
                started.set()
        assert release.wait(2)
        raise OSError("temporary object-store failure")

    service._read_completed_submission_scores = slow_read
    rows = [{**row, "round_id": "arena-%s" % index} for index in range(3)]
    try:
        for current_row in rows:
            assert service.completed_submission_scores(current_row) == {}
        assert started.wait(2)
        assert len(calls) == 2
    finally:
        release.set()
    for _ in range(100):
        if all(not service._completed_scores_cache[item["round_id"]]["refreshing"]
               for item in rows[:2]):
            break
        time.sleep(0.01)
    assert all(not service._completed_scores_cache[item["round_id"]]["refreshing"]
               for item in rows[:2])
    assert service.completed_submission_scores(rows[2]) == {}
    _wait_for_scores(service, rows[2], lambda _: not service._completed_scores_cache[rows[2]["round_id"]]["refreshing"])
    assert calls == ["arena-0", "arena-1", "arena-2"]


SAFE_CLOSURES = [
    ("stage1_scoring", "stage1_judged"),
    ("stage1_judged", "stage1_scored"),
    ("stage2_scoring", "stage2_judged"),
    ("stage2_judged", "scored"),
    *[(status, "cancelled") for status in (
        "stage1_scoring", "stage1_judged", "stage1_scored", "stage2",
        "stage2_closed", "stage2_scoring", "stage2_judged", "scored",
    )],
]


def _warm_verified_scores(service, row):
    scores = {"baseline": {"final_score": 1.75}, "miner": {"final_score": 0}}
    service._read_completed_submission_scores = Mock(return_value=scores)
    _wait_for_scores(service, row, lambda result: result == scores and not
                     service._completed_scores_cache[row["round_id"]]["refreshing"])
    return scores


@pytest.mark.parametrize("before,after", SAFE_CLOSURES)
def test_normal_closure_keeps_verified_scores_during_fresh_refresh(completed, before, after):
    service, row, _, _, _ = completed
    row["status"] = before
    prior = _warm_verified_scores(service, row)
    old_entry = service._completed_scores_cache[row["round_id"]]
    started, finish = threading.Event(), threading.Event()
    fresh = {**prior, "new-model": {"final_score": 12}}

    def read(_row, *, on_completed, should_continue):
        started.set()
        assert finish.wait(2)
        assert should_continue()
        for sid, score in fresh.items():
            on_completed(sid, score)
        return fresh

    service._read_completed_submission_scores = Mock(side_effect=read)
    row.update(status=after, status_generation=row["status_generation"] + 1)
    try:
        assert service.completed_submission_scores(row) == prior
        assert started.wait(2)
        entry = service._completed_scores_cache[row["round_id"]]
        assert entry is not old_entry
        assert entry["completed"] is not old_entry["completed"]
        assert entry["refreshed_at"] == 0
        assert service.completed_submission_scores(row) == prior
    finally:
        finish.set()
    assert _wait_for_scores(service, row, lambda result: result == fresh) == fresh
    assert service._read_completed_submission_scores.call_count == 1


@pytest.mark.parametrize("old_compact", [False, True])
def test_closure_identity_matches_existing_compact_projection(completed, old_compact):
    from lab_arena.store import PUBLIC_CONFIGURATION_COLUMN
    from tests.lab_arena.competition_summary_projection_test import _project

    service, row, _, _, _ = completed
    row["configuration_doc"].update({
        "mode": "live", "network_name": "finney", "netuid": 71,
        "schedule": {"final_scoring_close": "2026-10-19T22:00:00Z"},
        "baseline_hotkey": None, "scorer_policy": {"private": True},
        "deepline_catalog": {"private": True},
    })
    columns = "round_id,status,status_generation,participants," + PUBLIC_CONFIGURATION_COLUMN
    compact = _project(row, columns)
    service._round = Mock(return_value=row)
    first = compact if old_compact else row
    prior = _warm_verified_scores(service, first)
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(0)
    row.update(status="stage2_judged", status_generation=2)
    next_row = row if old_compact else _project(row, columns)
    assert service.completed_submission_scores(next_row) == prior
    # Explicit null stays distinct from an omitted field through that projection.
    next_row = deepcopy(next_row)
    next_row.update(status="scored", status_generation=3)
    next_row["configuration_doc"].pop("baseline_hotkey")
    assert service.completed_submission_scores(next_row) == {}


@pytest.mark.parametrize("key", PUBLIC_CONFIGURATION_FIELDS)
def test_closure_rejects_changed_public_configuration(completed, key):
    service, row, _, _, _ = completed
    _warm_verified_scores(service, row)
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(0)
    row.update(status="stage2_judged", status_generation=2)
    row["configuration_doc"][key] = "changed"
    assert service.completed_submission_scores(row) == {}


@pytest.mark.parametrize("change", ["participant", "nested_config", "config_type"])
@pytest.mark.parametrize("increment", [0, 1])
def test_closure_identity_is_a_snapshot_not_mutable_input(completed, change, increment):
    service, row, _, _, _ = completed
    row["configuration_doc"].update(schedule={"version": 0}, netuid=0)
    _warm_verified_scores(service, row)
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(0)
    row.update(status="stage2_judged" if increment else "stage2_scoring",
               status_generation=1 + increment)
    if change == "participant":
        row["participants"][0]["submission_id"] = "different"
    elif change == "nested_config":
        row["configuration_doc"]["schedule"]["version"] = 1
    else:
        row["configuration_doc"]["netuid"] = False
    assert service.completed_submission_scores(row) == {}


@pytest.mark.parametrize("after,increment", [
    ("stage2_judged", 0), ("stage2_judged", 2),  # Same generation or skipped generation.
    ("scored", 1), ("stage2_scoring", 1),       # Skipped phase or unrelated generation.
    ("stage1_scoring", 1), ("cancelled", 0), ("cancelled", 2), ("published", 1),
])
def test_other_phase_or_generation_changes_invalidate(completed, after, increment):
    service, row, _, _, _ = completed
    _warm_verified_scores(service, row)
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(0)
    row.update(status=after, status_generation=row["status_generation"] + increment)
    assert service.completed_submission_scores(row) == {}


@pytest.mark.parametrize("before,after", [
    ("stage1_scored", "stage2"), ("stage2", "stage2_closed"),
    ("stage2_closed", "stage2_scoring"), ("stage2_judged", "stage2_scoring"),
])
def test_unapproved_forward_edges_and_rewinds_invalidate(completed, before, after):
    service, row, _, _, _ = completed
    row["status"] = before
    _warm_verified_scores(service, row)
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(0)
    row.update(status=after, status_generation=row["status_generation"] + 1)
    assert service.completed_submission_scores(row) == {}


@pytest.mark.parametrize("old_status", ["stage2_scoring", "open", "cancelled"])
@pytest.mark.parametrize("compact", [False, True])
def test_delayed_old_request_cannot_evict_carried_scores_or_new_worker(completed, old_status, compact):
    service, row, _, _, _ = completed
    prior = _warm_verified_scores(service, row)
    old = deepcopy(row)
    old["status"] = old_status
    if compact:
        from lab_arena.store import PUBLIC_CONFIGURATION_COLUMN
        from tests.lab_arena.competition_summary_projection_test import _project
        old = _project(old, "round_id,status,status_generation,participants," + PUBLIC_CONFIGURATION_COLUMN)
    started, finish = threading.Event(), threading.Event()
    fresh = {"miner": {"final_score": 61}}

    def read(_row, *, on_completed, should_continue):
        started.set()
        assert finish.wait(2)
        on_completed("miner", fresh["miner"])
        return fresh

    service._read_completed_submission_scores = Mock(side_effect=read)
    row.update(status="stage2_judged", status_generation=2)
    try:
        assert service.completed_submission_scores(row) == prior
        assert started.wait(2)
        entry = service._completed_scores_cache[row["round_id"]]
        assert service.completed_submission_scores(old) == {}
        assert service._completed_scores_cache[row["round_id"]] is entry
        assert entry["refreshing"] and entry["completed"] == prior
    finally:
        finish.set()
    assert _wait_for_scores(service, row, lambda result: result == fresh) == fresh
    assert service._read_completed_submission_scores.call_count == 1


def test_old_refresh_cannot_overwrite_carried_entry(completed):
    service, row, _, _, _ = completed
    prior = _warm_verified_scores(service, row)
    old_started, release_old, old_finished = (threading.Event() for _ in range(3))
    new_started, release_new = threading.Event(), threading.Event()

    def read(refresh_row, *, on_completed, should_continue):
        if refresh_row["status_generation"] == 1:
            old_started.set()
            assert release_old.wait(2)
            assert not should_continue()
            on_completed("miner", {"final_score": 99})
            old_finished.set()
            return {"miner": {"final_score": 99}}
        new_started.set()
        assert release_new.wait(2)
        return prior

    service._read_completed_submission_scores = Mock(side_effect=read)
    service._completed_scores_cache[row["round_id"]]["refreshed_at"] -= 16
    try:
        assert service.completed_submission_scores(row) == prior
        assert old_started.wait(2)
        row = {**row, "status": "stage2_judged", "status_generation": 2}
        assert service.completed_submission_scores(row) == prior
        assert new_started.wait(2)
        entry = service._completed_scores_cache[row["round_id"]]
        release_old.set()
        assert old_finished.wait(2)
        assert entry["completed"] == prior
        assert service._completed_scores_cache[row["round_id"]] is entry
    finally:
        release_old.set()
        release_new.set()
    _wait_for_scores(service, row, lambda _: not entry["refreshing"])
    assert entry["completed"] == prior


def test_carried_score_removed_by_fresh_cost_check(completed):
    service, row, _, _, eligibility = completed
    assert _wait_for_scores(service, row, lambda scores: "miner" in scores)
    eligibility.update(eligible=False, eligibility_reason="provider_cost_uncertain")
    removed, finish = threading.Event(), threading.Event()
    original_read = service._read_completed_submission_scores

    def read(refresh_row, *, on_completed, should_continue):
        result = original_read(refresh_row, on_completed=on_completed, should_continue=should_continue)
        removed.set()
        assert finish.wait(2)
        return result

    service._read_completed_submission_scores = read
    row.update(status="stage2_judged", status_generation=2)
    # The request gets the previously verified value; its fresh eligibility check
    # still removes it, rather than treating carryover as permanent authority.
    try:
        assert service.completed_submission_scores(row)["miner"]["final_score"] == 60
        assert removed.wait(2)
        assert service._completed_scores_cache[row["round_id"]]["refreshing"]
        assert service.completed_submission_scores(row) == {}
    finally:
        finish.set()
    _wait_for_scores(service, row, lambda result: not result and not
                     service._completed_scores_cache[row["round_id"]]["refreshing"])
    assert service.completed_submission_scores(row) == {}


def test_successful_same_phase_refresh_replaces_prior_value(completed):
    service, row, _, _, _ = completed
    _warm_verified_scores(service, row)
    fresh = {"miner": {"final_score": 61}}
    service._read_completed_submission_scores = Mock(return_value=fresh)
    service._completed_scores_cache[row["round_id"]]["refreshed_at"] -= 16
    assert _wait_for_scores(service, row, lambda result: result == fresh) == fresh
    assert service._read_completed_submission_scores.call_count == 1


def _add_stored_peer(row, executions, judges):
    peer = deepcopy(executions)
    for run in peer:
        run.update(submission_id="stored-peer", run_id="peer-" + run["run_id"])
    executions.extend(peer)
    judges.extend([{**run, "kind": "score", "run_id": "judge-" + run["run_id"],
                    "scored_run_id": run["run_id"]} for run in peer])
    row["participants"].append({"submission_id": "stored-peer", "is_king": False})
    row["stage2_scoring_plan_doc"]["work_items"].extend([
        {"scored_run_id": run["run_id"], "submission_id": "stored-peer",
         "icp_position": run["icp_position"]} for run in peer
    ])
    return peer


def test_persisted_peer_is_released_before_blocked_unpublished_evidence(completed):
    service, row, executions, judges, _ = completed
    _add_stored_peer(row, executions, judges)
    for run in executions[:2]:
        run.update(per_icp_score=None, output_ref="slow-output")
    # A failed attempt's stored score must not prioritize an accepted attempt
    # whose evidence still requires assembly.
    executions.append({**executions[0], "run_id": "failed-stored", "attempt": 99,
                       "status": "failed", "per_icp_score": 999})
    before = deepcopy(executions)
    started, release, visible = threading.Event(), threading.Event(), threading.Event()
    results, errors = {}, []

    def read(*_args):
        started.set()
        assert release.wait(2)
        raise OSError("unpublished evidence temporarily unavailable")

    service._objects = SimpleNamespace(get_bounded=Mock(side_effect=read))
    service.evaluation_icps = Mock(return_value=[{}, {}])
    service._store.list_runs = Mock(wraps=service._store.list_runs)

    def callback(submission_id, value):
        if submission_id == "stored-peer" and value is not None:
            visible.set()

    def refresh():
        try:
            results.update(service._read_completed_submission_scores(row, on_completed=callback))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=refresh)
    thread.start()
    try:
        assert started.wait(2)
        assert visible.is_set()  # No timing threshold: evidence is still blocked.
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive() and not errors
    assert results["stored-peer"]["final_score"] == 60.0
    assert set(results) == {"stored-peer"}
    assert executions == before
    assert service._submission_cost_eligibility.call_count == 1
    assert service._store.list_runs.call_count == 2
    assert service._objects.get_bounded.call_count == 1


@pytest.mark.parametrize("fault", ["missing", "unpriced", "unfinished", "membership"])
def test_persisted_priority_does_not_bypass_completion_guards(completed, fault):
    service, row, executions, judges, eligibility = completed
    _add_stored_peer(row, executions, judges)
    if fault == "missing":
        executions.pop()
    elif fault == "unpriced":
        service._submission_cost_eligibility.side_effect = lambda _row, sid, *_args, **_kw: (
            {**eligibility, "eligible": False} if sid == "stored-peer" else eligibility
        )
    elif fault == "unfinished":
        judges[-1]["status"] = "leased"
    else:
        judges[-1]["submission_id"] = "changed-membership"
    assert set(service._read_completed_submission_scores(row)) == {"miner"}


def test_priority_uses_selected_accepted_attempt_and_is_stable(completed):
    service, row, executions, judges, _ = completed
    _add_stored_peer(row, executions, judges)
    # A later failed attempt with a persisted value cannot replace the accepted
    # attempt. Zero is persisted, and participants with the same priority keep order.
    executions.append({**executions[0], "run_id": "failed-later", "attempt": 99,
                       "status": "failed", "per_icp_score": 999})
    executions[0]["per_icp_score"] = 0
    callbacks = []
    result = service._read_completed_submission_scores(
        row, on_completed=lambda sid, value: callbacks.append(sid) if value else None,
    )
    assert callbacks == ["miner", "stored-peer"]
    assert result["miner"]["final_score"] == 40.0
    assert result["stored-peer"]["final_score"] == 60.0
    assert service._submission_cost_eligibility.call_count == 2


def _unpublished_complete_with_incomplete_peers(completed, peer_counts):
    """Real score arithmetic with a complete model and deadline-isolated peers."""
    import json
    from lab_arena import scoring
    from tests.lab_arena.test_lab_arena_scoring import make_icp, scored_company, breakdown

    service, row, executions, judges, eligibility = completed
    row['configuration_doc'].update(stage_1_icp_count=5, stage_2_icp_count=5,
                                    scorer_policy=scoring.build_scorer_policy())
    executions[:] = [{
        'run_id': 'execute-%d' % p, 'round_id': row['round_id'],
        'submission_id': 'miner', 'stage': 2, 'kind': 'execute',
        'icp_position': p, 'attempt': 1, 'status': 'accepted',
        'terminal_cause': 'accepted', 'per_icp_score': None,
        'output_ref': 'output-%d' % p,
    } for p in range(10)]
    for peer, accepted in peer_counts.items():
        row['participants'].append({'submission_id': peer, 'is_king': False})
        for p in range(10):
            ok = p < accepted
            executions.append({
                **executions[0], 'run_id': '%s-%d' % (peer, p),
                'submission_id': peer, 'icp_position': p,
                'status': 'accepted' if ok else 'failed',
                'terminal_cause': 'accepted' if ok else 'stage_closed',
                'output_ref': '%s-output-%d' % (peer, p) if ok else None,
                'terminal_doc': {} if ok else {'infrastructure_incomplete': True},
            })
    judges[:] = [{**r, 'kind': 'score', 'run_id': 'judge-' + r['run_id'],
                 'scored_run_id': r['run_id']} for r in executions if r['status'] == 'accepted']
    row['stage2_scoring_plan_doc'] = scoring.build_scoring_plan(
        round_id=row['round_id'], stage=2, runs=executions,
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        configuration=row['configuration_doc'],
    )
    body = json.dumps({'schema_version': contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                       'companies': [scored_company(0)]}).encode()
    service._objects = SimpleNamespace(get_bounded=Mock(return_value=body))
    service.evaluation_icps = Mock(return_value=[{**make_icp(p), 'max_companies': 1} for p in range(10)])
    service._verified_breakdowns = Mock(return_value=[breakdown(40)])
    eligibility['cost_summary']['per_icp'] = [{'icp_position': p, 'eligible': True} for p in range(10)]
    return service, row, executions, eligibility


@pytest.mark.parametrize('peer_counts', [
    {'partial': 2},
    {'missing-one': 0, 'missing-two': 0, 'missing-three': 0},
    {'partial': 2, 'missing-one': 0, 'missing-two': 0, 'missing-three': 0},
])
def test_completed_unpublished_score_is_not_blocked_by_other_incomplete_models(completed, peer_counts):
    service, row, executions, _ = _unpublished_complete_with_incomplete_peers(completed, peer_counts)
    before = deepcopy((row, executions))
    result = service._read_completed_submission_scores(row)
    assert set(result) == {'miner'}
    assert result['miner']['final_score'] == 40.0
    assert result['miner']['eligible'] is True
    assert len(result['miner']['execution_runs']) == 10
    assert (row, executions) == before


def test_completed_projection_keeps_current_model_incomplete_and_cost_guard(completed):
    service, row, executions, eligibility = _unpublished_complete_with_incomplete_peers(completed, {'partial': 2})
    last = executions[9]
    last.update(status='failed', terminal_cause='stage_closed', output_ref=None,
                terminal_doc={'infrastructure_incomplete': True})
    from lab_arena import scoring
    row['stage2_scoring_plan_doc'] = scoring.build_scoring_plan(
        round_id=row['round_id'], stage=2, runs=executions,
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        configuration=row['configuration_doc'],
    )
    assert service._read_completed_submission_scores(row) == {}
    service._submission_cost_eligibility.assert_not_called()


def test_other_incomplete_models_do_not_bypass_completed_model_cost_guard(completed):
    service, row, _, eligibility = _unpublished_complete_with_incomplete_peers(completed, {'partial': 2})
    eligibility.update(eligible=False, eligibility_reason='provider_cost_uncertain')
    assert service._read_completed_submission_scores(row) == {}
    service._submission_cost_eligibility.assert_called_once()


def test_cancelled_round_uses_same_cached_verified_scores(completed):
    service, row, _, _, _ = completed
    row['status'] = 'cancelled'
    read = Mock(wraps=service._read_completed_submission_scores)
    service._read_completed_submission_scores = read
    result = _wait_for_scores(service, row, lambda scores: 'miner' in scores)
    assert result['miner']['final_score'] == 60.0
    for _ in range(3):
        assert service.completed_submission_scores(row) == result
    assert read.call_count == 1


@pytest.mark.parametrize('invalid', ['unfinished', 'unsettled', 'new_floor'])
def test_cancelled_round_does_not_expose_invalid_or_stale_results(completed, invalid):
    service, row, _, judges, eligibility = completed
    row['status'] = 'cancelled'
    if invalid == 'unfinished':
        judges[0].update(status='queued', terminal_cause=None)
    elif invalid == 'unsettled':
        eligibility.update(eligible=False, eligibility_reason='provider_cost_uncertain')
    else:
        row['scoring_generation_floor'] = 1
    assert service._read_completed_submission_scores(row) == {}


@pytest.mark.parametrize('after', ['cancelled', 'stage2_scoring'])
def test_cancelled_score_cache_never_crosses_recovery_floor(completed, after):
    service, row, _, _, _ = completed
    row['status'] = 'cancelled'
    _warm_verified_scores(service, row)
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(0)
    row.update(status=after, status_generation=2, scoring_generation_floor=1)
    assert service.completed_submission_scores(row) == {}
