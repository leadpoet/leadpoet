"""Frozen judge history must not cause repeated publication attempts."""

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from types import SimpleNamespace

import pytest

from lab_arena.driver import drive_once
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStoreUnavailable


RUN_COLUMNS = "run_id,kind,stage,status,scored_run_id,per_icp_score"


def _round(*, day="2026-10-13", mode="live", created_at="2026-10-13T12:00:00Z",
           rewards_enabled=True):
    return {
        "round_id": "arena-test", "status": "scored", "evaluation_date": day,
        "created_at": created_at, "rewards_enabled": rewards_enabled,
        "configuration_doc": {
            "mode": mode,
            "schedule": {
                "submission_cutoff": "2026-10-13T13:00:00Z",
                "publication_deadline": "2026-10-13T20:30:00Z",
            },
        },
    }


def _runs(*, score=None, judge_status="failed", extra_status=None):
    rows = [
        {"run_id": "execute-1", "kind": "execute", "stage": 1,
         "status": "accepted", "scored_run_id": None, "per_icp_score": score},
        {"run_id": "judge-1", "kind": "score", "stage": 1,
         "status": judge_status, "scored_run_id": "execute-1", "per_icp_score": None},
    ]
    if extra_status:
        rows.append({"run_id": "judge-2", "kind": "score", "stage": 1,
                     "status": extra_status, "scored_run_id": "execute-1",
                     "per_icp_score": None})
    return rows


def _service(row, runs, *, billing=False, deepline_billing=False,
             read_error=False, pinned=False):
    events = []
    state = {"billing": billing, "deepline_billing": deepline_billing}
    original_row, original_runs = deepcopy(row), deepcopy(runs)

    class Store:
        def list_rounds(self, **kwargs):
            assert "evaluation_date,rewards_enabled" in kwargs["columns"]
            return [deepcopy(row)] if kwargs.get("offset") == 0 else []

        def get_round(self, round_id):
            assert round_id == row["round_id"]
            events.append("get_round")
            return deepcopy(row)

        def list_runs(self, round_id, *, columns="*", **kwargs):
            assert round_id == row["round_id"]
            assert columns == RUN_COLUMNS
            assert not kwargs
            events.append("compact_runs")
            return deepcopy(runs)

        def list_openrouter_cost_reconciliations(self, round_id, *, limit):
            assert round_id == row["round_id"] and limit == 1
            events.append("openrouter_candidates")
            if read_error:
                raise ArenaStoreUnavailable("synthetic read failure")
            return [{"uncertain_entry_id": 1}] if state["billing"] else []

        def list_deepline_cost_reconciliations(self, round_id, *, limit):
            assert round_id == row["round_id"] and limit == 1
            events.append("deepline_candidates")
            return [{"uncertain_entry_id": 2}] if state["deepline_billing"] else []

        def operator_hold_active(self):
            return False

        def expire_leases(self, round_id):
            assert round_id == row["round_id"]
            events.append("expire_leases")
            return {"status": "ok"}

    service = ArenaService.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(
        mode=row["configuration_doc"]["mode"],
        pinned_round_id=row["round_id"] if pinned else None,
        network_name="finney", netuid=71,
    )
    service.now = lambda: datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
    service._lock = RLock()
    service._invalidate_hot_round = lambda: None
    service._require_round_mode = lambda value: value
    service._public_round_output_policy = lambda config: {}
    def reconcile_deepline(round_id):
        events.append("deepline_billing")
        checked = int(state["deepline_billing"])
        state["deepline_billing"] = False
        return {"checked": checked}

    service._reconcile_active_deepline_cost = reconcile_deepline

    def reconcile_openrouter(round_id):
        events.append("openrouter_billing")
        state["billing"] = False
        return {"status": "settled" if billing else "none", "run_status": "failed"}

    service._reconcile_openrouter_cost = reconcile_openrouter
    service.publish = lambda round_id: events.append("publish") or {"status": "retry"}
    service.promote_pending_baselines = lambda: {"promoted": 0}
    service.ensure_daily_round = lambda: {"status": "existing"}
    service.activate_pending_rewards = lambda: {"activated": 0}
    service.reconcile_closed_provider_costs = lambda: {"status": "none"}
    return service, events, state, original_row, original_runs


def test_old_unfinished_round_is_not_polled_or_published_again():
    row, runs = _round(), _runs()
    service, events, _, before_row, before_runs = _service(row, runs)
    assert service.active_rounds(public_only=True) == []
    assert "get_round" not in events
    assert drive_once(service) == "idle"
    assert drive_once(service) == "idle"
    assert "publish" not in events
    assert "openrouter_billing" not in events
    assert row == before_row and runs == before_runs


def test_old_unfinished_round_reconciles_billing_without_publish_retry():
    service, events, _, _, _ = _service(_round(), _runs(), billing=True)
    assert service.active_rounds(public_only=True) == []
    assert "get_round" not in events
    assert "advanced arena-test" in drive_once(service)
    assert events.count("openrouter_billing") == 1
    assert events.count("publish") == 0
    assert drive_once(service) == "idle"
    assert events.count("openrouter_billing") == 1


def test_deepline_only_candidate_keeps_billing_without_publish_retry():
    service, events, _, _, _ = _service(_round(), _runs(), deepline_billing=True)
    assert service.active_rounds(public_only=True) == []
    assert "advanced arena-test" in drive_once(service)
    assert events.count("deepline_billing") == 1
    assert events.count("publish") == 0
    assert drive_once(service) == "idle"


def test_nine_accepted_judges_and_one_failed_after_retry_are_frozen():
    runs = []
    for position in range(10):
        run_id = "execute-%d" % position
        runs.append({"run_id": run_id, "kind": "execute", "stage": 1,
                     "status": "accepted", "scored_run_id": None,
                     "per_icp_score": 0 if position == 0 else (None if position == 9 else 42)})
        for attempt in ((1, 2) if position == 9 else (1,)):
            runs.append({"run_id": "judge-%d-%d" % (position, attempt),
                         "kind": "score", "stage": 1,
                         "status": "failed" if position == 9 else "accepted",
                         "scored_run_id": run_id, "per_icp_score": None})
    service, events, _, _, _ = _service(_round(), runs, billing=True)
    assert service.active_rounds(public_only=True) == []
    assert "advanced arena-test" in drive_once(service)
    assert "publish" not in events


@pytest.mark.parametrize("score,judge_status", [(0, "accepted"), (42, "accepted")])
def test_complete_scored_round_remains_publishable(score, judge_status):
    service, events, _, _, _ = _service(
        _round(), _runs(score=score, judge_status=judge_status))
    assert [row["round_id"] for row in service.active_rounds()] == ["arena-test"]
    assert "advanced arena-test" in drive_once(service)
    assert events.count("publish") == 1


@pytest.mark.parametrize("status", ["pending", "leased", "submitted"])
def test_active_score_work_keeps_old_round_eligible(status):
    service, events, _, _, _ = _service(_round(), _runs(extra_status=status))
    assert [row["round_id"] for row in service.active_rounds(public_only=True)] == ["arena-test"]
    assert "publish" not in events


def test_billing_read_error_keeps_driver_round_eligible():
    service, _, _, _, _ = _service(_round(), _runs(), read_error=True)
    assert [row["round_id"] for row in service.active_rounds()] == ["arena-test"]
    assert service.active_rounds(public_only=True) == []


@pytest.mark.parametrize("bad_field", ["missing_day", "malformed_day", "malformed_deadline"])
def test_incomplete_metadata_keeps_round_eligible_without_public_read_error(bad_field):
    row = _round()
    if bad_field == "missing_day":
        row.pop("evaluation_date")
    elif bad_field == "malformed_day":
        row["evaluation_date"] = "not-a-day"
    else:
        row["configuration_doc"]["schedule"]["publication_deadline"] = "not-a-time"
    service, events, _, _, _ = _service(row, _runs())
    assert [item["round_id"] for item in service.active_rounds(public_only=True)] == ["arena-test"]
    assert "compact_runs" not in events
    assert "advanced arena-test" in drive_once(service)
    assert events.count("publish") == 1


@pytest.mark.parametrize("row,pinned,expected", [
    (_round(day="2026-10-14"), False, True),
    (_round(mode="shadow", created_at="2026-10-13T16:31:00Z", rewards_enabled=False), True, True),
    (_round(mode="shadow", created_at="2026-10-13T16:29:00Z", rewards_enabled=False), True, False),
    (_round(mode="shadow", created_at="2026-10-13T16:31:00Z", rewards_enabled=True), True, False),
])
def test_rollout_and_pinned_shadow_discovery(row, pinned, expected):
    service, events, _, _, _ = _service(row, _runs(), pinned=pinned)
    assert bool(service.active_rounds(public_only=True)) is expected
    if expected:
        assert "compact_runs" not in events
