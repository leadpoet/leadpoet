"""Delayed judge billing cannot gate scoring, publication, or reward work."""
from types import SimpleNamespace

import pytest

from lab_arena.driver import drive_once
from lab_arena.service import ArenaService


def test_closed_billing_failure_runs_after_live_round_and_rewards():
    events = []

    def billing():
        events.append("billing")
        raise TimeoutError("provider pending")

    service = SimpleNamespace(
        promote_pending_baselines=lambda: {"promoted": 0},
        active_rounds=lambda: [{"round_id": "active", "status": "open"}],
        advance_round=lambda round_id: events.append("advance:" + round_id),
        activate_pending_rewards=lambda: events.append("rewards") or {"activated": 1},
        reconcile_closed_provider_costs=billing,
    )
    result = drive_once(service)
    assert events == ["advance:active", "rewards", "billing"]
    assert "activated rewards 1" in result
    assert "failed closed_provider_costs: TimeoutError" in result


@pytest.mark.parametrize("pinned", [None, "arena-pinned"])
def test_closed_billing_is_scoped_bounded_and_advances_cursor_on_missing_receipt(pinned):
    requests = []
    calls = []

    def candidate(**kwargs):
        requests.append(kwargs)
        return {"status": "ok", "uncertain_entry_id": 321,
                "round_id": pinned or "arena-closed", "run_id": "failed-score"}

    service = ArenaService.__new__(ArenaService)
    service._config = SimpleNamespace(mode="live", network_name="finney", netuid=71,
                                      pinned_round_id=pinned)
    service._store = SimpleNamespace(next_closed_deepline_reconciliation=candidate)
    service._closed_deepline_reconciliation_after = 123
    service._reconcile_deepline_cost = (
        lambda round_id, **kwargs: calls.append((round_id, kwargs)) or {"status": "pending"}
    )
    assert service.reconcile_closed_provider_costs() == {"status": "pending"}
    assert requests == [{"mode": "live", "network_name": "finney", "netuid": 71,
                         "round_id": pinned or "", "after_entry_id": 123}]
    assert calls == [(pinned or "arena-closed", {"run_id": "failed-score"})]
    assert service._closed_deepline_reconciliation_after == 321


def test_no_closed_candidate_performs_no_billing_read():
    service = ArenaService.__new__(ArenaService)
    service._config = SimpleNamespace(mode="live", network_name="finney", netuid=71)
    service._store = SimpleNamespace(
        next_closed_deepline_reconciliation=lambda **kwargs: {"status": "none"},
    )
    service._closed_deepline_reconciliation_after = 0
    assert service.reconcile_closed_provider_costs() == {"status": "none"}
