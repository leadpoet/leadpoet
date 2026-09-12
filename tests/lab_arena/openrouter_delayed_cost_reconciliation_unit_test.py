"""Service timing for exact OpenRouter billing reconciliation."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from types import SimpleNamespace

from lab_arena import service as svc
from scripts import validate_miner_testnet


def _candidate(entry_id: int):
    return {
        "uncertain_entry_id": entry_id,
        "round_id": "arena-2026-09-12",
        "run_id": "run-%d" % entry_id,
        "submission_id": "submission-1",
        "miner_hotkey": "miner-1",
        "assignment_id": "assignment-%d" % entry_id,
        "stage": 1,
        "icp_position": 0,
        "attempt": 1,
        "kind": "score",
        "call_identity": "sha256:" + ("%064x" % entry_id),
        "generation_id": "gen-%d" % entry_id,
        "credential_fingerprint": "sha256:" + "a" * 64,
        "funding_source": "host",
        "run_status": "leased",
        "lease_expires_at": "2026-09-12T01:20:00Z",
    }


class CursorStore:
    def __init__(self):
        self.items = [_candidate(10), _candidate(20)]
        self.after = []

    def list_openrouter_cost_reconciliations(
        self, round_id, *, run_id, after_entry_id, limit
    ):
        assert round_id == "arena-2026-09-12" and limit == 1
        self.after.append(after_entry_id)
        available = [
            item
            for item in self.items
            if not run_id or item["run_id"] == run_id
        ]
        return sorted(
            available,
            key=lambda item: (
                item["uncertain_entry_id"] <= after_entry_id,
                item["uncertain_entry_id"],
            ),
        )[:1]


class CursorBroker:
    def __init__(self, lock):
        self.lock = lock
        self.seen = []

    def reconcile_openrouter_cost(self, candidate):
        # The exact provider GET must run outside the service transition lock.
        assert self.lock.acquire(blocking=False)
        self.lock.release()
        self.seen.append(candidate["uncertain_entry_id"])
        return {"status": "pending" if len(self.seen) == 1 else "settled"}


def _bare_service(store, broker):
    service = object.__new__(svc.ArenaService)
    service._store = store
    service._lock = threading.RLock()
    service._brokers = {"arena-2026-09-12": broker}
    service._openrouter_reconciliation_after = {}
    return service


def test_unavailable_first_candidate_does_not_starve_the_next_candidate():
    store = CursorStore()
    lock = threading.RLock()
    broker = CursorBroker(lock)
    service = _bare_service(store, broker)
    service._lock = lock

    first = service._reconcile_openrouter_cost("arena-2026-09-12")
    second = service._reconcile_openrouter_cost("arena-2026-09-12")

    assert first["status"] == "pending"
    assert second["status"] == "settled"
    assert broker.seen == [10, 20]
    assert store.after == [0, 10]


def test_advance_defers_exhaustion_while_exact_billing_run_lease_is_active():
    service = object.__new__(svc.ArenaService)
    service._invalidate_hot_round = lambda: None
    service._round = lambda _round_id: {"status": "stage1_scoring"}
    service._reconcile_openrouter_cost = lambda _round_id: {
        "status": "pending",
        "run_status": "leased",
        "lease_expires_at": "2026-09-12T01:20:00Z",
    }
    service._clock = lambda: datetime(2026, 9, 12, 1, 10, tzinfo=timezone.utc)
    service._advance_round_locked = lambda _round_id: {
        "status": "would_cancel"
    }

    assert service.advance_round("arena-2026-09-12") == {
        "status": "retry",
        "round_status": "stage1_scoring",
        "reason": "provider_billing_pending",
    }


def test_completion_uses_existing_accounting_open_retry_before_new_attempt(monkeypatch):
    service = object.__new__(svc.ArenaService)
    service._request_round = lambda _envelope, scope, hot: (
        {
            "hotkey": "runner",
            "body": {"run_id": "run-1", "result": {"terminal_status": "judge_error"}},
        },
        {
            "round_id": "arena-2026-09-12",
            "configuration_doc": {},
        },
    )
    service._require_validator_authority = lambda _hotkey: None
    service._store = SimpleNamespace(
        get_run=lambda _run_id: {
            "run_id": "run-1",
            "round_id": "arena-2026-09-12",
            "runner_hotkey": "runner",
            "kind": "score",
            "status": "leased",
        }
    )
    service._lease_token_for_run = lambda _validated, _run: "lease"
    service._reconcile_openrouter_cost = lambda _round_id, run_id: {
        "status": "pending"
    }
    monkeypatch.setattr(
        svc.contracts, "validate_run_result", lambda document: dict(document)
    )

    assert service.handle_complete({}) == {
        "status": "accounting_open",
        "open_calls": 1,
    }


def test_fresh_testnet_bootstrap_installs_reconciliation_prerequisite_and_rpc():
    migrations = validate_miner_testnet.MIGRATIONS
    prerequisite = "scripts/223-lab-arena-cancelled-call-late-settlement.sql"
    reconciliation = "scripts/225-lab-arena-openrouter-delayed-cost-reconciliation.sql"
    assert migrations.count(prerequisite) == 1
    assert migrations.count(reconciliation) == 1
    assert migrations.index(prerequisite) < migrations.index(reconciliation)
