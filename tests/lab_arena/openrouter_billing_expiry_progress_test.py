"""An active billing wait preserves its lease but expires other overdue work."""

from __future__ import annotations

import threading
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lab_arena import contracts, service as svc
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import abandoned_host_recovery412_postgres_test as migration412
from tests.lab_arena.openrouter_delayed_cost_reconciliation_postgres_test import (
    _uncertain_call,
    database,
    resources,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, open_round


ACTIVE = ("stage1", "stage2", "stage1_scoring", "stage2_scoring")
NOW = datetime(2026, 10, 6, 3, 0, tzinfo=timezone.utc)


def _unit_service(status, *, paused=False, expired=False):
    service = object.__new__(svc.ArenaService)
    calls = []
    service._clock = lambda: NOW
    service._invalidate_hot_round = lambda: None
    service._round = lambda _round_id: {"status": status}
    service._reconcile_active_deepline_cost = lambda _round_id: {"status": "none"}
    service._reconcile_openrouter_cost = lambda _round_id: {
        "status": "pending",
        "run_status": "leased",
        "lease_expires_at": (NOW + timedelta(minutes=-1 if expired else 30)).isoformat().replace("+00:00", "Z"),
    }
    service._store = SimpleNamespace(
        operator_hold_active=lambda: paused,
        expire_leases=lambda round_id: calls.append(("expire", round_id)),
    )
    service._advance_round_locked = lambda round_id: (
        calls.append(("advance", round_id)) or {"status": "advanced"}
    )
    return service, calls


@pytest.mark.parametrize("status", ACTIVE)
def test_active_billing_wait_expires_overdue_work_without_closing(status):
    service, calls = _unit_service(status)
    assert service.advance_round("round") == {
        "status": "retry", "round_status": status,
        "reason": "provider_billing_pending",
    }
    assert calls == [("expire", "round")]


@pytest.mark.parametrize("status", ACTIVE)
def test_operator_hold_prevents_expiry_during_billing_wait(status):
    service, calls = _unit_service(status, paused=True)
    assert service.advance_round("round")["reason"] == "provider_billing_pending"
    assert calls == []


@pytest.mark.parametrize("status", (
    "committed", "stage1_closed", "stage1_judged", "stage1_scored",
    "stage2_closed", "stage2_judged", "scored",
))
def test_billing_wait_does_not_add_expiry_to_other_round_states(status):
    service, calls = _unit_service(status)
    assert service.advance_round("round")["reason"] == "provider_billing_pending"
    assert calls == []


@pytest.mark.parametrize("status", ACTIVE)
def test_expired_billing_lease_uses_normal_round_progress(status):
    service, calls = _unit_service(status, expired=True)
    assert service.advance_round("round") == {"status": "advanced"}
    assert calls == [("advance", "round")]


@pytest.mark.parametrize("status", ("stage1", "stage1_scoring"))
def test_live_billing_lease_still_defers_a_passed_close_deadline(status):
    service, calls = _unit_service(status)
    service._lock = threading.RLock()
    stage = "1"
    row = {"status": status, "configuration_doc": {"schedule": {
        "submission_cutoff": (NOW - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "stage_1_close": (NOW - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "stage_1_scoring_close": (NOW - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
    }}}
    service._round = lambda _round_id: row
    service._advance_round_locked = svc.ArenaService._advance_round_locked.__get__(service)
    service._store.list_runs = lambda *_args, **_kwargs: [{"status": "leased"}]
    service.close_stage = lambda *_args: calls.append(("close", stage)) or {"status": "closed"}
    service.close_scoring = service.close_stage
    assert service.advance_round("round")["reason"] == "provider_billing_pending"
    assert calls == [("expire", "round")]
    service._reconcile_openrouter_cost = lambda _: {
        "status": "pending", "run_status": "leased",
        "lease_expires_at": (NOW - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
    }
    assert service.advance_round("round") == {"status": "closed"}
    assert calls[-2:] == [("expire", "round"), ("close", stage)]


def _database_service(store, round_id, *, paused):
    service = object.__new__(svc.ArenaService)
    service._store = store
    service._store.operator_hold_active = lambda: paused
    service._lock = threading.RLock()
    service._clock = lambda: datetime.now(timezone.utc)
    service._invalidate_hot_round = lambda: None
    service._openrouter_reconciliation_after = {}
    service._reconcile_active_deepline_cost = lambda _: {"status": "none"}
    service._broker_for = lambda _: SimpleNamespace(
        reconcile_openrouter_cost=lambda _candidate: {"status": "pending"},
    )
    def round_row(_round_id):
        row = store.get_round(round_id)
        row["configuration_doc"]["schedule"].update({
            "submission_cutoff": (service.now() - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "stage_1_close": (service.now() + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        })
        return row
    service._round = round_row
    return service


@pytest.mark.parametrize("paused", (False, True))
def test_database_billing_wait_preserves_live_run_and_expired_retry(resources, paused):
    store, connect = resources
    label = "billpause" if paused else "billexpiry"
    round_id = "arena-2026-10-06-%s" % label
    runners, _ = open_round(store, round_id, participants=1, runners=2,
                            prefix=label, execution_cap_microusd=10_000_000)
    billed, billed_token, _, _ = claim(store, round_id, runners[0], parallelism=2)
    expired, _, _, _ = claim(store, round_id, runners[0], parallelism=2)
    assert billed["status"] == expired["status"] == "leased"
    identity = _uncertain_call(
        store, billed, hash_lease_token(billed_token), label=label, sequence=0,
        generation_id="gen-" + label, credential_fingerprint="sha256:" + "a" * 64,
    )
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at="
                           "clock_timestamp()-interval '1 second' WHERE run_id=%s",
                           (expired["run_id"],))
    before_billed = store.get_run(billed["run_id"])
    before_ledger = store.list_ledger(call_identity=identity)
    service = _database_service(store, round_id, paused=paused)
    assert service.advance_round(round_id)["reason"] == "provider_billing_pending"
    assert store.get_run(billed["run_id"]) == before_billed
    assert store.list_ledger(call_identity=identity) == before_ledger
    assert store.get_round(round_id)["status"] == "stage1"
    original = store.get_run(expired["run_id"])
    retry_id = expired["assignment_id"] + ":2"
    if paused:
        assert original["status"] == "leased"
        assert store.get_run(retry_id) is None
    else:
        assert original["status"] == "failed"
        assert original["terminal_cause"] == "lease_expired"
        assert store.get_run(retry_id)["status"] == "pending"
        reclaimed, _, _, _ = claim(store, round_id, runners[1])
        assert reclaimed["status"] == "leased"
        assert reclaimed["run_id"] == retry_id


@pytest.fixture(scope="module")
def database412():
    database_generator = migration412.database.__wrapped__()
    try:
        database = next(database_generator)
        migration412.migrated.__wrapped__(database)
        yield database
    finally:
        database_generator.close()


@pytest.mark.parametrize("paused", (False, True))
@pytest.mark.parametrize("recovery", ("overdue", "host_error"))
def test_migration412_billing_wait_preserves_paid_work_and_recovers_sibling(
    database412, paused, recovery,
):
    psycopg, dsn = database412
    connections = []
    def connect():
        connection = psycopg.connect(**dsn)
        connections.append(connection)
        return connection
    store = ArenaStore(PsycopgTransport(connect), lease_ttl_seconds=4500)
    prior = migration412.prior
    try:
        with connect() as connection:
            billed = migration412._seed(connection, kind="execute", event=False)
            sibling = migration412._claim(connection, prior.RUNNER_A, "b")
            assert sibling["status"] == "leased"
            if recovery == "overdue":
                with connection.cursor() as cursor:
                    cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at="
                                   "clock_timestamp()-interval '1 second' WHERE run_id=%s",
                                   (sibling["run_id"],))
                connection.commit()
            else:
                events = [{"event_id": str(uuid.uuid4()), "kind": "runtime.error",
                           "occurred_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                           "content": {"status": "abandoned", "failure_stage": "runtime",
                                       "error_class": "RuntimeHostError"}}]
                with connection.cursor() as cursor:
                    cursor.execute("SET ROLE lab_arena_service")
                    cursor.execute("SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)",
                                   (sibling["run_id"], "sha256:" + "b" * 64, json.dumps(events)))
                    assert cursor.fetchone()[0]["status"] == "accepted"
                    cursor.execute("RESET ROLE")
                connection.commit()
        identity = contracts.document_hash({"case": "billing412", "paused": paused,
                                            "recovery": recovery})
        arguments = {"run_id": billed["run_id"], "lease_token_hash": "sha256:" + "a" * 64,
                     "call_identity": identity}
        assert store.reserve_call(
            **arguments, operation_id="openrouter.chat", provider="openrouter",
            funding_source="host", amount_microusd=1000, call_doc={"model": "fixture"},
        )["status"] == "reserved"
        assert store.mark_dispatched(**arguments)["status"] == "dispatched"
        assert store.mark_uncertain(**arguments, call_doc={
            "reason": "missing_provider_cost", "call_succeeded": True,
            "provider_status": 200, "openrouter_generation_id": "gen-billing412",
            "credential_fingerprint": "sha256:" + "a" * 64,
        })["status"] == "uncertain"
        before_billed = store.get_run(billed["run_id"])
        before_ledger = store.list_ledger(call_identity=identity)
        before_sibling = store.get_run(sibling["run_id"])
        service = _database_service(store, prior.ROUND, paused=paused)
        assert service.advance_round(prior.ROUND)["reason"] == "provider_billing_pending"
        assert store.get_run(billed["run_id"]) == before_billed
        assert store.list_ledger(call_identity=identity) == before_ledger
        retry_id = sibling["assignment_id"] + ":2"
        if paused:
            assert store.get_run(sibling["run_id"]) == before_sibling
            assert store.get_run(retry_id) is None
        else:
            original = store.get_run(sibling["run_id"])
            assert original["status"] == "failed"
            assert original["terminal_cause"] == "lease_expired"
            if recovery == "host_error":
                assert original["terminal_doc"]["recovery_reason"] == "authenticated_zero_call_runtime_host_error"
                assert original["terminal_doc"]["original_lease_expires_at"] == before_sibling["lease_expires_at"]
            assert store.get_run(retry_id)["status"] == "pending"
            with connect() as connection:
                retry = migration412._claim(connection, prior.RUNNER_B, "c")
                assert retry["status"] == "leased"
                assert retry["run_id"] == retry_id
    finally:
        store.close()
        for connection in connections:
            if not connection.closed:
                connection.close()
