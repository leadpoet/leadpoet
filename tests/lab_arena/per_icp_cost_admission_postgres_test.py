"""Focused SQL admission checks for the per-ICP sourcing policy."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _start_parallel_round,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha
from tests.lab_arena.test_lab_arena_migration_postgres import open_round
from tests.lab_arena.test_lab_arena_service_round import Harness


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + ("289-lab-arena-per-icp-cost-policy.sql",)
    )


def _reserve(store, lease, token, label, amount):
    identity = contracts.provider_call_identity(
        attempt=lease["attempt"], assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"], action_sequence=0,
        operation_id="scrapingdog.scrape", request_hash=sha(label),
    )
    result = store.reserve_call(
        run_id=lease["run_id"], lease_token_hash=hash_lease_token(token),
        call_identity=identity, operation_id="scrapingdog.scrape",
        provider="scrapingdog",
        funding_source=store.provider_funding(
            lease["run_id"], "scrapingdog"
        )["funding_source"],
        amount_microusd=amount, call_doc={"request_hash": sha(label)},
    )
    return identity, result


def _settle(store, lease, token, identity, amount, succeeded=True):
    lease_hash = hash_lease_token(token)
    assert store.mark_dispatched(
        run_id=lease["run_id"], lease_token_hash=lease_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.settle_call(
        run_id=lease["run_id"], lease_token_hash=lease_hash,
        call_identity=identity, actual_microusd=amount,
        terminal_response={"status": 200, "call_succeeded": succeeded},
    )["status"] == "settled"


def _uncertain(store, lease, token, identity, *, succeeded):
    lease_hash = hash_lease_token(token)
    assert store.mark_dispatched(
        run_id=lease["run_id"], lease_token_hash=lease_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=lease["run_id"], lease_token_hash=lease_hash,
        call_identity=identity,
        call_doc={"reason": "transport_failure", "call_succeeded": succeeded},
    )["status"] == "uncertain"


def test_paid_admission_is_per_icp_and_counts_retries(database, tmp_path):
    connect = lambda: database[0].connect(**database[1])
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults, per_icp_cost_policy=True,
        integrity_from="2000-01-01T00:00:00Z",
    )
    _start_parallel_round(harness, "arena-2099-01-31-c1", slot_ceiling=2)
    store = harness.service.store
    runner = harness.runner_keys[0]
    (first, first_token), (other, other_token) = [
        claim(store, harness.round_id, runner, parallelism=2, ceiling=2)[:2]
        for _ in range(2)
    ]
    assert [first["icp_position"], other["icp_position"]] == [0, 1]

    identity, reserved = _reserve(store, first, first_token, "first", 3_900_000)
    assert reserved["status"] == "reserved"
    _, busy = _reserve(store, first, first_token, "parallel", 1)
    assert busy["status"] == "budget_busy"
    _settle(store, first, first_token, identity, 3_900_000)

    overshoot_id, admitted = _reserve(
        store, first, first_token, "overshoot", 250_000
    )
    assert admitted["status"] == "reserved"
    _settle(store, first, first_token, overshoot_id, 250_000)
    _, refused = _reserve(store, first, first_token, "after-cap", 1)
    assert (refused["status"], refused["reason"]) == ("refused", "money_cap")

    # Another ICP remains independent. A billed failed request still uses the
    # admission budget, although it does not enter final cost eligibility.
    failed_id, admitted = _reserve(store, other, other_token, "failed", 1_000_000)
    assert admitted["status"] == "reserved"
    _settle(store, other, other_token, failed_id, 1_000_000, succeeded=False)
    next_id, admitted = _reserve(store, other, other_token, "after-failed", 200_000)
    assert admitted["status"] == "reserved"
    _settle(store, other, other_token, next_id, 200_000)

    cost = store.icp_cost_eligibility(
        round_id=harness.round_id,
        submission_id=first["submission_id"], icp_position=0,
        qualified_company_count=5,
    )
    assert cost["competition_sourcing_microusd"] == 4_150_000
    assert cost["eligibility_reason"] == "cost_per_company_exceeded"
    other_cost = store.icp_cost_eligibility(
        round_id=harness.round_id,
        submission_id=other["submission_id"], icp_position=1,
        qualified_company_count=1,
    )
    assert other_cost["competition_sourcing_microusd"] == 200_000
    assert other_cost["eligible"] is True

    # A failed billed call can itself overshoot the admission cap. It remains
    # excluded from competition cost, but no later paid call can start.
    failed_id, admitted = _reserve(
        store, other, other_token, "failed-over", 1
    )
    assert admitted["status"] == "reserved"
    _settle(store, other, other_token, failed_id, 3_000_001, succeeded=False)
    _, refused = _reserve(
        store, other, other_token, "after-failed-over", 1
    )
    assert (refused["status"], refused["reason"]) == ("refused", "money_cap")


def test_zero_cost_calls_do_not_close_paid_admission(database, tmp_path):
    connect = lambda: database[0].connect(**database[1])
    harness = Harness(connect, tmp_path, challengers=[], runners=["zero"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults, per_icp_cost_policy=True,
        integrity_from="2000-01-01T00:00:00Z",
    )
    _start_parallel_round(harness, "arena-2099-02-01-c2", slot_ceiling=2)
    store = harness.service.store
    lease, token = claim(
        store, harness.round_id, harness.runner_keys[0], parallelism=2, ceiling=2
    )[:2]
    free_id, free = _reserve(store, lease, token, "free", 0)
    assert free["status"] == "reserved"
    paid_id, paid = _reserve(store, lease, token, "paid", 1)
    assert paid["status"] == "reserved"
    assert free_id != paid["call_identity"]
    _settle(store, lease, token, free_id, 0)
    _settle(store, lease, token, paid_id, 200_000)

    uncertain_id, uncertain = _reserve(
        store, lease, token, "small-uncertain", 100_000
    )
    assert uncertain["status"] == "reserved"
    _uncertain(store, lease, token, uncertain_id, succeeded=True)
    _, admitted = _reserve(store, lease, token, "after-small-uncertain", 1)
    assert admitted["status"] == "reserved"
    _uncertain(
        store, lease, token, admitted["call_identity"], succeeded=False
    )
    cost = store.icp_cost_eligibility(
        round_id=harness.round_id, submission_id=lease["submission_id"],
        icp_position=lease["icp_position"], qualified_company_count=5,
    )
    assert cost["eligibility_reason"] == "provider_cost_uncertain"

    large_id, large = _reserve(
        store, lease, token, "large-uncertain", 4_000_000
    )
    assert large["status"] == "reserved"
    _uncertain(store, lease, token, large_id, succeeded=False)
    _, refused = _reserve(store, lease, token, "after-large-uncertain", 1)
    assert (refused["status"], refused["reason"]) == ("refused", "money_cap")


def test_marker_absent_round_keeps_legacy_aggregate_money_cap(database):
    connect = lambda: database[0].connect(**database[1])
    from lab_arena.store import ArenaStore, PsycopgTransport

    store = ArenaStore(PsycopgTransport(connect))
    round_id = "arena-2099-02-02-legacy"
    runners, _ = open_round(
        store, round_id, participants=1, runners=1, prefix="legacy-per-icp",
        execution_cap_microusd=100,
    )
    lease, token = claim(store, round_id, runners[0])[:2]
    _, refused = _reserve(store, lease, token, "legacy-cap", 101)
    assert (refused["status"], refused["reason"]) == ("refused", "money_cap")


def test_migration_replays_and_exposes_scoped_capability(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    migration = Path(__file__).resolve().parents[2] / "scripts" / (
        "289-lab-arena-per-icp-cost-policy.sql"
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(migration.read_text(encoding="utf-8"))
            cursor.execute("SELECT public.lab_arena_per_icp_cost_schema_v1()")
            assert cursor.fetchone()[0] == {
                "schema_version": "leadpoet.lab_arena.per_icp_cost_schema.v1",
                "version": 289,
                "policy": "successful_calls_per_icp_v1",
            }
    finally:
        connection.close()
