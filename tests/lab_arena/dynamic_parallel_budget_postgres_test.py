"""Shared-budget admission for dynamic calls across parallel ICP leases."""

from __future__ import annotations

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _start_parallel_round,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha
from tests.lab_arena.test_lab_arena_service_round import (
    Harness,
    connect,
    database,
)


def _call_identity(lease: dict, label: str) -> str:
    return contracts.provider_call_identity(
        attempt=lease["attempt"],
        assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha(label),
    )


def _reserve(
    store,
    lease: dict,
    token: str,
    identity: str,
    *,
    amount_microusd: int,
    dynamic: bool = False,
) -> dict:
    call_doc = {"tool": "exa_search" if dynamic else "harvestapi_get_job"}
    if dynamic:
        call_doc["reserve_remaining_budget"] = True
    funding = store.provider_funding(lease["run_id"], "deepline")
    return store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        operation_id="deepline.execute",
        provider="deepline",
        funding_source=funding["funding_source"],
        amount_microusd=amount_microusd,
        call_doc=call_doc,
    )


def _settle(store, lease: dict, token: str, identity: str, amount: int) -> None:
    lease_hash = hash_lease_token(token)
    assert store.mark_dispatched(
        run_id=lease["run_id"],
        lease_token_hash=lease_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.settle_call(
        run_id=lease["run_id"],
        lease_token_hash=lease_hash,
        call_identity=identity,
        actual_microusd=amount,
        terminal_response={"status": 200},
    )["status"] == "settled"


def test_dynamic_reservation_serializes_ten_parallel_icps_without_identity_leakage(
    connect, tmp_path,
):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    participants = _start_parallel_round(
        harness, "arena-2099-01-01-db", slot_ceiling=10
    )
    assert len(participants) == 1
    store = harness.service.store
    runner = harness.runner_keys[0]

    leases = [
        claim(
            store,
            harness.round_id,
            runner,
            parallelism=10,
            ceiling=10,
        )[:2]
        for _ in range(10)
    ]
    assert all(lease["status"] == "leased" for lease, _token in leases)
    assert len({lease["run_id"] for lease, _token in leases}) == 10
    assert len({lease["submission_id"] for lease, _token in leases}) == 1
    assert [lease["icp_position"] for lease, _token in leases] == list(range(10))

    dynamic_lease, dynamic_token = leases[0]
    dynamic_identity = _call_identity(dynamic_lease, "dynamic-owner")
    dynamic = _reserve(
        store,
        dynamic_lease,
        dynamic_token,
        dynamic_identity,
        amount_microusd=0,
        dynamic=True,
    )
    execution_cap = int(
        store.get_round(harness.round_id)["configuration_doc"][
            "execution_cap_microusd"
        ]
    )
    assert dynamic == {
        "status": "reserved",
        "idempotent": False,
        "call_identity": dynamic_identity,
        "amount_microusd": execution_cap,
        "lease_expires_at": dynamic["lease_expires_at"],
    }
    assert store.mark_dispatched(
        run_id=dynamic_lease["run_id"],
        lease_token_hash=hash_lease_token(dynamic_token),
        call_identity=dynamic_identity,
    )["status"] == "dispatched"

    blocked_lease, blocked_token = leases[1]
    blocked_identity = _call_identity(blocked_lease, "fixed-after-dynamic")
    # 1,000 micro-USD is the existing fixed reservation for
    # harvestapi_get_job. The test does not invent a price for a dynamic tool.
    blocked = _reserve(
        store,
        blocked_lease,
        blocked_token,
        blocked_identity,
        amount_microusd=1_000,
    )
    assert blocked["status"] == "budget_busy"
    assert blocked["call_identity"] == blocked_identity
    assert store.list_ledger(call_identity=blocked_identity) == []
    assert store.mark_dispatched(
        run_id=blocked_lease["run_id"],
        lease_token_hash=hash_lease_token(blocked_token),
        call_identity=blocked_identity,
    )["status"] == "not_reserved"
    assert store.list_ledger(call_identity=blocked_identity) == []

    # This is a terminal accounted charge in the fixture, not a reservation
    # estimate or a proposed price for exa_search.
    assert store.settle_call(
        run_id=dynamic_lease["run_id"],
        lease_token_hash=hash_lease_token(dynamic_token),
        call_identity=dynamic_identity,
        actual_microusd=2_000,
        terminal_response={"status": 200},
    )["status"] == "settled"

    own_identities = [dynamic_identity, blocked_identity]
    assert _reserve(
        store,
        blocked_lease,
        blocked_token,
        blocked_identity,
        amount_microusd=1_000,
    )["status"] == "reserved"
    _settle(store, blocked_lease, blocked_token, blocked_identity, 1_000)

    settled_rows = store.list_ledger(call_identity=blocked_identity)
    assert [row["entry_kind"] for row in settled_rows] == [
        "reservation",
        "dispatch",
        "settlement",
    ]
    assert _reserve(
        store,
        blocked_lease,
        blocked_token,
        blocked_identity,
        amount_microusd=1_000,
    )["status"] == "settled"
    assert store.mark_dispatched(
        run_id=blocked_lease["run_id"],
        lease_token_hash=hash_lease_token(blocked_token),
        call_identity=blocked_identity,
    )["status"] == "settled"
    assert store.settle_call(
        run_id=blocked_lease["run_id"],
        lease_token_hash=hash_lease_token(blocked_token),
        call_identity=blocked_identity,
        actual_microusd=1_000,
        terminal_response={"status": 200},
    )["status"] == "settled"
    assert store.list_ledger(call_identity=blocked_identity) == settled_rows

    foreign_lease, foreign_token = leases[2]
    with pytest.raises(ArenaStoreError, match="lab_arena_call_identity_foreign"):
        _reserve(
            store,
            foreign_lease,
            foreign_token,
            blocked_identity,
            amount_microusd=1_000,
        )

    for index, (lease, token) in enumerate(leases[2:], start=2):
        identity = _call_identity(lease, "fixed-owner-%d" % index)
        own_identities.append(identity)
        reserved = _reserve(
            store,
            lease,
            token,
            identity,
            amount_microusd=1_000,
        )
        assert reserved["status"] == "reserved"
        _settle(store, lease, token, identity, 1_000)

    assert len(set(own_identities)) == 10
    for (lease, _token), identity in zip(leases, own_identities):
        rows = store.list_ledger(call_identity=identity)
        assert [row["entry_kind"] for row in rows] == [
            "reservation",
            "dispatch",
            "settlement",
        ]
        assert {row["run_id"] for row in rows} == {lease["run_id"]}
    settlements = store.list_ledger(
        submission_id=dynamic_lease["submission_id"],
        entry_kind="settlement",
    )
    assert len(settlements) == 10
    assert sum(row["amount_microusd"] for row in settlements) == 11_000
