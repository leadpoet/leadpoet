"""Exact 6300-second lease support on the real Arena PostgreSQL lifecycle."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStoreError, hash_lease_token, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import sha
from tests.lab_arena.test_lab_arena_service_round import Harness, _start_round, keypair


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/329-lab-arena-explicit-90m-lease.sql"
SIGNATURES = (
    "public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)",
    "public.lab_arena_mark_uncertain(text,text,text,jsonb,integer)",
    "public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)",
    "public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)",
)


@pytest.fixture(scope="module")
def database():
    migrations = CURRENT_SERVICE_MIGRATIONS + (
        "264-lab-arena-codex-cost-reconciliation.sql",
        "289-lab-arena-per-icp-cost-policy.sql",
        "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
        "312-lab-arena-temporary-hold-admission.sql",
        "314-lab-arena-openrouter-web-search-reservation.sql",
        "319-lab-arena-quota-sourcing-cost.sql",
        "321-lab-arena-confirmed-cost-admission.sql",
    )
    for psycopg2, dsn in database_with_lab_arena_migration(migrations):
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT p.oid::regprocedure::text,pg_get_userbyid(p.proowner),"
                "coalesce(p.proacl::text,'') FROM pg_proc p WHERE p.oid=ANY(%s::regprocedure[]) "
                "ORDER BY 1",
                (list(SIGNATURES),),
            )
            privileges_before = cursor.fetchall()
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
            cursor.execute(
                "SELECT p.oid::regprocedure::text,pg_get_userbyid(p.proowner),"
                "coalesce(p.proacl::text,'') FROM pg_proc p WHERE p.oid=ANY(%s::regprocedure[]) "
                "ORDER BY 1",
                (list(SIGNATURES),),
            )
            assert cursor.fetchall() == privileges_before
        yield psycopg2, dsn


def _service_claim(harness: Harness) -> dict:
    runner = keypair("svc-runner-alpha")
    policy = harness.service.store.get_round(harness.round_id)[
        "configuration_doc"
    ].get("checkpoint_deadline_policy")
    body = {"declared_parallelism": 1}
    if policy:
        body["checkpoint_deadline_policies"] = [policy]
    envelope = contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM,
        round_id=harness.round_id,
        hotkey=runner.ss58_address,
        body=body,
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: runner.sign(message.encode()).hex(),
    )
    return harness.service.handle_claim(envelope)


def _provider_identity(lease: dict, label: str) -> str:
    return contracts.provider_call_identity(
        attempt=lease["attempt"],
        assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"],
        action_sequence=900,
        operation_id="scrapingdog.scrape",
        request_hash=sha(label),
    )


def _exercise_cost_recovery(store, lease: dict, ttl: int, label: str) -> str:
    identity = _provider_identity(lease, label)
    token_hash = hash_lease_token(lease["lease_token"])
    assert store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        operation_id="scrapingdog.scrape",
        provider="scrapingdog",
        funding_source=store.provider_funding(
            lease["run_id"], "scrapingdog"
        )["funding_source"],
        amount_microusd=1234,
        call_doc={"request_hash": sha(label)},
        lease_ttl_seconds=ttl,
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=lease["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.settle_call(
        run_id=lease["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        actual_microusd=1234,
        terminal_response={"status": 200, "call_succeeded": True},
        lease_ttl_seconds=ttl,
    )["status"] == "settled"
    return identity


def _exercise_uncertain_recovery(store, lease: dict, ttl: int) -> str:
    identity = contracts.provider_call_identity(
        attempt=lease["attempt"], assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"], action_sequence=901,
        operation_id="scrapingdog.scrape", request_hash=sha("uncertain-90m"),
    )
    request_id = "ctx-tool-" + identity.removeprefix("sha256:")[:32]
    fingerprint = "sha256:" + "a" * 64
    token_hash = hash_lease_token(lease["lease_token"])
    assert store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        operation_id="scrapingdog.scrape", provider="deepline",
        funding_source=store.provider_funding(lease["run_id"], "deepline")[
            "funding_source"
        ],
        amount_microusd=1234,
        call_doc={
            "request_hash": sha("uncertain-90m"), "tool": "firecrawl_scrape",
            "deepline_request_id": request_id,
            "credential_fingerprint": fingerprint,
        },
        lease_ttl_seconds=ttl,
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=lease["run_id"], lease_token_hash=token_hash,
        call_identity=identity,
        call_doc={
            "reason": "transport_failure", "call_succeeded": False,
            "deepline_request_id": request_id,
            "deepline_operation": "firecrawl_scrape",
            "credential_fingerprint": fingerprint,
        },
        lease_ttl_seconds=ttl,
    )["status"] == "uncertain"
    candidate = next(
        item for item in store.list_deepline_cost_reconciliations(lease["round_id"])
        if item["call_identity"] == identity
    )
    assert store.reconcile_deepline_cost(
        round_id=candidate["round_id"], run_id=candidate["run_id"],
        call_identity=identity, uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=request_id, operation="firecrawl_scrape",
        credential_fingerprint=fingerprint, actual_microusd=1234,
        cost_units="0.01234",
    )["status"] == "settled"
    return identity


def test_exact_6300_lease_reaches_publication_and_old_profile_still_works(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)

    (tmp_path / "old").mkdir()
    old = Harness(connect, tmp_path / "old", challengers=[], runners=["alpha"])
    old.service.config.defaults = replace(
        old.service.config.defaults,
        checkpoint_deadline_enabled=True,
        runner_slot_ceiling=20,
        stage_minutes={
            "benchmark": 30, "stage_1": 420, "stage_1_scoring": 180,
            "stage_2": 420, "final_scoring": 180,
        },
    )
    _start_round(old, day=27, epoch=32500)
    assert old.service.store.get_round(old.round_id)["configuration_doc"][
        "lease_ttl_seconds"
    ] == 3600
    old.clock.advance_to(old.schedule()["stage_1_start"])
    assert old.service.advance_round(old.round_id)["status"] == "ok"
    old_lease = _service_claim(old)
    assert old_lease["lease_ttl_seconds"] == 3600
    _exercise_cost_recovery(old.service.store, old_lease, 3600, "old-profile")
    assert old.service.store.cancel_round(old.round_id, "operator_abort")["status"] == "cancelled"

    monkeypatch.setattr(
        contracts, "CHECKPOINT_DEADLINE_POLICY", contracts.CHECKPOINT_90M_DEADLINE_POLICY
    )
    monkeypatch.setattr(
        contracts, "CHECKPOINT_WALL_CLOCK_SECONDS", contracts.CHECKPOINT_90M_WALL_CLOCK_SECONDS
    )
    monkeypatch.setattr(
        contracts, "CHECKPOINT_LEASE_TTL_SECONDS", contracts.CHECKPOINT_90M_LEASE_TTL_SECONDS
    )
    (tmp_path / "new").mkdir()
    harness = Harness(connect, tmp_path / "new", challengers=[], runners=["alpha"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        checkpoint_deadline_enabled=True,
        per_icp_cost_policy=True,
        integrity_from="2000-01-01T00:00:00Z",
        runner_slot_ceiling=20,
        stage_minutes={
            "benchmark": 30, "stage_1": 420, "stage_1_scoring": 180,
            "stage_2": 420, "final_scoring": 180,
        },
    )
    _start_round(harness, day=28, epoch=32600)
    configuration = harness.service.store.get_round(harness.round_id)[
        "configuration_doc"
    ]
    assert (
        configuration["checkpoint_deadline_policy"],
        configuration["icp_wall_clock_seconds"],
        configuration["lease_ttl_seconds"],
    ) == ("atomic_checkpoint_90m_v1", 5400, 6300)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    lease = _service_claim(harness)
    assert lease["lease_ttl_seconds"] == 6300

    store = harness.service.store
    with pytest.raises(ArenaStoreError, match="claim_input_invalid"):
        store.claim_assignment(
            round_id=harness.round_id,
            runner_hotkey=harness.runner_keys[0],
            declared_parallelism=1,
            slot_ceiling=8,
            excluded_miner_hotkeys=[],
            request_id=contracts.new_request_id(),
            request_hash=sha("bad-claim"),
            lease_token_hash=hash_lease_token(new_lease_token()),
            lease_ttl_seconds=6301,
        )
    with pytest.raises(ArenaStoreError, match="reserve_input_invalid"):
        store.reserve_call(
            run_id=lease["run_id"], lease_token_hash=hash_lease_token(lease["lease_token"]),
            call_identity=_provider_identity(lease, "bad-reserve"),
            operation_id="scrapingdog.scrape", provider="scrapingdog",
            funding_source="host", amount_microusd=0, call_doc={},
            lease_ttl_seconds=6301,
        )

    settled_identity = _exercise_cost_recovery(store, lease, 6300, "90m-settle")
    with pytest.raises(ArenaStoreError, match="uncertain_input_invalid"):
        store.mark_uncertain(
            run_id=lease["run_id"], lease_token_hash=hash_lease_token(lease["lease_token"]),
            call_identity=settled_identity, call_doc={}, lease_ttl_seconds=6301,
        )
    with pytest.raises(ArenaStoreError, match="settle_input_invalid"):
        store.settle_call(
            run_id=lease["run_id"], lease_token_hash=hash_lease_token(lease["lease_token"]),
            call_identity=settled_identity, actual_microusd=1234,
            terminal_response={"status": 200}, lease_ttl_seconds=6301,
        )

    identity = _exercise_uncertain_recovery(store, lease, 6300)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT entry_kind,amount_microusd FROM public.lab_arena_ledger "
            "WHERE call_identity=%s ORDER BY entry_id",
            (identity,),
        )
        assert cursor.fetchall() == [
            ("reservation", 0),
            ("dispatch", 0),
            ("uncertain", 0),
            ("settlement", 1234),
        ]
    assert store.cancel_round(harness.round_id, "operator_abort")["status"] == "cancelled"

    (tmp_path / "published").mkdir()
    published_harness = Harness(
        connect, tmp_path / "published", challengers=[], runners=["alpha"]
    )
    published_harness.service.config.defaults = replace(
        published_harness.service.config.defaults,
        checkpoint_deadline_enabled=True,
        runner_slot_ceiling=20,
        stage_minutes={
            "benchmark": 30, "stage_1": 420, "stage_1_scoring": 180,
            "stage_2": 420, "final_scoring": 180,
        },
    )
    _start_round(published_harness, day=29, epoch=32700)
    published_harness.clock.advance_to(
        published_harness.schedule()["stage_1_start"]
    )
    assert published_harness.service.advance_round(
        published_harness.round_id
    )["status"] == "ok"
    published_harness.advance_until("published", runners=1)
    published = published_harness.service.store.get_round(
        published_harness.round_id
    )
    assert published["status"] == "published"
    assert published["configuration_doc"]["lease_ttl_seconds"] == 6300


def test_migration_is_exact_and_has_no_architecture_additions():
    body = MIGRATION.read_text()
    assert body.count("pg_get_functiondef") == 8
    assert body.count("<> 6300") == 4
    assert "CREATE TRIGGER" not in body
    assert "CREATE FUNCTION" not in body
    assert "CREATE OR REPLACE FUNCTION" not in body
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
