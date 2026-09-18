"""Opt-in twenty-ICP execution scheduling on the current PostgreSQL schema."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.service import ServiceError
from lab_arena.proxy_workers import (
    ProxyWorkerPool,
    VerifiedProxyWorker,
    VerifiedProxyWorkerInventory,
)
from lab_arena.service import ServiceError
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete
from tests.lab_arena.test_lab_arena_service_round import (
    Harness,
    connect,
    database,
    keypair,
)

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "255-lab-arena-parallel-twenty-icp-execution.sql"
)


def _verified_test_pool(process_capacity: int) -> ProxyWorkerPool:
    workers = tuple(
        VerifiedProxyWorker(
            slot_index=index + 1,
            source_name="test_proxy_%d" % (index + 1),
            proxy_url="http://worker:secret@proxy-%d.example:8080" % index,
            proxy_fingerprint="proxy%011d" % index,
            exit_ip="8.8.4.%d" % (index + 1),
            exit_ip_fingerprint="exit%012d" % index,
        )
        for index in range(process_capacity - 1)
    )
    return ProxyWorkerPool(
        VerifiedProxyWorkerInventory(
            workers=workers,
            native_exit_ip="1.1.1.1",
            native_exit_ip_fingerprint="nativeexit000000",
        )
    )


def _start_parallel_round(
    harness: Harness, round_id: str, *, slot_ceiling: int
) -> list[dict]:
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        runner_slot_ceiling=slot_ceiling,
        parallel_twenty_icp_execution=True,
    )
    harness.chain.epoch += int(round_id[-2:], 16)
    # Enter the review window after replacement freezes, as production does.
    cutoff = datetime.now(timezone.utc) + timedelta(minutes=30)
    configuration = harness.service.create_round(cutoff, round_id=round_id)
    assert configuration["runner_slot_ceiling"] == slot_ceiling
    assert configuration["parallel_twenty_icp_execution"] is True
    harness.round_id = round_id
    for flavor in harness.challengers:
        harness.submit(flavor, round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(round_id)["status"] == "ok"
    participants = harness.service.store.get_round(round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = harness.service.advance_round(round_id)
    assert opened["status"] == "ok"
    assert opened["assignments"] == len(participants) * contracts.BENCHMARK_ICP_COUNT
    return participants


def test_parallel_execution_migration_is_idempotent_and_keeps_score_guard(
    connect,
):
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            signature = (
                "public.lab_arena_claim_assignment(text,text,integer,integer,"
                "text[],text,text,text,integer)"
            )
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            before = cursor.fetchone()[0]
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            after = cursor.fetchone()[0]
            cursor.execute(
                "SELECT public.lab_arena_parallel_execution_schema_v1()"
            )
            capability = cursor.fetchone()[0]
            cursor.execute(
                "SELECT p.provolatile, p.prosecdef "
                "FROM pg_catalog.pg_proc AS p "
                "WHERE p.oid='public.lab_arena_parallel_execution_schema_v1()'"
                "::pg_catalog.regprocedure"
            )
            volatility, security_definer = cursor.fetchone()
            privileges = {}
            for role in (
                "lab_arena_service",
                "anon",
                "authenticated",
                "service_role",
            ):
                cursor.execute(
                    "SELECT has_function_privilege(%s, "
                    "'public.lab_arena_parallel_execution_schema_v1()', "
                    "'EXECUTE')",
                    (role,),
                )
                privileges[role] = cursor.fetchone()[0]
        assert after == before
        for marker in (
            "lab_arena_parallel_twenty_icp_execution",
            "lab_arena_score_submission_serialization",
            "lab_arena_closed_scoring_reservation_claim",
            "company_judgment_cache",
            "champion_funding_sources",
        ):
            assert marker in after
        assert capability == {
            "schema_version": "leadpoet.lab_arena.parallel_execution_schema.v1",
            "version": 255,
            "max_parallel_icps": 20,
        }
        assert (volatility, security_definer) == ("s", True)
        assert privileges == {
            "lab_arena_service": True,
            "anon": False,
            "authenticated": False,
            "service_role": False,
        }
    finally:
        connection.close()


def test_parallel_default_startup_requires_the_255_capability(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    observed = []
    real_capability = harness.service.store.parallel_execution_schema

    def checked_capability():
        observed.append("checked")
        return real_capability()

    harness.service.store.parallel_execution_schema = checked_capability
    harness.service.startup_checks()
    assert observed == []

    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        parallel_twenty_icp_execution=True,
    )
    harness.service.startup_checks()
    assert observed == ["checked"]

    def unavailable_capability():
        raise ArenaStoreError("function is unavailable")

    harness.service.store.parallel_execution_schema = unavailable_capability
    with pytest.raises(ServiceError) as caught:
        harness.service.startup_checks()
    assert caught.value.code == "parallel_execution_schema_unavailable"


def test_ten_slot_round_has_a_strict_ten_plus_ten_barrier_and_retry_recovery(
    connect, tmp_path
):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha", "beta"])
    participants = _start_parallel_round(
        harness, "arena-2099-01-01-d10", slot_ceiling=10
    )
    assert len(participants) == 1
    store = harness.service.store
    first_runner, second_runner = harness.runner_keys

    runner_key = keypair("svc-runner-alpha")
    old_claim = contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM, round_id=harness.round_id,
        hotkey=runner_key.ss58_address, body={"declared_parallelism": 8},
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: runner_key.sign(message.encode()).hex(),
    )
    with pytest.raises(ServiceError) as rejected:
        harness.service.handle_claim(old_claim)
    assert rejected.value.code == "validator_proxy_execution_upgrade_required"
    assert all(run["status"] == "pending" for run in store.list_runs(harness.round_id))
    first_claim = contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM,
        round_id=harness.round_id,
        hotkey=runner_key.ss58_address,
        body={"declared_parallelism": 10, "proxy_execution_version": contracts.PROXY_EXECUTION_VERSION},
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: runner_key.sign(message.encode()).hex(),
    )
    first_lease = harness.service.handle_claim(first_claim)
    assert first_lease["parallel_twenty_icp_execution"] is True
    first_batch = [(first_lease, first_lease["lease_token"])] + [
        claim(
            store,
            harness.round_id,
            first_runner,
            parallelism=10,
            ceiling=10,
        )[:2]
        for _ in range(9)
    ]
    assert [leased["icp_position"] for leased, _token in first_batch] == list(range(10))
    blocked, *_ = claim(
        store,
        harness.round_id,
        second_runner,
        parallelism=20,
        ceiling=10,
    )
    assert blocked["status"] == "no_pending"

    for leased, token in first_batch[:-1]:
        assert (
            complete(
                store,
                leased["run_id"],
                hash_lease_token(token),
                "accepted",
                output_ref="arena/test/outputs/%s.json" % leased["run_id"],
            )["status"]
            == "accepted"
        )
    failed, failed_token = first_batch[-1]
    failure = complete(
        store,
        failed["run_id"],
        hash_lease_token(failed_token),
        "provider_error",
    )
    assert failure["status"] == "failed"
    assert failure["confirmation_attempt"] == 2

    retry, retry_token, *_ = claim(
        store,
        harness.round_id,
        second_runner,
        parallelism=20,
        ceiling=10,
    )
    assert (retry["stage"], retry["icp_position"], retry["attempt"]) == (1, 9, 2)
    assert store.get_run(retry["run_id"])["runner_hotkey"] == second_runner
    assert (
        complete(
            store,
            retry["run_id"],
            hash_lease_token(retry_token),
            "accepted",
            output_ref="arena/test/outputs/%s.json" % retry["run_id"],
        )["status"]
        == "accepted"
    )

    second_batch, *_ = claim(
        store,
        harness.round_id,
        first_runner,
        parallelism=10,
        ceiling=10,
    )
    assert (second_batch["stage"], second_batch["icp_position"]) == (2, 10)
    with pytest.raises(Exception, match="parallel_execution_capacity_invalid"):
        claim(
            store,
            harness.round_id,
            first_runner,
            parallelism=20,
            ceiling=20,
        )


def test_twenty_slot_round_allows_two_validator_identities_independently(
    connect, tmp_path
):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha", "beta"])
    _start_parallel_round(harness, "arena-2099-01-01-d20", slot_ceiling=20)
    store = harness.service.store
    first_runner, second_runner = harness.runner_keys

    first_identity = [
        claim(
            store,
            harness.round_id,
            first_runner,
            parallelism=10,
            ceiling=20,
        )[0]
        for _ in range(10)
    ]
    full, *_ = claim(
        store,
        harness.round_id,
        first_runner,
        parallelism=10,
        ceiling=20,
    )
    assert full == {"status": "no_free_slot", "active_leases": 10, "slot_limit": 10}

    second_identity = [
        claim(
            store,
            harness.round_id,
            second_runner,
            parallelism=20,
            ceiling=20,
        )[0]
        for _ in range(10)
    ]
    assert {run["icp_position"] for run in first_identity + second_identity} == set(
        range(20)
    )
    assert {
        store.get_run(run["run_id"])["runner_hotkey"] for run in first_identity
    } == {first_runner}
    assert {
        store.get_run(run["run_id"])["runner_hotkey"] for run in second_identity
    } == {second_runner}
    drained, *_ = claim(
        store,
        harness.round_id,
        second_runner,
        parallelism=20,
        ceiling=20,
    )
    assert drained["status"] == "no_pending"


def test_parallel_execution_preserves_standard_twenty_icp_scoring_and_costs(
    connect, tmp_path
):
    harness = Harness(
        connect, tmp_path, challengers=["ParallelMiner"], runners=["alpha", "beta"]
    )
    participants = _start_parallel_round(
        harness, "arena-2099-01-01-e20", slot_ceiling=20
    )
    pending = harness.service.store.list_runs(harness.round_id, kind="execute")
    assert len(pending) == len(participants) * contracts.BENCHMARK_ICP_COUNT
    assert all(run["status"] == "pending" for run in pending)
    assert {run["stage"] for run in pending} == {1, 2}

    original_runner = harness.runner

    def runner_with_verified_pool(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner

    harness.runner = runner_with_verified_pool

    harness.advance_until("published", runners=2, max_steps=100)
    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    execute_runs = harness.service.store.list_runs(harness.round_id, kind="execute")
    score_runs = harness.service.store.list_runs(harness.round_id, kind="score")
    assert all(run["status"] == "accepted" for run in execute_runs + score_runs)
    assert len(execute_runs) == len(participants) * 20
    assert len(score_runs) == len(participants) * 20

    for participant in participants:
        public = harness.service.public_results(
            harness.round_id, participant["submission_id"]
        )
        rows = public["scores"]["stage_1"] + public["scores"]["stage_2"]
        assert len(rows) == 20
        assert {row["icp_position"] for row in rows} == set(range(20))
        costs = harness.service.store.submission_costs(participant["submission_id"])
        assert costs["submission_id"] == participant["submission_id"]
        assert sum(row["inflight_calls"] for row in costs["providers"]) == 0
        assert sum(row["uncertain_calls"] for row in costs["providers"]) == 0
