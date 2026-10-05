"""Frozen ICP banks limit each validator's concurrently leased model submissions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import hash_lease_token
from tests.lab_arena import untransitioned_host_fault_guard399_postgres_test as prior399
from tests.lab_arena.execute_host_cooldown396_postgres_test import _hash, _security
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete
from tests.lab_arena.test_lab_arena_service_round import Harness, daily_icps


SQL401 = Path(__file__).parents[2] / "scripts/401-lab-arena-proxy-model-capacity.sql"
migrated399 = prior399.migrated


@pytest.fixture(scope="module")
def database():
    # Reproduce the live claim function's 6300- and 4500-second lease guards.
    # Without migrations 329 and 354, the current fixture has another hash.
    migrations = CURRENT_SERVICE_MIGRATIONS + (
        "264-lab-arena-codex-cost-reconciliation.sql",
        "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
        "312-lab-arena-temporary-hold-admission.sql",
        "314-lab-arena-openrouter-web-search-reservation.sql",
        "319-lab-arena-quota-sourcing-cost.sql",
        "321-lab-arena-confirmed-cost-admission.sql",
        "329-lab-arena-explicit-90m-lease.sql",
        "354-lab-arena-60m-lease.sql",
        "359-lab-arena-setup-failure-model-retry.sql",
        "360-lab-arena-zero-setup-runner-handoff.sql",
        "365-lab-arena-trajectories.sql",
        "366-lab-arena-trajectory-capacity.sql",
    )
    yield from database_with_lab_arena_migration(migrations)


@pytest.fixture(scope="module")
def migrated(database, migrated399):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            before = _security(cursor)
            old_hash = _hash(cursor)
            assert old_hash == "03c4547d4acf78d85c5667978a5483af630c58109ec382c5bc8d331b1f4c1f25"
            cursor.execute(
                "SELECT pg_get_functiondef(%s::regprocedure)",
                ("public.lab_arena_claim_assignment(text,text,integer,integer,"
                 "text[],text,text,text,integer)",),
            )
            assert "hashtextextended('lab-arena-claim-control', 0)" in cursor.fetchone()[0]
            sql = SQL401.read_text()
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql.replace(old_hash, "0" * 64, 1))
            cursor.execute("ROLLBACK")
            assert _hash(cursor) == old_hash
            cursor.execute(sql)
            applied_hash = _hash(cursor)
            cursor.execute(sql)
            assert _hash(cursor) == applied_hash
            assert _security(cursor) == before
            cursor.execute("SELECT public.lab_arena_parallel_execution_schema_v1()")
            assert cursor.fetchone()[0]["max_parallel_icps"] == 251
    return True


@pytest.fixture(scope="module")
def connect(database, migrated):
    psycopg, dsn = database
    return lambda: psycopg.connect(**dsn)


def _start(
    connect, tmp_path, suffix: str, *, icps: int, ceiling: int,
    challengers=("FirstMiner", "SecondMiner"), runners=("alpha",),
):
    harness = Harness(
        connect, tmp_path, challengers=list(challengers), runners=list(runners),
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=icps,
        runner_slot_ceiling=ceiling,
        parallel_twenty_icp_execution=True,
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]),
        "icps": daily_icps()[:icps],
    }
    harness.chain.epoch += 1
    cutoff = datetime.now(timezone.utc) + timedelta(minutes=30)
    round_id = "arena-2099-01-01-" + suffix
    configuration = harness.service.create_round(cutoff, round_id=round_id)
    assert contracts.benchmark_icp_count(configuration) == icps
    assert configuration["runner_slot_ceiling"] == ceiling
    harness.round_id = round_id
    for flavor in harness.challengers:
        harness.submit(flavor, round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    first_advance = harness.service.advance_round(round_id)
    assert first_advance["status"] == "ok", first_advance
    participants = harness.service.store.get_round(round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(round_id)["status"] == "ok"
    assert len(participants) == 1 + len(challengers)
    return harness


@pytest.mark.parametrize(
    "suffix,icps,slots,expected_models",
    [
        ("c10", 10, 10, 1),
        ("c20", 10, 20, 2),
        ("c30", 10, 30, 3),
        ("c24", 12, 24, 2),
    ],
)
def test_model_admission_follows_frozen_bank_and_live_slots(
    connect, tmp_path, suffix, icps, slots, expected_models
):
    harness = _start(connect, tmp_path, suffix, icps=icps, ceiling=slots)
    store = harness.service.store
    hotkey = harness.runner_keys[0]
    leases = [
        claim(store, harness.round_id, hotkey, parallelism=slots, ceiling=slots)[:2]
        for _ in range(slots)
    ]
    assert all(lease["status"] == "leased" for lease, _ in leases)
    assert len({lease["submission_id"] for lease, _ in leases}) == expected_models
    assert all(
        sum(lease["submission_id"] == model for lease, _ in leases) == icps
        for model in {lease["submission_id"] for lease, _ in leases}
    )
    full, *_ = claim(store, harness.round_id, hotkey, parallelism=slots, ceiling=slots)
    assert full["status"] == "no_free_slot"

    first, token = leases[0]
    assert complete(
        store, first["run_id"], hash_lease_token(token), "accepted",
        output_ref="arena/test/outputs/%s.json" % first["run_id"],
    )["status"] == "accepted"
    blocked, *_ = claim(store, harness.round_id, hotkey, parallelism=slots, ceiling=slots)
    assert blocked["status"] == "no_pending"

    # A smaller local declaration cannot overrun its new slot limit.
    if slots == 20:
        reduced, *_ = claim(
            store, harness.round_id, hotkey, parallelism=10, ceiling=slots
        )
        assert reduced["status"] == "no_free_slot"
    for lease, token in leases[1:]:
        assert complete(
            store, lease["run_id"], hash_lease_token(token), "accepted",
            output_ref="arena/test/outputs/%s.json" % lease["run_id"],
        )["status"] == "accepted"
    if expected_models < 3:
        next_model, *_ = claim(
            store, harness.round_id, hotkey, parallelism=slots, ceiling=slots
        )
        assert next_model["status"] == "leased"
        assert next_model["submission_id"] not in {
            lease["submission_id"] for lease, _ in leases
        }


def test_frozen_old_round_cap_limits_new_validator_slots(connect, tmp_path):
    harness = _start(connect, tmp_path, "old", icps=10, ceiling=20)
    store = harness.service.store
    hotkey = harness.runner_keys[0]
    leases = [
        claim(store, harness.round_id, hotkey, parallelism=30, ceiling=20)[0]
        for _ in range(20)
    ]
    assert all(lease["status"] == "leased" for lease in leases)
    assert len({lease["submission_id"] for lease in leases}) == 2
    full, *_ = claim(store, harness.round_id, hotkey, parallelism=30, ceiling=20)
    assert full == {"status": "no_free_slot", "active_leases": 20, "slot_limit": 20}


def test_new_service_startup_accepts_expanded_schema(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        runner_slot_ceiling=contracts.RUNNER_SLOT_CEILING,
        parallel_twenty_icp_execution=True,
    )
    harness.service.startup_checks()


def test_interleaved_validators_admit_disjoint_models(connect, tmp_path):
    harness = _start(
        connect, tmp_path, "interleave", icps=10, ceiling=20,
        challengers=tuple("Miner%d" % index for index in range(5)),
        runners=("alpha", "beta", "gamma"),
    )
    held = {hotkey: [] for hotkey in harness.runner_keys}
    for _ in range(20):
        for hotkey in harness.runner_keys:
            response, *_ = claim(
                harness.service.store, harness.round_id, hotkey,
                parallelism=20, ceiling=20,
            )
            assert response["status"] == "leased"
            held[hotkey].append(response)
    models_by_runner = {
        hotkey: {lease["submission_id"] for lease in leases}
        for hotkey, leases in held.items()
    }
    assert all(len(models) == 2 for models in models_by_runner.values())
    assert len(set.union(*models_by_runner.values())) == 6
    assert all(
        all(sum(lease["submission_id"] == model for lease in held[hotkey]) == 10
            for model in models)
        for hotkey, models in models_by_runner.items()
    )


@pytest.mark.parametrize("slots,expected_leases", [(9, 9), (19, 10)])
def test_partial_model_slot_counts_keep_one_model(
    connect, tmp_path, slots, expected_leases
):
    harness = _start(
        connect, tmp_path, "partial%d" % slots,
        icps=10, ceiling=20,
    )
    hotkey = harness.runner_keys[0]
    leases = [
        claim(
            harness.service.store, harness.round_id, hotkey,
            parallelism=slots, ceiling=20,
        )[0]
        for _ in range(expected_leases)
    ]
    assert all(lease["status"] == "leased" for lease in leases)
    assert len({lease["submission_id"] for lease in leases}) == 1
    stopped, *_ = claim(
        harness.service.store, harness.round_id, hotkey,
        parallelism=slots, ceiling=20,
    )
    assert stopped["status"] == (
        "no_free_slot" if slots == 9 else "no_pending"
    )


@pytest.mark.parametrize("slots", [10, 20])
def test_scoring_claims_keep_independent_submission_throughput(
    connect, tmp_path, slots
):
    harness = _start(
        connect, tmp_path, "score%d" % slots, icps=10, ceiling=slots,
        challengers=tuple("ScoreMiner%d" % index for index in range(slots - 1)),
    )
    # This test isolates the claim scheduler; score serialization remains in
    # force and permits only one live score lease per submission.
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='stage1_scoring' "
                "WHERE round_id=%s", (harness.round_id,)
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET kind='score' "
                "WHERE round_id=%s", (harness.round_id,)
            )
            cursor.execute("SET session_replication_role=origin")
    store = harness.service.store
    hotkey = harness.runner_keys[0]
    leases = [
        claim(store, harness.round_id, hotkey, parallelism=slots, ceiling=slots)[0]
        for _ in range(slots)
    ]
    assert all(lease["status"] == "leased" for lease in leases)
    assert len({lease["submission_id"] for lease in leases}) == slots
    blocked, *_ = claim(
        store, harness.round_id, hotkey, parallelism=slots, ceiling=slots
    )
    assert blocked["status"] == "no_free_slot"
