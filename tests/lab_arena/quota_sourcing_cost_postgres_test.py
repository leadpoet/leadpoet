"""Real SQL cost snapshot: isolation, retries, auth, and unchanged v1 reads."""

from dataclasses import replace
import json

import pytest

from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _start_parallel_round,
)
from tests.lab_arena.per_icp_cost_admission_postgres_test import (
    _reserve,
    _settle,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    expire_now,
    sha,
)
from tests.lab_arena.test_lab_arena_service_round import Harness


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "319-lab-arena-quota-sourcing-cost.sql",
            "319-lab-arena-quota-sourcing-cost.sql",
        )
    )


def test_cost_snapshot_preserves_scope_and_history(database, tmp_path):
    connect = lambda: database[0].connect(**database[1])
    harness = Harness(connect, tmp_path, challengers=[], runners=["cost-view"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        per_icp_cost_policy=True,
        integrity_from="2000-01-01T00:00:00Z",
    )
    _start_parallel_round(harness, "arena-2099-02-07-c6", slot_ceiling=2)
    store = harness.service.store
    actor = harness.runner_keys[0]
    first, first_token = claim(
        store, harness.round_id, actor, parallelism=2, ceiling=2
    )[:2]
    other, other_token = claim(
        store, harness.round_id, actor, parallelism=2, ceiling=2
    )[:2]
    before = store.run_quota_snapshot(
        first["run_id"], hash_lease_token(first_token)
    )

    def read_state(run_id):
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_jsonb(runs) FROM public.lab_arena_runs AS runs "
                "WHERE run_id = %s",
                (run_id,),
            )
            run = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(*), COALESCE(MAX(entry_id), 0) "
                "FROM public.lab_arena_ledger WHERE run_id = %s",
                (run_id,),
            )
            ledger = cursor.fetchone()
        return run, ledger

    state_before = read_state(first["run_id"])
    initial = store.run_quota_snapshot(
        first["run_id"],
        hash_lease_token(first_token),
        include_sourcing_cost=True,
    )
    assert read_state(first["run_id"]) == state_before
    assert initial["providers"] == before["providers"]
    assert initial["sourcing_cost"]["successful_microusd"] == 0

    call, _ = _reserve(store, first, first_token, "first-cost", 200_000)
    _settle(store, first, first_token, call, 200_000)
    failed, _ = _reserve(store, first, first_token, "failed-cost", 50_000)
    _settle(
        store,
        first,
        first_token,
        failed,
        50_000,
        succeeded=False,
    )
    other_call, _ = _reserve(
        store, other, other_token, "other-cost", 600_000
    )
    _settle(store, other, other_token, other_call, 600_000)

    # Expire only this disposable test lease. Its costs remain in the retry.
    connection = connect()
    connection.autocommit = True
    try:
        expire_now(connection, first["run_id"])
    finally:
        connection.close()
    assert store.expire_leases(harness.round_id)["retried"] == 1
    retry, retry_token = claim(
        store, harness.round_id, actor, parallelism=2, ceiling=2
    )[:2]
    assert retry["attempt"] == 2
    assert retry["icp_position"] == first["icp_position"]

    # Add a successful OpenRouter call in the retry and a judge call for the
    # same ICP. The judge cost must not enter the execute-attempt total.
    judge_id = harness.round_id + ":cost-view-judge"
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,status,stage_generation,kind,scored_run_id) "
            "VALUES (%s,%s,%s,%s,%s,1,%s,1,'accepted',1,'score',%s)",
            (
                judge_id,
                judge_id,
                harness.round_id,
                retry["submission_id"],
                retry["miner_hotkey"],
                retry["icp_position"],
                retry["run_id"],
            ),
        )
        for run_id, amount in (
            (retry["run_id"], 100_000),
            (judge_id, 50_000_000),
        ):
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,"
                "amount_microusd,terminal_response,entry_doc) "
                "VALUES ('settlement',%s,%s,%s,%s,1,%s,'openrouter',"
                "'openrouter.responses','miner_key',%s,%s::jsonb,'{}'::jsonb)",
                (
                    retry["miner_hotkey"],
                    harness.round_id,
                    retry["submission_id"],
                    run_id,
                    sha(run_id),
                    amount,
                    json.dumps({"status": 200, "call_succeeded": True}),
                ),
            )
    held, reservation = _reserve(
        store, retry, retry_token, "pending-cost", 75_000
    )
    assert held
    assert reservation["status"] == "reserved"

    value = store.run_quota_snapshot(
        retry["run_id"],
        hash_lease_token(retry_token),
        include_sourcing_cost=True,
    )
    eligibility = store.icp_cost_eligibility(
        round_id=harness.round_id,
        submission_id=retry["submission_id"],
        icp_position=retry["icp_position"],
        qualified_company_count=0,
    )
    execution = eligibility["execution"]
    assert value["sourcing_cost"] == {
        "successful_microusd": execution["successful_microusd"],
        "settled_microusd": execution["settled_microusd"],
        "reserved_or_uncertain_microusd": execution[
            "reserved_or_uncertain_microusd"
        ],
        "success_unresolved_microusd": execution[
            "success_unresolved_microusd"
        ],
        "inflight_calls": execution["inflight_calls"],
        "success_unresolved_calls": execution["success_unresolved_calls"],
        "admission_cap_microusd": eligibility[
            "execution_icp_cap_microusd"
        ],
        "per_qualified_pair_cap_microusd": eligibility[
            "cost_per_company_cap_microusd"
        ],
    }
    # The reservation and settlement rows for one identity count once. Costs
    # from attempt one and attempt two are included; the judge is excluded.
    assert value["sourcing_cost"] == {
        "successful_microusd": 300_000,
        "settled_microusd": 350_000,
        "reserved_or_uncertain_microusd": 75_000,
        "success_unresolved_microusd": 75_000,
        "inflight_calls": 1,
        "success_unresolved_calls": 1,
        "admission_cap_microusd": 4_000_000,
        "per_qualified_pair_cap_microusd": 800_000,
    }
    other_value = store.run_quota_snapshot(
        other["run_id"],
        hash_lease_token(other_token),
        include_sourcing_cost=True,
    )
    assert other_value["sourcing_cost"]["successful_microusd"] == 600_000
    assert set(
        store.run_quota_snapshot(
            retry["run_id"], hash_lease_token(retry_token)
        )
    ) == {"schema_version", "providers"}

    for run_id, token in (
        (retry["run_id"], first_token),
        (other["run_id"], retry_token),
        (first["run_id"], first_token),
    ):
        with pytest.raises(ArenaStoreError):
            store.run_quota_snapshot(
                run_id,
                hash_lease_token(token),
                include_sourcing_cost=True,
            )
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT "
            "has_function_privilege('anon',"
            "'public.lab_arena_run_quota_snapshot_v2(text,text)','EXECUTE'),"
            "has_function_privilege('authenticated',"
            "'public.lab_arena_run_quota_snapshot_v2(text,text)','EXECUTE'),"
            "has_function_privilege('lab_arena_service',"
            "'public.lab_arena_run_quota_snapshot_v2(text,text)','EXECUTE')"
        )
        assert cursor.fetchone() == (False, False, True)
