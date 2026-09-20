"""Execute-quota profile boundaries against disposable current PostgreSQL."""

from __future__ import annotations

import json

import pytest

from lab_arena import broker as broker_module
from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_broker import (
    FakeTransport,
    HOST_KEYS,
    price_table,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    hotkey,
    sha,
)


LEASE_TOKEN_HASH = "sha256:" + "a" * 64


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "264-lab-arena-codex-cost-reconciliation.sql",
            "289-lab-arena-per-icp-cost-policy.sql",
            "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
            "312-lab-arena-temporary-hold-admission.sql",
            "314-lab-arena-openrouter-web-search-reservation.sql",
            "319-lab-arena-quota-sourcing-cost.sql",
            "321-lab-arena-confirmed-cost-admission.sql",
        )
    )


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database

    def _connect():
        return psycopg2.connect(**dsn)

    return _connect


@pytest.fixture()
def store(connect):
    transport = PsycopgTransport(connect)
    yield ArenaStore(transport, lease_ttl_seconds=3_600)
    transport.close()


def _exercise_profile(
    store, connect, *, suffix, quotas, limit, checkpoints, provider="openrouter",
    confirmed_cost_policy=False,
):
    round_id = "arena-2026-09-17-" + suffix
    submission_id = "quota-" + suffix + "-submission"
    run_id = "quota-" + suffix + "-run"
    assignment_id = "quota-" + suffix + "-assignment"
    miner = hotkey("quota-" + suffix + "-miner")
    configuration = {
        "call_quotas": quotas,
        "scoring_call_quotas": dict(
            contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM
        ),
        "stage_1_icp_count": 10,
        "stage_2_icp_count": 10,
        "max_attempts_per_assignment": 2,
        "execution_cap_microusd": 1_000_000,
        "scoring_cap_microusd": 50_000_000,
        "baseline_hotkey": hotkey("quota-profile-baseline"),
    }
    if confirmed_cost_policy:
        configuration.update(
            {
                "integrity_policy": "arena_integrity_v1",
                "sourcing_cost_eligibility_policy": (
                    contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
                ),
                "execution_icp_cap_microusd": 4_000_000,
                "cost_per_company_microusd": 800_000,
            }
        )
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds("
                "round_id,status,stage_generation,configuration_doc) "
                "VALUES (%s,'stage1',1,%s::jsonb)",
                (round_id, json.dumps(configuration)),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king) "
                "VALUES (%s,%s,%s,'frozen',FALSE)",
                (submission_id, round_id, miner),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,runner_hotkey,"
                "lease_token_hash,lease_generation,stage_generation,"
                "lease_expires_at) VALUES ("
                "%s,%s,%s,%s,%s,1,0,1,'execute','leased',%s,%s,1,1,"
                "clock_timestamp() + interval '1 hour')",
                (
                    run_id, assignment_id, round_id, submission_id, miner,
                    miner, LEASE_TOKEN_HASH,
                ),
            )
            cursor.execute("SET session_replication_role=origin")
    finally:
        connection.close()
    snapshots = {}
    statuses = []
    last_reserved_identity = None
    for index in range(limit + 1):
        identity = contracts.provider_call_identity(
            attempt=1,
            assignment_id=assignment_id,
            icp_position=0,
            action_sequence=index,
            operation_id={
                "openrouter": "openrouter.chat",
                "deepline": "deepline.execute",
                "scrapingdog": "scrapingdog.scrape",
            }[provider],
            request_hash=sha("%s-%d" % (suffix, index)),
        )
        reserved = store.reserve_call(
            run_id=run_id,
            lease_token_hash=LEASE_TOKEN_HASH,
            call_identity=identity,
            operation_id={
                "openrouter": "openrouter.chat",
                "deepline": "deepline.execute",
                "scrapingdog": "scrapingdog.scrape",
            }[provider],
            provider=provider,
            funding_source="miner_key",
            amount_microusd=0 if confirmed_cost_policy else 1,
            call_doc={},
        )
        statuses.append(reserved["status"])
        if reserved["status"] == "reserved":
            last_reserved_identity = identity
            if confirmed_cost_policy:
                assert reserved["amount_microusd"] == 0
            assert store.mark_dispatched(
                run_id=run_id,
                lease_token_hash=LEASE_TOKEN_HASH,
                call_identity=identity,
            )["status"] == "dispatched"
        else:
            assert reserved["reason"] == "per_icp_quota"
        ordinal = index + 1
        if ordinal in checkpoints:
            snapshots[ordinal] = store.run_quota_snapshot(
                run_id, LEASE_TOKEN_HASH
            )["providers"][provider]

    costs = store.submission_costs(submission_id)
    execute = next(
        item for item in costs["providers"]
        if item["kind"] == "execute" and item["provider"] == provider
    )
    return statuses, snapshots, execute, {
        "assignment_id": assignment_id,
        "icp_position": 0,
        "miner_hotkey": miner,
        "round_id": round_id,
        "run_id": run_id,
        "submission_id": submission_id,
        "last_reserved_identity": last_reserved_identity,
    }


def test_frozen_execute_profiles_enforce_dispatch_and_cost_boundaries(
    store, connect
):
    historical_statuses, historical_snapshots, historical_costs, _ = _exercise_profile(
        store, connect,
        suffix="historical60",
        quotas=dict(contracts.LEGACY_CALL_QUOTAS_PER_ICP),
        limit=60,
        checkpoints={60, 61},
    )
    assert historical_statuses[:60] == ["reserved"] * 60
    assert historical_statuses[60] == "refused"
    assert historical_snapshots == {
        60: {"limit": 60, "used": 60, "remaining": 0, "inflight": 60},
        61: {"limit": 60, "used": 60, "remaining": 0, "inflight": 60},
    }
    assert historical_costs["call_count"] == 61
    assert historical_costs["reserved_or_uncertain_microusd"] == 60
    assert historical_costs["refused_calls"] == 1

    all_provider_200_statuses, all_provider_200_snapshots, all_provider_200_costs, _ = _exercise_profile(
        store, connect,
        suffix="allprovider200",
        quotas=dict(contracts.ALL_PROVIDER_200_CALL_QUOTAS_PER_ICP),
        limit=200,
        checkpoints={60, 61, 199, 200, 201},
    )
    assert all_provider_200_statuses[:200] == ["reserved"] * 200
    assert all_provider_200_statuses[200] == "refused"
    assert all_provider_200_snapshots == {
        60: {"limit": 200, "used": 60, "remaining": 140, "inflight": 60},
        61: {"limit": 200, "used": 61, "remaining": 139, "inflight": 61},
        199: {"limit": 200, "used": 199, "remaining": 1, "inflight": 199},
        200: {"limit": 200, "used": 200, "remaining": 0, "inflight": 200},
        201: {"limit": 200, "used": 200, "remaining": 0, "inflight": 200},
    }
    assert all_provider_200_costs["call_count"] == 201
    assert all_provider_200_costs["reserved_or_uncertain_microusd"] == 200
    assert all_provider_200_costs["refused_calls"] == 1

    current_statuses, current_snapshots, current_costs, current = _exercise_profile(
        store, connect,
        suffix="frozen500",
        quotas=dict(contracts.OPENROUTER_500_CALL_QUOTAS_PER_ICP),
        limit=500,
        checkpoints={499, 500, 501},
        confirmed_cost_policy=True,
    )
    assert current_statuses[:500] == ["reserved"] * 500
    assert current_statuses[500] == "refused"
    assert current_snapshots == {
        499: {"limit": 500, "used": 499, "remaining": 1, "inflight": 499},
        500: {"limit": 500, "used": 500, "remaining": 0, "inflight": 500},
        501: {"limit": 500, "used": 500, "remaining": 0, "inflight": 500},
    }
    assert current_costs["call_count"] == 501
    assert current_costs["reserved_or_uncertain_microusd"] == 0
    assert current_costs["refused_calls"] == 1

    settled = store.settle_call(
        run_id=current["run_id"],
        lease_token_hash=LEASE_TOKEN_HASH,
        call_identity=current["last_reserved_identity"],
        actual_microusd=60_000,
        terminal_response={"status": 200, "call_succeeded": True},
    )
    assert settled["status"] == "settled"
    replay = store.settle_call(
        run_id=current["run_id"],
        lease_token_hash=LEASE_TOKEN_HASH,
        call_identity=current["last_reserved_identity"],
        actual_microusd=60_000,
        terminal_response={"status": 200, "call_succeeded": True},
    )
    assert (replay["status"], replay["idempotent"]) == ("settled", True)
    current_costs = next(
        item for item in store.submission_costs(current["submission_id"])["providers"]
        if item["kind"] == "execute" and item["provider"] == "openrouter"
    )
    assert current_costs["settled_microusd"] == 60_000
    assert current_costs["reserved_or_uncertain_microusd"] == 0

    # Each submission aggregates only its own frozen-profile run.
    assert (
        historical_costs["reserved_or_uncertain_microusd"]
        + all_provider_200_costs["reserved_or_uncertain_microusd"]
        + current_costs["reserved_or_uncertain_microusd"]
        == 260
    )


def test_current_2000_profile_crosses_real_broker_and_confirmed_cost_accounting(
    store, connect
):
    statuses, snapshots, costs, current = _exercise_profile(
        store,
        connect,
        suffix="current2000",
        quotas=dict(contracts.CALL_QUOTAS_PER_ICP),
        limit=1998,
        checkpoints={1999},
        confirmed_cost_policy=True,
    )
    assert statuses == ["reserved"] * 1999
    assert snapshots == {
        1999: {
            "limit": 2000,
            "used": 1999,
            "remaining": 1,
            "inflight": 1999,
        }
    }
    assert costs["call_count"] == 1999
    assert costs["reserved_or_uncertain_microusd"] == 0

    payload = {
        "id": "quota-2000",
        "model": "openai/gpt-4o-mini",
        "error": None,
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "cost": "0.0000021",
        },
    }
    transport = FakeTransport([(200, payload)])
    broker = broker_module.Broker(
        store=store,
        key_for=lambda provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: "miner_key",
        price_table=price_table(),
        transport=transport,
        lease_ttl_seconds=3600,
    )
    context = broker_module.RunContext(
        run_id=current["run_id"],
        assignment_id=current["assignment_id"],
        icp_position=current["icp_position"],
        lease_token_hash=LEASE_TOKEN_HASH,
        miner_hotkey=current["miner_hotkey"],
        submission_id=current["submission_id"],
        stage=1,
        round_id=current["round_id"],
    )
    parameters = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "finish"}],
        "max_tokens": 1,
    }
    accepted = broker.execute(
        context,
        operation_id="openrouter.chat",
        parameters=parameters,
        action_sequence=1999,
        timeout_ms=5000,
    )
    assert accepted.status == 200
    assert accepted.call["outcome"] == "settled"
    assert accepted.call["actual_microusd"] == 3
    assert len(transport.sent) == 1

    refused = broker.execute(
        context,
        operation_id="openrouter.chat",
        parameters=parameters,
        action_sequence=2000,
        timeout_ms=5000,
    )
    assert refused.status == 402
    assert refused.call["outcome"] == "refused"
    assert refused.call["reason"] == "per_icp_quota"
    assert len(transport.sent) == 1

    snapshot = store.run_quota_snapshot(current["run_id"], LEASE_TOKEN_HASH)
    assert snapshot["providers"]["openrouter"] == {
        "limit": 2000,
        "used": 2000,
        "remaining": 0,
        "inflight": 1999,
    }
    costs = next(
        item
        for item in store.submission_costs(current["submission_id"])["providers"]
        if item["kind"] == "execute" and item["provider"] == "openrouter"
    )
    assert costs["call_count"] == 2001
    assert costs["settled_microusd"] == 3
    assert costs["reserved_or_uncertain_microusd"] == 0
    assert costs["refused_calls"] == 1


@pytest.mark.parametrize("provider", ("deepline", "scrapingdog"))
def test_generic_profile_admits_provider_calls_past_old_thirty_call_limit(
    store, connect, provider
):
    statuses, snapshots, _costs, _ = _exercise_profile(
        store,
        connect,
        suffix={"deepline": "gdl", "scrapingdog": "gsd"}[provider],
        quotas=dict(contracts.CALL_QUOTAS_PER_ICP),
        provider=provider,
        limit=58,
        checkpoints={30, 31, 54, 58},
    )
    assert statuses == ["reserved"] * 59
    assert snapshots == {
        30: {"limit": 200, "used": 30, "remaining": 170, "inflight": 30},
        31: {"limit": 200, "used": 31, "remaining": 169, "inflight": 31},
        54: {"limit": 200, "used": 54, "remaining": 146, "inflight": 54},
        58: {"limit": 200, "used": 58, "remaining": 142, "inflight": 58},
    }


def test_frozen_openrouter_200_profile_keeps_deepline_at_thirty(store, connect):
    statuses, snapshots, _costs, _ = _exercise_profile(
        store,
        connect,
        suffix="fo2dl",
        quotas=dict(contracts.OPENROUTER_200_CALL_QUOTAS_PER_ICP),
        provider="deepline",
        limit=30,
        checkpoints={30, 31},
    )
    assert statuses[:30] == ["reserved"] * 30
    assert statuses[30] == "refused"
    assert snapshots == {
        30: {"limit": 30, "used": 30, "remaining": 0, "inflight": 30},
        31: {"limit": 30, "used": 30, "remaining": 0, "inflight": 30},
    }
