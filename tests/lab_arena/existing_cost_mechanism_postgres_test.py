"""PostgreSQL boundaries for the existing sourcing budget and eligibility."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    commit_round,
    complete,
    frozen_participants,
    hotkey,
    round_config,
    sha,
    source_submission_doc,
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _reserve(
    store: ArenaStore,
    run: dict,
    token: str,
    *,
    label: str,
    provider: str,
    operation: str,
    amount: int,
    funding_source: str,
) -> tuple[str, dict]:
    identity = sha("existing-cost-" + run["run_id"] + "-" + label)
    result = store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        operation_id=operation,
        provider=provider,
        funding_source=funding_source,
        amount_microusd=amount,
        call_doc={"request_hash": sha("request-" + run["run_id"] + "-" + label)},
    )
    return identity, result


def _settle(
    store: ArenaStore,
    run: dict,
    token: str,
    identity: str,
    amount: int,
    *,
    succeeded: bool,
) -> None:
    token_hash = hash_lease_token(token)
    assert store.mark_dispatched(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.settle_call(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        actual_microusd=amount,
        terminal_response={
            "status": 200 if succeeded else 503,
            "call_succeeded": succeeded,
        },
    )["status"] == "settled"


def _frozen_baseline(
    store: ArenaStore, round_id: str, configuration: dict
) -> dict:
    submission_id = "baseline-" + round_id.removeprefix("arena-")
    miner = configuration["baseline_hotkey"]
    assert store.register_submission(
        round_id,
        submission_id,
        miner,
        source_submission_doc(round_id, submission_id, is_king=True),
    )["status"] == "registered"
    assert store.update_submission(
        round_id, submission_id, "uploading", "accepted"
    )["status"] == "ok"
    assert store.update_submission(
        round_id,
        submission_id,
        "accepted",
        "frozen",
        {"is_king": True},
    )["status"] == "ok"
    return {
        "submission_id": submission_id,
        "miner_hotkey": miner,
        "is_king": True,
    }


@pytest.mark.parametrize(
    ("participant_kind", "funding_source"),
    [("baseline", "host"), ("miner", "miner_key")],
)
def test_eighty_dollar_runtime_cap_is_shared_across_icps_providers_retry_and_concurrency(
    database, participant_kind, funding_source
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    round_id = "arena-2098-09-14-e80" + participant_kind[0]
    try:
        runners = [
            hotkey("existing80-%s-runner-%d" % (participant_kind, index))
            for index in range(2)
        ]
        configuration = round_config(
            round_id,
            runners,
            quotas={"openrouter": 5, "deepline": 5, "scrapingdog": 5},
            execution_cap_microusd=80_000_000,
            cost_per_company_microusd=800_000,
        )
        assert store.create_round(round_id, configuration)["status"] == "created"
        participants = (
            [_frozen_baseline(store, round_id, configuration)]
            if participant_kind == "baseline"
            else frozen_participants(
                store, round_id, 1, prefix="existing80-miner"
            )
        )
        commit_round(store, round_id, participants)
        assert store.open_stage(
            round_id, 1, participants, list(contracts.stage_positions(1))
        )["status"] == "ok"
        first, first_token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8
        )
        assert first["status"] == "leased" and first["icp_position"] == 0
        failed_identity, reserved = _reserve(
            store,
            first,
            first_token,
            label="failed-first-attempt",
            provider="scrapingdog",
            operation="scrapingdog.scrape",
            amount=10_000_000,
            funding_source=funding_source,
        )
        assert reserved["status"] == "reserved"
        _settle(
            store,
            first,
            first_token,
            failed_identity,
            10_000_000,
            succeeded=False,
        )
        assert complete(
            store,
            first["run_id"],
            hash_lease_token(first_token),
            "model_error",
        )["status"] == "failed"

        retry, retry_token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8
        )
        other_icp, other_token, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8
        )
        last_icp, last_token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8
        )
        assert (retry["icp_position"], retry["attempt"]) == (0, 2)
        assert {other_icp["icp_position"], last_icp["icp_position"]} == {1, 2}

        calls = (
            (retry, retry_token, "retry-deepline", "deepline", "deepline.execute"),
            (
                other_icp,
                other_token,
                "other-openrouter",
                "openrouter",
                "openrouter.chat",
            ),
        )

        def reserve_concurrently(call):
            run, token, label, provider, operation = call
            local = ArenaStore(PsycopgTransport(connect))
            try:
                return run, token, _reserve(
                    local,
                    run,
                    token,
                    label=label,
                    provider=provider,
                    operation=operation,
                    amount=35_000_000,
                    funding_source=funding_source,
                )
            finally:
                local.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(reserve_concurrently, calls))
        assert [result[1]["status"] for _, _, result in concurrent] == [
            "reserved",
            "reserved",
        ]
        for run, token, (identity, _result) in concurrent:
            _settle(store, run, token, identity, 35_000_000, succeeded=True)

        _, refused = _reserve(
            store,
            last_icp,
            last_token,
            label="one-microdollar-over-runtime-cap",
            provider="openrouter",
            operation="openrouter.chat",
            amount=1,
            funding_source=funding_source,
        )
        assert refused["status"] == "refused"
        assert refused["reason"] == "money_cap"

        costs = store.submission_costs(participants[0]["submission_id"])
        execution = [row for row in costs["providers"] if row["kind"] == "execute"]
        assert {row["provider"] for row in execution} == {
            "deepline",
            "openrouter",
            "scrapingdog",
        }
        assert sum(row["settled_microusd"] for row in execution) == 80_000_000
        assert sum(row["refused_calls"] for row in execution) == 1
    finally:
        store.close()


def _insert_cost(
    cursor,
    *,
    round_id: str,
    participant: dict,
    label: str,
    amount: int,
    succeeded: bool,
    kind: str = "execute",
) -> None:
    run_id = participant["run_id"]
    if kind == "score":
        run_id = round_id + ":score:" + label
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,status,stage_generation,kind,scored_run_id) "
            "VALUES (%s,%s,%s,%s,%s,1,0,1,'pending',1,'score',%s)",
            (
                run_id,
                round_id + ":assignment:" + label,
                round_id,
                participant["submission_id"],
                participant["miner_hotkey"],
                participant["run_id"],
            ),
        )
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,"
        "entry_doc,terminal_response) "
        "VALUES ('settlement',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat',"
        "%s,%s,'{}'::jsonb,%s::jsonb)",
        (
            participant["miner_hotkey"],
            round_id,
            participant["submission_id"],
            run_id,
            sha("existing-eligibility-" + label),
            "host" if participant["is_king"] else "miner_key",
            amount,
            json.dumps(
                {
                    "status": 200 if succeeded else 503,
                    "call_succeeded": succeeded,
                    "choices": (
                        [{"message": {"content": ""}}] if succeeded else []
                    ),
                }
            ),
        ),
    )


def test_eighty_cent_final_cap_uses_successful_sourcing_and_verified_count(
    database,
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    round_id = "arena-2098-09-15-existing80c"
    config = round_config(
        round_id,
        [hotkey("existing80c-runner")],
        execution_cap_microusd=80_000_000,
        cost_per_company_microusd=800_000,
    )
    config["sourcing_cost_eligibility_policy"] = (
        contracts.SUCCESSFUL_CALLS_COST_POLICY
    )
    assert store.create_round(round_id, config)["status"] == "created"
    participants = [_frozen_baseline(store, round_id, config)]
    participants.extend(
        frozen_participants(store, round_id, 3, prefix="existing80c")
    )
    commit_round(store, round_id, participants)
    assert store.open_stage(
        round_id, 1, participants, list(contracts.stage_positions(1))
    )["status"] == "ok"

    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for participant in participants:
                cursor.execute(
                    "SELECT run_id FROM public.lab_arena_runs "
                    "WHERE submission_id=%s AND kind='execute' "
                    "ORDER BY run_id LIMIT 1",
                    (participant["submission_id"],),
                )
                participant["run_id"] = cursor.fetchone()[0]

            # Baseline: exact equality. The failed sourcing charge reaches the
            # runtime cap but does not enter successful-call eligibility.
            _insert_cost(
                cursor,
                round_id=round_id,
                participant=participants[0],
                label="baseline-empty-success",
                amount=1_600_000,
                succeeded=True,
            )
            _insert_cost(
                cursor,
                round_id=round_id,
                participant=participants[0],
                label="baseline-charged-failure",
                amount=78_400_000,
                succeeded=False,
            )
            _insert_cost(
                cursor,
                round_id=round_id,
                participant=participants[0],
                label="baseline-judge",
                amount=50_000_000,
                succeeded=True,
                kind="score",
            )

            # Miner: one microdollar above two verified pairs.
            _insert_cost(
                cursor,
                round_id=round_id,
                participant=participants[1],
                label="miner-one-over",
                amount=1_600_001,
                succeeded=True,
            )
            # Zero verified pairs: charged failure is excluded, while one
            # successful microdollar is enough to fail the zero-dollar cap.
            _insert_cost(
                cursor,
                round_id=round_id,
                participant=participants[2],
                label="zero-qualified-failure",
                amount=80_000_000,
                succeeded=False,
            )
            _insert_cost(
                cursor,
                round_id=round_id,
                participant=participants[3],
                label="zero-qualified-success",
                amount=1,
                succeeded=True,
            )

            results = []
            for participant, qualified in zip(participants, (2, 2, 0, 0)):
                cursor.execute(
                    "SELECT public.lab_arena__successful_call_eligibility(%s,%s,%s)",
                    (round_id, participant["submission_id"], qualified),
                )
                results.append(cursor.fetchone()[0])

        assert participants[0]["is_king"] is True
        assert [row["competition_sourcing_microusd"] for row in results] == [
            1_600_000,
            1_600_001,
            0,
            1,
        ]
        assert [row["eligibility_cap_microusd"] for row in results] == [
            1_600_000,
            1_600_000,
            0,
            0,
        ]
        assert [(row["eligible"], row["eligibility_reason"]) for row in results] == [
            (True, "eligible"),
            (False, "cost_per_company_exceeded"),
            (True, "eligible"),
            (False, "cost_per_company_exceeded"),
        ]
        assert results[0]["execution"]["settled_microusd"] == 80_000_000
        assert results[0]["execution"]["successful_calls"] == 1
        assert results[0]["judge"]["settled_microusd"] == 50_000_000
        assert results[0]["judge"]["successful_microusd"] == 50_000_000
    finally:
        connection.close()
        store.close()
