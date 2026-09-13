"""Score-claim serialization for a submission's shared provider budget."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    _commit_plan,
    _execute_everything,
    _scoring_items,
    claim,
    complete,
    hotkey,
    open_round,
    sha,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "236-lab-arena-score-submission-serialization.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _store(database) -> ArenaStore:
    psycopg2, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _open_scoring(
    store: ArenaStore,
    round_id: str,
    *,
    participants: int,
    runners: int,
) -> tuple[list[str], list[dict]]:
    runner_keys, participant_rows = open_round(
        store,
        round_id,
        participants=participants,
        runners=runners,
        prefix=round_id[-8:],
        max_attempts=1,
    )
    executed = _execute_everything(store, round_id, runner_keys[0])
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    items = _scoring_items(executed)
    assert store.open_scoring(round_id, 1, items)["status"] == "ok"
    return runner_keys, participant_rows


def _reserve_dynamic(
    store: ArenaStore, leased: dict, token: str, label: str
) -> tuple[str, dict]:
    call_identity = contracts.provider_call_identity(
        attempt=leased["attempt"],
        assignment_id=leased["assignment_id"],
        icp_position=leased["icp_position"],
        action_sequence=0,
        operation_id="exa.contents",
        request_hash=sha(label),
    )
    result = store.reserve_call(
        run_id=leased["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=call_identity,
        operation_id="exa.contents",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=0,
        call_doc={"reserve_remaining_budget": True, "tool": "exa_contents"},
    )
    return call_identity, result


def test_score_claims_serialize_per_submission_without_reducing_other_parallelism(
    database,
):
    store = _store(database)
    psycopg2, dsn = database
    migration = MIGRATION.read_text(encoding="utf-8")
    try:
        # The prior claim policy leases sibling ICP score work concurrently.
        before_round = "arena-2026-09-13-sb"
        before_runners, _ = _open_scoring(
            store, before_round, participants=1, runners=2
        )
        first, first_token, _, _ = claim(
            store, before_round, before_runners[0],
            parallelism=8, ceiling=8, excluded=[before_runners[0]],
        )
        second, second_token, _, _ = claim(
            store, before_round, before_runners[1],
            parallelism=8, ceiling=8, excluded=[before_runners[1]],
        )
        assert first["status"] == second["status"] == "leased"
        assert first["kind"] == second["kind"] == "score"
        assert first["submission_id"] == second["submission_id"]

        first_identity, first_reservation = _reserve_dynamic(
            store, first, first_token, "before-first"
        )
        assert first_reservation["status"] == "reserved"
        _second_identity, second_reservation = _reserve_dynamic(
            store, second, second_token, "before-second"
        )
        assert second_reservation["status"] == "budget_busy"
        assert store.mark_dispatched(
            run_id=first["run_id"],
            lease_token_hash=hash_lease_token(first_token),
            call_identity=first_identity,
        )["status"] == "dispatched"
        assert store.settle_call(
            run_id=first["run_id"],
            lease_token_hash=hash_lease_token(first_token),
            call_identity=first_identity,
            actual_microusd=2_000,
            terminal_response={"status": 200},
        )["status"] == "settled"

        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(migration)
                cursor.execute(
                    "SELECT pg_catalog.pg_get_functiondef("
                    "'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure)"
                )
                patched_definition = cursor.fetchone()[0]
                for marker in (
                    "lab_arena_score_submission_serialization",
                    "runner_authority_exclusions",
                    "judgment_group_leader",
                    "company_judgment_cache",
                    "champion_funding_sources",
                ):
                    assert marker in patched_definition
                assert patched_definition.count(
                    "lab_arena_score_submission_serialization"
                ) == 1
                cursor.execute(migration)
                cursor.execute(
                    "SELECT pg_catalog.pg_get_functiondef("
                    "'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure)"
                )
                assert cursor.fetchone()[0] == patched_definition

        # Execute assignments remain parallel for the same submission.
        after_round = "arena-2026-09-13-sa"
        after_runners, participants = open_round(
            store,
            after_round,
            participants=2,
            runners=4,
            prefix="serialize-after",
            max_attempts=1,
        )
        execute_leases = []
        for index in range(3):
            leased, token, _, _ = claim(
                store, after_round, after_runners[index],
                parallelism=8, ceiling=8, excluded=[after_runners[index]],
            )
            assert leased["status"] == "leased" and leased["kind"] == "execute"
            execute_leases.append((leased, token))
        assert execute_leases[0][0]["submission_id"] == execute_leases[2][0][
            "submission_id"
        ]
        for leased, token in execute_leases:
            assert complete(
                store, leased["run_id"], hash_lease_token(token), "accepted",
                output_ref="arena/x/outputs/%s.json" % leased["run_id"],
            )["status"] == "accepted"
        executed = {
            (leased["submission_id"], leased["icp_position"]): leased["run_id"]
            for leased, _token in execute_leases
        }
        executed.update(_execute_everything(store, after_round, after_runners[0]))
        assert store.close_stage(after_round, 1)["status"] == "closed"
        _commit_plan(store, after_round, 1)
        assert store.open_scoring(
            after_round, 1, _scoring_items(executed)
        )["status"] == "ok"

        # Different submissions can score in parallel. A third claim waits
        # because both submissions already have a live score lease.
        score_a, token_a, _, _ = claim(
            store, after_round, after_runners[0],
            parallelism=8, ceiling=8, excluded=[after_runners[0]],
        )
        score_b, _token_b, _, _ = claim(
            store, after_round, after_runners[1],
            parallelism=8, ceiling=8, excluded=[after_runners[1]],
        )
        assert score_a["status"] == score_b["status"] == "leased"
        assert score_a["submission_id"] != score_b["submission_id"]
        blocked, _, _, _ = claim(
            store, after_round, after_runners[2],
            parallelism=8, ceiling=8, excluded=[after_runners[2]],
        )
        assert blocked["status"] == "no_pending"

        # Settlement and terminal completion release this submission's next
        # ICP without changing the budget or retry policy.
        identity_a, reserved_a = _reserve_dynamic(
            store, score_a, token_a, "after-first"
        )
        assert reserved_a["status"] == "reserved"
        assert store.mark_dispatched(
            run_id=score_a["run_id"],
            lease_token_hash=hash_lease_token(token_a),
            call_identity=identity_a,
        )["status"] == "dispatched"
        assert store.settle_call(
            run_id=score_a["run_id"],
            lease_token_hash=hash_lease_token(token_a),
            call_identity=identity_a,
            actual_microusd=2_000,
            terminal_response={"status": 200},
        )["status"] == "settled"
        assert complete(
            store, score_a["run_id"], hash_lease_token(token_a), "judge_error"
        )["status"] == "failed"
        next_a, _next_token, _, _ = claim(
            store, after_round, after_runners[2],
            parallelism=8, ceiling=8, excluded=[after_runners[2]],
        )
        assert next_a["status"] == "leased"
        assert next_a["submission_id"] == score_a["submission_id"]

        # An expired lease does not block a sibling. The global claim lock
        # makes two validator claims race to exactly one new score lease.
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET lease_expires_at="
                    "pg_catalog.clock_timestamp() - interval '1 second' "
                    "WHERE run_id=%s",
                    (next_a["run_id"],),
                )
        barrier = threading.Barrier(2)

        def simultaneous_claim(index: int) -> dict:
            local_store = _store(database)
            try:
                barrier.wait(timeout=10)
                return claim(
                    local_store, after_round, after_runners[index],
                    parallelism=8, ceiling=8, excluded=[after_runners[index]],
                )[0]
            finally:
                local_store.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            raced = list(executor.map(simultaneous_claim, (2, 3)))
        assert sorted(result["status"] for result in raced) == [
            "leased", "no_pending"
        ]
        race_winner = next(result for result in raced if result["status"] == "leased")
        assert race_winner["submission_id"] == participants[0]["submission_id"]
    finally:
        store.close()
