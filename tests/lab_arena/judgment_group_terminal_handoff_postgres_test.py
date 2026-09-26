"""Terminal identical-judgment leader handoff in the current PostgreSQL schema."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache, scoring
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.company_judgments_postgres_test import (
    _frozen_participants,
)
from tests.lab_arena.judgment_cache_postgres_test import _items
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    _commit_plan,
    _execute_everything,
    claim,
    commit_round,
    hotkey,
    round_config,
)


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/363-lab-arena-judgment-group-terminal-handoff.sql"
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture()
def store(database):
    psycopg2, dsn = database
    yield ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _open_identical_group(
    store,
    round_id: str,
    runner: str,
    count: int,
    *,
    positions=(0,),
    alternate_second_leader=False,
):
    config = round_config(round_id, [runner])
    config["integrity_policy"] = "arena_integrity_v1"
    config["cost_per_company_microusd"] = 800_000
    config["scorer_policy"] = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2"
    )
    assert store.create_round(round_id, config)["status"] == "created"
    participants = _frozen_participants(
        store, round_id, count, prefix=round_id
    )
    commit_round(store, round_id, participants)
    assert store.open_stage(
        round_id, 1, participants, list(contracts.stage_positions(1))
    )["status"] == "ok"
    executed = _execute_everything(store, round_id, runner)
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    items = [
        item
        for item in _items(executed, participants, round_id=round_id)
        if item["icp_position"] in positions
    ]
    if alternate_second_leader:
        second_group = sorted(
            (
                item
                for item in items
                if item["icp_position"] == positions[1]
            ),
            key=lambda item: item["scored_run_id"],
        )
        assert len(second_group) == 2
        second_group[0]["judgment_group_leader"] = False
        second_group[1]["judgment_group_leader"] = True
    assert store.open_scoring(
        round_id, 1, items, integrity_cache=True
    )["assignments"] == count * len(positions)
    return participants


def _group(store, round_id: str):
    runs = store.list_runs(round_id, stage=1, kind="score")
    cache_keys = {run["judgment_cache_key"] for run in runs}
    assert len(cache_keys) == 1
    return runs


def _groups(store, round_id: str):
    grouped = {}
    for run in store.list_runs(round_id, stage=1, kind="score"):
        grouped.setdefault(run["judgment_cache_key"], []).append(run)
    return grouped


def _fail_judge(store, lease, token):
    return store.complete_attempt(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "judge_error"},
        terminal_cause="judge_error",
        output_ref="",
    )


def test_final_judge_failure_hands_off_then_success_fills_shared_cache(store):
    round_id = "arena-2026-09-26-jgc"
    runner_one = hotkey("jgroup-complete-runner-one")
    runner_two = hotkey("jgroup-complete-runner-two")
    _open_identical_group(store, round_id, runner_one, 3)

    first, first_token, _, _ = claim(
        store, round_id, runner_one, excluded=[runner_one]
    )
    first_result = _fail_judge(store, first, first_token)
    assert first_result["confirmation_attempt"] == 2
    after_first = _group(store, round_id)
    assert sum(
        run["status"] == "pending" and run["judgment_group_leader"]
        for run in after_first
    ) == 1
    assert sum(
        run["status"] == "pending" and not run["judgment_group_leader"]
        for run in after_first
    ) == 2

    retry, retry_token, _, _ = claim(
        store, round_id, runner_two, excluded=[runner_two]
    )
    assert retry["attempt"] == 2
    retry_result = _fail_judge(store, retry, retry_token)
    assert "confirmation_attempt" not in retry_result
    after_final = _group(store, round_id)
    promoted = [
        run
        for run in after_final
        if run["status"] == "pending" and run["judgment_group_leader"]
    ]
    assert len(promoted) == 1
    assert sum(
        run["status"] == "pending" and not run["judgment_group_leader"]
        for run in after_final
    ) == 1

    follower, follower_token, _, _ = claim(
        store, round_id, runner_one, excluded=[runner_one]
    )
    assert follower["run_id"] == promoted[0]["run_id"]
    follower_run = store.get_run(follower["run_id"])
    output = scoring.build_scoring_output(
        follower["scored_run_id"],
        [{"company_name": "same", "final_score": 82.0}],
    )
    evidence = judgment_cache.build_evidence_snapshot(
        output=output,
        cache_scope=follower_run["judgment_scope_doc"],
        source_score_run_id=follower["run_id"],
        source_scored_run_id=follower["scored_run_id"],
        source_output_ref="arena/x/scores/%s.json" % follower["run_id"],
        source_runner_hotkey=runner_one,
        runner_authority_exclusions=follower["runner_authority_exclusions"],
    )
    assert store.complete_attempt(
        run_id=follower["run_id"],
        lease_token_hash=hash_lease_token(follower_token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref=evidence["source_output_ref"],
        judgment_evidence=evidence,
        judgment_evidence_hash=contracts.document_hash(evidence),
    )["status"] == "accepted"
    final = _group(store, round_id)
    assert sorted(run["status"] for run in final) == [
        "accepted", "accepted", "failed", "failed"
    ]
    accepted = [run for run in final if run["status"] == "accepted"]
    assert {run["judgment_cache_source_run_id"] for run in accepted} == {
        follower["run_id"]
    }


def test_final_expired_leader_hands_off_without_early_promotion(store, database):
    round_id = "arena-2026-09-26-jge"
    runner_one = hotkey("jgroup-expiry-runner-one")
    runner_two = hotkey("jgroup-expiry-runner-two")
    _open_identical_group(store, round_id, runner_one, 2)

    first, _, _, _ = claim(store, round_id, runner_one, excluded=[runner_one])
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs "
                "SET lease_expires_at=clock_timestamp()-interval '1 second' "
                "WHERE run_id=%s",
                (first["run_id"],),
            )
    assert store.expire_leases(round_id)["retried"] == 1
    after_first = _group(store, round_id)
    assert sum(
        run["status"] == "pending" and run["judgment_group_leader"]
        for run in after_first
    ) == 1
    assert sum(
        run["status"] == "pending" and not run["judgment_group_leader"]
        for run in after_first
    ) == 1

    retry, _, _, _ = claim(store, round_id, runner_two, excluded=[runner_two])
    assert retry["attempt"] == 2
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs "
                "SET lease_expires_at=clock_timestamp()-interval '1 second' "
                "WHERE run_id=%s",
                (retry["run_id"],),
            )
    assert store.expire_leases(round_id)["retried"] == 0
    promoted = [
        run
        for run in _group(store, round_id)
        if run["status"] == "pending" and run["judgment_group_leader"]
    ]
    assert len(promoted) == 1
    follower, _, _, _ = claim(
        store, round_id, runner_one, excluded=[runner_one]
    )
    assert follower["run_id"] == promoted[0]["run_id"]


def test_concurrent_final_failures_in_distinct_groups_do_not_deadlock(store):
    round_id = "arena-2026-09-26-jgx"
    runners = [hotkey("jgroup-concurrent-runner-%d" % index) for index in range(4)]
    _open_identical_group(
        store,
        round_id,
        runners[0],
        2,
        positions=(0, 1),
        alternate_second_leader=True,
    )

    first_leases = [
        claim(store, round_id, runners[index], excluded=[runners[index]])
        for index in range(2)
    ]
    assert {lease[0]["icp_position"] for lease in first_leases} == {0, 1}
    for lease, token, _, _ in first_leases:
        assert _fail_judge(store, lease, token)["confirmation_attempt"] == 2

    retries = [
        claim(store, round_id, runners[index + 2], excluded=[runners[index + 2]])
        for index in range(2)
    ]
    assert {lease[0]["icp_position"] for lease in retries} == {0, 1}
    assert all(lease[0]["attempt"] == 2 for lease in retries)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_fail_judge, store, lease, token)
            for lease, token, _, _ in retries
        ]
        results = [future.result(timeout=10) for future in futures]
    assert all("confirmation_attempt" not in result for result in results)
    groups = _groups(store, round_id)
    assert len(groups) == 2
    assert all(
        sum(
            run["status"] == "pending" and run["judgment_group_leader"]
            for run in group
        ) == 1
        for group in groups.values()
    )


def test_migration_replay_repairs_one_current_orphan_only(store, database):
    round_id = "arena-2026-09-26-jra"
    runner = hotkey("jgroup-repair-runner")
    _open_identical_group(store, round_id, runner, 3)
    initial = _group(store, round_id)
    assert sum(
        run["status"] == "pending" and run["judgment_group_leader"]
        for run in initial
    ) == 1

    cache_round_id = "arena-2026-09-26-jrc"
    _open_identical_group(store, cache_round_id, runner, 2)
    cached_lease, cached_token, _, _ = claim(
        store, cache_round_id, runner, excluded=[runner]
    )
    cached_run = store.get_run(cached_lease["run_id"])
    cached_output = scoring.build_scoring_output(
        cached_lease["scored_run_id"],
        [{"company_name": "same", "final_score": 83.0}],
    )
    cached_evidence = judgment_cache.build_evidence_snapshot(
        output=cached_output,
        cache_scope=cached_run["judgment_scope_doc"],
        source_score_run_id=cached_lease["run_id"],
        source_scored_run_id=cached_lease["scored_run_id"],
        source_output_ref="arena/x/scores/%s.json" % cached_lease["run_id"],
        source_runner_hotkey=runner,
        runner_authority_exclusions=cached_lease[
            "runner_authority_exclusions"
        ],
    )
    assert store.complete_attempt(
        run_id=cached_lease["run_id"],
        lease_token_hash=hash_lease_token(cached_token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref=cached_evidence["source_output_ref"],
        judgment_evidence=cached_evidence,
        judgment_evidence_hash=contracts.document_hash(cached_evidence),
    )["status"] == "accepted"
    assert store.get_judgment_cache(cached_run["judgment_cache_key"])

    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            old_generation_run = next(
                run
                for run in initial
                if run["status"] == "pending"
                and not run["judgment_group_leader"]
            )
            cached_follower = next(
                run
                for run in _group(store, cache_round_id)
                if run["run_id"] != cached_lease["run_id"]
            )
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs "
                "SET stage_generation=stage_generation-1 "
                "WHERE run_id=%s",
                (old_generation_run["run_id"],),
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='pending', "
                "terminal_cause=NULL,result_doc=NULL,output_ref=NULL,"
                "runner_hotkey=NULL,lease_token_hash=NULL,"
                "lease_expires_at=NULL,judgment_group_leader=FALSE,"
                "judgment_cache_source_run_id=NULL "
                "WHERE run_id=%s",
                (cached_follower["run_id"],),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            # A healthy active leader, a cached follower, and an old-generation
            # follower are all controls that replay must leave unchanged.
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))

    active_control = _group(store, round_id)
    assert sum(
        run["stage_generation"] == initial[0]["stage_generation"]
        and run["status"] == "pending"
        and run["judgment_group_leader"]
        for run in active_control
    ) == 1
    assert next(
        run for run in active_control
        if run["run_id"] == old_generation_run["run_id"]
    )["judgment_group_leader"] is False
    assert next(
        run for run in _group(store, cache_round_id)
        if run["run_id"] == cached_follower["run_id"]
    )["judgment_group_leader"] is False

    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs "
                "SET judgment_group_leader=FALSE "
                "WHERE round_id=%s AND kind='score' AND status='pending' "
                "AND stage_generation=(SELECT stage_generation "
                "FROM public.lab_arena_rounds WHERE round_id=%s)",
                (round_id, round_id),
            )
            cursor.execute(
                MIGRATION.read_text(encoding="utf-8")
            )
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))

    repaired = _group(store, round_id)
    assert sum(
        run["stage_generation"] == initial[0]["stage_generation"]
        and run["status"] == "pending"
        and run["judgment_group_leader"]
        for run in repaired
    ) == 1
    assert sum(
        run["stage_generation"] == initial[0]["stage_generation"]
        and run["status"] == "pending"
        and not run["judgment_group_leader"]
        for run in repaired
    ) == 1
    assert next(
        run for run in repaired
        if run["run_id"] == old_generation_run["run_id"]
    )["judgment_group_leader"] is False
