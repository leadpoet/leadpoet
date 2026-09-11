"""Atomic accepted-judgment reuse in disposable PostgreSQL."""

from __future__ import annotations

from typing import Dict

import pytest

from lab_arena import contracts, judgment_cache, scoring
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    _commit_plan,
    _execute_everything,
    claim,
    commit_round,
    frozen_participants,
    hotkey,
    round_config,
)


MIGRATION = "212-lab-arena-accepted-judgment-cache.sql"


@pytest.fixture(scope="module")
def database():
    migrations = DEFAULT_MIGRATIONS
    if MIGRATION not in migrations:
        migrations = (*migrations, MIGRATION)
    yield from database_with_lab_arena_migration(migrations)


@pytest.fixture()
def store(database):
    psycopg2, dsn = database
    yield ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _scope(round_id: str, scored_run_id: str, position: int, marker: str) -> dict:
    document = scoring.build_scoring_input(
        scored_run_id=scored_run_id,
        icp={"icp_id": "icp-%d" % position, "prompt": "Find %s" % marker},
        companies=[{"company_name": marker, "website": "https://%s.example" % marker}],
        policy=scoring.build_scorer_policy(),
        evaluation_date="2026-09-02",
    )
    return judgment_cache.build_cache_scope(
        scoring_input=document,
        round_id=round_id,
        network_name="finney",
        netuid=71,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry.example/lab/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1",
    )


def _items(
    executed: Dict[tuple, str], participants: list[dict], *, round_id: str
) -> list[dict]:
    result = []
    for position, marker in ((0, "same"), (1, "failure")):
        group = []
        for participant in participants:
            submission_id = participant["submission_id"]
            scored_run_id = executed[(submission_id, position)]
            scope = _scope(round_id, scored_run_id, position, marker)
            group.append({
                "scored_run_id": scored_run_id,
                "submission_id": submission_id,
                "icp_position": position,
                "output_ref": "arena/x/outputs/%s.json" % scored_run_id,
                "judgment_cache_key": scope["cache_key"],
                "judgment_scope_doc": scope,
                "judgment_input_hash": scope["scoring_input_hash"],
            })
        miners = sorted(participant["miner_hotkey"] for participant in participants)
        for index, item in enumerate(sorted(group, key=lambda row: row["scored_run_id"])):
            item["judgment_group_leader"] = index == 0
            item["judgment_group_miner_hotkeys"] = miners
            result.append(item)
    return result


def test_first_accepted_judgment_atomically_accepts_identical_followers_and_failures_do_not_cache(store):
    round_id = "arena-2026-09-11-jc"
    runner = hotkey("jc-runner")
    config = round_config(round_id, [runner])
    config["integrity_policy"] = "arena_integrity_v1"
    assert store.create_round(round_id, config)["status"] == "created"
    participants = frozen_participants(store, round_id, 2, prefix="jc")
    commit_round(store, round_id, participants)
    assert store.open_stage(round_id, 1, participants, list(contracts.stage_positions(1)))["status"] == "ok"
    executed = _execute_everything(store, round_id, runner)
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)

    items = _items(executed, participants, round_id=round_id)
    opened = store.open_scoring(round_id, 1, items, integrity_cache=True)
    assert opened["assignments"] == 4 and opened["reused"] == 0

    refused, _, _, _ = claim(
        store, round_id, runner, excluded=[participants[0]["miner_hotkey"]]
    )
    assert refused["status"] == "no_pending"

    leased, token, _, _ = claim(store, round_id, runner)
    assert leased["status"] == "leased" and leased["kind"] == "score"
    source_run = store.get_run(leased["run_id"])
    assert source_run is not None
    output = scoring.build_scoring_output(
        leased["scored_run_id"], [{"company_name": "same", "final_score": 81.0}]
    )
    evidence = judgment_cache.build_evidence_snapshot(
        output=output,
        cache_scope=source_run["judgment_scope_doc"],
        source_score_run_id=leased["run_id"],
        source_scored_run_id=leased["scored_run_id"],
        source_output_ref="arena/x/scores/%s.json" % leased["run_id"],
        source_runner_hotkey=runner,
    )
    accepted = store.complete_attempt(
        run_id=leased["run_id"],
        lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref=evidence["source_output_ref"],
        judgment_evidence=evidence,
        judgment_evidence_hash=contracts.document_hash(evidence),
    )
    assert accepted["status"] == "accepted"
    repeated = store.complete_attempt(
        run_id=leased["run_id"],
        lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref=evidence["source_output_ref"],
        judgment_evidence=evidence,
        judgment_evidence_hash=contracts.document_hash(evidence),
    )
    assert repeated["status"] == "accepted" and repeated["idempotent"] is True
    same_key = source_run["judgment_cache_key"]
    same_runs = [
        run for run in store.list_runs(round_id, stage=1, kind="score")
        if run.get("judgment_cache_key") == same_key
    ]
    assert len(same_runs) == 2
    assert {run["status"] for run in same_runs} == {"accepted"}
    assert {run["judgment_cache_source_run_id"] for run in same_runs} == {
        leased["run_id"]
    }
    assert store.get_judgment_cache(same_key)["evidence_doc"] == evidence

    failed_lease, failed_token, _, _ = claim(store, round_id, runner)
    failed_source = store.get_run(failed_lease["run_id"])
    assert failed_source is not None
    assert store.complete_attempt(
        run_id=failed_lease["run_id"],
        lease_token_hash=hash_lease_token(failed_token),
        result={"terminal_status": "judge_error"},
        terminal_cause="judge_error",
        output_ref="",
    )["status"] == "failed"
    assert store.get_judgment_cache(failed_source["judgment_cache_key"]) is None
    retry = next(
        run for run in store.list_runs(round_id, stage=1, kind="score")
        if run["assignment_id"] == failed_source["assignment_id"]
        and run["attempt"] == 2
    )
    assert retry["status"] == "pending"
    assert retry["judgment_cache_key"] == failed_source["judgment_cache_key"]
    assert retry["judgment_group_miner_hotkeys"] == failed_source[
        "judgment_group_miner_hotkeys"
    ]

    retry_lease, retry_token, _, _ = claim(store, round_id, runner)
    assert retry_lease["run_id"] == retry["run_id"]
    retry_output = scoring.build_scoring_output(
        retry_lease["scored_run_id"],
        [{"company_name": "failure", "final_score": 55.0}],
    )
    retry_evidence = judgment_cache.build_evidence_snapshot(
        output=retry_output,
        cache_scope=retry["judgment_scope_doc"],
        source_score_run_id=retry["run_id"],
        source_scored_run_id=retry["scored_run_id"],
        source_output_ref="arena/x/scores/%s.json" % retry["run_id"],
        source_runner_hotkey=runner,
    )
    assert store.complete_attempt(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref=retry_evidence["source_output_ref"],
        judgment_evidence=retry_evidence,
        judgment_evidence_hash=contracts.document_hash(retry_evidence),
    )["status"] == "accepted"
    failure_runs = [
        run for run in store.list_runs(round_id, stage=1, kind="score")
        if run.get("judgment_cache_key") == failed_source["judgment_cache_key"]
    ]
    assert sorted(run["status"] for run in failure_runs) == [
        "accepted", "accepted", "failed"
    ]


def test_expired_judgment_leader_retry_keeps_the_exact_cache_identity(store, database):
    round_id = "arena-2026-09-11-je"
    runner = hotkey("je-runner")
    config = round_config(round_id, [runner])
    config["integrity_policy"] = "arena_integrity_v1"
    assert store.create_round(round_id, config)["status"] == "created"
    participants = frozen_participants(store, round_id, 1, prefix="je")
    commit_round(store, round_id, participants)
    assert store.open_stage(round_id, 1, participants, list(contracts.stage_positions(1)))["status"] == "ok"
    executed = _execute_everything(store, round_id, runner)
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    items = _items(executed, participants, round_id=round_id)[:1]
    assert store.open_scoring(round_id, 1, items, integrity_cache=True)["assignments"] == 1
    leased, _, _, _ = claim(store, round_id, runner)
    original = store.get_run(leased["run_id"])

    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE run_id = %s",
                (leased["run_id"],),
            )
    assert store.expire_leases(round_id)["retried"] == 1
    retry = next(
        run for run in store.list_runs(round_id, stage=1, kind="score")
        if run["assignment_id"] == original["assignment_id"] and run["attempt"] == 2
    )
    for field in (
        "judgment_cache_key", "judgment_input_hash", "judgment_scope_doc",
        "judgment_group_leader", "judgment_group_miner_hotkeys",
    ):
        assert retry[field] == original[field]
