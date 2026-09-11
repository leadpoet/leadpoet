"""PostgreSQL enforcement for the Arena Day 2 benchmark commitment."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_BENCHMARK_DISCLOSURE_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    hotkey,
    round_config,
)
from tests.postgres_migration_harness import SCRIPTS


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration()


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database

    def _connect():
        return psycopg2.connect(**dsn)

    return _connect


@pytest.fixture()
def store(connect):
    transport = PsycopgTransport(connect)
    yield ArenaStore(transport, lease_ttl_seconds=120)
    transport.close()


@pytest.fixture()
def superuser(connect):
    connection = connect()
    connection.autocommit = True
    yield connection
    connection.close()


def disclosure_config(
    round_id: str,
    runner_label: str,
    *,
    policy=True,
    mode="live",
    before_cutoff=False,
):
    configuration = round_config(round_id, [hotkey(runner_label)], mode=mode)
    configuration.update(
        {
            "network_name": "test",
            "netuid": 401,
            "schedule": {
                "submission_open": (
                    "2099-04-01T06:15:00Z"
                    if before_cutoff
                    else "2026-09-08T06:15:00Z"
                ),
                "submission_cutoff": (
                    "2099-04-02T06:15:00Z"
                    if before_cutoff
                    else "2026-09-09T06:15:00Z"
                ),
            },
        }
    )
    if policy is not False:
        configuration["benchmark_disclosure_policy"] = (
            "commit_reveal_day2_v1" if policy is True else policy
        )
    return configuration


def commitment_document(round_id: str, seed: str = "a", *, before_cutoff=False):
    manifest = {
        "schema_version": "leadpoet.lab_arena.benchmark_commitment.v1",
        "network_name": "test",
        "netuid": 401,
        "round_id": round_id,
        "icp_set_date": "2099-04-01" if before_cutoff else "2026-09-08",
        "evaluation_date": "2099-04-02" if before_cutoff else "2026-09-09",
        "public_at": (
            "2099-04-03T06:15:00Z"
            if before_cutoff
            else "2026-09-10T06:15:00Z"
        ),
        "disclosure_policy": "commit_reveal_day2_v1",
        "icp_count": 20,
        "entries": [
            {
                "icp_position": position,
                "icp_hash": "sha256:" + hashlib.sha256(
                    (seed + str(position)).encode("utf-8")
                ).hexdigest(),
            }
            for position in range(20)
        ],
    }
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return {
        "manifest": manifest,
        "manifest_hash": "sha256:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "canonical_manifest": canonical,
    }


def prepare_round(
    store: ArenaStore,
    superuser,
    suffix: str,
    *,
    participant_count=1,
    mode="live",
    before_cutoff=False,
):
    round_id = "arena-2099-04-02-" + suffix
    configuration = disclosure_config(
        round_id,
        suffix + "-runner",
        mode=mode,
        before_cutoff=before_cutoff,
    )
    assert store.create_round(round_id, configuration)["status"] == "created"
    participants = []
    with superuser.cursor() as cursor:
        for index in range(participant_count):
            participant = {
                "submission_id": "%s-sub-%d" % (suffix, index),
                "miner_hotkey": hotkey("%s-miner-%d" % (suffix, index)),
                "is_king": index == 0,
            }
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id, round_id, miner_hotkey, status, is_king) "
                "VALUES (%s, %s, %s, 'frozen', %s)",
                (
                    participant["submission_id"],
                    round_id,
                    participant["miner_hotkey"],
                    participant["is_king"],
                ),
            )
            participants.append(participant)
    return round_id, configuration, participants


def commit_v3(
    store,
    round_id,
    configuration,
    participants,
    document=None,
    seed="a",
    *,
    before_cutoff=False,
):
    return store.commit_round_v3(
        round_id,
        participants=participants,
        benchmark_ref="arena/%s/benchmarks/%s.json" % (round_id, seed * 64),
        evaluation_date="2099-04-02" if before_cutoff else "2026-09-09",
        icp_set_date="2099-04-01" if before_cutoff else "2026-09-08",
        scorer_image_digest=configuration["scorer_image_digest"],
        scorer_image_reference=configuration["scorer_image_reference"],
        benchmark_commitment_doc=document or commitment_document(
            round_id, seed, before_cutoff=before_cutoff
        ),
    )


def test_capability_columns_and_grants_are_exact(store, superuser):
    assert store.benchmark_disclosure_schema() == {
        "schema_version": "leadpoet.lab_arena.benchmark_disclosure.v1",
        "version": 210,
        "policy": "commit_reveal_day2_v1",
    }
    with superuser.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena_schema_version_v1()")
        assert cursor.fetchone()[0]["version"] == 197
        cursor.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'lab_arena_rounds' "
            "AND column_name LIKE 'benchmark_%' ORDER BY column_name"
        )
        assert cursor.fetchall() == [
            ("benchmark_commitment_doc", "jsonb"),
            ("benchmark_committed_at", "timestamp with time zone"),
            ("benchmark_ref", "text"),
            ("benchmark_reveal_at", "timestamp with time zone"),
        ]
        signature = (
            "public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)"
        )
        capability = "public.lab_arena_benchmark_disclosure_schema_v1()"
        for role, expected in (
            ("lab_arena_service", True),
            ("service_role", False),
            ("anon", False),
            ("authenticated", False),
        ):
            cursor.execute(
                "SELECT has_function_privilege(%s, %s, 'EXECUTE'), "
                "has_function_privilege(%s, %s, 'EXECUTE')",
                (role, signature, role, capability),
            )
            assert cursor.fetchone() == (expected, expected)
        for internal_signature in (
            "public.lab_arena__benchmark_commitment_valid_v1(text,text,bigint,date,text,timestamp with time zone,jsonb)",
            "public.lab_arena_benchmark_commitment_guard_v1()",
        ):
            for role in (
                "lab_arena_service",
                "service_role",
                "anon",
                "authenticated",
            ):
                cursor.execute(
                    "SELECT has_function_privilege(%s, %s, 'EXECUTE')",
                    (role, internal_signature),
                )
                assert cursor.fetchone() == (False,)


def test_creation_derives_reveal_time_and_rejects_unknown_or_malformed_policy(store):
    legacy_id = "arena-2099-04-02-legacy"
    assert store.create_round(
        legacy_id, disclosure_config(legacy_id, "legacy-runner", policy=False)
    )["status"] == "created"
    legacy = store.get_round(legacy_id)
    assert legacy["benchmark_reveal_at"] is None
    assert legacy["benchmark_commitment_doc"] is None
    assert legacy["benchmark_committed_at"] is None

    policy_id = "arena-2099-04-02-created"
    assert store.create_round(
        policy_id, disclosure_config(policy_id, "created-runner")
    )["status"] == "created"
    assert store.get_round(policy_id)["benchmark_reveal_at"] == (
        "2026-09-10T06:15:00+00:00"
    )

    for suffix, policy in (("unknown", "future_policy"), ("object", {"v": 1})):
        round_id = "arena-2099-04-02-" + suffix
        with pytest.raises(ArenaStoreError, match="disclosure_policy_invalid"):
            store.create_round(
                round_id,
                disclosure_config(round_id, suffix + "-runner", policy=policy),
            )


def test_v3_atomically_commits_metadata_and_preserves_effective_v2_budget(
    store, superuser
):
    round_id, configuration, participants = prepare_round(
        store, superuser, "atomic"
    )
    before = store.get_round(round_id)
    result = commit_v3(store, round_id, configuration, participants)
    assert result["status"] == "ok"
    row = store.get_round(round_id)
    assert row["status"] == "committed"
    assert row["status_generation"] == before["status_generation"] + 1
    assert row["benchmark_commitment_doc"] == commitment_document(round_id)
    assert row["benchmark_committed_at"] == result["benchmark_committed_at"]
    assert row["benchmark_reveal_at"] == "2026-09-10T06:15:00+00:00"
    assert row["configuration_doc"]["execution_cap_microusd"] == 50_000_000
    assert row["configuration_doc"]["cost_per_company_microusd"] == 500_000


def test_manifest_validation_fails_closed_without_changing_the_open_round(
    store, superuser
):
    round_id, configuration, participants = prepare_round(
        store, superuser, "malformed"
    )
    valid = commitment_document(round_id)
    invalid_documents = []

    extra_key = copy.deepcopy(valid)
    extra_key["unexpected"] = True
    invalid_documents.append(extra_key)
    wrong_scope = copy.deepcopy(valid)
    wrong_scope["manifest"]["round_id"] = "arena-2099-04-02-other"
    invalid_documents.append(wrong_scope)
    wrong_position = copy.deepcopy(valid)
    wrong_position["manifest"]["entries"][4]["icp_position"] = 5
    invalid_documents.append(wrong_position)
    wrong_hash = copy.deepcopy(valid)
    wrong_hash["manifest_hash"] = "sha256:" + "0" * 64
    invalid_documents.append(wrong_hash)
    wrong_canonical = copy.deepcopy(valid)
    wrong_canonical["canonical_manifest"] += " "
    invalid_documents.append(wrong_canonical)

    for document in invalid_documents:
        with pytest.raises(ArenaStoreError, match="benchmark_commitment_invalid"):
            commit_v3(store, round_id, configuration, participants, document)
        assert store.get_round(round_id)["status"] == "open"

    with pytest.raises(ArenaStoreError, match="round_commit_invalid"):
        store.commit_round_v3(
            round_id,
            participants=participants,
            benchmark_ref="arena/%s/benchmark.json" % round_id,
            evaluation_date="2026-09-09",
            icp_set_date="2026-09-08",
            scorer_image_digest=configuration["scorer_image_digest"],
            scorer_image_reference=configuration["scorer_image_reference"],
            benchmark_commitment_doc=valid,
        )


def test_v3_rejects_commitment_before_the_stored_cutoff(store, superuser):
    round_id, configuration, participants = prepare_round(
        store, superuser, "early", before_cutoff=True
    )
    with pytest.raises(ArenaStoreError, match="round_commit_too_early"):
        commit_v3(
            store,
            round_id,
            configuration,
            participants,
            before_cutoff=True,
        )
    assert store.get_round(round_id)["status"] == "open"
    with superuser.cursor() as cursor:
        with pytest.raises(Exception, match="commitment_transition_invalid"):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status = 'committed', "
                "participants = %s::jsonb, benchmark_ref = %s, "
                "evaluation_date = '2099-04-02', icp_set_date = '2099-04-01', "
                "benchmark_commitment_doc = %s::jsonb, "
                "benchmark_committed_at = clock_timestamp() "
                "WHERE round_id = %s",
                (
                    json.dumps(participants),
                    "arena/%s/benchmarks/%s.json" % (round_id, "a" * 64),
                    json.dumps(
                        commitment_document(round_id, before_cutoff=True)
                    ),
                    round_id,
                ),
            )
    assert store.get_round(round_id)["status"] == "open"


def test_v2_generic_and_privileged_updates_cannot_bypass_commitment(store, superuser):
    for suffix, use_v2 in (("v2bypass", True), ("generic", False)):
        round_id, configuration, participants = prepare_round(
            store, superuser, suffix
        )
        with pytest.raises(ArenaStoreError, match="benchmark_commitment_required"):
            if use_v2:
                store.commit_round_v2(
                    round_id,
                    participants=participants,
                    benchmark_ref="arena/%s/benchmark.json" % round_id,
                    evaluation_date="2026-09-09",
                    icp_set_date="2026-09-08",
                    scorer_image_digest=configuration["scorer_image_digest"],
                    scorer_image_reference=configuration[
                        "scorer_image_reference"
                    ],
                )
            else:
                store.transition_round(
                    round_id,
                    "open",
                    "committed",
                    {
                        "participants": participants,
                        "benchmark_ref": "arena/%s/benchmark.json" % round_id,
                        "evaluation_date": "2026-09-09",
                    },
                )
        assert store.get_round(round_id)["status"] == "open"

    round_id, configuration, participants = prepare_round(
        store, superuser, "immutable"
    )
    assert commit_v3(store, round_id, configuration, participants)["status"] == "ok"
    with superuser.cursor() as cursor:
        for assignment in (
            "benchmark_reveal_at = benchmark_reveal_at + interval '1 second'",
            "benchmark_committed_at = benchmark_committed_at + interval '1 second'",
            "benchmark_ref = benchmark_ref || '.changed'",
            "benchmark_commitment_doc = benchmark_commitment_doc || '{\"x\":1}'::jsonb",
            "icp_set_date = icp_set_date + 1",
            "evaluation_date = '2099-04-03'",
        ):
            with pytest.raises(Exception, match="immutable"):
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET %s WHERE round_id = %%s"
                    % assignment,
                    (round_id,),
                )
    assert store.cancel_round(round_id, "fixture cancellation")["status"] == "cancelled"
    cancelled = store.get_round(round_id)
    assert cancelled["benchmark_commitment_doc"] == commitment_document(round_id)

    open_id = "arena-2099-04-02-precommit"
    assert store.create_round(
        open_id, disclosure_config(open_id, "precommit-runner")
    )["status"] == "created"
    assert store.cancel_round(open_id, "no benchmark selected")["status"] == "cancelled"
    assert store.get_round(open_id)["benchmark_commitment_doc"] is None


def test_concurrent_commits_select_exactly_one_immutable_candidate(
    store, superuser
):
    round_id, configuration, participants = prepare_round(
        store, superuser, "concurrent"
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                commit_v3,
                store,
                round_id,
                configuration,
                participants,
                commitment_document(round_id, seed),
                seed,
            )
            for seed in ("a", "b")
        ]
        results = [future.result(timeout=10) for future in futures]
    assert sorted(result["status"] for result in results) == ["ok", "stale"]
    row = store.get_round(round_id)
    winning_seed = row["benchmark_ref"].rsplit("/", 1)[-1][0]
    assert winning_seed in ("a", "b")
    assert row["benchmark_commitment_doc"] == commitment_document(
        round_id, winning_seed
    )


def test_published_promotion_and_reward_updates_preserve_commitment(
    store, superuser
):
    round_id, configuration, participants = prepare_round(
        store, superuser, "published", participant_count=2, mode="shadow"
    )
    assert commit_v3(store, round_id, configuration, participants)["status"] == "ok"
    baseline, winner = participants
    with superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status = 'scored', "
            "finalists = '[]'::jsonb WHERE round_id = %s",
            (round_id,),
        )
    published_at = "2026-09-09T07:15:00Z"
    publication = {
        "schema_version": "leadpoet.lab_arena.publication.v1",
        "round_id": round_id,
        "participants": [
            {
                "submission_id": item["submission_id"],
                "miner_hotkey": item["miner_hotkey"],
                "is_baseline": item["is_king"],
            }
            for item in participants
        ],
        "stage1_ranking": [],
        "finalists": [],
        "final_ranking": [
            {
                "rank": 1,
                "submission_id": winner["submission_id"],
                "final_score": 60,
                "is_baseline": False,
            },
            {
                "rank": 2,
                "submission_id": baseline["submission_id"],
                "final_score": 50,
                "is_baseline": True,
            },
        ],
        "king_decision": {
            "outcome": "crowned",
            "king_submission_id": winner["submission_id"],
            "king_hotkey": winner["miner_hotkey"],
            "winner_submission_id": winner["submission_id"],
        },
        "published_at": published_at,
    }
    assert store.transition_round(
        round_id,
        "scored",
        "published",
        {"publication_doc": publication, "published_at": published_at},
    )["status"] == "ok"
    committed = store.get_round(round_id)["benchmark_commitment_doc"]

    promotion = {
        "commit": "1" * 40,
        "main_before": "2" * 40,
        "lab_before": "3" * 40,
        "timestamp": published_at,
    }
    with superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET promotion_doc = %s::jsonb "
            "WHERE round_id = %s",
            (json.dumps(promotion), round_id),
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET baseline_promoted_at = "
            "clock_timestamp() WHERE round_id = %s",
            (round_id,),
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET reward_activated_at = "
            "clock_timestamp(), effective_reward_epoch = 900001, "
            "reward_basis_hash = %s, reward_basis_doc = '{}'::jsonb, "
            "signing_key_doc = '{}'::jsonb, king_start_epoch = 900001 "
            "WHERE round_id = %s",
            ("sha256:" + "9" * 64, round_id),
        )
    updated = store.get_round(round_id)
    assert updated["benchmark_commitment_doc"] == committed
    assert updated["promotion_doc"] == promotion
    assert updated["baseline_promoted_at"] is not None
    assert updated["reward_activated_at"] is not None


def test_upgrade_preserves_legacy_rows_and_reapplication_is_idempotent():
    database = database_with_lab_arena_migration(DEFAULT_MIGRATIONS[:-1])
    connection = None
    transport = None
    try:
        psycopg2, dsn = next(database)
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True

        def connect():
            return psycopg2.connect(**dsn)

        transport = PsycopgTransport(connect)
        legacy_store = ArenaStore(transport)
        round_id = "arena-2099-04-02-upgrade"
        assert legacy_store.create_round(
            round_id,
            disclosure_config(round_id, "upgrade-runner", policy=False),
        )["status"] == "created"
        migration = (
            SCRIPTS / LAB_ARENA_BENCHMARK_DISCLOSURE_MIGRATION
        ).read_text(encoding="utf-8")
        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT benchmark_reveal_at, benchmark_commitment_doc, "
                "benchmark_committed_at FROM public.lab_arena_rounds "
                "WHERE round_id = %s",
                (round_id,),
            )
            assert cursor.fetchone() == (None, None, None)
    finally:
        if transport is not None:
            transport.close()
        if connection is not None:
            connection.close()
        database.close()
