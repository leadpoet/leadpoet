"""Participation credit stays with validators that produce original judgments."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.company_judgments_postgres_test import _complete, _open
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, hotkey


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/221-lab-arena-participation-original-judgments.sql"
)


@pytest.fixture(scope="module")
def database():
    migrations = tuple(POSTGREST_MIGRATIONS)
    if MIGRATION.name not in migrations:
        migrations += (MIGRATION.name,)
    yield from database_with_lab_arena_migration(migrations)


@pytest.fixture()
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


def _item(items: list[dict], submission_id: str) -> dict:
    return next(
        item for item in items if item["submission_id"] == submission_id
    )


def _participation(store: ArenaStore, run_id: str):
    return store.get_run(run_id)["participation_accepted_at"]


def test_only_original_company_judgment_work_earns_participation(
    store, database
):
    round_id = "arena-2026-09-11-pj"
    _participants, items = _open(
        store,
        round_id=round_id,
        markers_by_submission=[
            ["shared"],
            ["shared"],
            ["shared", "new"],
            ["failure"],
        ],
    )

    source_runner = hotkey("original-judgment-source")
    source, source_token, _, _ = claim(
        store, round_id, source_runner, excluded=[source_runner]
    )
    assert len(source["company_judgment_cache"]["hits"]) == 0
    assert len(source["company_judgment_cache"]["misses"]) == 1
    source_result = _complete(
        store,
        source,
        source_token,
        refs=_item(items, source["submission_id"])["company_judgment_refs"],
    )
    assert source_result["status"] == "accepted"
    source_at = _participation(store, source["run_id"])
    assert source_at is not None
    assert store.has_recent_participation("finney", 71, source_runner)

    cache_runner = hotkey("original-judgment-cache-reuser")
    cached, cached_token, _, _ = claim(
        store, round_id, cache_runner, excluded=[cache_runner]
    )
    assert len(cached["company_judgment_cache"]["hits"]) == 1
    assert len(cached["company_judgment_cache"]["misses"]) == 0
    cached_result = _complete(
        store,
        cached,
        cached_token,
        refs=_item(items, cached["submission_id"])["company_judgment_refs"],
        completion_marker="e",
    )
    assert cached_result["status"] == "accepted"
    assert _participation(store, cached["run_id"]) is None
    assert not store.has_recent_participation("finney", 71, cache_runner)

    mixed_runner = hotkey("original-judgment-mixed")
    mixed, mixed_token, _, _ = claim(
        store, round_id, mixed_runner, excluded=[mixed_runner]
    )
    assert len(mixed["company_judgment_cache"]["hits"]) == 1
    assert len(mixed["company_judgment_cache"]["misses"]) == 1
    mixed_result = _complete(
        store,
        mixed,
        mixed_token,
        refs=_item(items, mixed["submission_id"])["company_judgment_refs"],
        completion_marker="d",
    )
    assert mixed_result["status"] == "accepted"
    mixed_at = _participation(store, mixed["run_id"])
    assert mixed_at is not None
    assert store.has_recent_participation("finney", 71, mixed_runner)

    failed_runner = hotkey("original-judgment-failure")
    failed, failed_token, _, _ = claim(
        store, round_id, failed_runner, excluded=[failed_runner]
    )
    failed_result = store.complete_attempt(
        run_id=failed["run_id"],
        lease_token_hash=hash_lease_token(failed_token),
        result={"terminal_status": "judge_error"},
        terminal_cause="judge_error",
        output_ref="",
        output_hash="",
        company_judgment_evidence=[],
        completion_request_hash="sha256:" + "c" * 64,
    )
    assert failed_result["status"] == "failed"
    assert _participation(store, failed["run_id"]) is None

    assert _complete(
        store,
        source,
        source_token,
        refs=_item(items, source["submission_id"])["company_judgment_refs"],
    )["idempotent"] is True
    assert _complete(
        store,
        cached,
        cached_token,
        refs=_item(items, cached["submission_id"])["company_judgment_refs"],
        completion_marker="e",
    )["idempotent"] is True
    assert _complete(
        store,
        mixed,
        mixed_token,
        refs=_item(items, mixed["submission_id"])["company_judgment_refs"],
        completion_marker="d",
    )["idempotent"] is True
    replayed_failure = store.complete_attempt(
        run_id=failed["run_id"],
        lease_token_hash=hash_lease_token(failed_token),
        result={"terminal_status": "judge_error"},
        terminal_cause="judge_error",
        output_ref="",
        output_hash="",
        company_judgment_evidence=[],
        completion_request_hash="sha256:" + "c" * 64,
    )
    assert replayed_failure["status"] == "failed"
    assert replayed_failure["idempotent"] is True
    assert _participation(store, source["run_id"]) == source_at
    assert _participation(store, cached["run_id"]) is None
    assert _participation(store, mixed["run_id"]) == mixed_at
    assert _participation(store, failed["run_id"]) is None

    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                """
                SELECT proowner::regrole::text,
                       has_function_privilege(
                         'lab_arena_service',
                         'public.lab_arena_runs_participation_v1()',
                         'EXECUTE'
                       ),
                       has_schema_privilege(
                         'lab_arena_owner', 'public', 'CREATE'
                       )
                FROM pg_proc
                WHERE oid =
                  'public.lab_arena_runs_participation_v1()'::regprocedure
                """
            )
            assert cursor.fetchone() == ("lab_arena_owner", False, False)
            cursor.execute(
                "SELECT public.lab_arena_participation_schema_v1()"
            )
            assert cursor.fetchone()[0] == {
                "schema_version": (
                    "leadpoet.lab_arena.participation_schema.v1"
                ),
                "version": 221,
            }
            cursor.execute(
                """
                SELECT proowner::regrole::text,
                       has_function_privilege(
                         'lab_arena_service',
                         'public.lab_arena_participation_schema_v1()',
                         'EXECUTE'
                       ),
                       has_function_privilege(
                         'anon',
                         'public.lab_arena_participation_schema_v1()',
                         'EXECUTE'
                       ),
                       has_function_privilege(
                         'authenticated',
                         'public.lab_arena_participation_schema_v1()',
                         'EXECUTE'
                       ),
                       has_function_privilege(
                         'service_role',
                         'public.lab_arena_participation_schema_v1()',
                         'EXECUTE'
                       )
                FROM pg_proc
                WHERE oid =
                  'public.lab_arena_participation_schema_v1()'::regprocedure
                """
            )
            assert cursor.fetchone() == (
                "lab_arena_owner",
                True,
                False,
                False,
                False,
            )
    assert _participation(store, source["run_id"]) == source_at
    assert _participation(store, cached["run_id"]) is None
    assert _participation(store, mixed["run_id"]) == mixed_at
    assert _participation(store, failed["run_id"]) is None
