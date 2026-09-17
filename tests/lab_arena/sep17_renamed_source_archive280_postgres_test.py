"""Archive-only authority update after the public baseline repository rename."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.sep17_baseline_recovery278_postgres_test import (
    BANK_SHA,
    MIGRATION as RECOVERY278,
    ROUND,
    SCHEDULE,
    _restore,
)

ROOT = Path(__file__).parents[2]
RECOVERY279 = ROOT / "scripts/279-sep17-reviewed-source-before-activation.sql"
MIGRATION = ROOT / "scripts/280-sep17-renamed-source-archive-before-activation.sql"
AUTHORITY = "public.lab_arena_sep17_baseline_recovery278_authority"
COMMIT = "4ceae936b902433432a195f77af3f94559d80378"
OLD_SHA = "6c9eeaac41386204a7817b00038e5a3f3545e1c205a2dc32de6889927ffe7245"
OLD_SIZE = 563030
NEW_SHA = "f6d8ca05a33907489138b37b2089151d78a84a05e5c73020147a282a29383361"
NEW_SIZE = 563105
SOURCE_FIELDS = (
    "recovery_source_commit", "recovery_source_sha256", "recovery_source_size_bytes",
)


@pytest.fixture()
def connection():
    generator = database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)
    psycopg2, dsn = next(generator)
    database = psycopg2.connect(**dsn)
    try:
        _restore(database)
        with database.cursor() as cursor:
            cursor.execute(RECOVERY278.read_text())
            cursor.execute(RECOVERY279.read_text())
        database.commit()
        yield database
    finally:
        database.close()
        generator.close()


def authority_row(cursor) -> dict:
    cursor.execute(
        "SELECT to_jsonb(a) FROM " + AUTHORITY + " a WHERE round_id=%s",
        (ROUND,),
    )
    return cursor.fetchone()[0]


def state_seals(connection) -> dict:
    """Hash protected fixture state so assertion output cannot expose its rows."""
    result = {}
    with connection.cursor() as cursor:
        for table, order in (
            ("lab_arena_rounds", "round_id"),
            ("lab_arena_submissions", "submission_id"),
            ("lab_arena_runs", "run_id"),
            ("lab_arena_ledger", "entry_id"),
            ("lab_arena_sep17_baseline_recovery278_audit", "round_id"),
            ("lab_arena_sep17_baseline_recovery278_authority", "round_id"),
        ):
            cursor.execute(
                "SELECT encode(extensions.digest(COALESCE(jsonb_agg(to_jsonb(t) ORDER BY "
                + order + ")::text,'[]'),'sha256'),'hex') FROM public." + table + " t"
            )
            result[table] = cursor.fetchone()[0]
        cursor.execute(
            "SELECT encode(extensions.digest(COALESCE(jsonb_agg(jsonb_build_array("
            "conname,pg_get_constraintdef(oid)) ORDER BY conname)::text,'[]'),"
            "'sha256'),'hex') FROM pg_constraint WHERE conrelid=%s::regclass",
            (AUTHORITY,),
        )
        result["authority_constraints"] = cursor.fetchone()[0]
    return result


def apply(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    connection.commit()


def prepare(cursor, *, digest: str, size: int) -> dict:
    cursor.execute(
        "SELECT public.lab_arena_prepare_sep17_baseline_recovery278_v1("
        "%s,%s,%s,%s,%s::jsonb)",
        (size, digest, COMMIT, BANK_SHA, json.dumps(SCHEDULE)),
    )
    return cursor.fetchone()[0]


def test_archive_only_update_prepare_and_active_exact_replay(connection):
    before = state_seals(connection)
    with connection.cursor() as cursor:
        authority_before = authority_row(cursor)

    apply(connection)
    updated = state_seals(connection)
    with connection.cursor() as cursor:
        authority_after = authority_row(cursor)
    assert {
        key: value for key, value in before.items()
        if key not in {"authority_constraints", "lab_arena_sep17_baseline_recovery278_authority"}
    } == {
        key: value for key, value in updated.items()
        if key not in {"authority_constraints", "lab_arena_sep17_baseline_recovery278_authority"}
    }
    assert tuple(authority_after[field] for field in SOURCE_FIELDS) == (
        COMMIT, NEW_SHA, NEW_SIZE,
    )
    assert all(
        authority_before[key] == authority_after[key]
        for key in authority_before
        if key not in {"recovery_source_sha256", "recovery_source_size_bytes"}
    )

    with connection.cursor() as cursor:
        result = prepare(cursor, digest=NEW_SHA, size=NEW_SIZE)
        assert result["status"] == "prepared"
        cursor.execute(
            "SELECT submission_doc->>'source_commit',submission_doc->>'source_sha256',"
            "source_size_bytes FROM public.lab_arena_submissions WHERE submission_id=%s",
            ("baseline-2026-09-17",),
        )
        assert cursor.fetchone() == (COMMIT, NEW_SHA, NEW_SIZE)
        cursor.execute(
            "SELECT count(*),count(DISTINCT assignment_id) FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='execute' AND assignment_id LIKE '%%:rerun278'",
            (ROUND,),
        )
        assert cursor.fetchone() == (20, 20)
    connection.commit()
    active = state_seals(connection)

    apply(connection)
    assert state_seals(connection) == active


def test_old_archive_already_activated_refuses_without_changes(connection):
    with connection.cursor() as cursor:
        assert prepare(cursor, digest=OLD_SHA, size=OLD_SIZE)["status"] == "prepared"
    connection.commit()
    before = state_seals(connection)
    with pytest.raises(Exception, match="exact unused recovery278"):
        apply(connection)
    connection.rollback()
    assert state_seals(connection) == before


@pytest.mark.parametrize(
    "defect", ["authority_drift", "terminal_drift", "source_constraint_drift"],
)
def test_archive_update_drift_refusal_rolls_back(connection, defect):
    with connection.cursor() as cursor:
        if defect == "authority_drift":
            cursor.execute(
                "ALTER TABLE " + AUTHORITY
                + " DROP CONSTRAINT recovery279_exact_reviewed_source"
            )
            cursor.execute(
                "UPDATE " + AUTHORITY + " SET recovery_source_sha256=%s",
                ("c" * 64,),
            )
        elif defect == "terminal_drift":
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET cancel_reason="
                "'fixture-terminal-drift' WHERE round_id=%s",
                (ROUND,),
            )
            cursor.execute("SET session_replication_role=origin")
        else:
            cursor.execute(
                "ALTER TABLE " + AUTHORITY
                + " DROP CONSTRAINT recovery279_exact_reviewed_source"
            )
            cursor.execute(
                "ALTER TABLE " + AUTHORITY
                + " ADD CONSTRAINT recovery279_exact_reviewed_source "
                "CHECK (recovery_source_size_bytes > 0)"
            )
    connection.commit()
    before = state_seals(connection)
    reason = (
        "recovery279 source constraint differs"
        if defect == "source_constraint_drift"
        else "exact unused recovery278"
    )
    with pytest.raises(Exception, match=reason):
        apply(connection)
    connection.rollback()
    assert state_seals(connection) == before
