"""Source-only authority update before the sealed September17 recovery starts."""
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
    ROUND,
    SCHEDULE,
    _prepare as prepare_old,
    _restore,
    historical_migration_text,
)

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "scripts/279-sep17-reviewed-source-before-activation.sql.template"
MIGRATION = TEMPLATE.with_suffix("")
AUTHORITY = "public.lab_arena_sep17_baseline_recovery278_authority"
SOURCE_FIELDS = (
    "recovery_source_commit", "recovery_source_sha256", "recovery_source_size_bytes",
)


def new_source() -> tuple[str, str, int]:
    """The committed 279 identity is independent of today's operator."""
    return (
        "4ceae936b902433432a195f77af3f94559d80378",
        "6c9eeaac41386204a7817b00038e5a3f3545e1c205a2dc32de6889927ffe7245",
        563030,
    )


def migration_text() -> str:
    commit, digest, size = new_source()
    rendered = TEMPLATE.read_text().replace("__SOURCE_COMMIT__", commit)
    rendered = rendered.replace("__SOURCE_SHA256__", digest).replace("__SOURCE_SIZE__", str(size))
    assert "__SOURCE_" not in rendered
    if MIGRATION.exists():
        assert MIGRATION.read_text() == rendered, "rendered279_template_or_operator_mismatch"
    return rendered


@pytest.fixture()
def connection():
    generator = database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)
    psycopg2, dsn = next(generator)
    database = psycopg2.connect(**dsn)
    try:
        _restore(database)
        with database.cursor() as cursor:
            cursor.execute(historical_migration_text())
        database.commit()
        yield database
    finally:
        database.close()
        generator.close()


def source_row(cursor) -> dict:
    cursor.execute("SELECT to_jsonb(a) FROM " + AUTHORITY + " a WHERE round_id=%s", (ROUND,))
    return cursor.fetchone()[0]


def state_seals(connection) -> dict:
    """Only hashes reach assertion output; never private fixture documents."""
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
            "conname,pg_get_constraintdef(oid)) ORDER BY conname)::text,'[]'),'sha256'),'hex') "
            "FROM pg_constraint WHERE conrelid=%s::regclass", (AUTHORITY,),
        )
        result["authority_constraints"] = cursor.fetchone()[0]
    return result


def apply(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(migration_text())
    connection.commit()


def drop_source_check(cursor, column) -> None:
    assert column in SOURCE_FIELDS
    cursor.execute(
        "SELECT c.conname FROM pg_constraint c JOIN pg_attribute a ON a.attrelid=c.conrelid "
        "AND c.conkey=ARRAY[a.attnum]::smallint[] WHERE c.conrelid=%s::regclass "
        "AND c.contype='c' AND a.attname=%s", (AUTHORITY, column),
    )
    names = cursor.fetchall()
    assert len(names) == 1
    cursor.execute("SELECT format('ALTER TABLE %%s DROP CONSTRAINT %%I',%s::regclass,%s)", (AUTHORITY, names[0][0]))
    cursor.execute(cursor.fetchone()[0])


def test_source_only_update_exact_replay_and_normal_prepare(connection):
    before = state_seals(connection)
    with connection.cursor() as cursor:
        authority_before = source_row(cursor)
    apply(connection)
    after = state_seals(connection)
    with connection.cursor() as cursor:
        authority_after = source_row(cursor)
    assert {key: value for key, value in before.items() if key not in {
        "authority_constraints", "lab_arena_sep17_baseline_recovery278_authority",
    }} == {key: value for key, value in after.items() if key not in {
        "authority_constraints", "lab_arena_sep17_baseline_recovery278_authority",
    }}
    assert tuple(authority_after[field] for field in SOURCE_FIELDS) == new_source()
    # Booleans avoid printing protected authority records on failure.
    assert all(authority_before[key] == authority_after[key] for key in authority_before if key not in SOURCE_FIELDS)
    with connection.cursor() as cursor:
        for field, value in zip(SOURCE_FIELDS, ("c" * 40, "d" * 64, new_source()[2] + 1)):
            cursor.execute("SAVEPOINT reject_source_drift")
            with pytest.raises(Exception, match="recovery279_exact_reviewed_source"):
                cursor.execute("UPDATE " + AUTHORITY + " SET " + field + "=%s", (value,))
            cursor.execute("ROLLBACK TO SAVEPOINT reject_source_drift")
    apply(connection)
    assert state_seals(connection) == after

    commit, digest, size = new_source()
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep17_baseline_recovery278_v1(%s,%s,%s,%s,%s::jsonb)",
            (size, digest, commit, BANK_SHA, json.dumps(SCHEDULE)),
        )
        assert cursor.fetchone()[0]["status"] == "prepared"
        cursor.execute(
            "SELECT submission_doc->>'source_commit',submission_doc->>'source_sha256',"
            "source_size_bytes FROM public.lab_arena_submissions WHERE submission_id=%s",
            ("baseline-2026-09-17",),
        )
        assert cursor.fetchone() == (commit, digest, size)
        cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute'", (ROUND,))
        assert cursor.fetchone() == (20,)
        cursor.execute("SELECT public.lab_arena_sep17_recovery278_archive_valid_v1(),public.lab_arena_sep17_recovery278_nonbaseline_ledger_valid_v1()")
        assert cursor.fetchone() == (True, True)
    connection.commit()
    prepared = state_seals(connection)
    apply(connection)  # Exact replay cannot modify a now-active recovery.
    assert state_seals(connection) == prepared


def test_old_source_already_prepared_refuses_without_changes(connection):
    with connection.cursor() as cursor:
        assert prepare_old(cursor)["status"] == "prepared"
    connection.commit()
    before = state_seals(connection)
    with pytest.raises(Exception, match="exact unused recovery278"):
        apply(connection)
    connection.rollback()
    assert state_seals(connection) == before


@pytest.mark.parametrize("defect", ["old_authority", "terminal_drift", "source_constraint_missing"])
def test_source_update_refusal_rolls_back_every_change(connection, defect):
    with connection.cursor() as cursor:
        if defect == "old_authority":
            drop_source_check(cursor, "recovery_source_commit")
            cursor.execute("UPDATE " + AUTHORITY + " SET recovery_source_commit=%s", ("c" * 40,))
        elif defect == "terminal_drift":
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET cancel_reason='fixture-terminal-drift' WHERE round_id=%s", (ROUND,))
            cursor.execute("SET session_replication_role=origin")
        else:
            drop_source_check(cursor, "recovery_source_size_bytes")
    connection.commit()
    before = state_seals(connection)
    reason = "original source constraints differ" if defect == "source_constraint_missing" else "exact unused recovery278"
    with pytest.raises(Exception, match=reason):
        apply(connection)
    connection.rollback()
    assert state_seals(connection) == before
