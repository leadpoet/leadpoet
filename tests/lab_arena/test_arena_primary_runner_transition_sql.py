from __future__ import annotations

from pathlib import Path

import pytest
from psycopg2.extras import Json

from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "204-arena-primary-runner-transition.sql"
ROUND_ID = "arena-2026-09-11"
OLD = "5GsGcRyR4kWCcsa1qEAwxtbDq34ZwkQt3rHAGniPFjv1JoXW"
PRIMARY = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
LIVE_HASH = "0001ab8b055e176aff4d9e98ac6a9221f91328fe89208369afad38018614b8af"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration()


def _migration_for_fixture(cursor) -> str:
    cursor.execute(
        """
        SELECT encode(extensions.digest(convert_to(configuration_doc::text, 'UTF8'), 'sha256'), 'hex')
        FROM public.lab_arena_rounds WHERE round_id = %s
        """,
        (ROUND_ID,),
    )
    fixture_hash = cursor.fetchone()[0]
    return MIGRATION.read_text(encoding="utf-8").replace(LIVE_HASH, fixture_hash)


def _reset_round(cursor, config, *, status="open"):
    cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once")
    cursor.execute("DELETE FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND_ID,))
    cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once")
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds(round_id, status, configuration_doc) "
        "VALUES (%s, %s, %s::jsonb)",
        (ROUND_ID, status, Json(config)),
    )


def test_exact_open_round_transition_is_narrow_and_idempotent(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    config = {
        "runner_hotkeys": [OLD],
        "banned_hotkeys": [],
        "schedule": {"submission_cutoff": "2099-09-11T00:00:00Z"},
        "unchanged": {"scorer": "sha256:" + "a" * 64, "pool_percent": 15},
    }
    with connection.cursor() as cursor:
        _reset_round(cursor, config)
        cursor.execute("SELECT updated_at FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND_ID,))
        before_updated_at = cursor.fetchone()[0]
        sql = _migration_for_fixture(cursor)
        cursor.execute(sql)
        cursor.execute(sql)
        cursor.execute(
            "SELECT configuration_doc, updated_at FROM public.lab_arena_rounds WHERE round_id = %s",
            (ROUND_ID,),
        )
        document, updated_at = cursor.fetchone()
        assert document["runner_hotkeys"] == [OLD, PRIMARY]
        assert document["unchanged"] == config["unchanged"]
        assert {**document, "runner_hotkeys": [OLD]} == config
        assert updated_at >= before_updated_at
        cursor.execute(
            "SELECT tgenabled FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass "
            "AND tgname='lab_arena_rounds_write_once'"
        )
        assert cursor.fetchone()[0] == "O"
        with pytest.raises(psycopg2.Error, match="round configuration is write-once"):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc="
                "jsonb_set(configuration_doc, '{runner_hotkeys}', '[]'::jsonb) WHERE round_id=%s",
                (ROUND_ID,),
            )
        cursor.execute("ROLLBACK")
    connection.close()


def test_transition_rejects_changed_configuration_and_restores_trigger(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        config = {
            "runner_hotkeys": [OLD, PRIMARY],
            "banned_hotkeys": [],
            "schedule": {"submission_cutoff": "2099-09-11T00:00:00Z"},
            "unchanged": {"pool_percent": 16},
        }
        _reset_round(cursor, config)
        sql = MIGRATION.read_text(encoding="utf-8")
        with pytest.raises(psycopg2.Error, match="post-state differs"):
            cursor.execute(sql)
        cursor.execute("ROLLBACK")
        cursor.execute(
            "SELECT tgenabled FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass "
            "AND tgname='lab_arena_rounds_write_once'"
        )
        assert cursor.fetchone()[0] == "O"
    connection.close()


@pytest.mark.parametrize(
    ("status", "cutoff", "banned", "message"),
    [
        ("committed", "2099-09-11T00:00:00Z", [], "round has started"),
        ("open", "2020-09-11T00:00:00Z", [], "submission cutoff passed"),
        ("open", "2099-09-11T00:00:00Z", [PRIMARY], "runner is banned"),
    ],
)
def test_transition_rejects_unsafe_round_state(database, status, cutoff, banned, message):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    config = {
        "runner_hotkeys": [OLD],
        "banned_hotkeys": banned,
        "schedule": {"submission_cutoff": cutoff},
        "unchanged": {"pool_percent": 15},
    }
    with connection.cursor() as cursor:
        _reset_round(cursor, config, status=status)
        sql = _migration_for_fixture(cursor)
        with pytest.raises(psycopg2.Error, match=message):
            cursor.execute(sql)
        cursor.execute("ROLLBACK")
        cursor.execute(
            "SELECT configuration_doc->'runner_hotkeys', tgenabled "
            "FROM public.lab_arena_rounds, pg_trigger "
            "WHERE round_id=%s AND tgrelid='public.lab_arena_rounds'::regclass "
            "AND tgname='lab_arena_rounds_write_once'",
            (ROUND_ID,),
        )
        runners, enabled = cursor.fetchone()
        assert runners == [OLD]
        assert enabled == "O"
    connection.close()
