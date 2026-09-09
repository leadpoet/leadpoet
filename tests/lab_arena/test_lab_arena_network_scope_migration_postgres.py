"""PostgreSQL proof for queryable, backwards-compatible Arena chain scope."""

from __future__ import annotations

import json

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_NETWORK_SCOPE_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.test_source_add_end_to_end_postgres import SCRIPTS


def test_network_scope_migration_is_idempotent_and_keeps_legacy_rows_on_finney():
    generator = database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS[
            : DEFAULT_MIGRATIONS.index(LAB_ARENA_NETWORK_SCOPE_MIGRATION)
        ]
    )
    psycopg2, dsn = next(generator)
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc, rewards_enabled) VALUES "
                "(%s, 'open', %s::jsonb, FALSE)",
                ("arena-2026-09-07-legacy", json.dumps({"mode": "live"})),
            )
            migration = (SCRIPTS / LAB_ARENA_NETWORK_SCOPE_MIGRATION).read_text(
                encoding="utf-8"
            )
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT arena_network_name, arena_netuid "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                ("arena-2026-09-07-legacy",),
            )
            assert cursor.fetchone() == ("finney", 71)
            cursor.execute("SELECT public.lab_arena_schema_version_v1()")
            assert cursor.fetchone()[0]["version"] == 189

            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc, rewards_enabled) VALUES "
                "(%s, 'open', %s::jsonb, TRUE)",
                (
                    "arena-2026-09-07-testnet",
                    json.dumps({
                        "mode": "live", "network_name": "test", "netuid": 401,
                    }),
                ),
            )
            cursor.execute(
                "SELECT round_id FROM public.lab_arena_rounds "
                "WHERE arena_network_name='test' AND arena_netuid=401"
            )
            assert cursor.fetchall() == [("arena-2026-09-07-testnet",)]
            with pytest.raises(psycopg2.Error):
                cursor.execute(
                    "INSERT INTO public.lab_arena_rounds "
                    "(round_id, status, configuration_doc, rewards_enabled) VALUES "
                    "(%s, 'open', %s::jsonb, TRUE)",
                    (
                        "arena-2026-09-07-unpaired",
                        json.dumps({"mode": "live", "network_name": "test"}),
                    ),
                )
    finally:
        connection.close()
        generator.close()
