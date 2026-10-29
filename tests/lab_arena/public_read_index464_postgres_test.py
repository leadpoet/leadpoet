"""Exercise the public-read index against real toasted round documents."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import public_dashboard, service as service_module, store as store_module
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.competition_publication443_postgres_test import _insert_rows, _security_state
from tests.lab_arena.competition_summary_projection_test import _row
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, current_service_database, database_with_lab_arena_migration


MIGRATION = Path(__file__).resolve().parents[2] / "scripts/464-lab-arena-public-read-index.sql"
INDEX = "lab_arena_rounds_chain_mode_created_idx"
# Migration 501 adds company_judgment_policy to this projection. These exact
# migration-464 checks must keep the original public field contract.
PUBLIC_CONFIGURATION_FIELDS_464 = (
    'mode', 'network_name', 'netuid', 'schedule', 'benchmark_disclosure_policy',
    'stage_1_icp_count', 'stage_2_icp_count', 'promotion_margin',
    'execution_sequence_policy', 'sourcing_cost_eligibility_policy',
    'recovery_source_round_id', 'integrity_policy', 'contact_policy',
    'company_quality_policy', 'intent_details_policy', 'baseline_hotkey',
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        tuple(name for name in CURRENT_SERVICE_MIGRATIONS if name != MIGRATION.name)
        + ("443-lab-arena-competition-publication-projection.sql",)
    )


@pytest.fixture
def current_database_without_index():
    for database in current_service_database():
        with database[0].connect(**database[1]) as connection,connection.cursor() as cursor:
            cursor.execute('DROP INDEX public.' + INDEX)
        yield database


def _plan(cursor, sql):
    cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)
    return cursor.fetchone()[0][0]


def _nodes(plan):
    yield plan
    for child in plan.get("Plans", []):
        yield from _nodes(child)


def test_index_bounds_large_round_reads_preserves_responses_and_security(current_database_without_index, monkeypatch):
    pg, dsn = current_database_without_index
    connection = pg.connect(**dsn)
    connection.autocommit = True
    transport = PsycopgTransport(lambda: pg.connect(**dsn))
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime(2026, 10, 14, tzinfo=timezone.utc)
    service._store = ArenaStore(transport)
    service._clock = lambda: datetime(2026, 10, 13, tzinfo=timezone.utc)
    service._config = SimpleNamespace(mode="live", pinned_round_id=None,
        chain=SimpleNamespace(current_settlement_epoch=lambda: 100))
    # Deliberately difficult to compress: public reads must not walk shadow
    # catalogs simply because they share the live round's chain scope.
    catalog = "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(1024))
    rows = []
    for index in range(480):
        day = (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).date().isoformat()
        row = _row(day, mode="live" if index % 12 == 0 else "shadow")
        row["configuration_doc"]["deepline_catalog"] = {"private": catalog}
        rows.append(row)
    # A second chain must not leak into either public response.
    other = _row("2028-01-01", netuid=72)
    rows.append(other)
    try:
        _insert_rows(connection, rows)
        with connection.cursor() as cursor:
            cursor.execute("ANALYZE public.lab_arena_rounds")
            security_before = _security_state(cursor)
            cursor.execute("SELECT md5(string_agg(to_jsonb(r)::text, '' ORDER BY round_id)) FROM public.lab_arena_rounds r")
            data_before = cursor.fetchone()
            query = "SELECT round_id FROM public.lab_arena_rounds WHERE arena_network_name='finney' AND arena_netuid=71 AND configuration_doc->>'mode'='live' ORDER BY created_at DESC LIMIT 30"
            cursor.execute("SET ROLE lab_arena_service")
            before = _plan(cursor, query)
            cursor.execute(query)
            ids_before = cursor.fetchall()
            cursor.execute("RESET ROLE")
        with monkeypatch.context() as old:
            old.setattr(public_dashboard, "_COMPETITION_ROUND_COLUMNS", public_dashboard._ROUND_COLUMNS)
            old.setattr(service_module, "PUBLIC_CONFIGURATION_COLUMN", "configuration_doc")
            old.setattr(store_module, "PUBLIC_CONFIGURATION_COLUMN", "configuration_doc")
            competition_before = service.public_competition()
            current_before = service.public_current()
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
            # Indexes do not alter table/schema privileges or row policies.
            security_after = _security_state(cursor)
            assert security_after[0] == security_before[0]
            assert [r for r in security_after[1] if r[0] != INDEX] == security_before[1]
            assert security_after[2] == security_before[2]
            cursor.execute("SELECT md5(string_agg(to_jsonb(r)::text, '' ORDER BY round_id)) FROM public.lab_arena_rounds r")
            assert cursor.fetchone() == data_before
            cursor.execute("SET ROLE lab_arena_service")
            after = _plan(cursor, query)
            cursor.execute(query)
            assert cursor.fetchall() == ids_before
            assert any(node.get("Index Name") == INDEX for node in _nodes(after["Plan"]))
            assert after["Plan"]["Shared Hit Blocks"] < before["Plan"]["Shared Hit Blocks"] / 2
            # Repeat a PostgREST-shaped prepared read beyond PostgreSQL's first
            # five custom plans, including its parameterized JSON path key.
            cursor.execute("PREPARE public_round_page(text,text,text,bigint) AS SELECT round_id FROM public.lab_arena_rounds WHERE configuration_doc->>$1=$2 AND arena_network_name=$3 AND arena_netuid=$4 ORDER BY created_at DESC LIMIT 30")
            for _ in range(12):
                cursor.execute("EXECUTE public_round_page('mode','live','finney',71)")
                assert cursor.fetchall() == ids_before
            prepared = _plan(cursor, "EXECUTE public_round_page('mode','live','finney',71)")
            assert any(node.get("Index Name") == INDEX for node in _nodes(prepared["Plan"]))
            cursor.execute("RESET ROLE")
        assert service.public_competition() == competition_before
        assert service.public_current() == current_before
        print(json.dumps({"before_ms": before["Execution Time"], "after_ms": after["Execution Time"],
            "before_buffers": before["Plan"]["Shared Hit Blocks"], "after_buffers": after["Plan"]["Shared Hit Blocks"]}))
    finally:
        transport.close()
        connection.close()


def test_compact_configuration_preserves_types_missing_null_and_permissions(database):
    pg, dsn = database
    connection = pg.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            for role in ("anon", "authenticated", "service_role"):
                cursor.execute("SELECT has_function_privilege(%s, 'public.lab_arena_public_configuration_v1(public.lab_arena_rounds)', 'EXECUTE')", (role,))
                assert cursor.fetchone() == (False,)
            cursor.execute("SELECT prosecdef, provolatile FROM pg_proc WHERE oid='public.lab_arena_public_configuration_v1(public.lab_arena_rounds)'::regprocedure")
            assert cursor.fetchone() == (False, "i")
            cursor.execute("SET ROLE lab_arena_service")
            for configuration in ({}, {"mode": None, "schedule": {}, "stage_1_icp_count": 0},
                    {key: {"value": [None, False, 0, ""]} for key in store_module.PUBLIC_CONFIGURATION_FIELDS},
                    {"mode": "live", "deepline_catalog": {"private": True}, "scorer_policy": {"private": True}},
                    None, [], "legacy"):
                cursor.execute("SELECT public.lab_arena_public_configuration_v1(jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb))", (json.dumps({"configuration_doc": configuration}),))
                actual = cursor.fetchone()[0]
                assert actual == ({key: value for key, value in configuration.items()
                    if key in PUBLIC_CONFIGURATION_FIELDS_464} if isinstance(configuration, dict) else {})
    finally:
        connection.close()
