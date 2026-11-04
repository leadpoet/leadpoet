"""Empty legacy contracts retain source proof without exposing private routes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from lab_arena import public_dashboard as dashboard
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.competition_publication443_postgres_test import _insert_rows, _security_state
from tests.lab_arena.competition_summary_projection_test import _row, _service
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.test_lab_arena_postgrest_route import DOCKER, make_transport, stack


MIGRATION = Path(__file__).parents[2] / "scripts/601-lab-arena-defending-champion-proof.sql"
SIGNATURE = "public.lab_arena_public_configuration_v1(public.lab_arena_rounds)"
MARKER = "baseline_champion_source_frozen"
MISSING = object()
CONTRACTS = [MISSING, None, False, 1, "legacy", [], {},
             {"reasoning_capture_policy": "full_generated_thinking_v1",
              "reasoning_route_registry": {"private_route": "must stay private"}}]


def _rows():
    rows = []
    for index, contract in enumerate(CONTRACTS):
        row = _row("2026-09-%02d" % (index + 1))
        if contract is MISSING:
            del row["configuration_doc"]["baseline_reasoning_configuration"]
        else:
            row["configuration_doc"]["baseline_reasoning_configuration"] = contract
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def database():
    database = current_service_database()
    try:
        value = next(database)
        with value[0].connect(**value[1]) as connection:
            connection.autocommit = True
            _insert_rows(connection, _rows())
        yield value
    finally:
        database.close()


def _project(cursor, row):
    cursor.execute("SELECT public.lab_arena_public_configuration_v1("
                   "jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb))",
                   (json.dumps(row),))
    return cursor.fetchone()[0]


def test_migration_preserves_safe_fields_rows_and_security(database):
    pg, dsn = database
    rows = _rows()
    with pg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            security = _security_state(cursor)
            cursor.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) FROM public.lab_arena_rounds r")
            saved_rows = cursor.fetchone()[0]
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            original = cursor.fetchone()[0]
            before = [_project(cursor, row) for row in rows]
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
            assert _security_state(cursor) == security
            cursor.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) FROM public.lab_arena_rounds r")
            assert cursor.fetchone()[0] == saved_rows
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            amended = cursor.fetchone()[0]
            # The exact reader edit leaves its existing safe-key whitelist intact.
            assert amended[amended.index("  FROM pg_catalog.jsonb_each"):] == original[original.index("  FROM pg_catalog.jsonb_each"):]
            cursor.execute("SELECT pg_get_userbyid(proowner),proacl::text,prosecdef,provolatile,proconfig "
                           "FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURE,))
            assert cursor.fetchone() == (
                "lab_arena_owner", "{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",
                False, "i", ["search_path=pg_catalog, public"])
            for role in ("anon", "authenticated", "service_role"):
                cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, SIGNATURE))
                assert cursor.fetchone() == (False,)
            cursor.execute("SET ROLE lab_arena_service")
            for row, previous in zip(rows, before):
                compact = _project(cursor, row)
                assert compact.pop(MARKER) is isinstance(row["configuration_doc"].get("baseline_reasoning_configuration"), dict)
                assert compact == previous
            cursor.execute("RESET ROLE")
            # Fail closed on an unknown predecessor, then restore the reader.
            cursor.execute(amended.replace("  SELECT COALESCE", "  SELECT /* unknown */ COALESCE", 1))
            try:
                with pytest.raises(pg.Error, match="preimage differs"):
                    cursor.execute(MIGRATION.read_text())
                cursor.execute("ROLLBACK")
            finally:
                cursor.execute(amended)


def test_real_compact_and_full_summaries_match_for_all_source_proofs(database, monkeypatch):
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    transport = PsycopgTransport(lambda: pg.connect(**dsn))
    store = ArenaStore(transport)
    try:
        rows = store.list_rounds(limit=30)
        projected = store.list_rounds(limit=30, columns=dashboard._COMPETITION_ROUND_COLUMNS)
        assert "private_route" not in json.dumps(projected)
        assert "reasoning_route_registry" not in json.dumps(projected)
        for full, compact in zip(rows, projected):
            summary = dashboard.round_summary(full)
            assert dashboard.round_summary(compact) == summary
            contract = full["configuration_doc"].get("baseline_reasoning_configuration")
            expected = ({"submission_id": "prior-champion", "miner_hotkey": "5Prior"}
                        if isinstance(contract, dict) else None)
            assert summary["defending_champion"] == expected
        service, _ = _service(rows)
        service._store = store
        columns = dashboard._COMPETITION_ROUND_COLUMNS
        monkeypatch.setattr(dashboard, "_COMPETITION_ROUND_COLUMNS", dashboard._ROUND_COLUMNS)
        expected = service.public_competition()
        monkeypatch.setattr(dashboard, "_COMPETITION_ROUND_COLUMNS", columns)
        assert service.public_competition() == expected
        with pg.connect(**dsn) as connection, connection.cursor() as cursor:
            for funding in (True, False, None):
                for contract in CONTRACTS:
                    row = deepcopy(rows[0])
                    row["champion_funding_frozen"] = funding
                    if contract is MISSING:
                        row["configuration_doc"].pop("baseline_reasoning_configuration", None)
                    else:
                        row["configuration_doc"]["baseline_reasoning_configuration"] = contract
                    compact = {**row, "configuration_doc": _project(cursor, row)}
                    assert compact["configuration_doc"][MARKER] is (funding is True and isinstance(contract, dict))
                    assert dashboard.round_summary(compact) == dashboard.round_summary(row)
    finally:
        transport.close()


@pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")
def test_postgrest_preserves_empty_contract_proof_and_excludes_private_routes(stack):
    with stack["connection"].cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    _insert_rows(stack["connection"], _rows())
    transport = make_transport(stack, "lab_arena_service")
    store = ArenaStore(transport)
    try:
        full = store.list_rounds(limit=30, columns=dashboard._ROUND_COLUMNS)
        compact = store.list_rounds(limit=30, columns=dashboard._COMPETITION_ROUND_COLUMNS)
        assert [dashboard.round_summary(row) for row in compact] == [
            dashboard.round_summary(row) for row in full]
        by_id = {row["round_id"]: dashboard.round_summary(row) for row in compact}
        assert by_id["arena-2026-09-07"]["defending_champion"] == {
            "submission_id": "prior-champion", "miner_hotkey": "5Prior"}
        assert by_id["arena-2026-09-01"]["defending_champion"] is None
        assert by_id["arena-2026-09-02"]["defending_champion"] is None
        assert "reasoning_route_registry" not in json.dumps(compact)
        assert "private_route" not in json.dumps(compact)
    finally:
        transport.close()
