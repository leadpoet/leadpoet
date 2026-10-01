"""The canonical restart guard preserves an independent operator hold."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_restart_claim_drain_postgres import (
    GUARD, OWNER, _acquire, _rpc,
)


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _reason(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT operator_paused,pause_reason,guard_commitment "
            "FROM public.lab_arena_restart_claim_control WHERE singleton"
        )
        return cursor.fetchone()


def _release(connection):
    _rpc(connection, "lab_arena_authorize_restart_phase_v1", GUARD, OWNER, 1, "gateway_destructive")
    _rpc(connection, "lab_arena_mark_restart_ready_v1", GUARD, OWNER, 1, "gateway_ready")
    return _rpc(connection, "lab_arena_release_restart_guard_v1", GUARD, OWNER, 1, "test-release")


def test_existing_operator_reason_survives_restart_acquire_and_release(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_restart_claim_control "
                "SET operator_paused=TRUE,pause_reason='oct01_deepline_outage',"
                "actor_ref='oct01-provider-recovery369' WHERE singleton"
            )
        assert _reason(connection) == (True, "oct01_deepline_outage", "")
        assert _acquire(connection, scope="gateway")["guard_present"] is True
        paused, reason, commitment = _reason(connection)
        assert paused is True and reason == "oct01_deepline_outage" and commitment
        assert _release(connection)["guard_present"] is False
        assert _reason(connection) == (True, "oct01_deepline_outage", "")


def test_ordinary_restart_reason_is_cleared_on_release(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute((Path(__file__).parents[2] /
                            "scripts/374-lab-arena-preserve-operator-pause-reason.sql").read_text())
        assert _reason(connection) == (False, "", "")
        assert _acquire(connection, scope="gateway")["guard_present"] is True
        paused, reason, commitment = _reason(connection)
        assert paused is False and reason == "canonical_restart_guard" and commitment
        assert _release(connection)["guard_present"] is False
        assert _reason(connection) == (False, "", "")
