"""Exercise SOURCE_ADD claim/pause serialization in disposable PostgreSQL."""

from __future__ import annotations

import hashlib
import threading

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    SOURCE_ADD_CLAIM_CONTROL_FUNCTION_AUTHORITY_SHA256,
)
from tests.test_source_add_duplicate_privacy_postgres import (
    MIGRATIONS as PRE_CLAIM_CONTROL_MIGRATIONS,
)
from tests.test_source_add_end_to_end_postgres import _database_with_migrations


MIGRATION = "172-research-lab-source-add-claim-control.sql"
MIGRATIONS = PRE_CLAIM_CONTROL_MIGRATIONS + (MIGRATION,)
GUARD_A = "source_add_restart_guard:" + "a" * 64
GUARD_B = "source_add_restart_guard:" + "b" * 64
GUARD_C = "source_add_restart_guard:" + "c" * 64


def _guard_commitment(guard_id: str) -> str:
    return "sha256:" + hashlib.sha256(guard_id.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def database():
    yield from _database_with_migrations(MIGRATIONS)


@pytest.fixture(scope="module")
def pre_claim_control_database():
    yield from _database_with_migrations(PRE_CLAIM_CONTROL_MIGRATIONS)


def _pause(cursor, paused: bool, suffix: str):
    cursor.execute(
        "SELECT public.research_lab_source_add_set_paused(%s, %s, %s)",
        (
            paused,
            f"claim control postgres {suffix}",
            "operator:claim-control-postgres",
        ),
    )
    return cursor.fetchone()[0]


def _acquire_guard(cursor, guard_id: str, suffix: str):
    cursor.execute(
        """
        SELECT public.research_lab_source_add_acquire_restart_guard_v1(
            %s, 300, %s
        )
        """,
        (guard_id, f"operator:claim-control-{suffix}"),
    )
    return cursor.fetchone()[0]


def _quiescence(cursor, guard_id: str):
    cursor.execute(
        "SELECT public.research_lab_source_add_restart_quiescence_v1(%s)",
        (guard_id,),
    )
    return cursor.fetchone()[0]


def _release_guard(cursor, guard_id: str, suffix: str):
    cursor.execute(
        """
        SELECT public.research_lab_source_add_release_restart_guard_v1(
            %s, %s
        )
        """,
        (guard_id, f"operator:claim-control-{suffix}"),
    )
    return cursor.fetchone()[0]


def _insert_work(cursor, *, suffix: str, status: str = "queued") -> None:
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_work_items (
            work_id, submission_id, adapter_id, work_kind, work_status,
            lease_token, leased_by, lease_expires_at, job_doc
        ) VALUES (
            %s, %s, %s, 'provenance', %s,
            CASE WHEN %s = 'leased' THEN gen_random_uuid() ELSE NULL END,
            CASE WHEN %s = 'leased' THEN 'worker:expired' ELSE '' END,
            CASE WHEN %s = 'leased'
                 THEN NOW() - INTERVAL '1 hour' ELSE NULL END,
            '{}'::JSONB
        )
        """,
        (
            f"source_add_work:{suffix}",
            f"source_add_submission:{suffix}",
            f"adapter:claim-control-{suffix}",
            status,
            status,
            status,
            status,
        ),
    )


def test_migration_rejects_active_or_any_leased_handoff(
    pre_claim_control_database,
) -> None:
    psycopg2, dsn = pre_claim_control_database
    migration_sql = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / MIGRATION
    ).read_text(encoding="utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _pause(cursor, False, "active migration rejection")
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD must be paused before claim-control migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")

            _pause(cursor, True, "leased migration rejection")
            _insert_work(cursor, suffix="7200000000000001", status="leased")
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD work is leased during claim-control migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")
            cursor.execute(
                "DELETE FROM public.research_lab_source_add_work_items "
                "WHERE work_id = %s",
                ("source_add_work:7200000000000001",),
            )
    finally:
        connection.close()


def test_contract_quiescence_acl_and_migration_idempotency(database) -> None:
    psycopg2, dsn = database
    migration_sql = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / MIGRATION
    ).read_text(encoding="utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.research_lab_source_add_claim_control_contract_v1()"
            )
            initial_contract = cursor.fetchone()[0]
            assert initial_contract["function_authority_sha256"] == (
                SOURCE_ADD_CLAIM_CONTROL_FUNCTION_AUTHORITY_SHA256
            )
            assert initial_contract["lock_before_paused_read"] is True
            assert initial_contract["leased_scope"] == (
                "all_leased_regardless_of_expiry"
            )
            assert all(initial_contract["functions"].values())
            assert initial_contract["permissions"] == {
                "service_role_exists": True,
                "acquire_guard_service_role_callable": True,
                "claim_service_role_callable": True,
                "pause_service_role_callable": True,
                "quiescence_service_role_callable": True,
                "release_guard_service_role_callable": True,
                "contract_service_role_callable": True,
                "anon_callable": False,
                "authenticated_callable": False,
            }
            acquired = _acquire_guard(cursor, GUARD_A, "contract")
            assert set(acquired) == {
                "schema_version",
                "paused",
                "guard_active",
                "guard_commitment",
                "guard_expires_at",
            }
            assert acquired["schema_version"] == (
                "leadpoet.source_add_restart_guard.v1"
            )
            assert acquired["paused"] is True
            assert acquired["guard_active"] is True
            assert acquired["guard_commitment"] == _guard_commitment(GUARD_A)

            quiescence = _quiescence(cursor, GUARD_A)
            assert set(quiescence) == {
                "schema_version",
                "paused",
                "guard_active",
                "guard_matches",
                "guard_commitment",
                "guard_expires_at",
                "leased_work_count",
                "quiescent",
            }
            assert quiescence["paused"] is True
            assert quiescence["guard_active"] is True
            assert quiescence["guard_matches"] is True
            assert quiescence["leased_work_count"] == 0
            assert quiescence["quiescent"] is True

            cursor.execute(migration_sql)
            cursor.execute(
                "SELECT public.research_lab_source_add_claim_control_contract_v1()"
            )
            assert cursor.fetchone()[0] == initial_contract
            released = _release_guard(cursor, GUARD_A, "contract")
            assert released == {
                "schema_version": (
                    "leadpoet.source_add_restart_guard_release.v1"
                ),
                "released": True,
                "paused": True,
                "guard_active": False,
            }
            cursor.execute(
                """
                SELECT paused, restart_guard_commitment,
                       restart_guard_expires_at
                FROM public.research_lab_source_add_control
                WHERE singleton
                """
            )
            assert cursor.fetchone() == (True, "", None)
    finally:
        connection.close()


def test_claim_then_pause_exposes_lease_until_worker_drains(database) -> None:
    psycopg2, dsn = database
    setup = psycopg2.connect(**dsn)
    setup.autocommit = True
    with setup.cursor() as cursor:
        _insert_work(cursor, suffix="7200000000000002")
        _pause(cursor, False, "claim before pause")
    setup.close()

    claim_connection = psycopg2.connect(**dsn)
    claim_connection.autocommit = False
    with claim_connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout = '5s'")
        cursor.execute(
            "SELECT public.research_lab_source_add_claim_work(%s, %s)",
            ("worker:claim-before-pause", 300),
        )
        claimed = cursor.fetchone()[0]
    assert claimed["status"] == "claimed"

    started = threading.Event()
    finished = threading.Event()
    outcomes: list[dict] = []
    errors: list[BaseException] = []

    def acquire_guard() -> None:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET statement_timeout = '5s'")
                started.set()
                outcomes.append(
                    _acquire_guard(cursor, GUARD_B, "after-active-claim")
                )
        except BaseException as exc:  # surfaced in the parent test thread
            errors.append(exc)
        finally:
            finished.set()
            connection.close()

    thread = threading.Thread(target=acquire_guard)
    thread.start()
    assert started.wait(timeout=2)
    assert not finished.wait(timeout=0.25)
    claim_connection.commit()
    claim_connection.close()
    thread.join(timeout=6)
    assert not thread.is_alive()
    assert errors == []
    assert outcomes[0]["guard_active"] is True

    verify = psycopg2.connect(**dsn)
    verify.autocommit = True
    try:
        with verify.cursor() as cursor:
            quiescence = _quiescence(cursor, GUARD_B)
            assert quiescence["paused"] is True
            assert quiescence["guard_active"] is True
            assert quiescence["guard_matches"] is True
            assert quiescence["leased_work_count"] == 1
            assert quiescence["quiescent"] is False
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_work_items
                SET work_status = 'completed', completed_at = NOW(),
                    lease_token = NULL, leased_by = '', lease_expires_at = NULL
                WHERE work_id = %s
                """,
                ("source_add_work:7200000000000002",),
            )
            assert _quiescence(cursor, GUARD_B)["quiescent"] is True
            assert _release_guard(
                cursor, GUARD_B, "after-active-claim"
            )["paused"] is True
    finally:
        verify.close()


def test_guard_commit_rejects_waiting_resume_and_prevents_claim(database) -> None:
    psycopg2, dsn = database
    setup = psycopg2.connect(**dsn)
    setup.autocommit = True
    with setup.cursor() as cursor:
        _insert_work(cursor, suffix="7200000000000003")
        _pause(cursor, False, "prepare pause before claim")
    setup.close()

    pause_connection = psycopg2.connect(**dsn)
    pause_connection.autocommit = False
    with pause_connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout = '5s'")
        acquired = _acquire_guard(cursor, GUARD_C, "before-waiting-resume")
    assert acquired["guard_active"] is True

    started = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    def resume() -> None:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET statement_timeout = '5s'")
                started.set()
                _pause(cursor, False, "concurrent guarded resume")
        except BaseException as exc:  # expected guard rejection, asserted below
            errors.append(exc)
        finally:
            finished.set()
            connection.close()

    thread = threading.Thread(target=resume)
    thread.start()
    assert started.wait(timeout=2)
    assert not finished.wait(timeout=0.25)
    pause_connection.commit()
    pause_connection.close()
    thread.join(timeout=6)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert "must be explicitly reacquired and released" in str(errors[0])

    verify = psycopg2.connect(**dsn)
    verify.autocommit = True
    try:
        with verify.cursor() as cursor:
            cursor.execute(
                "SELECT work_status FROM public.research_lab_source_add_work_items "
                "WHERE work_id = %s",
                ("source_add_work:7200000000000003",),
            )
            assert cursor.fetchone()[0] == "queued"
            cursor.execute(
                "SELECT public.research_lab_source_add_claim_work(%s, %s)",
                ("worker:after-guard", 300),
            )
            assert cursor.fetchone()[0] == {"status": "paused"}
            assert _quiescence(cursor, GUARD_C)["quiescent"] is True
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD restart guard identity does not match",
            ):
                _release_guard(cursor, GUARD_A, "wrong-release")
            released = _release_guard(cursor, GUARD_C, "exact-release")
            assert released["paused"] is True
            assert _pause(cursor, False, "resume after exact release")[
                "paused"
            ] is False
            _pause(cursor, True, "cleanup after exact release")
    finally:
        verify.close()


def test_expired_guard_requires_explicit_reacquire_and_exact_release(
    database,
) -> None:
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _acquire_guard(cursor, GUARD_A, "expiry-original")
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_control
                SET restart_guard_expires_at = NOW() - INTERVAL '1 second'
                WHERE singleton
                """
            )
            expired = _quiescence(cursor, GUARD_A)
            assert expired["guard_active"] is False
            assert expired["guard_matches"] is True
            assert expired["quiescent"] is False
            with pytest.raises(
                psycopg2.Error,
                match="must be explicitly reacquired and released",
            ):
                _pause(cursor, False, "expired guard cannot auto resume")

            reacquired = _acquire_guard(cursor, GUARD_B, "expiry-recovery")
            assert reacquired["guard_active"] is True
            assert reacquired["guard_commitment"] == _guard_commitment(GUARD_B)
            assert _quiescence(cursor, GUARD_A)["guard_matches"] is False
            assert _quiescence(cursor, GUARD_B)["quiescent"] is True
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD restart guard identity does not match",
            ):
                _release_guard(cursor, GUARD_A, "stale-expired-owner")
            assert _release_guard(
                cursor, GUARD_B, "expiry-recovery"
            ) == {
                "schema_version": (
                    "leadpoet.source_add_restart_guard_release.v1"
                ),
                "released": True,
                "paused": True,
                "guard_active": False,
            }
    finally:
        connection.close()
