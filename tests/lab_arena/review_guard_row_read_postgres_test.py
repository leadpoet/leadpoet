"""The review gate waits for guard writes, not unrelated lease sweeps."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import select
import time

import pytest

from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration
from tests.postgres_migration_harness import SCRIPTS


RPC = "lab_arena_review_guard_state_v1"
MIGRATION = "574-lab-arena-review-guard-row-read.sql"
GUARD = "lab_arena_restart_guard:" + "b" * 64
OWNER = "lab_arena_restart_owner:" + "c" * 64


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(DEFAULT_MIGRATIONS + (
        MIGRATION,
    ))


def connect(database, *, service=False, transactional=False):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = not transactional
    with connection.cursor() as cursor:
        if service:
            cursor.execute("SET ROLE lab_arena_service")
        cursor.execute("SET lock_timeout='250ms'; SET statement_timeout='3s'")
    return connection


def rpc(connection, function=RPC):
    with connection.cursor() as cursor:
        cursor.execute("SELECT public.%s()" % function)
        return cursor.fetchone()[0]


def acquire(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton")
        generation = cursor.fetchone()[0]
        cursor.execute(
            "SELECT public.lab_arena_acquire_restart_guard_v1(%s,%s,%s,60,%s,'gateway','test')",
            (GUARD, OWNER, generation, "a" * 40),
        )
        return cursor.fetchone()[0]


def test_review_guard_does_not_queue_behind_claim_control_but_canonical_state_still_does(database):
    psycopg2, _ = database
    with closing(connect(database, transactional=True)) as sweep, closing(connect(database, service=True)) as reader:
        with sweep.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0))")
        with pytest.raises(psycopg2.errors.LockNotAvailable):
            rpc(reader, "lab_arena_restart_guard_state_v1")
        assert rpc(reader) == {
            "schema_version": "leadpoet.lab_arena.review_guard_state.v1", "guard_present": False,
        }
        sweep.rollback()
        assert rpc(reader, "lab_arena_restart_guard_state_v1")["guard_present"] is False


@pytest.mark.parametrize("commit", [False, True])
def test_real_guard_update_blocks_read_until_commit_or_rollback(database, commit):
    with closing(connect(database, transactional=True)) as writer, closing(connect(database, service=True)) as reader, closing(connect(database)) as observer:
        acquired = acquire(writer)
        with reader.cursor() as cursor:
            cursor.execute("SET lock_timeout='2s'")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(rpc, reader)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                with observer.cursor() as cursor:
                    cursor.execute(
                        "SELECT %s=ANY(pg_blocking_pids(%s))",
                        (writer.get_backend_pid(), reader.get_backend_pid()),
                    )
                    if cursor.fetchone()[0]:
                        break
                time.sleep(0.01)
            else:
                pytest.fail("review guard read did not wait for the actual guard writer")
            assert not future.done()
            if commit:
                writer.commit()
            else:
                writer.rollback()
            assert future.result(timeout=2)["guard_present"] is commit
        if commit:
            with observer.cursor() as cursor:
                cursor.execute(
                    "SELECT public.lab_arena_abort_restart_guard_v1(%s,%s,%s,'test')",
                    (GUARD, OWNER, acquired["guard_generation"]),
                )
        assert rpc(reader)["guard_present"] is False


def test_guard_update_lock_timeout_fails_closed(database):
    psycopg2, _ = database
    with closing(connect(database, transactional=True)) as writer, closing(connect(database, service=True)) as reader:
        acquire(writer)
        with pytest.raises(psycopg2.errors.LockNotAvailable):
            rpc(reader)
        writer.rollback()
        assert rpc(reader)["guard_present"] is False


def test_retained_expired_guard_still_blocks_and_operator_pause_semantics_stay_same(database):
    with closing(connect(database, transactional=True)) as admin:
        with admin.cursor() as cursor:
            cursor.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton")
        assert rpc(admin)["guard_present"] is False
        assert rpc(admin, "lab_arena_restart_guard_state_v1")["paused"] is True
        acquire(admin)
        with admin.cursor() as cursor:
            cursor.execute("UPDATE public.lab_arena_restart_claim_control SET guard_expires_at=clock_timestamp()-interval '1 second' WHERE singleton")
        assert rpc(admin)["guard_present"] is True
        canonical = rpc(admin, "lab_arena_restart_guard_state_v1")
        assert canonical["guard_present"] is True and canonical["guard_active"] is False
        admin.rollback()


def test_missing_control_row_errors_instead_of_permitting_review(database):
    psycopg2, _ = database
    with closing(connect(database, transactional=True)) as admin:
        with admin.cursor() as cursor:
            cursor.execute("DELETE FROM public.lab_arena_restart_claim_control WHERE singleton")
        with pytest.raises(psycopg2.errors.NoDataFound):
            rpc(admin)
        admin.rollback()
        assert rpc(admin)["guard_present"] is False


def test_idempotent_install_preserves_control_canonical_function_and_permissions(database):
    with closing(connect(database)) as admin, closing(connect(database)) as listener:
        with listener.cursor() as cursor:
            cursor.execute("LISTEN pgrst")
        with admin.cursor() as cursor:
            cursor.execute("SELECT row_to_json(c) FROM public.lab_arena_restart_claim_control c")
            before = cursor.fetchone()[0]
            cursor.execute("SELECT pg_get_functiondef('public.lab_arena_restart_guard_state_v1()'::regprocedure)")
            canonical_before = cursor.fetchone()[0]
            cursor.execute((SCRIPTS / MIGRATION).read_text())
            assert select.select([listener], [], [], 2)[0]
            listener.poll()
            assert [(n.channel, n.payload) for n in listener.notifies] == [("pgrst", "reload schema")]
            cursor.execute("SELECT row_to_json(c) FROM public.lab_arena_restart_claim_control c")
            assert cursor.fetchone()[0] == before
            cursor.execute("SELECT pg_get_functiondef('public.lab_arena_restart_guard_state_v1()'::regprocedure)")
            assert cursor.fetchone()[0] == canonical_before
            cursor.execute("SELECT pg_get_userbyid(proowner),prosecdef,provolatile,proconfig FROM pg_proc WHERE oid='public.lab_arena_review_guard_state_v1()'::regprocedure")
            assert cursor.fetchone() == ("lab_arena_owner", True, "v", ["search_path=pg_catalog, public"])
            for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
                cursor.execute("SELECT has_function_privilege(%s,'public.lab_arena_review_guard_state_v1()','EXECUTE')", (role,))
                assert cursor.fetchone()[0] is (role == "lab_arena_service")
