"""An expired restart-guard lease stops withholding paid review work."""
import pytest

from lab_arena.store import hash_lease_token, new_lease_token
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    accepted_recovery_submission, recovery_connect, recovery_store, recovery_superuser,
)
from tests.lab_arena.unfinished_review_recovery585_postgres_test import _begin
from tests.postgres_migration_harness import SCRIPTS

REVIEW_READ = '574-lab-arena-review-guard-row-read.sql'
MIGRATION = SCRIPTS / '595-lab-arena-review-guard-lease-expiry.sql'
EXPIRED = ("UPDATE public.lab_arena_restart_claim_control "
           "SET guard_expires_at=clock_timestamp()-interval '1 second' WHERE singleton")


@pytest.fixture(scope='module')
def recovery_database():
    from tests.lab_arena import atomic_review_restart_guard587_postgres_test as prior
    for psycopg2, dsn in prior.recovery_database.__wrapped__():
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as c:
                c.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                acl = c.fetchone()[0]
                c.execute('SET ROLE review587_migrator')
                c.execute((SCRIPTS / REVIEW_READ).read_text())
                c.execute(MIGRATION.read_text())
                c.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                assert c.fetchone()[0] == acl
        yield psycopg2, dsn


@pytest.fixture(autouse=True)
def clear_guard(recovery_superuser):
    yield
    with recovery_superuser.cursor() as c:
        c.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,"
                  "guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',"
                  "restart_scope='',restart_phase='',captured_leases='[]'::jsonb WHERE singleton")


def acquire(connection, seconds=600):
    with connection.cursor() as c:
        c.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
        generation = c.fetchone()[0]
        c.execute("SELECT public.lab_arena_acquire_restart_guard_v1(%s,%s,%s,%s,%s,'gateway','review595-test')",
                  ('lab_arena_restart_guard:' + 'a'*64, 'lab_arena_restart_owner:' + 'b'*64,
                   generation, seconds, 'c'*40))
        return c.fetchone()[0]


def review_state(connection):
    with connection.cursor() as c:
        c.execute('SELECT public.lab_arena_review_guard_state_v1()')
        return c.fetchone()[0]


def test_review_read_reports_presence_and_lease_separately(recovery_superuser):
    assert review_state(recovery_superuser) == {
        'schema_version': 'leadpoet.lab_arena.review_guard_state.v1',
        'guard_present': False, 'guard_active': False,
    }
    acquire(recovery_superuser)
    assert review_state(recovery_superuser) == {
        'schema_version': 'leadpoet.lab_arena.review_guard_state.v1',
        'guard_present': True, 'guard_active': True,
    }
    with recovery_superuser.cursor() as c:
        c.execute(EXPIRED)
    assert review_state(recovery_superuser) == {
        'schema_version': 'leadpoet.lab_arena.review_guard_state.v1',
        'guard_present': True, 'guard_active': False,
    }


def test_expired_lease_admits_review_reservation(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'g01')
    acquire(recovery_superuser)
    with recovery_superuser.cursor() as c:
        c.execute(EXPIRED)
    assert _begin(recovery_store, submission, miner)['status'] == 'claimed'


def test_live_lease_still_pauses_review_reservation(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'g02')
    acquire(recovery_superuser)
    before = recovery_store.get_submission(submission), recovery_store.list_ledger(submission_id=submission)
    assert _begin(recovery_store, submission, miner)['status'] == 'paused'
    assert (recovery_store.get_submission(submission),
            recovery_store.list_ledger(submission_id=submission)) == before


def test_restart_semantics_and_operator_pause_are_unchanged(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'g03')
    acquire(recovery_superuser)
    with recovery_superuser.cursor() as c:
        c.execute(EXPIRED)
        c.execute('SELECT public.lab_arena_restart_guard_state_v1()')
        canonical = c.fetchone()[0]
    # A restart controller still sees its own retained, expired guard.
    assert canonical['guard_present'] is True and canonical['guard_active'] is False
    with recovery_superuser.cursor() as c:
        c.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton')
    assert review_state(recovery_superuser)['guard_present'] is True
    assert _begin(recovery_store, submission, miner)['status'] == 'claimed'


def test_missing_singleton_still_fails_closed(recovery_store, recovery_connect):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'g04')
    connection = recovery_connect()
    try:
        with connection.cursor() as c:
            c.execute('DELETE FROM public.lab_arena_restart_claim_control WHERE singleton')
            with pytest.raises(Exception, match='no rows'):
                c.execute('SELECT public.lab_arena_review_guard_state_v1()')
        connection.rollback()
        with connection.cursor() as c:
            c.execute('DELETE FROM public.lab_arena_restart_claim_control WHERE singleton')
            with pytest.raises(Exception, match='no rows'):
                c.execute("SELECT public.lab_arena_begin_submission_review(%s,%s,%s,50000,"
                          "'anthropic/claude-sonnet-5',2,1000)",
                          (submission, miner, hash_lease_token(new_lease_token())))
    finally:
        connection.rollback(); connection.close()


def test_migration_reapply_preserves_definition_and_security(recovery_superuser):
    query = ("SELECT oid,pg_get_functiondef(oid),to_jsonb(pg_proc)-'prosrc' FROM pg_proc WHERE oid IN ("
             "'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::regprocedure,"
             "'public.lab_arena_review_guard_state_v1()'::regprocedure,"
             "'public.lab_arena_restart_guard_state_v1()'::regprocedure) ORDER BY oid")
    with recovery_superuser.cursor() as c:
        c.execute(query)
        before = c.fetchall()
        c.execute(MIGRATION.read_text())
        c.execute(query)
        assert c.fetchall() == before
