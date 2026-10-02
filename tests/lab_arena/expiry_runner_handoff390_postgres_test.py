"""Expiry retry keeps validator identity and hands zero-call score failures off."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior


database = prior.database
MIGRATION = Path(__file__).parents[2] / 'scripts/390-lab-arena-expired-retry-runner-handoff.sql'
CLAIM_HASH = '0252fa3efafe895a7cade9169a36493dc8be70a9148a5ca037d0d0dde870a7c0'


def _hash(cursor, signature):
    cursor.execute("SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),"
                   "'sha256'),'hex')", (signature,))
    return cursor.fetchone()[0]


@pytest.fixture(scope='module')
def migration_sql(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = MIGRATION.read_text()
            assert CLAIM_HASH in sql
            cursor.execute("SELECT owner.rolname,p.proacl,p.prosecdef,p.provolatile,p.proconfig "
                           "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                           "JOIN pg_roles owner ON owner.oid=p.proowner "
                           "WHERE n.nspname='public' AND p.proname='lab_arena_claim_assignment' "
                           "AND p.pronargs=9")
            security_before = cursor.fetchone()
            sql = sql.replace(CLAIM_HASH, _hash(cursor,
                'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'))
            cursor.execute('BEGIN')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cursor.execute(sql.replace(
                    _hash(cursor, 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'), '0' * 64, 1))
            cursor.execute('ROLLBACK')
            cursor.execute(sql)
            cursor.execute(sql)
            cursor.execute("SELECT owner.rolname,p.proacl,p.prosecdef,p.provolatile,p.proconfig "
                           "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                           "JOIN pg_roles owner ON owner.oid=p.proowner "
                           "WHERE n.nspname='public' AND p.proname='lab_arena_claim_assignment' "
                           "AND p.pronargs=9")
            assert cursor.fetchone() == security_before
            yield sql


def _leased_prior(connection, *, kind='score'):
    prior._seed(connection, {'terminal_status':'judge_error'}, 'judge_error', kind=kind)
    with connection.cursor() as cursor:
        cursor.execute('SET session_replication_role=replica')
        cursor.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s", (prior.ASSIGNMENT+':2',))
        cursor.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,"
                       "result_doc=NULL,lease_token_hash=%s,lease_expires_at=now()-interval '1 second' "
                       "WHERE run_id=%s", ('sha256:'+'a'*64, prior.ASSIGNMENT+':1'))
        cursor.execute('SET session_replication_role=origin')
    connection.commit()


def test_expiry_retains_prior_runner_and_prevents_zero_call_score_reclaim(database, migration_sql):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        _leased_prior(conn)
        with conn.cursor() as cursor:
            cursor.execute('SELECT public.lab_arena_expire_leases(%s)', (prior.ROUND,))
            assert cursor.fetchone()[0]['retried'] == 1
            cursor.execute("SELECT previous_runner_hotkey,status,kind FROM public.lab_arena_runs "
                           "WHERE run_id=%s", (prior.ASSIGNMENT+':2',))
            assert cursor.fetchone() == (prior.RUNNER_A, 'pending', 'score')
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, '1') == {'status':'no_pending'}
        claim = prior._claim(conn, prior.RUNNER_B, '2')
        assert claim['status']=='leased' and claim['run_id']==prior.ASSIGNMENT+':2'


@pytest.mark.parametrize('cause,with_ledger,expect_block', [
    ('lease_expired', False, True),
    ('lease_expired', True, False),
    ('judge_error', False, False),
], ids=['zero_call_expiry','provider_spend','ordinary_judge_failure'])
def test_only_exact_zero_call_expiry_blocks_same_runner(
    database, migration_sql, cause, with_ledger, expect_block
):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        prior._seed(conn, {'terminal_status':cause}, cause,
                    kind='score', prior_has_ledger=with_ledger)
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET result_doc=NULL,"
                           "lease_expires_at=now()-interval '1 second' "
                           "WHERE run_id=%s", (prior.ASSIGNMENT+':1',))
            cursor.execute('SET session_replication_role=origin')
        conn.commit()
        response = prior._claim(conn, prior.RUNNER_A, '3')
        assert (response == {'status':'no_pending'}) is expect_block
        if expect_block:
            assert prior._claim(conn, prior.RUNNER_B, '4')['status']=='leased'
        else:
            assert response['run_id']==prior.ASSIGNMENT+':2'


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb WHERE attempt=1",
    "UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted' WHERE attempt=1",
    "UPDATE public.lab_arena_runs SET lease_expires_at=now()+interval '1 hour' WHERE attempt=1",
    "UPDATE public.lab_arena_runs SET attempt=3 WHERE attempt=2",
], ids=['has_result','not_failed','not_expired','not_immediate'])
def test_unproved_predecessor_keeps_existing_lone_runner_fallback(
    database, migration_sql, mutation
):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        prior._seed(conn, {'terminal_status':'lease_expired'}, 'lease_expired', kind='score')
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET result_doc=NULL,"
                           "lease_expires_at=now()-interval '1 second' WHERE attempt=1")
            cursor.execute(mutation)
            cursor.execute('SET session_replication_role=origin')
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, '7')['status']=='leased'


def test_execute_expiry_keeps_existing_claim_fallback(database, migration_sql):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        _leased_prior(conn, kind='execute')
        with conn.cursor() as cursor:
            cursor.execute('SELECT public.lab_arena_expire_leases(%s)', (prior.ROUND,))
            assert cursor.fetchone()[0]['retried']==1
            cursor.execute('SELECT previous_runner_hotkey FROM public.lab_arena_runs '
                           'WHERE run_id=%s', (prior.ASSIGNMENT+':2',))
            assert cursor.fetchone()[0]==prior.RUNNER_A
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, '5')['status']=='leased'
