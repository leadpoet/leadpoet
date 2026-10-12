"""Migration468 works under hosted authority and restores restricted privileges."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena import zero_call_setup_recovery467_postgres_test as deployed

SQL = Path(__file__).parents[2] / 'scripts/468-lab-arena-legacy-review-checkpoint.sql'
RPCS = (
    'public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
    'public.lab_arena_retry_credit_failures_v1(text,text,text,text)',
)
HELPER = 'public.lab_arena__legacy_partial_bound(public.lab_arena_runs,jsonb,text)'


def _rpc_snapshot(cursor):
    result = {}
    for signature in RPCS:
        cursor.execute('SELECT pg_get_functiondef(oid),pg_get_userbyid(proowner),proacl::text FROM pg_proc WHERE oid=%s::regprocedure', (signature,))
        result[signature] = cursor.fetchone()
    return result


def test_non_superuser_apply_replay_and_failed_old_migration_restore_authority():
    generator = deployed.database.__wrapped__()
    psycopg, dsn = next(generator)
    control = psycopg.connect(**dsn)
    control.autocommit = True
    try:
        with control.cursor() as cursor:
            cursor.execute('CREATE ROLE arena_checkpoint_migrator NOLOGIN NOSUPERUSER INHERIT')
            cursor.execute('GRANT lab_arena_owner TO arena_checkpoint_migrator')
            cursor.execute('GRANT USAGE, CREATE ON SCHEMA public TO arena_checkpoint_migrator WITH GRANT OPTION')
            cursor.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            original_schema_acl = cursor.fetchone()[0]
            original_rpcs = _rpc_snapshot(cursor)
            cursor.execute('SET ROLE arena_checkpoint_migrator')
            cursor.execute("SELECT rolsuper,has_schema_privilege(current_user,'public','CREATE'),has_schema_privilege('lab_arena_owner','public','CREATE'),pg_has_role(current_user,'lab_arena_owner','MEMBER') FROM pg_roles WHERE rolname=current_user")
            assert cursor.fetchone() == (False, True, False, True)
            literal = SQL.read_text()
            grant = 'GRANT CREATE ON SCHEMA public TO lab_arena_owner;'
            revoke = 'REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;'
            assert literal.count(grant) == literal.count(revoke) == 1
            # The exact prior migration must fail and roll back on this hosted
            # authority. A superuser-only fixture would hide the production bug.
            previous = literal.replace(grant, '').replace(revoke, '')
            with pytest.raises(psycopg.Error) as denied:
                cursor.execute(previous)
            assert denied.value.pgcode == '42501'
            cursor.execute('ROLLBACK')
            assert _rpc_snapshot(cursor) == original_rpcs
            cursor.execute('SELECT to_regprocedure(%s)', (HELPER,))
            assert cursor.fetchone()[0] is None
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cursor.fetchone()[0] == original_schema_acl

            first_rpcs = None
            first_helper = None
            for _ in range(2):
                cursor.execute(literal)
                cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                assert cursor.fetchone()[0] == original_schema_acl
                cursor.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_schema_privilege('lab_arena_service','public','CREATE')")
                assert cursor.fetchone() == (False, False)
                installed = _rpc_snapshot(cursor)
                for signature, row in installed.items():
                    assert row[1:] == original_rpcs[signature][1:]
                    assert row[1] == 'lab_arena_owner'
                cursor.execute('SELECT pg_get_functiondef(oid),pg_get_userbyid(proowner),prosecdef,proacl::text FROM pg_proc WHERE oid=%s::regprocedure', (HELPER,))
                helper = cursor.fetchone()
                assert helper[1:] == ('lab_arena_owner', True, '{lab_arena_owner=X/lab_arena_owner}')
                for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
                    cursor.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, HELPER))
                    assert cursor.fetchone()[0] is False
                if first_rpcs is None:
                    first_rpcs, first_helper = installed, helper
                else:
                    assert installed == first_rpcs
                    assert helper == first_helper
            cursor.execute('RESET ROLE')
    finally:
        control.close()
        generator.close()
