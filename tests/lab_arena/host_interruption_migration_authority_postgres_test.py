"""Migration474 runs with hosted authority and restores schema permissions."""
import hashlib
from pathlib import Path

import pytest

from tests.lab_arena import host_interruption_retry474_postgres_test as functional

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/474-lab-arena-host-interruption-execution-retries.sql'
RPCS = (
    'public.lab_arena_close_stage(text,smallint)',
    'public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
    'public.lab_arena_expire_leases(text)',
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',
)
HELPERS = (
    'public.lab_arena_execution_attempt_count(public.lab_arena_runs)',
    'public.lab_arena__execute_assignment_calls_v1(text,text)',
    'public.lab_arena__execute_host_retry_ready_v1(public.lab_arena_runs)',
)


def snapshot(cursor, signatures):
    result = {}
    for signature in signatures:
        cursor.execute('SELECT pg_get_functiondef(oid),pg_get_userbyid(proowner),proacl::text,prosecdef,proconfig FROM pg_proc WHERE oid=to_regprocedure(%s)', (signature,))
        result[signature] = cursor.fetchone()
    return result


def test_non_superuser_old_failure_fixed_apply_and_replay_preserve_authority():
    generator = functional.previous.database.__wrapped__()
    db = next(generator)
    psycopg, dsn = db
    control = psycopg.connect(**dsn)
    control.autocommit = True
    try:
        with control.cursor() as cursor:
            cursor.execute((ROOT / 'scripts/473-lab-arena-recovered-execution-judge-continuation.sql').read_text())
            cursor.execute('CREATE ROLE arena_host_retry_migrator NOLOGIN NOSUPERUSER INHERIT')
            cursor.execute('GRANT lab_arena_owner TO arena_host_retry_migrator')
            cursor.execute('GRANT USAGE,CREATE ON SCHEMA public TO arena_host_retry_migrator WITH GRANT OPTION')
            cursor.execute('ALTER DEFAULT PRIVILEGES FOR ROLE arena_host_retry_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cursor.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            original_schema_acl = cursor.fetchone()[0]
            original_rpcs = snapshot(cursor, RPCS)
            cursor.execute('SET ROLE arena_host_retry_migrator')
            cursor.execute("SELECT rolsuper,has_schema_privilege(current_user,'public','CREATE'),has_schema_privilege('lab_arena_owner','public','CREATE'),pg_has_role(current_user,'lab_arena_owner','MEMBER') FROM pg_roles WHERE rolname=current_user")
            assert cursor.fetchone() == (False, True, False, True)
            literal = SQL.read_text()
            grant = 'GRANT CREATE ON SCHEMA public TO lab_arena_owner;\n'
            revoke = 'REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;\n'
            assert literal.count(grant) == literal.count(revoke) == 1
            old = literal.replace(grant, '').replace(revoke, '')
            assert hashlib.sha256(old.encode()).hexdigest() == 'aad125cbe73c749281d2f3a98ff090467db8510fe130c6b92bc2a93d78dc92f9'
            with pytest.raises(psycopg.Error) as denied:
                cursor.execute(old)
            assert denied.value.pgcode == '42501'
            cursor.execute('ROLLBACK')
            assert snapshot(cursor, RPCS) == original_rpcs
            assert all(row is None for row in snapshot(cursor, HELPERS).values())
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cursor.fetchone()[0] == original_schema_acl
            first = None
            for _ in range(2):
                cursor.execute(literal)
                cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                assert cursor.fetchone()[0] == original_schema_acl
                cursor.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_schema_privilege('lab_arena_service','public','CREATE')")
                assert cursor.fetchone() == (False, False)
                installed = snapshot(cursor, RPCS)
                for signature, row in installed.items():
                    assert row[1:] == original_rpcs[signature][1:]
                helpers = snapshot(cursor, HELPERS)
                for signature, row in helpers.items():
                    assert row[1] == 'lab_arena_owner' and row[3] is True
                    assert row[4] == ['search_path=pg_catalog, public']
                    for role in ('anon', 'authenticated', 'service_role'):
                        cursor.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, signature))
                        assert cursor.fetchone()[0] is False
                    cursor.execute('SELECT has_function_privilege(\'lab_arena_service\',%s,\'EXECUTE\')', (signature,))
                    assert cursor.fetchone()[0] is (signature == HELPERS[0])
                current = (installed, helpers)
                if first is None:
                    first = current
                else:
                    assert current == first
            cursor.execute('RESET ROLE')
        # Use the resulting schema for exact recovery, exhaustion, and billing
        # controls. All databases and transitions here are disposable and local.
        functional.test_host_then_provider_then_accepted_uses_original_queue_and_preserves_history(db)
        functional.test_two_genuine_provider_failures_still_exhaust(db)
        functional.test_unsettled_host_fee_does_not_turn_missing_evaluation_into_zero(db)
    finally:
        control.close()
        generator.close()
