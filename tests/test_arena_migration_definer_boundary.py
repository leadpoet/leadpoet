"""Native ACL regression controls; only disposable PostgreSQL is written."""
from pathlib import Path
import os
import re
import shutil

import pytest

from scripts.arena_rds_migration import ROLE_QUERY
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.postgres_migration_harness import _database_with_migrations

MIGRATION = Path(__file__).resolve().parents[1] / 'scripts/692-arena-migration-definer-boundary.sql'
SQL = MIGRATION.read_text()
SIGNATURES = re.findall(r"'((?:public\.)[a-z_]+\([^']*\))'", SQL)


@pytest.fixture
def database():
    # Role membership options are PostgreSQL 17 production semantics. The
    # repository's older default container is PostgreSQL 15.
    local = os.environ.get('LAB_ARENA_PG_LOCAL') == '1' or shutil.which('docker') is None
    generator = (database_with_lab_arena_migration(migrations=()) if local else
                 _database_with_migrations((), image='postgres:17'))
    for pg, dsn in generator:
        c = pg.connect(**dsn)
        c.autocommit = True
        with c.cursor() as q:
            q.execute('CREATE ROLE lab_arena_owner NOLOGIN NOINHERIT; '
                      'CREATE ROLE arena_admin LOGIN INHERIT NOSUPERUSER; '
                      'GRANT postgres TO arena_admin WITH ADMIN FALSE, INHERIT TRUE, SET TRUE; '
                      'CREATE ROLE arena_migrator LOGIN NOINHERIT; '
                      'GRANT lab_arena_owner TO arena_migrator '
                      'WITH ADMIN FALSE, INHERIT TRUE, SET TRUE; '
                      'GRANT CREATE ON SCHEMA public TO arena_migrator WITH GRANT OPTION')
            for signature in SIGNATURES:
                # Harmless owner functions model the real ACL boundary.
                q.execute(f'CREATE FUNCTION {signature} RETURNS integer LANGUAGE sql '
                          'SECURITY DEFINER AS $$ SELECT 1 $$; '
                          f'GRANT EXECUTE ON FUNCTION {signature} TO anon,authenticated,service_role')
        yield pg, c, dsn
        c.close()


def catalog(c):
    with c.cursor() as q:
        q.execute("SELECT oid,proowner,prosrc,proacl::text,proconfig FROM pg_proc "
                  "WHERE oid=ANY(%s::regprocedure[]) ORDER BY oid", (SIGNATURES,))
        return q.fetchall()


def test_removes_only_default_access_preserves_app_grants_and_owner_bodies(database):
    pg, c, dsn = database
    with pg.connect(**{**dsn, 'user': 'arena_migrator'}) as restricted:
        with restricted.cursor() as q:
            q.execute(ROLE_QUERY)
            assert q.fetchone()[0]['external_definers'] is True
    before = catalog(c)
    with pg.connect(**{**dsn, 'user': 'arena_admin'}) as admin:
        admin.autocommit = True
        with admin.cursor() as q:
            q.execute(SQL)
            q.execute(SQL)  # repeat is safe
    with c.cursor() as q:
        for signature in SIGNATURES:
            q.execute('SELECT has_function_privilege(%s,%s,%s)',
                      ('lab_arena_owner', signature, 'EXECUTE'))
            assert q.fetchone() == (False,)
            for role in ('anon', 'authenticated', 'service_role', 'postgres'):
                q.execute('SELECT has_function_privilege(%s,%s,%s)', (role,signature,'EXECUTE'))
                assert q.fetchone() == (True,)
    after = catalog(c)
    assert [row[:3] + row[4:] for row in after] == [row[:3] + row[4:] for row in before]
    for old, new in zip(before, after):
        assert new[3] == old[3].replace('=X/postgres,', '', 1)
    with pg.connect(**{**dsn, 'user': 'arena_migrator'}) as restricted:
        with restricted.cursor() as q:
            q.execute(ROLE_QUERY)
            assert q.fetchone()[0]['external_definers'] is False
            with pytest.raises(pg.errors.InsufficientPrivilege):
                q.execute('SELECT public.debug_validator_check(%s)', ('test',))
    with c.cursor() as q:
        q.execute('SET ROLE service_role; SELECT public.debug_validator_check(%s)', ('test',))
        assert q.fetchone() == (1,)
        q.execute('RESET ROLE')


@pytest.mark.parametrize('drift', ['owner', 'missing_explicit_grant'])
def test_unknown_acl_or_owner_is_atomic(database, drift):
    pg, c, _ = database
    with c.cursor() as q:
        if drift == 'owner':
            q.execute('ALTER FUNCTION public.setup_rls() OWNER TO lab_arena_owner')
        else:
            q.execute('REVOKE EXECUTE ON FUNCTION public.setup_rls() FROM anon')
        before = catalog(c)
        with pytest.raises(pg.Error):
            q.execute(SQL)
        q.execute('ROLLBACK')
    assert catalog(c) == before
