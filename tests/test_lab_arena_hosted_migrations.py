"""Arena installation under the non-superuser role used by hosted Supabase."""

from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.postgres_migration_harness import SCRIPTS


def test_hosted_owner_transfers_and_idempotent_upgrade():
    # Earlier tests installed as a superuser, which bypasses schema ownership
    # checks. This role has migration authority but no superuser privilege.
    # PostgreSQL 17 scopes CREATEROLE operations by ADMIN OPTION. Supabase's
    # managed roles already exist, so model that hosted reapply shape directly.
    setup = """
    CREATE ROLE hosted_migrator LOGIN CREATEDB CREATEROLE INHERIT;
    GRANT anon, authenticated, service_role TO hosted_migrator WITH ADMIN OPTION;
    CREATE ROLE lab_arena_owner NOLOGIN;
    CREATE ROLE lab_arena_service NOLOGIN;
    GRANT lab_arena_owner, lab_arena_service TO hosted_migrator WITH ADMIN OPTION;
    ALTER SCHEMA public OWNER TO hosted_migrator;
    GRANT USAGE ON SCHEMA extensions TO hosted_migrator WITH GRANT OPTION;
    SET ROLE hosted_migrator;
    CREATE TABLE public.qualification_private_icp_sets (
      set_id BIGINT PRIMARY KEY, icps JSONB NOT NULL,
      active_from TIMESTAMPTZ, active_until TIMESTAMPTZ,
      is_active BOOLEAN NOT NULL DEFAULT FALSE
    );
    ALTER TABLE public.qualification_private_icp_sets ENABLE ROW LEVEL SECURITY;
    RESET ROLE;
    """
    database = database_with_lab_arena_migration((), setup_sql=setup)
    try:
        psycopg2, dsn = next(database)
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("SET ROLE hosted_migrator")
                # Applied historical view definitions need not support replay
                # after later migrations add columns. Reapply the current
                # migration, not obsolete definitions that would drop columns.
                for migrations in (DEFAULT_MIGRATIONS, DEFAULT_MIGRATIONS[-1:]):
                    for migration in migrations:
                        try:
                            cursor.execute((SCRIPTS / migration).read_text())
                        except Exception as exc:
                            raise AssertionError(
                                "hosted migration failed: %s" % migration
                            ) from exc
                        cursor.execute(
                            "SELECT has_schema_privilege('lab_arena_owner', 'public', 'CREATE'), "
                            "has_schema_privilege('lab_arena_service', 'public', 'CREATE')"
                        )
                        assert cursor.fetchone() == (False, False)
                cursor.execute("RESET ROLE; SET ROLE lab_arena_service")
                cursor.execute("SELECT public.lab_arena_schema_version_v1()")
                assert cursor.fetchone()[0]["version"] == 197
    finally:
        database.close()
