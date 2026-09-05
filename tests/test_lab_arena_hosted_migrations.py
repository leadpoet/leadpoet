"""Arena installation under the non-superuser role used by hosted Supabase."""

from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS
from tests.test_source_add_end_to_end_postgres import SCRIPTS, _database_with_migrations


def test_hosted_owner_transfers_and_idempotent_upgrade():
    # Earlier tests installed as a superuser, which bypasses schema ownership
    # checks. This role has migration authority but no superuser privilege.
    setup = """
    CREATE ROLE hosted_migrator LOGIN CREATEROLE INHERIT;
    GRANT anon, authenticated, service_role TO hosted_migrator WITH ADMIN OPTION;
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
    database = _database_with_migrations((), setup_sql=setup)
    try:
        psycopg2, dsn = next(database)
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("SET ROLE hosted_migrator")
                for _ in range(2):
                    for migration in DEFAULT_MIGRATIONS:
                        cursor.execute((SCRIPTS / migration).read_text())
                        cursor.execute(
                            "SELECT has_schema_privilege('lab_arena_owner', 'public', 'CREATE'), "
                            "has_schema_privilege('lab_arena_service', 'public', 'CREATE')"
                        )
                        assert cursor.fetchone() == (False, False)
                cursor.execute("RESET ROLE; SET ROLE lab_arena_service")
                cursor.execute("SELECT public.lab_arena_schema_version_v1()")
                assert cursor.fetchone()[0]["version"] == 185
    finally:
        database.close()
