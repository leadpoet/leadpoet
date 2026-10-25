"""Exercise migration ownership under the production schema privilege boundary."""
from contextlib import closing

import pytest

from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = SCRIPTS / "574-lab-arena-review-guard-row-read.sql"
FUNCTION = "public.lab_arena_review_guard_state_v1()"


@pytest.fixture()
def database():
    for psycopg2, dsn in database_with_lab_arena_migration(DEFAULT_MIGRATIONS):
        with closing(psycopg2.connect(**dsn)) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                # PostgreSQL 18 forbids demoting its bootstrap superuser.
                # Model the release role separately, as migration 567 does.
                cursor.execute("CREATE ROLE review_guard574_migrator NOLOGIN NOSUPERUSER INHERIT")
                cursor.execute("GRANT lab_arena_owner TO review_guard574_migrator")
                cursor.execute("ALTER SCHEMA public OWNER TO review_guard574_migrator")
                cursor.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC,lab_arena_owner")
                cursor.execute("SET ROLE review_guard574_migrator")
                cursor.execute("SELECT rolsuper,pg_has_role(current_user,'lab_arena_owner','MEMBER'),has_schema_privilege('lab_arena_owner','public','CREATE') FROM pg_roles WHERE rolname=current_user")
                assert cursor.fetchone() == (False, True, False)
            yield psycopg2, connection


def schema_acl(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        return cursor.fetchone()[0]


def function_security(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_get_userbyid(proowner),proacl::text,prosecdef,provolatile,proconfig FROM pg_proc WHERE oid=to_regprocedure(%s)", (FUNCTION,))
        return cursor.fetchone()


@pytest.mark.parametrize("had_create", [False, True])
def test_non_superuser_migration_restores_schema_acl_on_first_apply_and_reapply(database, had_create):
    _, connection = database
    if had_create:
        with connection.cursor() as cursor:
            cursor.execute("GRANT CREATE ON SCHEMA public TO lab_arena_owner")
    before = schema_acl(connection)
    assert function_security(connection) is None
    for _ in range(2):
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        assert schema_acl(connection) == before
        security = function_security(connection)
        assert security[0] == "lab_arena_owner"
        with connection.cursor() as cursor:
            cursor.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_function_privilege('lab_arena_service',%s,'EXECUTE'),has_function_privilege('service_role',%s,'EXECUTE')", (FUNCTION, FUNCTION))
            assert cursor.fetchone() == (had_create, True, False)


@pytest.mark.parametrize("preinstalled", [False, True])
def test_non_superuser_failed_migration_rolls_back_schema_and_function_privileges(database, preinstalled):
    psycopg2, connection = database
    if preinstalled:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
    before_acl = schema_acl(connection)
    before_function = function_security(connection)
    failed = MIGRATION.read_text().replace(
        "DO $restore574$",
        "REVOKE EXECUTE ON FUNCTION public.lab_arena_review_guard_state_v1() FROM lab_arena_service;\n"
        "SELECT 1/0;\nDO $restore574$",
        1,
    )
    with pytest.raises(psycopg2.errors.DivisionByZero):
        with connection.cursor() as cursor:
            cursor.execute(failed)
    # The migration owns BEGIN on an autocommit connection; end that explicit
    # failed transaction with SQL rather than psycopg's autocommit no-op.
    with connection.cursor() as cursor:
        cursor.execute("ROLLBACK")
    assert schema_acl(connection) == before_acl
    assert function_security(connection) == before_function
    with connection.cursor() as cursor:
        cursor.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
        assert cursor.fetchone()[0] is False
    # A retry on the same migration connection must still install correctly.
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE')", (FUNCTION,))
        assert cursor.fetchone()[0] is True
    assert schema_acl(connection) == before_acl
