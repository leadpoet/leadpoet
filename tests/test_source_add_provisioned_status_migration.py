from pathlib import Path

import pytest

from tests.test_source_add_end_to_end_postgres import _database_with_migrations

MIGRATION = Path(__file__).parents[1] / "scripts/186-research-lab-source-add-provisioned-status.sql"


def test_source_add_status_migration_is_idempotent_and_preserves_states():
    sql = MIGRATION.read_text(encoding="utf-8")

    assert sql.startswith("-- 186-")
    assert "BEGIN;" in sql and "COMMIT;" in sql
    assert "UPDATE public.research_lab_source_add_provisioning_events" in sql
    assert "SET provision_status = 'provisioned'" in sql
    assert "provision_status IN (" in sql
    assert "'approved_pending_provision'" in sql
    assert "'provisioned'" in sql
    assert "'disabled'" in sql
    assert "pg_get_functiondef" in sql
    assert "CREATE TABLE" not in sql
    assert "DROP TABLE" not in sql


def test_source_add_status_migration_rewrites_deployed_predicates():
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "LIKE '%provisioned_autoresearch_eligible%'" in sql
    assert "'''provisioned_autoresearch_eligible'''" in sql
    assert "'''provisioned'''" in sql
    assert sql.index("UPDATE public.research_lab_source_add_provisioning_events") < sql.index(
        "$source_add_status_functions$"
    )


@pytest.fixture(scope="module")
def migration_database():
    setup_sql = """
    CREATE TABLE public.research_lab_source_add_provisioning_events (
        provision_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        provision_status TEXT NOT NULL CHECK (
            provision_status IN (
                'approved_pending_provision',
                'provisioned_autoresearch_eligible',
                'disabled'
            )
        )
    );
    INSERT INTO public.research_lab_source_add_provisioning_events (provision_status)
    VALUES ('provisioned_autoresearch_eligible'), ('disabled');
    CREATE FUNCTION public.source_add_test_status_predicate(p_status TEXT)
    RETURNS BOOLEAN LANGUAGE plpgsql AS $$
    BEGIN
        RETURN p_status = 'provisioned_autoresearch_eligible';
    END;
    $$;
    """
    yield from _database_with_migrations(
        ("186-research-lab-source-add-provisioned-status.sql",),
        setup_sql=setup_sql,
    )


def test_source_add_status_migration_runs_on_postgres_and_is_idempotent(
    migration_database,
):
    psycopg2, dsn = migration_database
    connection = psycopg2.connect(**dsn)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
                cursor.execute(
                    "SELECT array_agg(provision_status ORDER BY provision_status) "
                    "FROM public.research_lab_source_add_provisioning_events"
                )
                assert cursor.fetchone()[0] == ["disabled", "provisioned"]
                cursor.execute(
                    "SELECT pg_get_functiondef('public.source_add_test_status_predicate(text)'::regprocedure)"
                )
                assert "'provisioned'" in cursor.fetchone()[0]
    finally:
        connection.close()
