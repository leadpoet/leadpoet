from pathlib import Path

import pytest

from tests.test_source_add_end_to_end_postgres import _database_with_migrations
from tests.test_source_add_provenance_leg1_postgres import PRE_MIGRATIONS

MIGRATION = Path(__file__).parents[1] / "scripts/186-research-lab-source-add-provisioned-status.sql"


def test_source_add_status_migration_is_idempotent_and_preserves_states():
    sql = MIGRATION.read_text(encoding="utf-8")

    assert sql.startswith("-- 186-")
    assert "BEGIN;" in sql and "COMMIT;" in sql
    assert "CASE" in sql
    assert "provision_status IN (" in sql
    assert "'approved_pending_provision'" in sql
    assert "'provisioned'" in sql
    assert "'disabled'" in sql
    assert "pg_get_functiondef" in sql
    assert "supported_functions CONSTANT TEXT[]" in sql
    assert "public.research_lab_source_add_final_approval_catalog_v2(text)" in sql
    assert "IS NOT TRUE" in sql
    assert "CREATE OR REPLACE VIEW public.research_lab_source_add_provisioning_current" in sql
    assert "provision_ref/provenance" in sql
    assert "UPDATE public.research_lab_source_add_provisioning_events" not in sql
    assert "CREATE TABLE" not in sql
    assert "DROP TABLE" not in sql


def test_source_add_status_migration_rewrites_deployed_predicates():
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "LIKE '%provisioned_autoresearch_eligible%'" in sql
    assert "provisioned_autoresearch_eligible''" in sql
    assert "research_lab_source_add_provision_status_is_eligible_v1" in sql
    assert sql.index(
        "CREATE OR REPLACE VIEW public.research_lab_source_add_provisioning_current"
    ) < sql.index("$source_add_status_functions$")


@pytest.fixture(scope="module")
def migration_database():
    yield from _database_with_migrations(
        PRE_MIGRATIONS
        + (
            "175-research-lab-source-add-provenance-leg1.sql",
            "176-research-lab-source-add-provenance-origin-repair.sql",
        )
    )


def test_source_add_status_migration_runs_on_postgres_and_is_idempotent(
    migration_database,
):
    psycopg2, dsn = migration_database
    connection = psycopg2.connect(**dsn)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE FUNCTION public.source_add_test_status_predicate(p_status TEXT)
                    RETURNS BOOLEAN LANGUAGE plpgsql AS $function$
                    BEGIN
                        RETURN p_status = 'provisioned_autoresearch_eligible';
                    END;
                    $function$;
                    """
                )
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
                cursor.execute(
                    "SELECT array_agg(provision_status ORDER BY provision_status) "
                    "FROM public.research_lab_source_add_provisioning_current"
                )
                assert cursor.fetchone()[0] is None
                cursor.execute(
                    "SELECT pg_get_functiondef('public.source_add_test_status_predicate(text)'::regprocedure)"
                )
                assert "provisioned_autoresearch_eligible" in cursor.fetchone()[0]
                cursor.execute(
                    "SELECT public.research_lab_source_add_provision_status_is_eligible_v1(NULL) IS NOT TRUE"
                )
                assert cursor.fetchone()[0] is True
    finally:
        connection.close()
