from pathlib import Path

import pytest

from tests.test_source_add_end_to_end_postgres import _database_with_migrations

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
    setup_sql = """
    CREATE TABLE public.research_lab_source_catalog (
        catalog_id TEXT PRIMARY KEY,
        source_name TEXT,
        source_kind TEXT,
        declared_base_domains JSONB,
        accepted_at TIMESTAMPTZ,
        catalog_doc JSONB
    );
    INSERT INTO public.research_lab_source_catalog
        (catalog_id, source_name, source_kind, declared_base_domains, catalog_doc)
    VALUES ('source_catalog:test', 'Test', 'web', '[]'::JSONB, '{}'::JSONB);
    CREATE TABLE public.research_lab_source_add_provisioning_events (
        provision_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        provision_ref TEXT NOT NULL,
        catalog_id TEXT NOT NULL,
        submission_id TEXT NOT NULL,
        adapter_id TEXT NOT NULL,
        miner_hotkey TEXT NOT NULL,
        source_identity_hash TEXT NOT NULL,
        registry_provider_id TEXT NOT NULL,
        seq INTEGER NOT NULL DEFAULT 0,
        provision_doc JSONB NOT NULL DEFAULT '{}'::JSONB,
        credential_envelope JSONB NOT NULL DEFAULT '{}'::JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        provision_status TEXT NOT NULL CHECK (
            provision_status IN (
                'approved_pending_provision',
                'provisioned_autoresearch_eligible',
                'disabled'
            )
        )
    );
    INSERT INTO public.research_lab_source_add_provisioning_events (
        provision_ref, catalog_id, submission_id, adapter_id, miner_hotkey,
        source_identity_hash, registry_provider_id, provision_status
    ) VALUES
        ('source_add_provision:test', 'source_catalog:test', 'submission:test',
         'adapter:test', 'miner:test', '', 'provider:test',
         'provisioned_autoresearch_eligible'),
        ('source_add_provision:disabled', 'source_catalog:test', 'submission:test',
         'adapter:disabled', 'miner:test', '', 'provider:test', 'disabled');
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
                    "FROM public.research_lab_source_add_provisioning_current"
                )
                assert cursor.fetchone()[0] == ["disabled", "provisioned"]
                cursor.execute(
                    "SELECT pg_get_functiondef('public.source_add_test_status_predicate(text)'::regprocedure)"
                )
                assert "provisioned_autoresearch_eligible" in cursor.fetchone()[0]
    finally:
        connection.close()
