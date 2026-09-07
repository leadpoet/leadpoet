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
    assert "IS NULL OR" in sql
    assert "CREATE OR REPLACE VIEW public.research_lab_source_add_provisioning_current" in sql
    assert "provision_ref/provenance" in sql
    assert "UPDATE public.research_lab_source_add_provisioning_events" not in sql
    assert "CREATE TABLE" not in sql
    assert "DROP TABLE" not in sql


def test_source_add_status_migration_rewrites_deployed_predicates():
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "LIKE '%provisioned_autoresearch_eligible%'" in sql
    assert "provisioned_autoresearch_eligible''" in sql
    assert "research_lab_source_add_provision_status_is_eligible_v1" not in sql
    assert "provision_status IN (''provisioned'', ''provisioned_autoresearch_eligible'')" in sql
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
                cursor.execute(
                    """
                    -- Seed an already-deployed append-only row.  The
                    -- production-origin trigger needs a full admission
                    -- history, which is outside this migration test.
                    SET session_replication_role = replica;
                    INSERT INTO public.research_lab_source_catalog (
                        catalog_id, adapter_id, miner_ref, source_name, source_kind,
                        declared_base_domains, registry_provider_id,
                        measured_trial_yield, catalog_doc
                    ) VALUES (
                        'source_catalog:1111111111111111',
                        'adapter:status-migration-test', '5StatusMigrationMiner',
                        'Status migration test', 'registry', '["status.test"]'::JSONB,
                        'status-migration', 0, '{"immutable":true}'::JSONB
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_source_add_provisioning_events (
                        provision_ref, catalog_id, submission_id, adapter_id,
                        miner_hotkey, source_identity_hash, registry_provider_id,
                        provision_status, seq, provision_doc, credential_envelope
                    ) VALUES (
                        'source_add_provision:1111111111111111',
                        'source_catalog:1111111111111111',
                        'source_add_submission:1111111111111111',
                        'adapter:status-migration-test', '5StatusMigrationMiner',
                        'sha256:' || repeat('1', 64), 'status-migration',
                        'provisioned_autoresearch_eligible', 0,
                        '{"immutable":"provision-doc"}'::JSONB,
                        '{"immutable":"credential-envelope"}'::JSONB
                        )
                    """
                )
                cursor.execute("SET session_replication_role = origin")
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
                cursor.execute(
                    """
                    SELECT provision_ref, provision_status, provision_doc,
                           credential_envelope
                    FROM public.research_lab_source_add_provisioning_events
                    WHERE adapter_id = 'adapter:status-migration-test'
                    """
                )
                assert cursor.fetchone() == (
                    "source_add_provision:1111111111111111",
                    "provisioned_autoresearch_eligible",
                    {"immutable": "provision-doc"},
                    {"immutable": "credential-envelope"},
                )
                cursor.execute(
                    """
                    SELECT provision_ref, provision_status, provision_doc,
                           credential_envelope
                    FROM public.research_lab_source_add_provisioning_current
                    WHERE adapter_id = 'adapter:status-migration-test'
                    """
                )
                assert cursor.fetchone() == (
                    "source_add_provision:1111111111111111",
                    "provisioned",
                    {"immutable": "provision-doc"},
                    {"immutable": "credential-envelope"},
                )
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_source_add_work_items (
                        work_id, submission_id, adapter_id, work_kind, work_status,
                        attempt_count, lease_token, leased_by, job_doc
                    ) VALUES (
                        'source_add_work:1111111111111111',
                        'source_add_submission:1111111111111111',
                        'adapter:status-migration-test', 'provisioning_smoke',
                        'leased', 0, '11111111-1111-1111-1111-111111111111',
                        'status-migration-test',
                        jsonb_build_object(
                            'config_ref', 'source_add_probe_config:1111111111111111',
                            'host_hash', 'sha256:' || repeat('a', 64),
                            'catalog_row', '{}'::JSONB,
                            'provision_row', jsonb_build_object(
                                'provision_ref', 'source_add_provision:1111111111111111'
                            )
                        )
                    )
                    """
                )
                with pytest.raises(psycopg2.errors.RaiseException, match="binding differs"):
                    cursor.execute(
                        """
                        SELECT public.research_lab_source_add_finalize_provision_smoke_v3(
                            'source_add_work:1111111111111111',
                            '11111111-1111-1111-1111-111111111111',
                            'source_add_submission:1111111111111111',
                            '{}'::JSONB,
                            '{"provision_status":"provisioned"}'::JSONB,
                            '{"work_id":"source_add_work:1111111111111111",'
                            '"attempt_number":0}'::JSONB
                        )
                        """
                    )
                connection.rollback()
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_work_items
                    WHERE work_id = 'source_add_work:1111111111111111'
                      AND work_status = 'leased'
                    """
                )
                assert cursor.fetchone()[0] == 1
                cursor.execute(
                    "SELECT public.research_lab_source_add_post_accept_leg1_contract_v4()"
                )
                assert cursor.fetchone()[0]["function_authority_sha256"] == (
                    "sha256:f17fab75262f612bf6aa5ca1dc4cb7dfe60d08f4b4cbf7b95fa5e7ea28084fb3"
                )
                cursor.execute(
                    "SELECT array_agg(provision_status ORDER BY provision_status) "
                    "FROM public.research_lab_source_add_provisioning_current"
                )
                assert cursor.fetchone()[0] == ["provisioned"]
                cursor.execute(
                    "SELECT pg_get_functiondef('public.source_add_test_status_predicate(text)'::regprocedure)"
                )
                assert "provisioned_autoresearch_eligible" in cursor.fetchone()[0]
    finally:
        connection.close()
