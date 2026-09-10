from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS as ARENA_MIGRATIONS,
    _DAILY_SOURCE_SHIM_SQL,
)
from tests.postgres_migration_harness import (
    HISTORICAL_SOURCE_ADD_UPGRADE_MIGRATIONS,
    _database_with_migrations,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "198-retire-research-lab-source-add-schema.sql"


def test_source_add_retirement_is_narrow_and_idempotent() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "CASCADE" not in "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    ).upper()
    assert "DROP TABLE IF EXISTS public.research_lab_source_add_submissions" in sql
    assert "DROP TABLE IF EXISTS public.research_lab_source_add_work_items" in sql
    assert "DROP VIEW IF EXISTS public.research_lab_source_add_submission_current" in sql
    assert "pg_get_function_identity_arguments" in sql


def test_source_add_retirement_removes_reward_schema_but_preserves_replay() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    retired = (
        "research_lab_source_catalog",
        "research_lab_source_add_reward_obligations",
        "research_lab_source_add_reward_events",
        "research_lab_source_add_reward_current",
    )
    for relation in retired:
        assert (
            f"DROP TABLE IF EXISTS public.{relation}" in sql
            or f"DROP VIEW IF EXISTS public.{relation}" in sql
        )

    assert "source_add_alpha_percent" in sql
    assert "to_regclass('public.lab_arena_submissions')" in sql


def test_source_add_retirement_does_not_rewrite_applied_migrations() -> None:
    source_add_migrations = sorted(ROOT.glob("scripts/*source-add*.sql"))

    assert MIGRATION in source_add_migrations
    assert (ROOT / "scripts" / "72-research-lab-source-experiments.sql").exists()
    assert any(path.name.startswith("186-") for path in source_add_migrations)


def test_historical_upgrade_retires_source_add_and_preserves_arena() -> None:
    generator = _database_with_migrations(
        HISTORICAL_SOURCE_ADD_UPGRADE_MIGRATIONS
        + tuple(name for name in ARENA_MIGRATIONS if not name.startswith("203-")),
        setup_sql=_DAILY_SOURCE_SHIM_SQL,
    )
    psycopg2, dsn = next(generator)
    try:
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_attested_boot_identities_v2 (
                        boot_identity_hash, schema_version, role, physical_role,
                        commit_sha, pcr0, build_manifest_hash,
                        dependency_lock_hash, config_hash, signing_pubkey,
                        transport_pubkey, transport_certificate_hash, boot_nonce,
                        attestation_user_data_hash, attestation_document_ref,
                        attestation_document_hash, identity_doc, issued_at
                    ) VALUES (
                        'sha256:' || repeat('1', 64),
                        'leadpoet.attested_boot_identity.v2',
                        'gateway_coordinator', 'gateway_coordinator',
                        repeat('a', 40), repeat('b', 96),
                        'sha256:' || repeat('2', 64),
                        'sha256:' || repeat('3', 64),
                        'sha256:' || repeat('4', 64), repeat('5', 64),
                        repeat('6', 64), 'sha256:' || repeat('7', 64),
                        repeat('8', 32), 'sha256:' || repeat('9', 64),
                        'attestation:historical', 'sha256:' || repeat('a', 64),
                        '{}'::jsonb, now()
                    );
                    INSERT INTO public.research_lab_attested_execution_receipts_v2 (
                        receipt_hash, schema_version, role, purpose, job_id,
                        epoch_id, sequence, commit_sha, pcr0,
                        build_manifest_hash, dependency_lock_hash, config_hash,
                        boot_identity_hash, input_root, output_root,
                        transport_root, host_operation_root, artifact_root,
                        receipt_status, enclave_pubkey, enclave_signature,
                        receipt_doc, issued_at
                    ) VALUES (
                        'sha256:' || repeat('b', 64),
                        'leadpoet.attested_execution_receipt.v2',
                        'gateway_coordinator',
                        'research_lab.source_add_reward_input.v2',
                        'historical-source-reward', 1, 1, repeat('a', 40),
                        repeat('b', 96), 'sha256:' || repeat('2', 64),
                        'sha256:' || repeat('3', 64),
                        'sha256:' || repeat('4', 64),
                        'sha256:' || repeat('1', 64),
                        'sha256:' || repeat('c', 64),
                        'sha256:' || repeat('d', 64),
                        'sha256:' || repeat('e', 64),
                        'sha256:' || repeat('f', 64),
                        'sha256:' || repeat('0', 64), 'succeeded',
                        repeat('1', 64), repeat('2', 128), '{}'::jsonb, now()
                    );
                    INSERT INTO public.research_lab_attested_execution_results_v2 (
                        receipt_hash, schema_version, role, operation, purpose,
                        job_id, epoch_id, sequence, release_hash, input_root,
                        output_root, artifact_root, result_hash,
                        artifact_hashes, result_doc
                    ) VALUES (
                        'sha256:' || repeat('b', 64),
                        'leadpoet.attested_execution_result.v2',
                        'gateway_coordinator', 'attest_weight_input',
                        'research_lab.source_add_reward_input.v2',
                        'historical-source-reward', 1, 1,
                        'sha256:' || repeat('1', 64),
                        'sha256:' || repeat('c', 64),
                        'sha256:' || repeat('d', 64),
                        'sha256:' || repeat('0', 64),
                        'sha256:' || repeat('3', 64), '[]'::jsonb, '{}'::jsonb
                    );
                    """
                )
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
                # A second application proves the cleanup is idempotent on the
                # deployed-history upgrade path.
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM public.research_lab_attested_execution_results_v2
                    WHERE purpose = 'research_lab.source_add_reward_input.v2'
                    """
                )
                assert cursor.fetchone() == (1,)
                with pytest.raises(psycopg2.errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO public.research_lab_attested_execution_receipts_v2
                        SELECT 'sha256:' || repeat('4', 64), schema_version,
                               role, purpose, 'new-source-reward', epoch_id,
                               sequence + 1, commit_sha, pcr0,
                               build_manifest_hash, dependency_lock_hash,
                               config_hash, boot_identity_hash, input_root,
                               output_root, transport_root, host_operation_root,
                               artifact_root, receipt_status, failure_code,
                               enclave_pubkey, enclave_signature, receipt_doc,
                               issued_at, now()
                        FROM public.research_lab_attested_execution_receipts_v2
                        WHERE job_id = 'historical-source-reward'
                        """
                    )
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_attested_execution_receipts_v2
                    SELECT 'sha256:' || repeat('5', 64), schema_version,
                           role, 'research_lab.allocation.v2',
                           'new-generic-receipt', epoch_id, sequence + 2,
                           commit_sha, pcr0, build_manifest_hash,
                           dependency_lock_hash, config_hash,
                           boot_identity_hash, input_root, output_root,
                           transport_root, host_operation_root, artifact_root,
                           receipt_status, failure_code, enclave_pubkey,
                           enclave_signature, receipt_doc, issued_at, now()
                    FROM public.research_lab_attested_execution_receipts_v2
                    WHERE job_id = 'historical-source-reward'
                    """
                )
                with pytest.raises(psycopg2.errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO public.research_lab_attested_execution_results_v2
                        SELECT 'sha256:' || repeat('5', 64), schema_version,
                               role, 'attest_weight_input', purpose,
                               'new-source-result', epoch_id, sequence + 2,
                               release_hash, input_root, output_root,
                               artifact_root, 'sha256:' || repeat('6', 64),
                               artifact_hashes, result_doc, now()
                        FROM public.research_lab_attested_execution_results_v2
                        WHERE job_id = 'historical-source-reward'
                        """
                    )
                cursor.execute(
                    """
                    SELECT
                        count(*) FILTER (
                            WHERE c.relname LIKE 'research_lab_source_add%'
                               OR c.relname = 'research_lab_source_catalog'
                        ),
                        to_regclass('public.lab_arena_submissions') IS NOT NULL
                    FROM pg_catalog.pg_class c
                    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public'
                    """
                )
                assert cursor.fetchone() == (0, True)
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM pg_catalog.pg_proc p
                    JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = 'public'
                      AND (
                          p.proname LIKE 'research_lab_source_add%'
                          OR p.proname LIKE 'research_lab_source_catalog%'
                      )
                    """
                )
                assert cursor.fetchone() == (0,)
                cursor.execute(
                    """
                    SELECT count(*), bool_and(NOT convalidated)
                    FROM pg_catalog.pg_constraint
                    WHERE conname IN (
                        'research_lab_attested_receipts_source_add_retired_check',
                        'research_lab_attested_results_source_add_retired_check',
                        'research_lab_attested_transport_source_add_retired_check'
                    )
                    """
                )
                assert cursor.fetchone() == (3, True)
    finally:
        generator.close()
