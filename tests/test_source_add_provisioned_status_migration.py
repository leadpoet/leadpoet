from pathlib import Path

import pytest

from tests.test_source_add_end_to_end_postgres import (
    _database_with_migrations,
    _json,
    _scalar,
)
from tests.test_source_add_provenance_leg1_postgres import (
    PRE_MIGRATIONS,
    _claim_reward,
    _finalize_reward,
    _provision_after_leg1,
    _seed_boot_identity,
    _seed_case,
    _set_paused,
)
from leadpoet_canonical.attested_v2 import sha256_json

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


@pytest.fixture(scope="module")
def migration_database_after_status():
    yield from _database_with_migrations(
        PRE_MIGRATIONS
        + (
            "175-research-lab-source-add-provenance-leg1.sql",
            "176-research-lab-source-add-provenance-origin-repair.sql",
            "177-research-lab-source-add-provenance-authority-acl.sql",
            "178-research-lab-source-add-miner-status.sql",
            "186-research-lab-source-add-provisioned-status.sql",
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
                cursor.execute("BEGIN")
                _set_paused(cursor, False, "status migration test")
                _seed_boot_identity(cursor)
                case = _seed_case(cursor, 0x1860000000000000)
                smoke_work_id = (
                    "source_add_work:"
                    + sha256_json({"smoke": case["record"]["submission_id"]})[7:23]
                )
                rejection_sql, rejection_args = _provision_after_leg1(
                    cursor,
                    case,
                    reject_current_builtin=True,
                    allow_unrewarded=True,
                    stop_before_rpc=True,
                )
                assert rejection_sql
                cursor.execute(
                    """
                    SELECT job_doc->'catalog_row', job_doc->'provision_row'
                    FROM public.research_lab_source_add_work_items
                    WHERE work_id = %s
                    """,
                    (smoke_work_id,),
                )
                catalog_row, eligible_row = cursor.fetchone()
                eligible_row = dict(eligible_row)
                eligible_row["provision_ref"] = (
                    "source_add_provision:"
                    + sha256_json({"migration": case["record"]["submission_id"]})[7:23]
                )
                cursor.execute(
                    """
                    SELECT public.research_lab_source_add_finalize_provision_v3(
                        %s, %s::JSONB, %s::JSONB, %s::JSONB
                    )
                    """,
                    (
                        case["record"]["submission_id"],
                        _json(catalog_row),
                        _json(eligible_row),
                        _json(rejection_args[-1].adapted),
                    ),
                )
                assert cursor.fetchone()[0]["status"] == "provisioned"
                cursor.execute("SAVEPOINT before_status_rpc")
                assert cursor.execute(rejection_sql, rejection_args) is None
                assert cursor.fetchone()[0] == {"status": "not_eligible"}
                cursor.execute("ROLLBACK TO SAVEPOINT before_status_rpc")
                cursor.execute(
                    """
                    SELECT provision_ref, provision_status, provision_doc,
                           credential_envelope
                        FROM public.research_lab_source_add_provisioning_events
                        WHERE adapter_id = %s
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                    (case["record"]["adapter_id"],),
                )
                historical_row = cursor.fetchone()
                assert historical_row[1] == "provisioned_autoresearch_eligible"
                assert historical_row[2]["provider_registry_entry"]["active"] is True
                assert historical_row[3] == {}
                cursor.execute(
                    """
                    SELECT provision_ref, provision_status, provision_doc,
                           credential_envelope
                    FROM public.research_lab_source_add_provisioning_current
                    WHERE adapter_id = %s
                    """,
                    (case["record"]["adapter_id"],),
                )
                current_row = cursor.fetchone()
                assert current_row[1] == "provisioned"
                assert current_row[2] == historical_row[2]
                assert current_row[3] == historical_row[3]
                cursor.execute(
                    """
                    UPDATE public.research_lab_source_add_work_items
                    SET job_doc = job_doc #- '{provision_row,provision_status}'
                    WHERE work_id = %s
                    """,
                    (
                            smoke_work_id,
                    ),
                )
                cursor.execute(
                    """
                    SELECT to_jsonb(work),
                           (SELECT COUNT(*) FROM public.research_lab_source_add_provisioning_events),
                           (SELECT COUNT(*) FROM public.research_lab_source_add_submissions)
                    FROM public.research_lab_source_add_work_items work
                    WHERE work.work_id = %s
                    """,
                    (smoke_work_id,),
                )
                missing_status = cursor.fetchone()
                cursor.execute("SAVEPOINT before_missing_status_rpc")
                with pytest.raises(psycopg2.errors.RaiseException, match="current-provider smoke binding differs"):
                    cursor.execute(
                        rejection_sql, rejection_args
                    )
                cursor.execute("ROLLBACK TO SAVEPOINT before_missing_status_rpc")
                cursor.execute(
                    """
                    SELECT to_jsonb(work),
                           (SELECT COUNT(*) FROM public.research_lab_source_add_provisioning_events),
                           (SELECT COUNT(*) FROM public.research_lab_source_add_submissions)
                    FROM public.research_lab_source_add_work_items work
                    WHERE work.work_id = %s
                    """,
                    (smoke_work_id,),
                )
                assert cursor.fetchone() == missing_status
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


def test_source_add_status_migration_preserves_successful_leg1_provision(
    migration_database_after_status,
):
    psycopg2, dsn = migration_database_after_status
    connection = psycopg2.connect(**dsn)
    try:
        with connection:
            with connection.cursor() as cursor:
                _set_paused(cursor, False, "post-186 Leg 1 success")
                _seed_boot_identity(cursor)
                case = _seed_case(cursor, 0x1860000000000100)

                assert (
                    _provision_after_leg1(
                        cursor, case, allow_unrewarded=True
                    )
                    is None
                )
                reward_work = _claim_reward(cursor)
                finalized = _finalize_reward(
                    cursor, work=reward_work, case=case, caller_cap=1
                )
                assert finalized["status"] == "created"
                assert _scalar(
                    cursor,
                    "SELECT provision_status FROM public.research_lab_source_add_provisioning_current WHERE adapter_id=%s",
                    (case["record"]["adapter_id"],),
                ) == "provisioned"
                assert _scalar(
                    cursor,
                    "SELECT work_status FROM public.research_lab_source_add_work_items WHERE work_id=%s",
                    (reward_work["work_id"],),
                ) == "completed"
                assert _scalar(
                    cursor,
                    "SELECT count(*) FROM public.research_lab_source_add_reward_obligations WHERE adapter_id=%s AND leg=1",
                    (case["record"]["adapter_id"],),
                ) == 1
    finally:
        connection.close()
