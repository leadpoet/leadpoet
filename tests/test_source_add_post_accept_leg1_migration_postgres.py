"""Execute migration-168 upgrade preflights against nonempty N-1 state."""

from __future__ import annotations

import json
import threading
import time

import pytest

from tests.test_source_add_end_to_end_postgres import (
    PRE_ORIGIN_MIGRATIONS,
    SCRIPTS,
    database as base_database,
)


MIGRATION = SCRIPTS / "169-research-lab-source-add-post-accept-leg1.sql"
ORIGIN_MIGRATION = (
    SCRIPTS / "170-research-lab-source-add-provider-origin-uniqueness.sql"
)
PRE_MIGRATIONS = PRE_ORIGIN_MIGRATIONS[:-1]


def _install_test_extensions(cursor) -> None:
    cursor.execute(
        """
        CREATE SCHEMA IF NOT EXISTS extensions;
        CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
        """
    )


def _seed_n_minus_one_history(
    cursor,
    *,
    accepted_at: str,
    provisioned_at: str,
    intent_at: str,
    reward_at: str,
    mismatched_catalog: bool,
) -> None:
    submission_id = "source_add_submission:0000000000000167"
    adapter_id = "adapter:migration-167"
    miner_hotkey = "5Migration167Miner"
    catalog_id = "source_catalog:0000000000000167"
    other_catalog_id = "source_catalog:9999999999999167"
    identity_hash = "sha256:" + "1" * 64
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_submissions (
            submission_id, adapter_id, miner_hotkey, stage, seq,
            submission_doc, precheck_status, precheck_doc,
            source_identity_hash, source_identity_version, created_at
        ) VALUES (
            %s, %s, %s, 'accepted', 0, %s::JSONB,
            'provenance_precheck_passed', '{}'::JSONB,
            %s, 'v2', %s::TIMESTAMPTZ
        )
        """,
        (
            submission_id,
            adapter_id,
            miner_hotkey,
            json.dumps(
                {
                    "manifest": {
                        "source_name": "Migration 167 fixture",
                        "source_kind": "registry",
                        "declared_base_domains": ["api.migration-167.test"],
                    },
                    "source_metadata": {
                        "api_base_url": "https://api.migration-167.test/v1"
                    },
                }
            ),
            identity_hash,
            accepted_at,
        ),
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_catalog (
            catalog_id, adapter_id, miner_ref, source_name, source_kind,
            declared_base_domains, registry_provider_id,
            measured_trial_yield, accepted_at, catalog_doc,
            source_identity_hash
        ) VALUES (
            %s, %s, %s, 'Migration 167 fixture', 'registry',
            '["api.migration-167.test"]'::JSONB, 'migration_167_provider',
            0, %s::TIMESTAMPTZ, '{}'::JSONB, %s
        )
        """,
        (catalog_id, adapter_id, miner_hotkey, provisioned_at, identity_hash),
    )
    if mismatched_catalog:
        cursor.execute(
            """
            INSERT INTO public.research_lab_source_catalog (
                catalog_id, adapter_id, miner_ref, source_name, source_kind,
                declared_base_domains, registry_provider_id,
                measured_trial_yield, accepted_at, catalog_doc,
                source_identity_hash
            ) VALUES (
                %s, 'adapter:unrelated-167', '5Unrelated167',
                'Unrelated fixture', 'registry',
                '["api.unrelated-167.test"]'::JSONB,
                'unrelated_167_provider', 0, %s::TIMESTAMPTZ,
                '{}'::JSONB, %s
            )
            """,
            (other_catalog_id, provisioned_at, "sha256:" + "2" * 64),
        )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_functional_probe_attempts (
            attempt_ref, submission_id, adapter_id, work_id, attempt_number,
            evaluation_mode, config_ref, result_status, created_at
        ) VALUES (
            'source_add_probe_attempt:0000000000000167', %s, %s,
            'source_add_work:0000000000000167', 1,
            'provisioning_smoke', 'source_add_probe_config:0000000000000167',
            'passed', %s::TIMESTAMPTZ
        )
        """,
        (submission_id, adapter_id, provisioned_at),
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_provisioning_events (
            provision_ref, catalog_id, submission_id, adapter_id,
            miner_hotkey, source_identity_hash, registry_provider_id,
            provision_status, seq, provision_doc, credential_envelope,
            created_at
        ) VALUES (
            'source_add_provision:0000000000000167', %s, %s, %s, %s, %s,
            'migration_167_provider', 'provisioned_autoresearch_eligible', 0,
            '{}'::JSONB, '{}'::JSONB, %s::TIMESTAMPTZ
        )
        """,
        (
            catalog_id,
            submission_id,
            adapter_id,
            miner_hotkey,
            identity_hash,
            provisioned_at,
        ),
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_reward_intents (
            intent_id, submission_id, adapter_id, miner_hotkey, intent_status,
            functional_receipt_hash, business_artifact_hash, created_at
        ) VALUES (
            'source_add_reward_intent:0000000000000167', %s, %s, %s,
            'queued', %s, %s, %s::TIMESTAMPTZ
        )
        """,
        (
            submission_id,
            adapter_id,
            miner_hotkey,
            "sha256:" + "3" * 64,
            "sha256:" + "4" * 64,
            intent_at,
        ),
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_reward_obligations (
            reward_ref, adapter_id, catalog_id, miner_hotkey, leg,
            reward_kind, alpha_percent, reward_epochs, start_epoch,
            trigger_evidence_doc, created_at
        ) VALUES (
            'source_add_reward:0000000000000167', %s, %s, %s, 1,
            'source_acceptance', 1, 20, 100, '{}'::JSONB,
            %s::TIMESTAMPTZ
        )
        """,
        (
            adapter_id,
            other_catalog_id if mismatched_catalog else catalog_id,
            miner_hotkey,
            reward_at,
        ),
    )


@pytest.mark.parametrize(
    ("case_name", "accepted_at", "provisioned_at", "intent_at", "reward_at", "mismatched_catalog", "message"),
    (
        (
            "early_acceptance",
            "2026-08-01T00:00:00Z",
            "2026-08-02T00:00:00Z",
            "2026-08-03T00:00:00Z",
            "2026-08-04T00:00:00Z",
            False,
            "pre-final acceptance",
        ),
        (
            "early_obligation",
            "2026-08-02T00:00:00Z",
            "2026-08-02T00:00:00Z",
            "2026-08-02T00:00:00Z",
            "2026-08-01T00:00:00Z",
            False,
            "pre-accept Leg 1 obligation",
        ),
        (
            "wrong_catalog",
            "2026-08-02T00:00:00Z",
            "2026-08-02T00:00:00Z",
            "2026-08-02T00:00:00Z",
            "2026-08-03T00:00:00Z",
            True,
            "pre-accept Leg 1 obligation",
        ),
    ),
)
def test_migration_rejects_n_minus_one_preapproval_history(
    base_database,
    case_name,
    accepted_at,
    provisioned_at,
    intent_at,
    reward_at,
    mismatched_catalog,
    message,
):
    psycopg2, dsn = base_database
    database_name = "source_add_167_" + case_name
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        cursor.execute("CREATE DATABASE " + database_name)
    admin.close()

    case_dsn = {**dsn, "dbname": database_name}
    connection = psycopg2.connect(**case_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _install_test_extensions(cursor)
            cursor.execute(
                """
                CREATE TABLE public.research_lab_auto_research_loop_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type TEXT NOT NULL,
                    CONSTRAINT research_lab_auto_research_loop_events_event_type_check
                        CHECK (event_type = 'loop_started')
                )
                """
            )
            for migration in PRE_MIGRATIONS:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            _seed_n_minus_one_history(
                cursor,
                accepted_at=accepted_at,
                provisioned_at=provisioned_at,
                intent_at=intent_at,
                reward_at=reward_at,
                mismatched_catalog=mismatched_catalog,
            )
            with pytest.raises(psycopg2.Error, match=message):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT pg_catalog.to_regprocedure(%s)",
                (
                    "public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)",
                ),
            )
            assert cursor.fetchone()[0] is None
    finally:
        connection.close()
        admin = psycopg2.connect(**dsn)
        admin.autocommit = True
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        admin.close()


def test_migration_preserves_terminal_legacy_obligation_as_audit_history(
    base_database,
):
    psycopg2, dsn = base_database
    database_name = "source_add_167_terminal_history"
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        cursor.execute("CREATE DATABASE " + database_name)
    admin.close()

    case_dsn = {**dsn, "dbname": database_name}
    connection = psycopg2.connect(**case_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _install_test_extensions(cursor)
            cursor.execute(
                """
                CREATE TABLE public.research_lab_auto_research_loop_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type TEXT NOT NULL,
                    CONSTRAINT research_lab_auto_research_loop_events_event_type_check
                        CHECK (event_type = 'loop_started')
                )
                """
            )
            for migration in PRE_MIGRATIONS:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
                ("terminal fixture", "operator:terminal-fixture"),
            )
            terminal_submission = {
                "submission_id": "source_add_submission:0000000000000167",
                "adapter_id": "adapter:migration-167-terminal",
                "miner_hotkey": "5Migration167TerminalMiner",
                "credential_envelope": {},
                "manifest": {
                    "credential_policy": "no_credentials",
                    "credential_ref": "",
                    "source_name": "Migration 167 terminal fixture",
                    "source_kind": "registry",
                    "declared_base_domains": ["api.migration-167-terminal.test"],
                },
                "source_metadata": {
                    "api_base_url": "https://api.migration-167-terminal.test/v1",
                    "documentation_url": (
                        "https://docs.migration-167-terminal.test/api"
                    ),
                    "auth_type": "none",
                    "endpoint_examples": [
                        {
                            "method": "GET",
                            "path": "/records",
                            "purpose": "Return current registry records",
                            "example_query": "limit=1",
                        }
                    ],
                    "rate_limit_notes": "bounded",
                },
            }
            cursor.execute(
                """
                SELECT public.research_lab_source_add_admit(
                    %s::JSONB, %s, %s, %s, %s, 10, 20, 30
                )
                """,
                (
                    json.dumps(terminal_submission, sort_keys=True),
                    "sha256:" + "1" * 64,
                    "",
                    "",
                    "source_add_work:0000000000000167",
                ),
            )
            assert cursor.fetchone()[0]["status"] == "admitted"
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_reward_obligations (
                    reward_ref, adapter_id, catalog_id, miner_hotkey, leg,
                    reward_kind, alpha_percent, reward_epochs, start_epoch,
                    trigger_evidence_doc
                ) VALUES (
                    'source_add_reward:0000000000000167',
                    'adapter:migration-167-terminal', NULL,
                    '5Migration167TerminalMiner', 1, 'source_acceptance',
                    1, 20, 100, '{}'::JSONB
                )
                """
            )
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
                ("terminal migration", "operator:terminal-fixture"),
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_reward_events (
                    reward_ref, seq, reward_status, reason
                ) VALUES
                    ('source_add_reward:0000000000000167', 0, 'active',
                     'legacy_pre_accept_reward'),
                    ('source_add_reward:0000000000000167', 1,
                     'stopped_forward', 'legacy_reward_retired')
                """
            )

            cursor.execute(MIGRATION.read_text(encoding="utf-8"))

            cursor.execute(
                """
                SELECT current_reward_status
                FROM public.research_lab_source_add_reward_current
                WHERE reward_ref = 'source_add_reward:0000000000000167'
                """
            )
            assert cursor.fetchone() == ("stopped_forward",)
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*)
                     FROM public.research_lab_source_catalog
                     WHERE adapter_id = 'adapter:migration-167-terminal'),
                    (SELECT COUNT(*)
                     FROM public.research_lab_source_add_provisioning_events
                     WHERE adapter_id = 'adapter:migration-167-terminal'),
                    (SELECT COUNT(*)
                     FROM public.research_lab_source_add_submissions
                     WHERE adapter_id = 'adapter:migration-167-terminal'
                       AND stage = 'accepted')
                """
            )
            assert cursor.fetchone() == (0, 0, 0)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_reward_current
                WHERE reward_ref = 'source_add_reward:0000000000000167'
                  AND current_reward_status IN (
                      'active', 'queued', 'partially_paid'
                  )
                """
            )
            assert cursor.fetchone() == (0,)
            with pytest.raises(psycopg2.Error, match="stopped reward is terminal"):
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_source_add_reward_events (
                        reward_ref, seq, reward_status, reason
                    ) VALUES (
                        'source_add_reward:0000000000000167', 2, 'active',
                        'invalid_legacy_reactivation'
                    )
                    """
                )
            cursor.execute("ROLLBACK")
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(ORIGIN_MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                """
                SELECT current_reward_status, current_event_seq
                FROM public.research_lab_source_add_reward_current
                WHERE reward_ref = 'source_add_reward:0000000000000167'
                """
            )
            assert cursor.fetchone() == ("stopped_forward", 1)
            cursor.execute(
                """
                SELECT submission_id, reservation_status
                FROM public.research_lab_source_add_provider_origin_current
                WHERE adapter_id = 'adapter:migration-167-terminal'
                """
            )
            assert cursor.fetchone() == (
                terminal_submission["submission_id"],
                "reserved",
            )
            cursor.execute(
                "SELECT pg_catalog.to_regprocedure(%s)",
                (
                    "public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)",
                ),
            )
            assert cursor.fetchone()[0] is not None
    finally:
        connection.close()
        admin = psycopg2.connect(**dsn)
        admin.autocommit = True
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        admin.close()


def test_migration_rejects_reactivated_terminal_legacy_history(base_database):
    psycopg2, dsn = base_database
    database_name = "source_add_167_reactivated_terminal"
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        cursor.execute("CREATE DATABASE " + database_name)
    admin.close()

    case_dsn = {**dsn, "dbname": database_name}
    connection = psycopg2.connect(**case_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _install_test_extensions(cursor)
            cursor.execute(
                """
                CREATE TABLE public.research_lab_auto_research_loop_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type TEXT NOT NULL,
                    CONSTRAINT research_lab_auto_research_loop_events_event_type_check
                        CHECK (event_type = 'loop_started')
                )
                """
            )
            for migration in PRE_MIGRATIONS:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            _seed_n_minus_one_history(
                cursor,
                accepted_at="2026-08-01T00:00:00Z",
                provisioned_at="2026-08-01T00:00:00Z",
                intent_at="2026-08-01T00:00:00Z",
                reward_at="2026-08-02T00:00:00Z",
                mismatched_catalog=False,
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_reward_events (
                    reward_ref, seq, reward_status, reason
                ) VALUES
                    ('source_add_reward:0000000000000167', 0, 'active',
                     'legacy_reward_created'),
                    ('source_add_reward:0000000000000167', 1,
                     'stopped_forward', 'legacy_reward_retired'),
                    ('source_add_reward:0000000000000167', 2, 'active',
                     'invalid_legacy_reactivation')
                """
            )
            with pytest.raises(
                psycopg2.Error,
                match="terminal history requires adjudication",
            ):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT pg_catalog.to_regprocedure(%s)",
                (
                    "public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)",
                ),
            )
            assert cursor.fetchone()[0] is None
    finally:
        connection.close()
        admin = psycopg2.connect(**dsn)
        admin.autocommit = True
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        admin.close()


def test_pre_migration_fixture_ends_immediately_before_167():
    assert PRE_ORIGIN_MIGRATIONS[-1] == MIGRATION.name
    assert PRE_MIGRATIONS[-1] == "145-research-lab-source-add-admission-control.sql"


def test_migration_reapplies_after_atomic_intent_precedes_final_approval(
    base_database,
):
    psycopg2, dsn = base_database
    database_name = "source_add_167_atomic_intent"
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        cursor.execute("CREATE DATABASE " + database_name)
    admin.close()

    case_dsn = {**dsn, "dbname": database_name}
    connection = psycopg2.connect(**case_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _install_test_extensions(cursor)
            cursor.execute(
                """
                CREATE TABLE public.research_lab_auto_research_loop_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type TEXT NOT NULL,
                    CONSTRAINT research_lab_auto_research_loop_events_event_type_check
                        CHECK (event_type = 'loop_started')
                )
                """
            )
            for migration in PRE_MIGRATIONS:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            _seed_n_minus_one_history(
                cursor,
                intent_at="2026-08-01T00:00:00Z",
                provisioned_at="2026-08-02T00:00:00Z",
                accepted_at="2026-08-03T00:00:00Z",
                reward_at="2026-08-04T00:00:00Z",
                mismatched_catalog=False,
            )
            migration_sql = MIGRATION.read_text(encoding="utf-8")
            cursor.execute(migration_sql)
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_provisioning_events (
                    provision_ref, catalog_id, submission_id, adapter_id,
                    miner_hotkey, source_identity_hash, registry_provider_id,
                    provision_status, seq, provision_doc, credential_envelope,
                    created_at
                ) VALUES (
                    'source_add_provision:0000000000001167',
                    'source_catalog:0000000000000167',
                    'source_add_submission:0000000000000167',
                    'adapter:migration-167', '5Migration167Miner', %s,
                    'migration_167_provider', 'disabled', 1,
                    '{}'::JSONB, '{}'::JSONB, '2026-08-05T00:00:00Z'
                )
                """,
                ("sha256:" + "1" * 64,),
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_functional_probe_attempts (
                    attempt_ref, submission_id, adapter_id, work_id,
                    attempt_number, evaluation_mode, config_ref,
                    result_status, created_at
                ) VALUES (
                    'source_add_probe_attempt:0000000000001167',
                    'source_add_submission:0000000000000167',
                    'adapter:migration-167',
                    'source_add_work:0000000000001167', 1,
                    'provisioning_smoke',
                    'source_add_probe_config:0000000000000167',
                    'failed', '2026-08-06T00:00:00Z'
                )
                """
            )
            cursor.execute(migration_sql)
            cursor.execute(
                "SELECT pg_catalog.to_regprocedure(%s)",
                (
                    "public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)",
                ),
            )
            assert cursor.fetchone()[0] is not None
    finally:
        connection.close()
        admin = psycopg2.connect(**dsn)
        admin.autocommit = True
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        admin.close()


def test_migration_rejects_claim_that_already_read_unpaused_state(base_database):
    psycopg2, dsn = base_database
    database_name = "source_add_167_claim_fence"
    work_id = "source_add_work:1671671671671671"
    host_hash = "sha256:" + "9" * 64
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        cursor.execute("CREATE DATABASE " + database_name)
    admin.close()

    case_dsn = {**dsn, "dbname": database_name}
    setup = psycopg2.connect(**case_dsn)
    setup.autocommit = True
    blocker = None
    observer = None
    claim_thread = None
    claim_started = threading.Event()
    claim_pid: list[int] = []
    claim_results: list[dict] = []
    claim_errors: list[BaseException] = []

    def claim_work() -> None:
        connection = psycopg2.connect(**case_dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                claim_pid.append(int(cursor.fetchone()[0]))
                claim_started.set()
                cursor.execute(
                    "SELECT public.research_lab_source_add_claim_work(%s, %s)",
                    ("worker:migration-167-race", 60),
                )
                claim_results.append(cursor.fetchone()[0])
        except BaseException as exc:  # surfaced in the parent test thread
            claim_errors.append(exc)
        finally:
            connection.close()

    try:
        with setup.cursor() as cursor:
            _install_test_extensions(cursor)
            cursor.execute(
                """
                CREATE TABLE public.research_lab_auto_research_loop_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type TEXT NOT NULL,
                    CONSTRAINT research_lab_auto_research_loop_events_event_type_check
                        CHECK (event_type = 'loop_started')
                )
                """
            )
            for migration in PRE_MIGRATIONS:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
                ("claim fence fixture", "operator:migration-167-race"),
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_work_items (
                    work_id, submission_id, adapter_id, work_kind,
                    work_status, priority, job_doc
                ) VALUES (
                    %s, 'source_add_submission:1671671671671671',
                    'adapter:migration-167-race', 'functional_probe',
                    'queued', 10, jsonb_build_object('host_hash', %s)
                )
                """,
                (work_id, host_hash),
            )

        blocker = psycopg2.connect(**case_dsn)
        with blocker.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                ("source-add-host:" + host_hash,),
            )

        claim_thread = threading.Thread(target=claim_work)
        claim_thread.start()
        assert claim_started.wait(timeout=5)
        observer = psycopg2.connect(**case_dsn)
        observer.autocommit = True
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with observer.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT wait_event_type, wait_event
                    FROM pg_stat_activity
                    WHERE pid = %s
                    """,
                    (claim_pid[0],),
                )
                wait_state = cursor.fetchone()
            if wait_state == ("Lock", "advisory"):
                break
            time.sleep(0.02)
        else:
            pytest.fail("claim did not reach the post-control advisory wait")

        with setup.cursor() as cursor:
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
                ("apply migration 168", "operator:migration-167-race"),
            )
            with pytest.raises(psycopg2.errors.LockNotAvailable):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT pg_catalog.to_regprocedure(%s)",
                (
                    "public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)",
                ),
            )
            assert cursor.fetchone()[0] is None

        blocker.commit()
        claim_thread.join(timeout=5)
        assert not claim_thread.is_alive()
        assert claim_errors == []
        assert [result["status"] for result in claim_results] == ["claimed"]
        with setup.cursor() as cursor:
            cursor.execute(
                """
                SELECT work_status, lease_token IS NOT NULL
                FROM public.research_lab_source_add_work_items
                WHERE work_id = %s
                """,
                (work_id,),
            )
            assert cursor.fetchone() == ("leased", True)
            with pytest.raises(psycopg2.Error, match="affected work is leased"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        if observer is not None:
            observer.close()
        if claim_thread is not None and claim_thread.is_alive():
            claim_thread.join(timeout=5)
        setup.close()
        admin = psycopg2.connect(**dsn)
        admin.autocommit = True
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            cursor.execute("DROP DATABASE IF EXISTS " + database_name)
        admin.close()
