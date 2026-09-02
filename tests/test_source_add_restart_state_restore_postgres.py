"""Exercise SOURCE_ADD restart-state restoration in disposable PostgreSQL."""

from __future__ import annotations

import hashlib
import json

import pytest

from gateway.research_lab.source_add_provenance import (
    sanitize_source_add_precheck_doc,
)
from gateway.tee.supabase_schema_preflight_v2 import (
    SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256,
)
from scripts import production_parity_snapshot as parity_snapshot
from tests.test_source_add_claim_control_postgres import (
    _insert_work,
)
from tests.test_source_add_end_to_end_postgres import (
    SCRIPTS,
    _database_with_migrations,
)
from tests.test_source_add_leg1_release_policy_postgres import (
    LATEST_MIGRATIONS as PRE_RESTORE_MIGRATIONS,
)


MIGRATION = "174-research-lab-source-add-restart-state-restore.sql"
MIGRATIONS = PRE_RESTORE_MIGRATIONS + (MIGRATION,)
PROVENANCE_LEG1_MIGRATION = (
    "175-research-lab-source-add-provenance-leg1.sql"
)
PROVENANCE_ORIGIN_REPAIR_MIGRATION = (
    "176-research-lab-source-add-provenance-origin-repair.sql"
)
PROVENANCE_AUTHORITY_ACL_MIGRATION = (
    "177-research-lab-source-add-provenance-authority-acl.sql"
)
ACL_MIGRATIONS = MIGRATIONS + (
    PROVENANCE_LEG1_MIGRATION,
    PROVENANCE_ORIGIN_REPAIR_MIGRATION,
    PROVENANCE_AUTHORITY_ACL_MIGRATION,
)
GUARD_A = "source_add_restart_guard:" + "a" * 64
GUARD_B = "source_add_restart_guard:" + "b" * 64
GUARD_C = "source_add_restart_guard:" + "c" * 64
OWNER_A = "source_add_restart_owner:" + "1" * 64
OWNER_B = "source_add_restart_owner:" + "2" * 64
OWNER_C = "source_add_restart_owner:" + "3" * 64


def _commitment(identity: str) -> str:
    return "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _owner_generation_commitment(owner_id: str, generation: int) -> str:
    value = f"{_commitment(owner_id)}:{generation}".encode("utf-8")
    return "sha256:" + hashlib.sha256(value).hexdigest()


@pytest.fixture(scope="module")
def pre_restore_database():
    yield from _database_with_migrations(PRE_RESTORE_MIGRATIONS)


@pytest.fixture(scope="module")
def schema_only_clone_database():
    yield from _database_with_migrations(PRE_RESTORE_MIGRATIONS)


@pytest.fixture(scope="module")
def schema_only_provenance_clone_database():
    yield from _database_with_migrations(MIGRATIONS)


@pytest.fixture(scope="module")
def database():
    yield from _database_with_migrations(MIGRATIONS)


@pytest.fixture(scope="module")
def acl_database():
    yield from _database_with_migrations(ACL_MIGRATIONS)


def _set_paused(cursor, paused: bool, reason: str, actor: str) -> dict:
    cursor.execute(
        "SELECT public.research_lab_source_add_set_paused(%s, %s, %s)",
        (paused, reason, actor),
    )
    return cursor.fetchone()[0]


def _state(cursor) -> dict:
    cursor.execute(
        "SELECT public.research_lab_source_add_restart_guard_state_v2()"
    )
    return cursor.fetchone()[0]


def _source_add_acl_contracts(cursor) -> dict:
    cursor.execute(
        """
        SELECT pg_catalog.json_build_object(
            'duplicate_privacy',
                public.research_lab_source_add_duplicate_privacy_contract_v1()
                    -> 'permissions',
            'post_accept_leg1',
                public.research_lab_source_add_post_accept_leg1_contract_v3()
                    -> 'permissions',
            'claim_control',
                public.research_lab_source_add_claim_control_contract_v2()
                    -> 'permissions'
        )
        """
    )
    return cursor.fetchone()[0]


def _source_add_function_acl(cursor, signature: str) -> dict[str, bool]:
    cursor.execute(
        """
        WITH function_row AS (
            SELECT function_catalog.*
            FROM pg_catalog.pg_proc AS function_catalog
            WHERE function_catalog.oid = pg_catalog.to_regprocedure(%s)
        )
        SELECT
            pg_catalog.has_function_privilege(
                'service_role', function_row.oid, 'EXECUTE'
            ),
            EXISTS (
                SELECT 1
                FROM pg_catalog.aclexplode(
                    COALESCE(
                        function_row.proacl,
                        pg_catalog.acldefault('f', function_row.proowner)
                    )
                ) AS privilege
                WHERE privilege.grantee = 0
                  AND privilege.privilege_type = 'EXECUTE'
            ),
            pg_catalog.has_function_privilege(
                'anon', function_row.oid, 'EXECUTE'
            ),
            pg_catalog.has_function_privilege(
                'authenticated', function_row.oid, 'EXECUTE'
            )
        FROM function_row
        """,
        (signature,),
    )
    row = cursor.fetchone()
    assert row is not None, signature
    return {
        "service_role_callable": row[0],
        "public_callable": row[1],
        "anon_callable": row[2],
        "authenticated_callable": row[3],
    }


def _assert_complete_source_add_acl(cursor) -> None:
    expected = parity_snapshot._schema_only_source_add_acl_expectations()
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM pg_catalog.pg_proc AS function_row
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = function_row.pronamespace
        WHERE namespace.nspname = 'public'
          AND (
                pg_catalog.strpos(function_row.proname, 'source_add') > 0
                OR function_row.proname =
                    'enforce_research_lab_source_catalog_provider_origin'
          )
        """
    )
    assert cursor.fetchone()[0] == len(expected)
    assert {
        signature: _source_add_function_acl(cursor, signature)
        for signature in expected
    } == expected


def _acquire(
    cursor,
    guard_id: str,
    owner_id: str,
    expected_generation: int,
    actor: str,
    *,
    lease_seconds: int = 300,
) -> dict:
    cursor.execute(
        """
        SELECT public.research_lab_source_add_acquire_restart_guard_v2(
            %s, %s, %s, %s, %s
        )
        """,
        (guard_id, owner_id, expected_generation, lease_seconds, actor),
    )
    return cursor.fetchone()[0]


def _release(
    cursor,
    guard_id: str,
    owner_id: str,
    generation: int,
    actor: str,
) -> dict:
    cursor.execute(
        """
        SELECT public.research_lab_source_add_release_restart_guard_v2(
            %s, %s, %s, %s
        )
        """,
        (guard_id, owner_id, generation, actor),
    )
    return cursor.fetchone()[0]


def _state_shape(
    *,
    paused: bool,
    generation: int,
    restore_paused: bool | None,
    guard_id: str = "",
    owner_id: str = "",
    expires_at=None,
) -> dict:
    return {
        "schema_version": "leadpoet.source_add_restart_guard_state.v2",
        "paused": paused,
        "guard_active": bool(guard_id),
        "guard_commitment": _commitment(guard_id) if guard_id else "",
        "owner_commitment": _commitment(owner_id) if owner_id else "",
        "guard_generation": generation,
        "owner_generation_commitment": (
            _owner_generation_commitment(owner_id, generation)
            if owner_id
            else ""
        ),
        "guard_expires_at": expires_at,
        "restore_paused": restore_paused,
    }


def _reserve_provider_origin(
    cursor,
    *,
    submission_id: str,
    adapter_id: str,
    miner_hotkey: str,
    api_base_url: str,
) -> None:
    cursor.execute(
        "SELECT public.research_lab_source_add_provider_origin_hash_v1(%s)",
        (api_base_url,),
    )
    provider_origin_hash = cursor.fetchone()[0]
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_provider_origin_events (
            origin_version, provider_origin_hash, submission_id, adapter_id,
            miner_hotkey, reservation_status, seq, reason
        ) VALUES ('v1', %s, %s, %s, %s, 'reserved', 0, %s)
        """,
        (
            provider_origin_hash,
            submission_id,
            adapter_id,
            miner_hotkey,
            "restart_state_restore_credential_constraint_test",
        ),
    )


def test_migration_rejects_active_guarded_or_leased_state(
    pre_restore_database,
) -> None:
    psycopg2, dsn = pre_restore_database
    migration_sql = (SCRIPTS / MIGRATION).read_text(encoding="utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _set_paused(
                cursor,
                False,
                "restart restore active precondition",
                "operator:restart-restore-test",
            )
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD must be paused before restart-state migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")

            _set_paused(
                cursor,
                True,
                "restart restore guard precondition",
                "operator:restart-restore-test",
            )
            cursor.execute(
                "SELECT public.research_lab_source_add_restart_guard_state_v1()"
            )
            generation = cursor.fetchone()[0]["guard_generation"]
            cursor.execute(
                """
                SELECT public.research_lab_source_add_acquire_restart_guard_v1(
                    %s, %s, %s, 300, %s
                )
                """,
                (GUARD_A, OWNER_A, generation, "operator:pre-migration-guard"),
            )
            acquired = cursor.fetchone()[0]
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD restart guard is active during restart-state migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")
            cursor.execute(
                """
                SELECT public.research_lab_source_add_release_restart_guard_v1(
                    %s, %s, %s, %s
                )
                """,
                (
                    GUARD_A,
                    OWNER_A,
                    acquired["guard_generation"],
                    "operator:pre-migration-guard",
                ),
            )

            _insert_work(cursor, suffix="1740000000000001", status="leased")
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD work is leased during restart-state migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")
            cursor.execute(
                "DELETE FROM public.research_lab_source_add_work_items "
                "WHERE work_id = %s",
                ("source_add_work:1740000000000001",),
            )
    finally:
        connection.close()


def test_schema_only_parity_stages_paused_empty_clone_before_migration(
    schema_only_clone_database,
) -> None:
    psycopg2, dsn = schema_only_clone_database
    migration_path = SCRIPTS / MIGRATION
    migration_identity = {
        "path": parity_snapshot._SOURCE_ADD_RESTART_STATE_MIGRATION,
        "sequence": 174,
        "sha256": parity_snapshot.file_sha256(migration_path),
        "transaction_mode": "candidate-file",
    }
    staging_sql = parity_snapshot._schema_only_source_add_maintenance_sql(
        migration_identity
    ).decode("utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT public.research_lab_source_add_duplicate_privacy_contract_v1()
                    -> 'permissions'
                """
            )
            source_duplicate_permissions = cursor.fetchone()[0]
            assert source_duplicate_permissions["anon_callable"] is False
            assert source_duplicate_permissions["authenticated_callable"] is False

            with pytest.raises(
                psycopg2.Error,
                match="schema-only SOURCE_ADD control state is not empty",
            ):
                cursor.execute(staging_sql)
            cursor.execute("ROLLBACK")
            cursor.execute(
                "DELETE FROM public.research_lab_source_add_control WHERE singleton"
            )
            cursor.execute(
                "SELECT COUNT(*) FROM public.research_lab_source_add_work_items"
            )
            assert cursor.fetchone()[0] == 0

            # pg_dump/pg_restore --no-acl recreates PostgreSQL's default
            # PUBLIC EXECUTE privilege; the parity role bootstrap also grants
            # service_role every function before this clone-only repair.
            cursor.execute(
                "GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO PUBLIC"
            )
            cursor.execute(
                """
                SELECT public.research_lab_source_add_duplicate_privacy_contract_v1()
                    -> 'permissions'
                """
            )
            leaked_duplicate_permissions = cursor.fetchone()[0]
            assert leaked_duplicate_permissions["anon_callable"] is True
            assert leaked_duplicate_permissions["authenticated_callable"] is True

            cursor.execute(staging_sql)
            cursor.execute(
                """
                SELECT paused, reason, actor_ref,
                       restart_guard_commitment = '',
                       restart_guard_owner_commitment = '',
                       restart_guard_generation,
                       restart_guard_expires_at IS NULL,
                       restart_guard_acquired_at IS NULL,
                       restart_guard_actor_ref = ''
                FROM public.research_lab_source_add_control
                WHERE singleton
                """
            )
            assert cursor.fetchone() == (
                True,
                parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_REASON,
                parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_ACTOR,
                True,
                True,
                0,
                True,
                True,
                True,
            )

            cursor.execute(migration_path.read_text(encoding="utf-8"))
            cursor.execute(
                (SCRIPTS / PROVENANCE_LEG1_MIGRATION).read_text(
                    encoding="utf-8"
                )
            )
            cursor.execute(
                (SCRIPTS / PROVENANCE_ORIGIN_REPAIR_MIGRATION).read_text(
                    encoding="utf-8"
                )
            )
            cursor.execute(
                (SCRIPTS / PROVENANCE_AUTHORITY_ACL_MIGRATION).read_text(
                    encoding="utf-8"
                )
            )
            cursor.execute(
                "GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role"
            )
            cursor.execute(
                "GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO PUBLIC"
            )
            for leaked_signature in (
                "public.research_lab_source_add_finish_work"
                "(text,uuid,text,text,jsonb,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb,"
                "timestamp with time zone,boolean)",
                "public.research_lab_source_add_requeue_provenance_v2"
                "(text,text,text,text,text,text,text)",
                "public.enforce_research_lab_source_add_leg1_obligation_v2()",
            ):
                assert _source_add_function_acl(cursor, leaked_signature) == {
                    "service_role_callable": True,
                    "public_callable": True,
                    "anon_callable": True,
                    "authenticated_callable": True,
                }
            cursor.execute(
                parity_snapshot._schema_only_source_add_acl_sql(
                    parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS
                ).decode("utf-8")
            )
            _assert_complete_source_add_acl(cursor)
            assert _source_add_function_acl(
                cursor,
                "public.research_lab_source_add_finish_work"
                "(text,uuid,text,text,jsonb,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb,"
                "timestamp with time zone,boolean)",
            )["service_role_callable"] is True
            assert _source_add_function_acl(
                cursor,
                "public.enforce_research_lab_source_add_leg1_obligation_v2()",
            ) == {
                "service_role_callable": False,
                "public_callable": False,
                "anon_callable": False,
                "authenticated_callable": False,
            }
            assert _source_add_function_acl(
                cursor,
                "public.prevent_research_lab_source_add_reward_mutation()",
            ) == {
                "service_role_callable": True,
                "public_callable": True,
                "anon_callable": True,
                "authenticated_callable": True,
            }
            repaired_contracts = _source_add_acl_contracts(cursor)
            assert repaired_contracts["duplicate_privacy"] == (
                source_duplicate_permissions
            )
            assert repaired_contracts["post_accept_leg1"] == {
                "service_role_exists": True,
                "candidate_callable": True,
                "rollback_v2_callable": True,
                "internal_not_callable": True,
            }
            assert repaired_contracts["claim_control"] == {
                "service_role_exists": True,
                "service_role_callable": True,
                "anon_callable": False,
                "authenticated_callable": False,
            }
            assert _state(cursor) == _state_shape(
                paused=True,
                generation=0,
                restore_paused=None,
            )
    finally:
        connection.close()


def test_schema_only_parity_pauses_exact_174_clone_before_migration_175(
    schema_only_provenance_clone_database,
) -> None:
    psycopg2, dsn = schema_only_provenance_clone_database
    migration_path = SCRIPTS / PROVENANCE_LEG1_MIGRATION
    migration_identity = next(
        dict(migration)
        for migration in parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_CUTOVER_MIGRATIONS
        if migration["path"] == parity_snapshot._SOURCE_ADD_PROVENANCE_LEG1_MIGRATION
    )
    staging_sql = parity_snapshot._schema_only_source_add_maintenance_sql(
        migration_identity
    ).decode("utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM public.research_lab_source_add_control WHERE singleton"
            )
            cursor.execute(
                "SELECT COUNT(*) FROM public.research_lab_source_add_work_items"
            )
            assert cursor.fetchone()[0] == 0

            cursor.execute(staging_sql)
            cursor.execute(
                """
                SELECT paused, reason, actor_ref,
                       restart_guard_commitment = '',
                       restart_guard_owner_commitment = '',
                       restart_guard_generation,
                       restart_guard_restore_paused IS NULL
                FROM public.research_lab_source_add_control
                WHERE singleton
                """
            )
            assert cursor.fetchone() == (
                True,
                parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_REASON,
                parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_ACTOR,
                True,
                True,
                0,
                True,
            )

            cursor.execute(migration_path.read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT public.research_lab_source_add_post_accept_leg1_contract_v3()"
            )
            contract = cursor.fetchone()[0]
            assert contract["schema_version"] == (
                "leadpoet.source_add_post_accept_leg1_contract.v3"
            )
            assert contract["daily_cap"] == 50
            assert contract["leg1_alpha_percent"] == 0.2
            assert contract["leg1_reward_epochs"] == 20
            assert _state(cursor) == _state_shape(
                paused=True,
                generation=0,
                restore_paused=None,
            )
    finally:
        connection.close()


def test_schema_only_acl_repair_rejects_unreviewed_function_surface(
    acl_database,
) -> None:
    psycopg2, dsn = acl_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE FUNCTION public.research_lab_source_add_unreviewed()
                RETURNS VOID
                LANGUAGE plpgsql
                AS 'BEGIN RETURN; END'
                """
            )
            with pytest.raises(
                psycopg2.Error,
                match="schema-only SOURCE_ADD ACL function inventory differs",
            ):
                cursor.execute(
                    parity_snapshot._schema_only_source_add_acl_sql(
                        parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS
                    ).decode("utf-8")
                )
            cursor.execute("ROLLBACK")
            cursor.execute(
                "DROP FUNCTION public.research_lab_source_add_unreviewed()"
            )
            _assert_complete_source_add_acl(cursor)
    finally:
        connection.close()


def test_contract_acl_and_migration_are_idempotent(database) -> None:
    psycopg2, dsn = database
    migration_sql = (SCRIPTS / MIGRATION).read_text(encoding="utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            initial_state = _state(cursor)
            assert initial_state == _state_shape(
                paused=True,
                generation=0,
                restore_paused=None,
            )
            cursor.execute(
                "SELECT public.research_lab_source_add_claim_control_contract_v2()"
            )
            initial_contract = cursor.fetchone()[0]
            cursor.execute(
                "SELECT public.research_lab_source_add_claim_control_contract_v1()"
            )
            rollback_v1_contract = cursor.fetchone()[0]
            cursor.execute(
                """
                SELECT 'sha256:' || encode(
                    extensions.digest(
                        convert_to(
                            public.research_lab_source_add_claim_control_contract_v1()::TEXT,
                            'UTF8'
                        ),
                        'sha256'
                    ),
                    'hex'
                )
                """
            )
            expected_rollback_hash = cursor.fetchone()[0]
            assert initial_contract["rollback_v1_contract_schema_version"] == (
                rollback_v1_contract["schema_version"]
            )
            assert initial_contract["rollback_v1_contract_sha256"] == (
                expected_rollback_hash
            )
            assert initial_contract["function_authority_sha256"] == (
                SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256
            )
            assert initial_contract["acquire_captures_pre_restart_paused"] is True
            assert initial_contract["renewal_preserves_restore_state"] is True
            assert initial_contract["expired_takeover_preserves_restore_state"] is True
            assert initial_contract["operator_pause_wins"] is True
            assert initial_contract["release_restores_pre_restart_state"] is True
            assert initial_contract["failed_restart_keeps_paused"] is True
            assert all(initial_contract["functions"].values())
            assert initial_contract["permissions"] == {
                "service_role_exists": True,
                "service_role_callable": True,
                "anon_callable": False,
                "authenticated_callable": False,
            }

            cursor.execute(migration_sql)
            cursor.execute(
                "SELECT public.research_lab_source_add_claim_control_contract_v2()"
            )
            assert cursor.fetchone()[0] == initial_contract
            assert _state(cursor) == initial_state
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'research_lab_source_add_control'
                  AND column_name = 'restart_guard_restore_paused'
                """
            )
            assert cursor.fetchone()[0] == 1
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM pg_catalog.pg_trigger
                WHERE tgname = 'trg_source_add_restart_restore_pause_v2'
                  AND NOT tgisinternal
                """
            )
            assert cursor.fetchone()[0] == 1
    finally:
        connection.close()


def test_active_before_restart_is_restored_active_after_exact_release(database) -> None:
    psycopg2, dsn = database
    actor = "gateway-restart:active-restore"
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _set_paused(
                cursor,
                False,
                "operator source add active",
                "operator:source-add-active",
            )
            generation = _state(cursor)["guard_generation"]
            acquired = _acquire(cursor, GUARD_A, OWNER_A, generation, actor)
            assert acquired["schema_version"] == (
                "leadpoet.source_add_restart_guard.v2"
            )
            assert acquired["paused"] is True
            assert acquired["restore_paused"] is False
            assert acquired["guard_generation"] == generation + 1
            released = _release(
                cursor,
                GUARD_A,
                OWNER_A,
                acquired["guard_generation"],
                actor,
            )
            assert released == {
                "schema_version": (
                    "leadpoet.source_add_restart_guard_release.v2"
                ),
                "released": True,
                "paused": False,
                "guard_active": False,
                "guard_generation": acquired["guard_generation"],
                "owner_generation_commitment": (
                    _owner_generation_commitment(
                        OWNER_A, acquired["guard_generation"]
                    )
                ),
                "restored_pre_restart_state": True,
            }
            state = _state(cursor)
            assert state == _state_shape(
                paused=False,
                generation=acquired["guard_generation"],
                restore_paused=None,
            )
            _set_paused(
                cursor,
                True,
                "active restoration test cleanup",
                "operator:restart-restore-test",
            )
    finally:
        connection.close()


def test_paused_before_restart_remains_paused_after_exact_release(database) -> None:
    psycopg2, dsn = database
    actor = "gateway-restart:paused-restore"
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _set_paused(
                cursor,
                True,
                "operator source add paused",
                "operator:source-add-paused",
            )
            generation = _state(cursor)["guard_generation"]
            acquired = _acquire(cursor, GUARD_B, OWNER_B, generation, actor)
            assert acquired["restore_paused"] is True
            released = _release(
                cursor,
                GUARD_B,
                OWNER_B,
                acquired["guard_generation"],
                actor,
            )
            assert released["paused"] is True
            assert released["restored_pre_restart_state"] is True
            assert _state(cursor)["paused"] is True
            assert _state(cursor)["restore_paused"] is None
    finally:
        connection.close()


def test_renewal_and_expired_takeover_preserve_active_restore_state(database) -> None:
    psycopg2, dsn = database
    first_actor = "gateway-restart:renewal"
    takeover_actor = "gateway-restart:takeover"
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _set_paused(
                cursor,
                False,
                "operator source add active before renewal",
                "operator:source-add-active",
            )
            generation = _state(cursor)["guard_generation"]
            acquired = _acquire(
                cursor,
                GUARD_A,
                OWNER_A,
                generation,
                first_actor,
                lease_seconds=60,
            )
            renewed = _acquire(
                cursor,
                GUARD_A,
                OWNER_A,
                acquired["guard_generation"],
                first_actor,
                lease_seconds=300,
            )
            assert renewed["guard_generation"] == acquired["guard_generation"]
            assert renewed["restore_paused"] is False
            assert renewed["guard_expires_at"] >= acquired["guard_expires_at"]

            cursor.execute(
                """
                UPDATE public.research_lab_source_add_control
                SET restart_guard_expires_at = NOW() - INTERVAL '1 second'
                WHERE singleton
                """
            )
            takeover = _acquire(
                cursor,
                GUARD_C,
                OWNER_C,
                renewed["guard_generation"],
                takeover_actor,
            )
            assert takeover["guard_generation"] == renewed["guard_generation"] + 1
            assert takeover["restore_paused"] is False
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD restart guard owner or generation does not match",
            ):
                _release(
                    cursor,
                    GUARD_A,
                    OWNER_A,
                    renewed["guard_generation"],
                    first_actor,
                )
            current = _state(cursor)
            assert current["guard_commitment"] == _commitment(GUARD_C)
            assert current["owner_commitment"] == _commitment(OWNER_C)
            assert current["restore_paused"] is False
            released = _release(
                cursor,
                GUARD_C,
                OWNER_C,
                takeover["guard_generation"],
                takeover_actor,
            )
            assert released["paused"] is False
            _set_paused(
                cursor,
                True,
                "takeover restoration test cleanup",
                "operator:restart-restore-test",
            )
    finally:
        connection.close()


def test_explicit_operator_pause_while_guarded_wins(database) -> None:
    psycopg2, dsn = database
    restart_actor = "gateway-restart:operator-pause"
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _set_paused(
                cursor,
                False,
                "operator source add active before restart",
                "operator:source-add-active",
            )
            generation = _state(cursor)["guard_generation"]
            acquired = _acquire(
                cursor, GUARD_B, OWNER_B, generation, restart_actor
            )
            assert acquired["restore_paused"] is False
            paused = _set_paused(
                cursor,
                True,
                "operator incident pause",
                "operator:incident-response",
            )
            assert paused["paused"] is True
            assert paused["restart_guard_restore_paused"] is True
            assert _state(cursor)["restore_paused"] is True
            released = _release(
                cursor,
                GUARD_B,
                OWNER_B,
                acquired["guard_generation"],
                restart_actor,
            )
            assert released["paused"] is True
            assert _state(cursor)["paused"] is True
    finally:
        connection.close()


@pytest.mark.parametrize(
    "submission_doc",
    [
        {
            "endpoint_example": (
                "https://api.builtwith.com/v21/api.json?"
                "KEY=fake-builtwith-value&LOOKUP=example.com"
            )
        },
        {"request": {"api_key": "fake-api-key-value"}},
        {"headers": {"Authorization": "Bearer fake-bearer-value"}},
        {"headers": {"Authorization": "API 00000000-0000-0000-0000-000000000000"}},
        {"headers": {"X-RapidAPI-Key": "fake-rapid-api-key-value"}},
        {"headers": {"xApiKey": "fake-camel-api-key-value"}},
        {"headers": {"xRapidApiKey": "fake-camel-rapid-api-key-value"}},
        {"request": {"clientSecret": "fake-client-secret-value"}},
        {"request": {"apiToken": "fake-api-token-value"}},
        {"request": {"clientToken": "fake-client-token-value"}},
        {"request": {"providerKey": "fake-provider-key-value"}},
        {"headers": {"X-Custom-Auth": "fake-custom-auth-value"}},
        {"request": {"credential": "fake-credential-value"}},
        {"request": {"clientCredentials": "fake-client-credentials-value"}},
        {"request": {"privateKey": "fake-private-key-value"}},
    ],
    ids=(
        "builtwith-key-query",
        "nested-api-key",
        "authorization-header",
        "builtwith-authorization-header",
        "rapidapi-header",
        "camel-api-header",
        "camel-rapidapi-header",
        "camel-client-secret",
        "camel-api-token",
        "camel-client-token",
        "camel-provider-key",
        "custom-auth-header",
        "credential-field",
        "camel-client-credentials",
        "camel-private-key",
    ),
)
def test_submission_constraint_rejects_credential_material(
    database,
    submission_doc,
) -> None:
    psycopg2, dsn = database
    suffix = hashlib.sha256(
        json.dumps(submission_doc, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    submission_id = f"source_add_submission:{suffix}"
    adapter_id = f"adapter:credential-{suffix}"
    miner_hotkey = "5RestartRestoreCredentialGuardMiner"
    api_base_url = f"https://credential-{suffix}.example.test/v1"
    submission_doc = {
        **submission_doc,
        "source_metadata": {"api_base_url": api_base_url},
    }
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _reserve_provider_origin(
                cursor,
                submission_id=submission_id,
                adapter_id=adapter_id,
                miner_hotkey=miner_hotkey,
                api_base_url=api_base_url,
            )
            with pytest.raises(
                psycopg2.Error,
                match=(
                    "research_lab_source_add_submission_no_credential_material_v2"
                ),
            ):
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_source_add_submissions (
                        submission_id, adapter_id, miner_hotkey, stage, seq,
                        submission_doc
                    ) VALUES (%s, %s, %s, 'submitted', 0, %s::JSONB)
                    """,
                    (
                        submission_id,
                        adapter_id,
                        miner_hotkey,
                        json.dumps(submission_doc, sort_keys=True),
                    ),
                )
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_submissions
                WHERE submission_id = %s
                """,
                (submission_id,),
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()


def test_submission_constraint_allows_credential_free_document(database) -> None:
    psycopg2, dsn = database
    submission_id = "source_add_submission:1740000000000002"
    adapter_id = "adapter:credential-free-174"
    miner_hotkey = "5RestartRestoreCredentialFreeMiner"
    api_base_url = "https://credential-free-174.example.test/v1"
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _reserve_provider_origin(
                cursor,
                submission_id=submission_id,
                adapter_id=adapter_id,
                miner_hotkey=miner_hotkey,
                api_base_url=api_base_url,
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_submissions (
                    submission_id, adapter_id, miner_hotkey, stage, seq,
                    submission_doc
                ) VALUES (%s, %s, %s, 'submitted', 0, %s::JSONB)
                """,
                (
                    submission_id,
                    adapter_id,
                    miner_hotkey,
                    json.dumps(
                        {
                            "source_metadata": {
                                "api_base_url": api_base_url,
                            },
                            "documentation_url": "https://example.com/docs",
                            "endpoint_example": "/v1/lookup?domain=example.com",
                            "summary": (
                                "API key authentication is managed by the operator"
                            ),
                        },
                        sort_keys=True,
                    ),
                ),
            )
            cursor.execute(
                """
                SELECT submission_doc
                FROM public.research_lab_source_add_submissions
                WHERE submission_id = %s
                """,
                (submission_id,),
            )
            assert cursor.fetchone()[0]["endpoint_example"] == (
                "/v1/lookup?domain=example.com"
            )
            with pytest.raises(
                psycopg2.Error,
                match=(
                    "research_lab_source_add_submission_no_credential_material_v2"
                ),
            ):
                cursor.execute(
                    """
                    UPDATE public.research_lab_source_add_submissions
                    SET submission_doc = submission_doc || %s::JSONB
                    WHERE submission_id = %s
                    """,
                    (
                        json.dumps(
                            {"request": {"api_key": "late-credential-value"}},
                            sort_keys=True,
                        ),
                        submission_id,
                    ),
                )
    finally:
        connection.close()


def test_submission_constraint_allows_sanitized_builtwith_precheck(
    database,
) -> None:
    psycopg2, dsn = database
    submission_id = "source_add_submission:1740000000000003"
    adapter_id = "adapter:sanitized-builtwith-174"
    miner_hotkey = "5RestartRestoreSanitizedBuiltWithMiner"
    api_base_url = "https://api.builtwith.com/trends/v6"
    precheck_doc = sanitize_source_add_precheck_doc(
        {
            "docs_fetch": {
                "text_excerpt": (
                    "Call /api.json?KEY=YOUR_API_KEY&TECH=Shopify for trend metadata."
                )
            }
        }
    )
    assert precheck_doc["docs_fetch"]["text_excerpt"] == "[redacted]"
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _reserve_provider_origin(
                cursor,
                submission_id=submission_id,
                adapter_id=adapter_id,
                miner_hotkey=miner_hotkey,
                api_base_url=api_base_url,
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_submissions (
                    submission_id, adapter_id, miner_hotkey, stage, seq,
                    submission_doc, precheck_status, precheck_doc
                ) VALUES (
                    %s, %s, %s, 'provenance_precheck_passed', 0,
                    %s::JSONB, 'provenance_precheck_passed', %s::JSONB
                )
                """,
                (
                    submission_id,
                    adapter_id,
                    miner_hotkey,
                    json.dumps(
                        {
                            "source_metadata": {"api_base_url": api_base_url},
                            "precheck_doc": precheck_doc,
                        },
                        sort_keys=True,
                    ),
                    json.dumps(precheck_doc, sort_keys=True),
                ),
            )
            cursor.execute(
                """
                SELECT submission_doc->'precheck_doc'->'docs_fetch'->>'text_excerpt'
                FROM public.research_lab_source_add_submissions
                WHERE submission_id = %s
                """,
                (submission_id,),
            )
            assert cursor.fetchone()[0] == "[redacted]"
    finally:
        connection.close()
