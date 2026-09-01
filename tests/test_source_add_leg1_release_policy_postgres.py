"""Execute the rollback-safe SOURCE_ADD Leg 1 policy migration."""

from __future__ import annotations

import json

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    SOURCE_ADD_POST_ACCEPT_LEG1_FUNCTION_AUTHORITY_SHA256,
)
from tests.test_source_add_claim_control_postgres import (
    MIGRATIONS as PRE_POLICY_MIGRATIONS,
    _insert_work,
)
from tests.test_source_add_end_to_end_postgres import (
    SCRIPTS,
    _create_seed_leg1_reward,
    _database_with_migrations,
    _finalize_seed_smoke_to_leg1,
    _scalar,
    _seed_leased_smoke_case,
)


MIGRATION = "173-research-lab-source-add-leg1-release-policy.sql"
LATEST_MIGRATIONS = PRE_POLICY_MIGRATIONS + (MIGRATION,)


@pytest.fixture(scope="module")
def pre_policy_database():
    yield from _database_with_migrations(PRE_POLICY_MIGRATIONS)


@pytest.fixture(scope="module")
def policy_database():
    yield from _database_with_migrations(LATEST_MIGRATIONS)


def _set_paused(cursor, paused: bool, suffix: str) -> None:
    cursor.execute(
        "SELECT public.research_lab_source_add_set_paused(%s, %s, %s)",
        (paused, f"policy migration {suffix}", "operator:policy-migration-test"),
    )


def test_migration_requires_quiescence_is_idempotent_and_preserves_old_reward(
    pre_policy_database,
) -> None:
    psycopg2, dsn = pre_policy_database
    migration_sql = (SCRIPTS / MIGRATION).read_text(encoding="utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _set_paused(cursor, False, "active rejection")
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD must be paused before Leg 1 policy migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")

            _set_paused(cursor, True, "leased rejection")
            _insert_work(cursor, suffix="1730000000000001", status="leased")
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD work is leased during Leg 1 policy migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")
            cursor.execute(
                "DELETE FROM public.research_lab_source_add_work_items "
                "WHERE work_id = %s",
                ("source_add_work:1730000000000001",),
            )

            _set_paused(cursor, False, "grandfather setup")
            seed = 0x1730000000001000
            created = _create_seed_leg1_reward(
                cursor,
                case=_seed_leased_smoke_case(
                    cursor,
                    seed=seed,
                    base_url="https://grandfathered-173.example/v1",
                ),
                seed=seed,
            )
            cursor.execute(
                "SELECT to_jsonb(reward) FROM "
                "public.research_lab_source_add_reward_obligations reward "
                "WHERE reward_ref = %s",
                (created["reward_ref"],),
            )
            before = cursor.fetchone()[0]
            assert float(before["alpha_percent"]) == pytest.approx(1.0)

            _set_paused(cursor, True, "apply")
            cursor.execute(migration_sql)
            cursor.execute(migration_sql)
            cursor.execute(
                "SELECT to_jsonb(reward) FROM "
                "public.research_lab_source_add_reward_obligations reward "
                "WHERE reward_ref = %s",
                (created["reward_ref"],),
            )
            assert cursor.fetchone()[0] == before

            v1 = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_post_accept_leg1_contract_v1()",
            )
            v2 = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_post_accept_leg1_contract_v2()",
            )
            assert (v1["leg1_alpha_percent"], v1["daily_cap"]) == (1.0, 10)
            assert (v2["leg1_alpha_percent"], v2["daily_cap"]) == (0.2, 50)
            assert v2["function_authority_sha256"] == (
                SOURCE_ADD_POST_ACCEPT_LEG1_FUNCTION_AUTHORITY_SHA256
            )
    finally:
        connection.close()


def test_v3_economics_and_acl_are_exact_while_v2_remains_rollback_safe(
    policy_database,
) -> None:
    psycopg2, dsn = policy_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    zero_uuid = "00000000-0000-0000-0000-000000000000"
    try:
        with connection.cursor() as cursor:
            contract = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_post_accept_leg1_contract_v2()",
            )
            assert contract["schema_version"] == (
                "leadpoet.source_add_post_accept_leg1_contract.v2"
            )
            assert all(contract["functions"].values())
            assert all(contract["triggers"].values())
            assert all(contract["permissions"].values())
            assert contract["function_authority_sha256"] == (
                SOURCE_ADD_POST_ACCEPT_LEG1_FUNCTION_AUTHORITY_SHA256
            )
            cursor.execute(
                """
                SELECT
                    has_function_privilege(
                        'service_role',
                        'public.research_lab_source_add_finalize_leg1_v3(text,text,uuid,uuid,integer,jsonb,jsonb)',
                        'EXECUTE'
                    ),
                    has_function_privilege(
                        'anon',
                        'public.research_lab_source_add_finalize_leg1_v3(text,text,uuid,uuid,integer,jsonb,jsonb)',
                        'EXECUTE'
                    ),
                    has_function_privilege(
                        'authenticated',
                        'public.research_lab_source_add_post_accept_leg1_contract_v2()',
                        'EXECUTE'
                    )
                """
            )
            assert cursor.fetchone() == (True, False, False)

            reward = {
                "reward_ref": "source_add_reward:" + "0" * 16,
                "reward_kind": "source_acceptance",
                "state": "active",
                "reward_epochs": 20,
                "start_epoch": 0,
                "decision_receipt_hash": "sha256:" + "1" * 64,
                "decision_artifact_hash": "sha256:" + "2" * 64,
                "trigger_evidence_doc": {"functional_probe_passed": True},
            }
            old_reward = json.dumps({**reward, "alpha_percent": 1.0})
            new_reward = json.dumps({**reward, "alpha_percent": 0.2})
            assert _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_finalize_leg1_v2(
                    'intent:none', 'work:none', %s::UUID, %s::UUID,
                    100, %s::JSONB, '{}'::JSONB
                )
                """,
                (zero_uuid, zero_uuid, old_reward),
            ) == {"status": "lease_lost"}
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD Leg 1 release economics differs",
            ):
                _scalar(
                    cursor,
                    """
                    SELECT public.research_lab_source_add_finalize_leg1_v2(
                        'intent:none', 'work:none', %s::UUID, %s::UUID,
                        100, %s::JSONB, '{}'::JSONB
                    )
                    """,
                    (zero_uuid, zero_uuid, new_reward),
                )
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD Leg 1 release economics differs",
            ):
                _scalar(
                    cursor,
                    """
                    SELECT public.research_lab_source_add_finalize_leg1_v3(
                        'intent:none', 'work:none', %s::UUID, %s::UUID,
                        1, %s::JSONB, '{}'::JSONB
                    )
                    """,
                    (zero_uuid, zero_uuid, old_reward),
                )
            assert _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_finalize_leg1_v3(
                    'intent:none', 'work:none', %s::UUID, %s::UUID,
                    1, %s::JSONB, '{}'::JSONB
                )
                """,
                (zero_uuid, zero_uuid, new_reward),
            ) == {"status": "lease_lost"}
    finally:
        connection.close()


def test_v3_server_cap_creates_fifty_then_fifo_defers_fifty_first(
    policy_database,
) -> None:
    psycopg2, dsn = policy_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _set_paused(cursor, False, "cap fifty")
            for index in range(50):
                seed = 0x1731000000000000 + index * 0x100
                _create_seed_leg1_reward(
                    cursor,
                    case=_seed_leased_smoke_case(
                        cursor,
                        seed=seed,
                        base_url=f"https://cap-173-{index}.example/v1",
                    ),
                    seed=seed,
                    alpha_percent=0.2,
                    economics_rpc_version=3,
                )
            assert _scalar(
                cursor,
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_reward_events
                WHERE reason = 'leg1_functional_probe_passed'
                  AND created_at >= (
                      (NOW() AT TIME ZONE 'UTC')::DATE::TIMESTAMP
                      AT TIME ZONE 'UTC'
                  )
                """,
            ) == 50

            blocked_seed = 0x1731000000010000
            blocked = _finalize_seed_smoke_to_leg1(
                cursor,
                case=_seed_leased_smoke_case(
                    cursor,
                    seed=blocked_seed,
                    base_url="https://cap-173-fifty-first.example/v1",
                ),
                seed=blocked_seed,
            )
            claimed = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_claim_work(%s, 180)",
                ("postgres-cap-173-fifty-first",),
            )["work"]
            assert claimed["work_id"] == blocked["reward_work_id"]
            result = _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_reserve_leg1_slot_v3(
                    %s, %s, %s::UUID, 1, 300
                )
                """,
                (
                    blocked["intent_id"],
                    blocked["reward_work_id"],
                    claimed["lease_token"],
                ),
            )
            assert result["status"] == "daily_cap_fifo"
            assert _scalar(
                cursor,
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_reward_obligations
                WHERE adapter_id = %s AND leg = 1
                """,
                (blocked["adapter_id"],),
            ) == 0
    finally:
        connection.close()
