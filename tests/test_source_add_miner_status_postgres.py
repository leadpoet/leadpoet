"""Disposable-PostgreSQL coverage for private SOURCE_ADD miner status."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from leadpoet_canonical.attested_v2 import sha256_json
from tests.test_source_add_end_to_end_postgres import (
    _database_with_migrations,
    _json,
    _scalar,
    _seed_boot_identity,
)
from tests.test_source_add_provenance_leg1_postgres import (
    PRE_MIGRATIONS,
    _admit_and_claim,
    _claim_reward,
    _finalize_reward,
    _finish_provenance,
    _record,
    _set_paused,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "178-research-lab-source-add-miner-status.sql"
MIGRATIONS = PRE_MIGRATIONS + (
    "175-research-lab-source-add-provenance-leg1.sql",
    "176-research-lab-source-add-provenance-origin-repair.sql",
    "177-research-lab-source-add-provenance-authority-acl.sql",
    MIGRATION,
)


@pytest.fixture(scope="module")
def database():
    yield from _database_with_migrations(MIGRATIONS)


def _admit_without_claim(cursor, seed: int) -> dict:
    record = _record(seed)
    submission_id = record["submission_id"]
    result = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_admit_v2(
            %s::JSONB,%s,%s,%s,%s,%s,3,5,10
        )
        """,
        (
            _json(record),
            sha256_json({"primary": submission_id}),
            sha256_json({"documentation": submission_id}),
            sha256_json({"legacy": submission_id}),
            record["provider_origin_hash"],
            "source_add_work:" + f"{seed + 10_000:016x}",
        ),
    )
    assert result["status"] == "admitted"
    return record


def _status_row(cursor, submission_id: str) -> dict:
    cursor.execute(
        """
        SELECT decision_status, decision_reason_code, decision_reason,
               reward_status, alpha_percent, reward_epochs,
               start_epoch, end_epoch
        FROM public.research_lab_source_add_miner_status_v1
        WHERE submission_id=%s
        """,
        (submission_id,),
    )
    row = cursor.fetchone()
    assert row is not None
    return {
        "decision_status": row[0],
        "decision_reason_code": row[1],
        "decision_reason": row[2],
        "reward_status": row[3],
        "alpha_percent": row[4],
        "reward_epochs": row[5],
        "start_epoch": row[6],
        "end_epoch": row[7],
    }


def _finalized_case(cursor, seed: int) -> tuple[dict, str]:
    record, work = _admit_and_claim(cursor, seed)
    case = _finish_provenance(cursor, record=record, work=work)
    reward_work = _claim_reward(cursor)
    result = _finalize_reward(
        cursor,
        work=reward_work,
        case=case,
        caller_cap=50,
    )
    assert result["status"] == "created"
    return record, result["reward_ref"]


def test_exact_status_and_reward_projection(database):
    _, dsn = database
    import psycopg2

    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            _set_paused(cursor, False, "miner-status-projection")
            _seed_boot_identity(cursor)

            active, _ = _finalized_case(cursor, 0x1780000000000001)
            completed, completed_reward = _finalized_case(
                cursor, 0x1780000000000002
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_reward_events (
                    reward_ref, seq, reward_status, reason
                ) VALUES (%s,1,'stopped_forward','reward_window_complete')
                """,
                (completed_reward,),
            )

            approved_pending_record, approved_pending_work = _admit_and_claim(
                cursor, 0x1780000000000003
            )
            _finish_provenance(
                cursor,
                record=approved_pending_record,
                work=approved_pending_work,
            )

            review_record, review_work = _admit_and_claim(
                cursor, 0x1780000000000004
            )
            _finish_provenance(
                cursor,
                record=review_record,
                work=review_work,
                status="needs_manual_review",
            )

            rejected_record, rejected_work = _admit_and_claim(
                cursor, 0x1780000000000005
            )
            _finish_provenance(
                cursor,
                record=rejected_record,
                work=rejected_work,
                status="rejected_precheck",
                historical_routing_reason=(
                    "documentation_contains_fake_or_test_markers"
                ),
            )
            queued = _admit_without_claim(cursor, 0x1780000000000006)
            private_duplicate = _insert_page_submission(
                cursor,
                seed=0x1780000000000007,
                hotkey="5MinerStatusPrivateDuplicate",
                created_at="2026-09-03T12:07:00Z",
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_submissions (
                    submission_id, adapter_id, miner_hotkey, stage, seq,
                    submission_doc, precheck_status, precheck_doc,
                    source_identity_hash, source_identity_version, created_at
                )
                SELECT
                    current.submission_id, current.adapter_id,
                    current.miner_hotkey, 'rejected_precheck', 1,
                    current.submission_doc, 'rejected_precheck',
                    '{"reason_codes":["duplicate_provider_origin_existing_owner"]}'::JSONB,
                    current.source_identity_hash, current.source_identity_version,
                    NOW()
                FROM public.research_lab_source_add_submission_current current
                WHERE current.submission_id=%s
                """,
                (private_duplicate,),
            )

            assert _status_row(cursor, active["submission_id"]) == {
                "decision_status": "approved",
                "decision_reason_code": "leg1_reward_active",
                "decision_reason": (
                    "The source passed automated checks and the Leg 1 reward "
                    "is active."
                ),
                "reward_status": "active",
                "alpha_percent": Decimal("0.200000"),
                "reward_epochs": 20,
                "start_epoch": 50_000,
                "end_epoch": 50_019,
            }
            assert _status_row(cursor, completed["submission_id"]) == {
                "decision_status": "approved",
                "decision_reason_code": "leg1_reward_stopped",
                "decision_reason": (
                    "The source passed automated checks. Future Leg 1 reward "
                    "payments have stopped."
                ),
                "reward_status": "stopped",
                "alpha_percent": Decimal("0.200000"),
                "reward_epochs": 20,
                "start_epoch": 50_000,
                "end_epoch": 50_019,
            }
            assert _status_row(
                cursor, approved_pending_record["submission_id"]
            ) == {
                "decision_status": "approved",
                "decision_reason_code": "leg1_reward_pending",
                "decision_reason": (
                    "The source passed automated checks. Leg 1 reward setup "
                    "is in progress."
                ),
                "reward_status": "pending",
                "alpha_percent": None,
                "reward_epochs": None,
                "start_epoch": None,
                "end_epoch": None,
            }
            assert _status_row(cursor, review_record["submission_id"]) == {
                "decision_status": "pending",
                "decision_reason_code": "additional_review_needed",
                "decision_reason": (
                    "Automated verification was inconclusive and needs "
                    "additional review."
                ),
                "reward_status": "not_decided",
                "alpha_percent": None,
                "reward_epochs": None,
                "start_epoch": None,
                "end_epoch": None,
            }
            assert _status_row(cursor, rejected_record["submission_id"]) == {
                "decision_status": "rejected",
                "decision_reason_code": "source_credibility_not_verified",
                "decision_reason": (
                    "The source did not pass the public credibility checks."
                ),
                "reward_status": "not_eligible",
                "alpha_percent": None,
                "reward_epochs": None,
                "start_epoch": None,
                "end_epoch": None,
            }
            assert _status_row(cursor, queued["submission_id"]) == {
                "decision_status": "pending",
                "decision_reason_code": "automated_checks_in_progress",
                "decision_reason": (
                    "Automated Source Add checks are still in progress."
                ),
                "reward_status": "not_decided",
                "alpha_percent": None,
                "reward_epochs": None,
                "start_epoch": None,
                "end_epoch": None,
            }
            assert _status_row(cursor, private_duplicate) == {
                "decision_status": "rejected",
                "decision_reason_code": "automated_checks_not_passed",
                "decision_reason": (
                    "The submission did not pass automated Source Add checks."
                ),
                "reward_status": "not_eligible",
                "alpha_percent": None,
                "reward_epochs": None,
                "start_epoch": None,
                "end_epoch": None,
            }


def _insert_page_submission(
    cursor,
    *,
    seed: int,
    hotkey: str,
    created_at: str,
) -> str:
    submission_id = "source_add_submission:" + f"{seed:016x}"
    adapter_id = "adapter:miner-status-page-" + f"{seed:016x}"
    api_base_url = f"https://api-{seed:016x}.miner-status.test/v1"
    provider_origin_hash = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_provider_origin_hash_v1(%s)",
        (api_base_url,),
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_provider_origin_events (
            origin_version, provider_origin_hash, submission_id, adapter_id,
            miner_hotkey, reservation_status, seq, reason
        ) VALUES ('v1',%s,%s,%s,%s,'reserved',0,'miner_status_test')
        """,
        (provider_origin_hash, submission_id, adapter_id, hotkey),
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_submissions (
            submission_id, adapter_id, miner_hotkey, stage, seq,
            submission_doc, precheck_status, precheck_doc,
            source_identity_hash, source_identity_version, created_at
        ) VALUES (
            %s,%s,%s,'submitted',0,%s::JSONB,'','{}'::JSONB,%s,'v2',%s
        )
        """,
        (
            submission_id,
            adapter_id,
            hotkey,
            _json(
                {
                    "manifest": {"source_name": f"Page source {seed}"},
                    "source_metadata": {"api_base_url": api_base_url},
                }
            ),
            sha256_json({"page-seed": seed}),
            created_at,
        ),
    )
    return submission_id


def _page(cursor, hotkey: str, cursor_id: str | None, limit: int) -> list[str]:
    cursor.execute(
        """
        SELECT submission_id
        FROM public.research_lab_source_add_miner_status_page_v1(%s,%s,%s)
        """,
        (hotkey, cursor_id, limit),
    )
    return [row[0] for row in cursor.fetchall()]


def test_owner_filter_and_keyset_pagination(database):
    _, dsn = database
    import psycopg2

    owner = "5MinerStatusPaginationOwner"
    other = "5MinerStatusPaginationOther"
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            owned = [
                _insert_page_submission(
                    cursor,
                    seed=0x1781000000000000 + offset,
                    hotkey=owner,
                    created_at=f"2026-09-03T12:0{offset}:00Z",
                )
                for offset in range(4)
            ]
            foreign = _insert_page_submission(
                cursor,
                seed=0x1782000000000000,
                hotkey=other,
                created_at="2026-09-03T12:10:00Z",
            )

            first_page_with_lookahead = _page(cursor, owner, None, 2)
            assert first_page_with_lookahead == [owned[3], owned[2], owned[1]]
            second_page = _page(cursor, owner, owned[2], 2)
            assert second_page == [owned[1], owned[0]]
            assert not set(first_page_with_lookahead[:2]) & set(second_page)
            assert foreign not in first_page_with_lookahead + second_page
            assert _page(cursor, other, None, 50) == [foreign]
            assert _page(cursor, owner, foreign, 50) == []


def test_idempotency_and_public_role_denial(database):
    psycopg2, dsn = database
    migration_sql = (ROOT / "scripts" / MIGRATION).read_text(encoding="utf-8")

    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT json_agg(column_name ORDER BY ordinal_position)
                FROM information_schema.columns
                WHERE table_schema='public'
                  AND table_name='research_lab_source_add_miner_status_v1'
                """
            )
            assert cursor.fetchone()[0] == [
                "schema_version",
                "submission_id",
                "miner_hotkey",
                "source_name",
                "submitted_at",
                "updated_at",
                "decision_status",
                "decision_reason_code",
                "decision_reason",
                "reward_status",
                "alpha_percent",
                "reward_epochs",
                "start_epoch",
                "end_epoch",
            ]
            cursor.execute(
                """
                SELECT
                    'public.research_lab_source_add_miner_status_v1'::REGCLASS::OID,
                    'public.research_lab_source_add_miner_status_page_v1(text,text,integer)'::REGPROCEDURE::OID,
                    'public.research_lab_source_add_miner_status_contract_v1()'::REGPROCEDURE::OID,
                    (SELECT COUNT(*)
                     FROM public.research_lab_source_add_miner_status_v1)
                """
            )
            authority_before = cursor.fetchone()
            contract_before = _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_miner_status_contract_v1()
                """,
            )
            cursor.execute(migration_sql)
            cursor.execute(migration_sql)
            cursor.execute(
                """
                SELECT
                    'public.research_lab_source_add_miner_status_v1'::REGCLASS::OID,
                    'public.research_lab_source_add_miner_status_page_v1(text,text,integer)'::REGPROCEDURE::OID,
                    'public.research_lab_source_add_miner_status_contract_v1()'::REGPROCEDURE::OID,
                    (SELECT COUNT(*)
                     FROM public.research_lab_source_add_miner_status_v1)
                """
            )
            assert cursor.fetchone() == authority_before
            assert _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_miner_status_contract_v1()
                """,
            ) == contract_before

            for role in ("anon", "authenticated"):
                assert _scalar(
                    cursor,
                    """
                    SELECT has_table_privilege(
                        %s,
                        'public.research_lab_source_add_miner_status_v1',
                        'SELECT'
                    )
                    """,
                    (role,),
                ) is False
                assert _scalar(
                    cursor,
                    """
                    SELECT has_function_privilege(
                        %s,
                        'public.research_lab_source_add_miner_status_contract_v1()',
                        'EXECUTE'
                    )
                    """,
                    (role,),
                ) is False
                assert _scalar(
                    cursor,
                    """
                    SELECT has_function_privilege(
                        %s,
                        'public.research_lab_source_add_miner_status_page_v1(text,text,integer)',
                        'EXECUTE'
                    )
                    """,
                    (role,),
                ) is False
                cursor.execute(f"SET ROLE {role}")
                try:
                    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                        cursor.execute(
                            """
                            SELECT *
                            FROM public.research_lab_source_add_miner_status_v1
                            LIMIT 1
                            """
                        )
                    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                        cursor.execute(
                            """
                            SELECT *
                            FROM public.research_lab_source_add_miner_status_page_v1(
                                '5UnauthorizedMiner', NULL, 20
                            )
                            """
                        )
                    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                        cursor.execute(
                            """
                            SELECT public.research_lab_source_add_miner_status_contract_v1()
                            """
                        )
                finally:
                    cursor.execute("RESET ROLE")

            cursor.execute(
                """
                ALTER ROLE service_role BYPASSRLS;
                GRANT USAGE ON SCHEMA public, extensions TO service_role;
                GRANT SELECT ON ALL TABLES IN SCHEMA public TO service_role;
                """
            )
            cursor.execute("SET ROLE service_role")
            try:
                contract = _scalar(
                    cursor,
                    """
                    SELECT public.research_lab_source_add_miner_status_contract_v1()
                    """,
                )
                assert set(contract) == {
                    "schema_version",
                    "view_name",
                    "page_rpc",
                    "page_signature",
                    "view_columns",
                    "view_security_invoker",
                    "view_security_barrier",
                    "page_security_invoker",
                    "page_stable",
                    "view_authority_sha256",
                    "page_authority_sha256",
                    "contract_authority_sha256",
                    "permissions",
                }
                assert contract["view_authority_sha256"] == (
                    "sha256:8096dcc13409b33b56ad70f9606c9fe8ac7c644583b02b9c70f97322dfe86e26"
                )
                assert contract["page_authority_sha256"] == (
                    "sha256:fefb9294135f34d9e0f329288f9ee11c42b54e36eaa4941d92e20b69e1a9d2e1"
                )
                assert contract["contract_authority_sha256"] == (
                    "sha256:b2d1ba1bf1062a911dc4ab3d6619d93b5cf282d4daa3896c553e99e0520b2c11"
                )
                assert contract["permissions"] == {
                    "view_service_role_select": True,
                    "view_anon_select": False,
                    "view_authenticated_select": False,
                    "view_public_select": False,
                    "page_service_role_callable": True,
                    "page_anon_callable": False,
                    "page_authenticated_callable": False,
                    "page_public_callable": False,
                    "contract_service_role_callable": True,
                    "contract_anon_callable": False,
                    "contract_authenticated_callable": False,
                }
                assert _scalar(
                    cursor,
                    """
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_miner_status_page_v1(
                        '5UnauthorizedMiner', NULL, 50
                    )
                    """,
                ) == 0
            finally:
                cursor.execute("RESET ROLE")
