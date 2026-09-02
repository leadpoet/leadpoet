"""Disposable-PostgreSQL regression for migration 176 host collisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.research_lab.source_add_workflow import (
    source_add_reward_intent_id,
    source_add_work_id,
)
from leadpoet_canonical.attested_v2 import sha256_json
from research_lab.source_add_identity import (
    normalize_source_add_provider_origin,
    source_provider_origin_hash,
)
from tests.test_source_add_end_to_end_postgres import (
    _database_with_migrations,
    _json,
    _scalar,
    _seed_boot_identity,
)
from tests.test_source_add_provenance_leg1_postgres import (
    PRE_MIGRATIONS,
    _claim_reward,
    _finalize_reward,
    _finish_provenance,
    _record,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_175 = "175-research-lab-source-add-provenance-leg1.sql"
MIGRATION_176 = "176-research-lab-source-add-provenance-origin-repair.sql"
MIGRATION_177 = "177-research-lab-source-add-provenance-authority-acl.sql"


@pytest.fixture(scope="module")
def pre_repair_database():
    yield from _database_with_migrations(PRE_MIGRATIONS)


def _record_on_host(seed: int, *, host: str, path: str) -> dict:
    record = _record(seed)
    base_url = f"https://{host}{path}"
    origin = normalize_source_add_provider_origin(base_url)
    record["source_metadata"]["api_base_url"] = base_url
    record["manifest"]["declared_base_domains"] = [origin]
    record["provider_origin_host"] = origin
    record["provider_origin_hash"] = source_provider_origin_hash(base_url)
    return record


def _admit_and_finish(cursor, *, record: dict, seed: int) -> dict:
    work_id = "source_add_work:" + f"{seed + 10_000:016x}"[-16:]
    admitted = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_admit_v2(
            %s::JSONB,%s,%s,%s,%s,%s,3,5,10
        )
        """,
        (
            _json(record),
            sha256_json({"primary": record["submission_id"]}),
            sha256_json({"documentation": record["submission_id"]}),
            sha256_json({"legacy": record["submission_id"]}),
            record["provider_origin_hash"],
            work_id,
        ),
    )
    assert admitted["status"] == "admitted"
    cursor.execute(
        """
        UPDATE public.research_lab_source_add_work_items
        SET available_at=NOW() + INTERVAL '1 hour'
        WHERE work_kind='leg1_reward'
          AND work_status='queued'
        """
    )
    claimed = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s,180)",
        ("migration-176-provenance-" + record["submission_id"][-16:],),
    )["work"]
    assert claimed["work_id"] == work_id
    cursor.execute(
        """
        UPDATE public.research_lab_source_add_work_items
        SET available_at=NOW()
        WHERE work_kind='leg1_reward'
          AND work_status='queued'
        """
    )
    return _finish_provenance(cursor, record=record, work=claimed)


def _append_functional_failure(cursor, case: dict) -> None:
    submission_id = case["record"]["submission_id"]
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_submissions (
            submission_id,schema_version,adapter_id,miner_hotkey,stage,seq,
            measured_trial_yield,submission_doc,precheck_status,precheck_doc,
            source_identity_hash,source_identity_version
        )
        SELECT
            current.submission_id,current.schema_version,current.adapter_id,
            current.miner_hotkey,'functional_probe_failed',current.seq + 1,
            current.measured_trial_yield,
            current.submission_doc || jsonb_build_object(
                'stage','functional_probe_failed'
            ),
            current.precheck_status,current.precheck_doc,
            current.source_identity_hash,current.source_identity_version
        FROM public.research_lab_source_add_submission_current current
        WHERE current.submission_id=%s
        """,
        (submission_id,),
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_identity_events (
            identity_version,source_identity_hash,submission_id,adapter_id,
            miner_hotkey,reservation_status,seq,reason
        )
        SELECT
            identity_version,source_identity_hash,submission_id,adapter_id,
            miner_hotkey,'released',seq + 1,'terminal_rejection'
        FROM public.research_lab_source_add_identity_current
        WHERE submission_id=%s AND reservation_status='reserved'
        """,
        (submission_id,),
    )


def test_migration_176_repairs_historical_host_collision_and_is_idempotent(
    pre_repair_database,
):
    psycopg2, dsn = pre_repair_database
    sql_175 = (ROOT / "scripts" / MIGRATION_175).read_text(encoding="utf-8")
    sql_176 = (ROOT / "scripts" / MIGRATION_176).read_text(encoding="utf-8")
    sql_177 = (ROOT / "scripts" / MIGRATION_177).read_text(encoding="utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _scalar(
                cursor,
                "SELECT public.research_lab_source_add_set_paused(FALSE,%s,%s)",
                ("seed migration 176", "operator:migration-176-test"),
            )
            _seed_boot_identity(cursor)

            earlier = _admit_and_finish(
                cursor,
                record=_record_on_host(
                    0x1760000000000001,
                    host="collision.migration-176.test",
                    path="/v1",
                ),
                seed=0x1760000000000001,
            )
            _append_functional_failure(cursor, earlier)
            later = _admit_and_finish(
                cursor,
                record=_record_on_host(
                    0x1760000000000002,
                    host="collision.migration-176.test",
                    path="/v2",
                ),
                seed=0x1760000000000002,
            )
            unique_terminal = _admit_and_finish(
                cursor,
                record=_record_on_host(
                    0x1760000000000003,
                    host="unique.migration-176.test",
                    path="/v1",
                ),
                seed=0x1760000000000003,
            )
            _append_functional_failure(cursor, unique_terminal)

            _scalar(
                cursor,
                "SELECT public.research_lab_source_add_set_paused(TRUE,%s,%s)",
                ("apply migration 175", "operator:migration-176-test"),
            )
            cursor.execute(sql_175)
            for case in (earlier, later, unique_terminal):
                assert _scalar(
                    cursor,
                    """
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_intents
                    WHERE submission_id=%s
                      AND approval_kind='provenance_precheck_passed'
                    """,
                    (case["record"]["submission_id"],),
                ) == 1
            broken = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_provider_origin_contract_v1()",
            )
            assert broken["coverage_complete"] is False
            assert broken["collision_free"] is False

            cursor.execute(sql_176)
            cursor.execute(sql_177)
            contract = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_provider_origin_contract_v1()",
            )
            assert contract["coverage_complete"] is True
            assert contract["collision_free"] is True
            assert contract["owner_count"] == contract["reserved_count"]
            leg1_contract = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_post_accept_leg1_contract_v4()",
            )
            assert leg1_contract["schema_version"] == (
                "leadpoet.source_add_post_accept_leg1_contract.v4"
            )
            assert leg1_contract["required_migration"] == (
                "scripts/176-research-lab-source-add-provenance-origin-repair.sql"
            )
            assert leg1_contract["backfill_policy"] == (
                "earliest_exact_attested_provenance_per_provider_origin"
            )
            assert leg1_contract["cancelled_intents_are_authority"] is False

            authority = _scalar(
                cursor,
                """
                SELECT jsonb_agg(submission_id ORDER BY submission_id)
                FROM public.research_lab_source_add_provenance_leg1_authority_v1
                WHERE submission_id IN (%s,%s,%s)
                """,
                (
                    earlier["record"]["submission_id"],
                    later["record"]["submission_id"],
                    unique_terminal["record"]["submission_id"],
                ),
            )
            assert authority == sorted(
                [
                    earlier["record"]["submission_id"],
                    unique_terminal["record"]["submission_id"],
                ]
            )
            cursor.execute(
                """
                ALTER ROLE service_role BYPASSRLS;
                GRANT USAGE ON SCHEMA public, extensions TO service_role;
                GRANT SELECT ON ALL TABLES IN SCHEMA public TO service_role;
                """
            )
            cursor.execute("SET ROLE service_role")
            try:
                service_role_authority = _scalar(
                    cursor,
                    """
                    SELECT jsonb_agg(submission_id ORDER BY submission_id)
                    FROM public.research_lab_source_add_provenance_leg1_authority_v1
                    WHERE submission_id IN (%s,%s,%s)
                    """,
                    (
                        earlier["record"]["submission_id"],
                        later["record"]["submission_id"],
                        unique_terminal["record"]["submission_id"],
                    ),
                )
            finally:
                cursor.execute("RESET ROLE")
            assert service_role_authority == authority
            for helper in (
                "public.research_lab_source_add_provider_origin_host_v1(text)",
                "public.research_lab_source_add_provider_origin_hash_v1(text)",
            ):
                assert _scalar(
                    cursor,
                    "SELECT has_function_privilege('service_role', %s, 'EXECUTE')",
                    (helper,),
                ) is True
                assert _scalar(
                    cursor,
                    "SELECT has_function_privilege('anon', %s, 'EXECUTE')",
                    (helper,),
                ) is False
                assert _scalar(
                    cursor,
                    "SELECT has_function_privilege('authenticated', %s, 'EXECUTE')",
                    (helper,),
                ) is False
            assert _scalar(
                cursor,
                """
                SELECT jsonb_build_object(
                    'stage',current.stage,
                    'precheck',current.precheck_doc,
                    'intent',(SELECT intent_status
                              FROM public.research_lab_source_add_reward_intents
                              WHERE submission_id=current.submission_id),
                    'work',(SELECT work_status
                            FROM public.research_lab_source_add_work_items
                            WHERE submission_id=current.submission_id
                              AND work_kind='leg1_reward'),
                    'rewards',(SELECT COUNT(*)
                               FROM public.research_lab_source_add_reward_obligations
                               WHERE adapter_id=current.adapter_id AND leg=1)
                )
                FROM public.research_lab_source_add_submission_current current
                WHERE current.submission_id=%s
                """,
                (later["record"]["submission_id"],),
            ) == {
                "stage": "rejected_precheck",
                "precheck": {
                    "status": "rejected_precheck",
                    "reason_codes": ["submission_not_eligible"],
                },
                "intent": "cancelled",
                "work": "cancelled",
                "rewards": 0,
            }
            for case in (earlier, unique_terminal):
                assert _scalar(
                    cursor,
                    """
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_provider_origin_current
                    WHERE provider_origin_hash=%s
                      AND submission_id=%s
                      AND reservation_status='reserved'
                    """,
                    (
                        case["record"]["provider_origin_hash"],
                        case["record"]["submission_id"],
                    ),
                ) == 1

            _scalar(
                cursor,
                "SELECT public.research_lab_source_add_set_paused(FALSE,%s,%s)",
                ("finalize repaired rewards", "operator:migration-176-test"),
            )
            winner_cases = {
                case["record"]["submission_id"]: case
                for case in (earlier, unique_terminal)
            }
            for _ in winner_cases:
                reward_work = _claim_reward(cursor)
                finalized = _finalize_reward(
                    cursor,
                    work=reward_work,
                    case=winner_cases[reward_work["submission_id"]],
                    caller_cap=999,
                )
                assert finalized["status"] == "created"
            assert _scalar(
                cursor,
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_reward_obligations
                WHERE adapter_id IN (%s,%s) AND leg=1
                  AND catalog_id IS NULL
                  AND alpha_percent=0.2
                  AND reward_epochs=20
                """,
                (
                    earlier["record"]["adapter_id"],
                    unique_terminal["record"]["adapter_id"],
                ),
            ) == 2
            assert _scalar(
                cursor,
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_reward_obligations
                WHERE adapter_id=%s AND leg=1
                """,
                (later["record"]["adapter_id"],),
            ) == 0
            _scalar(
                cursor,
                "SELECT public.research_lab_source_add_set_paused(TRUE,%s,%s)",
                ("replay migration 176", "operator:migration-176-test"),
            )

            before_reapply = _scalar(
                cursor,
                """
                SELECT jsonb_build_object(
                    'history',(SELECT COUNT(*) FROM public.research_lab_source_add_submissions),
                    'origins',(SELECT COUNT(*) FROM public.research_lab_source_add_provider_origin_events),
                    'identities',(SELECT COUNT(*) FROM public.research_lab_source_add_identity_events),
                    'intents',(SELECT COUNT(*) FROM public.research_lab_source_add_reward_intents),
                    'works',(SELECT COUNT(*) FROM public.research_lab_source_add_work_items),
                    'rewards',(SELECT COUNT(*) FROM public.research_lab_source_add_reward_obligations),
                    'reward_events',(SELECT COUNT(*) FROM public.research_lab_source_add_reward_events),
                    'slots',(SELECT COUNT(*) FROM public.research_lab_source_add_reward_slots)
                )
                """,
            )
            cursor.execute(sql_176)
            cursor.execute(sql_177)
            after_reapply = _scalar(
                cursor,
                """
                SELECT jsonb_build_object(
                    'history',(SELECT COUNT(*) FROM public.research_lab_source_add_submissions),
                    'origins',(SELECT COUNT(*) FROM public.research_lab_source_add_provider_origin_events),
                    'identities',(SELECT COUNT(*) FROM public.research_lab_source_add_identity_events),
                    'intents',(SELECT COUNT(*) FROM public.research_lab_source_add_reward_intents),
                    'works',(SELECT COUNT(*) FROM public.research_lab_source_add_work_items),
                    'rewards',(SELECT COUNT(*) FROM public.research_lab_source_add_reward_obligations),
                    'reward_events',(SELECT COUNT(*) FROM public.research_lab_source_add_reward_events),
                    'slots',(SELECT COUNT(*) FROM public.research_lab_source_add_reward_slots)
                )
                """,
            )
            assert after_reapply == before_reapply

            _scalar(
                cursor,
                "SELECT public.research_lab_source_add_set_paused(FALSE,%s,%s)",
                ("future auto reward", "operator:migration-176-test"),
            )
            future = _admit_and_finish(
                cursor,
                record=_record_on_host(
                    0x1760000000000004,
                    host="future.migration-176.test",
                    path="/v1",
                ),
                seed=0x1760000000000004,
            )
            future_intent = source_add_reward_intent_id(
                future["record"]["submission_id"],
                future["record"]["adapter_id"],
            )
            future_work = source_add_work_id(
                future["record"]["submission_id"],
                "leg1_reward",
                future_intent,
            )
            assert _scalar(
                cursor,
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_reward_intents intent
                JOIN public.research_lab_source_add_work_items work
                  ON work.job_doc->>'intent_id'=intent.intent_id
                WHERE intent.intent_id=%s AND intent.intent_status='queued'
                  AND work.work_id=%s AND work.work_status='queued'
                """,
                (future_intent, future_work),
            ) == 1
            duplicate = _record_on_host(
                0x1760000000000005,
                host="collision.migration-176.test",
                path="/v3",
            )
            duplicate_result = _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_admit_v2(
                    %s::JSONB,%s,%s,%s,%s,%s,3,5,10
                )
                """,
                (
                    _json(duplicate),
                    sha256_json({"primary": duplicate["submission_id"]}),
                    sha256_json({
                        "documentation": duplicate["submission_id"]
                    }),
                    sha256_json({"legacy": duplicate["submission_id"]}),
                    duplicate["provider_origin_hash"],
                    "source_add_work:1760000000000005",
                ),
            )
            assert duplicate_result["status"] == "duplicate"
            assert _scalar(
                cursor,
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_reward_intents
                WHERE submission_id=%s
                """,
                (duplicate["submission_id"],),
            ) == 0
    finally:
        connection.close()
