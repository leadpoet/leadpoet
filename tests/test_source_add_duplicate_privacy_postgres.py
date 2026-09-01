"""Execute duplicate-private SOURCE_ADD admission against disposable PostgreSQL."""

from __future__ import annotations

import json
import threading

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    SOURCE_ADD_DUPLICATE_PRIVACY_FUNCTION_AUTHORITY_SHA256,
)
from research_lab.source_add_identity import source_provider_origin_hash
from tests.test_source_add_end_to_end_postgres import (
    MIGRATIONS as PRE_PRIVACY_MIGRATIONS,
    _database_with_migrations,
)


MIGRATION = "171-research-lab-source-add-duplicate-privacy.sql"
MIGRATIONS = PRE_PRIVACY_MIGRATIONS + (MIGRATION,)


@pytest.fixture(scope="module")
def database():
    generator = _database_with_migrations(MIGRATIONS)
    value = next(generator)
    psycopg2, dsn = value
    setup = psycopg2.connect(**dsn)
    setup.autocommit = True
    with setup.cursor() as cursor:
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
            (
                "duplicate privacy postgres",
                "operator:duplicate-privacy-postgres",
            ),
        )
    setup.close()
    try:
        yield value
    finally:
        try:
            next(generator)
        except StopIteration:
            pass


def _json(value):
    from psycopg2.extras import Json

    return Json(value, dumps=lambda item: json.dumps(item, sort_keys=True))


def _record(*, suffix: str, host: str, miner: str) -> dict:
    api_base_url = f"https://{host}/v1"
    return {
        "submission_id": f"source_add_submission:{suffix}",
        "adapter_id": f"adapter:privacy-{suffix}",
        "miner_hotkey": miner,
        "credential_envelope": {},
        "provider_origin_host": host,
        "provider_origin_hash": source_provider_origin_hash(api_base_url),
        "manifest": {
            "credential_policy": "no_credentials",
            "credential_ref": "",
            "source_name": f"Privacy {suffix}",
            "source_kind": "registry",
            "declared_base_domains": [host],
        },
        "source_metadata": {
            "api_base_url": api_base_url,
            "documentation_url": f"https://docs.{host}/reference",
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


def _admit(
    cursor,
    record: dict,
    *,
    identity_character: str,
    documentation_character: str,
    legacy_character: str,
    work_suffix: str,
    cooldown_seconds: int = 20,
):
    cursor.execute(
        """
        SELECT public.research_lab_source_add_admit_v3(
            %s::JSONB, %s, %s, %s, %s, %s,
            3, 5, 10, %s
        )
        """,
        (
            _json(record),
            "sha256:" + identity_character * 64,
            "sha256:" + documentation_character * 64,
            "sha256:" + legacy_character * 64,
            record["provider_origin_hash"],
            f"source_add_work:{work_suffix}",
            cooldown_seconds,
        ),
    )
    return cursor.fetchone()[0]


def _admit_v2(
    cursor,
    record: dict,
    *,
    identity_character: str,
    documentation_character: str,
    legacy_character: str,
    work_suffix: str,
):
    cursor.execute(
        """
        SELECT public.research_lab_source_add_admit_v2(
            %s::JSONB, %s, %s, %s, %s, %s,
            3, 5, 10
        )
        """,
        (
            _json(record),
            "sha256:" + identity_character * 64,
            "sha256:" + documentation_character * 64,
            "sha256:" + legacy_character * 64,
            record["provider_origin_hash"],
            f"source_add_work:{work_suffix}",
        ),
    )
    return cursor.fetchone()[0]


def test_concurrent_duplicates_precede_cooldown_and_survive_restart(database):
    psycopg2, dsn = database

    record = _record(
        suffix="6900000000000001",
        host="api.concurrent-privacy.example",
        miner="5DuplicatePrivacyMiner",
    )
    duplicate_request_count = 20
    barrier = threading.Barrier(duplicate_request_count)
    outcomes: list[str] = []
    errors: list[BaseException] = []
    outcome_lock = threading.Lock()

    def submit() -> None:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                barrier.wait(timeout=5)
                result = _admit(
                    cursor,
                    record,
                    identity_character="1",
                    documentation_character="2",
                    legacy_character="3",
                    work_suffix="6900000000000001",
                )
                with outcome_lock:
                    outcomes.append(result["status"])
        except BaseException as exc:  # surfaced in the parent test thread
            with outcome_lock:
                errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=submit) for _ in range(duplicate_request_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(outcomes) == ["admitted"] + ["duplicate"] * (
        duplicate_request_count - 1
    )

    verify = psycopg2.connect(**dsn)
    verify.autocommit = True
    with verify.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_submission_current
                 WHERE submission_id = %s),
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_provider_origin_current
                 WHERE provider_origin_hash = %s
                   AND reservation_status = 'reserved'),
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_work_items
                 WHERE submission_id = %s AND work_kind = 'provenance'),
                (SELECT COUNT(DISTINCT submission_id)
                 FROM public.research_lab_source_add_identity_current
                 WHERE source_identity_hash IN (%s, %s, %s)
                   AND reservation_status = 'reserved')
            """,
            (
                record["submission_id"],
                record["provider_origin_hash"],
                record["submission_id"],
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
                "sha256:" + "3" * 64,
            ),
        )
        assert cursor.fetchone() == (1, 1, 1, 1)
        cursor.execute(
            """
            SELECT
                has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_admit_v3('
                    'jsonb,text,text,text,text,text,'
                    'integer,integer,integer,integer)',
                    'EXECUTE'
                ),
                has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_admit_v3('
                    'jsonb,text,text,text,text,text,'
                    'integer,integer,integer,integer)',
                    'EXECUTE'
                ),
                has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_admit_v3('
                    'jsonb,text,text,text,text,text,'
                    'integer,integer,integer,integer)',
                    'EXECUTE'
                )
            """
        )
        assert cursor.fetchone() == (True, False, False)
        cursor.execute(
            "SELECT public.research_lab_source_add_duplicate_privacy_contract_v1()"
        )
        contract = cursor.fetchone()[0]
        assert contract["function_authority_sha256"] == (
            SOURCE_ADD_DUPLICATE_PRIVACY_FUNCTION_AUTHORITY_SHA256
        )
        assert contract["admission_signature"] == (
            "jsonb,text,text,text,text,text,integer,integer,integer,integer"
        )
        assert contract["compatibility_cooldown_seconds"] == 20
        assert contract["duplicate_precedes_cooldown"] is True
        assert all(contract["functions"].values())
        assert contract["permissions"] == {
            "service_role_exists": True,
            "v3_service_role_callable": True,
            "v2_service_role_callable": True,
            "contract_service_role_callable": True,
            "anon_callable": False,
            "authenticated_callable": False,
        }
    verify.close()

    # A fresh connection represents a restarted gateway process: both the
    # duplicate decision and distinct-source cooldown come from durable rows.
    restarted = psycopg2.connect(**dsn)
    restarted.autocommit = True
    with restarted.cursor() as cursor:
        assert (
            _admit(
                cursor,
                record,
                identity_character="1",
                documentation_character="2",
                legacy_character="3",
                work_suffix="6900000000000001",
            )["status"]
            == "duplicate"
        )

        distinct = _record(
            suffix="6900000000000002",
            host="api.distinct-privacy.example",
            miner=record["miner_hotkey"],
        )
        cooldown = _admit(
            cursor,
            distinct,
            identity_character="4",
            documentation_character="5",
            legacy_character="6",
            work_suffix="6900000000000002",
        )
        assert cooldown["status"] == "route_cooldown"
        assert cooldown["cooldown_seconds"] == 20
        assert 1 <= cooldown["wait_seconds"] <= 20

        cursor.execute(
            """
            SELECT
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_submission_current
                 WHERE submission_id = %s),
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_provider_origin_current
                 WHERE provider_origin_hash = %s),
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_work_items
                 WHERE submission_id = %s)
            """,
            (
                distinct["submission_id"],
                distinct["provider_origin_hash"],
                distinct["submission_id"],
            ),
        )
        assert cursor.fetchone() == (0, 0, 0)
    restarted.close()


def test_mixed_v2_v3_cross_key_duplicate_is_atomic(database):
    psycopg2, dsn = database
    miner = "5MixedRollingPrivacyMiner"
    first = _record(
        suffix="6900000000000011",
        host="api.mixed-v2-privacy.example",
        miner=miner,
    )
    second = _record(
        suffix="6900000000000011",
        host="api.mixed-v3-privacy.example",
        miner=miner,
    )
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    errors: list[BaseException] = []
    outcome_lock = threading.Lock()

    def submit(*, compatibility_v2: bool) -> None:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                barrier.wait(timeout=5)
                if compatibility_v2:
                    result = _admit_v2(
                        cursor,
                        first,
                        identity_character="7",
                        documentation_character="8",
                        legacy_character="9",
                        work_suffix="6900000000000011",
                    )
                else:
                    result = _admit(
                        cursor,
                        second,
                        identity_character="a",
                        documentation_character="b",
                        legacy_character="c",
                        work_suffix="6900000000000011",
                    )
                with outcome_lock:
                    outcomes.append(result["status"])
        except BaseException as exc:
            with outcome_lock:
                errors.append(exc)
        finally:
            connection.close()

    threads = [
        threading.Thread(target=submit, kwargs={"compatibility_v2": True}),
        threading.Thread(target=submit, kwargs={"compatibility_v2": False}),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(outcomes) == ["admitted", "duplicate"]

    verify = psycopg2.connect(**dsn)
    verify.autocommit = True
    with verify.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_submission_current
                 WHERE submission_id = %s),
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_provider_origin_current
                 WHERE provider_origin_hash IN (%s, %s)
                   AND reservation_status = 'reserved'),
                (SELECT COUNT(*)
                 FROM public.research_lab_source_add_work_items
                 WHERE work_id = %s),
                (SELECT COUNT(DISTINCT submission_id)
                 FROM public.research_lab_source_add_identity_current
                 WHERE source_identity_hash IN (%s, %s, %s, %s, %s, %s)
                   AND reservation_status = 'reserved')
            """,
            (
                first["submission_id"],
                first["provider_origin_hash"],
                second["provider_origin_hash"],
                "source_add_work:6900000000000011",
                "sha256:" + "7" * 64,
                "sha256:" + "8" * 64,
                "sha256:" + "9" * 64,
                "sha256:" + "a" * 64,
                "sha256:" + "b" * 64,
                "sha256:" + "c" * 64,
            ),
        )
        assert cursor.fetchone() == (1, 1, 1, 1)
    verify.close()


def test_cooldown_uses_wall_clock_after_older_transaction_start(database):
    psycopg2, dsn = database
    miner = "5OldTransactionPrivacyMiner"
    early = psycopg2.connect(**dsn)
    early.autocommit = False
    with early.cursor() as cursor:
        cursor.execute("SELECT NOW()")
        transaction_started_at = cursor.fetchone()[0]

    admitted = _record(
        suffix="6900000000000021",
        host="api.later-admission-privacy.example",
        miner=miner,
    )
    later = psycopg2.connect(**dsn)
    later.autocommit = True
    with later.cursor() as cursor:
        assert (
            _admit_v2(
                cursor,
                admitted,
                identity_character="d",
                documentation_character="e",
                legacy_character="f",
                work_suffix="6900000000000021",
            )["status"]
            == "admitted"
        )
        cursor.execute(
            """
            SELECT created_at
            FROM public.research_lab_source_add_work_items
            WHERE work_id = %s
            """,
            ("source_add_work:6900000000000021",),
        )
        assert cursor.fetchone()[0] > transaction_started_at
    later.close()

    distinct = _record(
        suffix="6900000000000022",
        host="api.older-transaction-privacy.example",
        miner=miner,
    )
    with early.cursor() as cursor:
        cooldown = _admit(
            cursor,
            distinct,
            identity_character="4",
            documentation_character="5",
            legacy_character="6",
            work_suffix="6900000000000022",
        )
        assert cooldown["status"] == "route_cooldown"
        assert cooldown["cooldown_seconds"] == 20
        assert 1 <= cooldown["wait_seconds"] <= 20
    early.rollback()
    early.close()
