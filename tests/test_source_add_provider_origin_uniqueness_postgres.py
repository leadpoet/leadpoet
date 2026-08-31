"""Execute SOURCE_ADD exact-host uniqueness against real PostgreSQL."""

from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from research_lab.source_add_identity import (
    normalize_source_add_provider_origin,
    source_provider_origin_hash,
)
from tests.test_source_add_end_to_end_postgres import (
    PRE_ORIGIN_MIGRATIONS as MIGRATIONS,
    SCRIPTS,
    pre_origin_database as base_database,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "scripts"
    / "170-research-lab-source-add-provider-origin-uniqueness.sql"
)
BACKFILL_SUFFIX = "0000000000000168"
BACKFILL_IDENTITY_HASH = "sha256:" + "e" * 64


def _json(value):
    from psycopg2.extras import Json

    return Json(value, dumps=lambda item: json.dumps(item, sort_keys=True))


def _record(*, suffix: str, host: str, path: str, miner: str) -> dict:
    api_base_url = f"https://{host}{path}"
    origin_hash = source_provider_origin_hash(api_base_url)
    return {
        "submission_id": f"source_add_submission:{suffix}",
        "adapter_id": f"adapter:origin-{suffix}",
        "miner_hotkey": miner,
        "credential_envelope": {},
        "provider_origin_host": host,
        "provider_origin_hash": origin_hash,
        "manifest": {
            "credential_policy": "no_credentials",
            "credential_ref": "",
            "source_name": f"Origin {suffix}",
            "source_kind": "registry",
            "declared_base_domains": [host],
        },
        "source_metadata": {
            "api_base_url": api_base_url,
            "documentation_url": f"https://docs.{host}/docs",
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


@pytest.fixture(scope="module")
def origin_database(base_database):
    psycopg2, dsn = base_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS extensions")
        cursor.execute(
            "CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions"
        )
        legacy = _record(
            suffix=BACKFILL_SUFFIX,
            host="api.pre168-owner.example",
            path="/v1",
            miner="5Pre168Owner",
        )
        legacy.pop("provider_origin_host")
        legacy.pop("provider_origin_hash")
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
            ("pre-168 backfill fixture", "operator:pre-168-backfill"),
        )
        cursor.execute(
            """
            SELECT public.research_lab_source_add_admit(
                %s::JSONB, %s, %s, %s, %s, 10, 20, 30
            )
            """,
            (
                _json(legacy),
                BACKFILL_IDENTITY_HASH,
                "sha256:" + "d" * 64,
                "sha256:" + "c" * 64,
                "source_add_work:0000000000000168",
            ),
        )
        assert cursor.fetchone()[0]["status"] == "admitted"
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
            ("apply migration 169", "operator:pre-168-backfill"),
        )
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.close()
    return psycopg2, dsn


def _admit(
    cursor,
    record: dict,
    *,
    identity_character: str,
    work_suffix: str,
    include_secondary_identities: bool = True,
):
    origin_hash = str(record["provider_origin_hash"])
    cursor.execute(
        """
        SELECT public.research_lab_source_add_admit_v2(
            %s::JSONB, %s, %s, %s, %s, %s, 10, 20, 30
        )
        """,
        (
            _json(record),
            "sha256:" + identity_character * 64,
            (
                "sha256:" + chr(ord(identity_character) + 1) * 64
                if include_secondary_identities
                else ""
            ),
            (
                "sha256:" + chr(ord(identity_character) + 2) * 64
                if include_secondary_identities
                else ""
            ),
            origin_hash,
            f"source_add_work:{work_suffix}",
        ),
    )
    return cursor.fetchone()[0]


def _append_terminal(cursor, record: dict, *, stage: str = "rejected") -> None:
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_submissions (
            submission_id, adapter_id, miner_hotkey, stage, seq,
            submission_doc, precheck_status, precheck_doc,
            source_identity_hash, source_identity_version
        )
        SELECT
            current.submission_id, current.adapter_id, current.miner_hotkey,
            %s, current.seq + 1,
            current.submission_doc || jsonb_build_object('stage', %s),
            current.precheck_status, current.precheck_doc,
            current.source_identity_hash, current.source_identity_version
        FROM public.research_lab_source_add_submission_current current
        WHERE current.submission_id = %s
        """,
        (stage, stage, record["submission_id"]),
    )


def test_python_and_postgres_provider_origin_vectors_are_exact(origin_database):
    psycopg2, dsn = origin_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    vectors = (
        "https://api.parity.example/v1",
        "https://api.parity.example/v2/search",
        "https://data.parity.example/v1",
        "https://api.parity.example:443/v1",
        "https://api.parity.example:/v1",
        "https://api.parity.example:0443/v1",
        "https://api.parity.example:0/v1",
        "https://[2001:0db8:0000::1]/v1",
        "https://[::ffff:192.0.2.1]/v1",
        "https://[::ffff:c000:201]/v1",
        "https://[fe80::1%25eth0]/v1",
        "https://b\N{LATIN SMALL LETTER U WITH DIAERESIS}cher.example/v1",
        "https://localhost/v1",
        "https://api.parity.example:8443/v1",
        "https://127.1/v1",
        "https://api.parity.example/path with space",
    )
    with connection.cursor() as cursor:
        for value in vectors:
            cursor.execute(
                """
                SELECT
                    public.research_lab_source_add_provider_origin_host_v1(%s),
                    public.research_lab_source_add_provider_origin_hash_v1(%s)
                """,
                (value, value),
            )
            sql_host, sql_hash = cursor.fetchone()
            assert sql_host == normalize_source_add_provider_origin(value)
            assert sql_hash == source_provider_origin_hash(value)
    connection.close()


def test_pre168_backfilled_owner_can_enter_catalog_and_provisioning(
    origin_database,
):
    psycopg2, dsn = origin_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    submission_id = f"source_add_submission:{BACKFILL_SUFFIX}"
    adapter_id = f"adapter:origin-{BACKFILL_SUFFIX}"
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT submission_doc ? 'provider_origin_hash'
            FROM public.research_lab_source_add_submission_current
            WHERE submission_id = %s
            """,
            (submission_id,),
        )
        assert cursor.fetchone()[0] is False
        cursor.execute(
            """
            INSERT INTO public.research_lab_source_catalog (
                catalog_id, adapter_id, miner_ref, source_name, source_kind,
                declared_base_domains, registry_provider_id,
                measured_trial_yield, catalog_doc, source_identity_hash
            ) VALUES (
                'source_catalog:0000000000000168', %s, '5Pre168Owner',
                'Origin 0000000000000168', 'registry',
                '["api.pre168-owner.example"]'::JSONB,
                'pre168_owner', 0.0, '{}'::JSONB, %s
            )
            """,
            (adapter_id, BACKFILL_IDENTITY_HASH),
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_source_add_provisioning_events (
                provision_ref, catalog_id, submission_id, adapter_id,
                miner_hotkey, source_identity_hash, registry_provider_id,
                provision_status, seq, provision_doc, credential_envelope
            ) VALUES (
                'source_add_provision:0000000000000168',
                'source_catalog:0000000000000168', %s, %s,
                '5Pre168Owner', %s, 'pre168_owner',
                'approved_pending_provision', 0, '{}'::JSONB, '{}'::JSONB
            )
            """,
            (submission_id, adapter_id, BACKFILL_IDENTITY_HASH),
        )
        cursor.execute(
            """
            SELECT provision_status
            FROM public.research_lab_source_add_provisioning_current
            WHERE submission_id = %s
            """,
            (submission_id,),
        )
        assert cursor.fetchone()[0] == "approved_pending_provision"
    connection.close()


def test_same_host_alias_is_duplicate_and_terminal_release_is_pre_reward_only(
    origin_database,
):
    psycopg2, dsn = origin_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    first = _record(
        suffix="1000000000000001",
        host="api.origin-one.example",
        path="/v1",
        miner="5OriginMinerOne",
    )
    alias = _record(
        suffix="1000000000000002",
        host="api.origin-one.example",
        path="/v2/search",
        miner="5OriginMinerTwo",
    )
    subdomain = _record(
        suffix="1000000000000003",
        host="data.origin-one.example",
        path="/v1",
        miner="5OriginMinerThree",
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
            ("provider-origin postgres", "operator:provider-origin-postgres"),
        )
        n_minus_one = _record(
            suffix="1000000000000000",
            host="api.n-minus-one.example",
            path="/v1",
            miner="5NMinusOneMiner",
        )
        n_minus_one.pop("provider_origin_host")
        n_minus_one.pop("provider_origin_hash")
        with pytest.raises(psycopg2.Error, match="provider-origin submission ownership"):
            cursor.execute(
                """
                SELECT public.research_lab_source_add_admit(
                    %s::JSONB, %s, %s, %s, %s, 10, 20, 30
                )
                """,
                (
                    _json(n_minus_one),
                    "sha256:" + "f" * 64,
                    "",
                    "",
                    "source_add_work:1000000000000000",
                ),
            )
        assert _admit(
            cursor,
            first,
            identity_character="1",
            work_suffix="1000000000000001",
        )["status"] == "admitted"
        assert _admit(
            cursor,
            alias,
            identity_character="4",
            work_suffix="1000000000000002",
        )["status"] == "duplicate"
        assert _admit(
            cursor,
            subdomain,
            identity_character="7",
            work_suffix="1000000000000003",
        )["status"] == "admitted"

        _append_terminal(cursor, first)
        cursor.execute(
            """
            SELECT reservation_status
            FROM public.research_lab_source_add_provider_origin_current
            WHERE provider_origin_hash = %s
            """,
            (first["provider_origin_hash"],),
        )
        assert cursor.fetchone()[0] == "released"
        assert _admit(
            cursor,
            alias,
            identity_character="4",
            work_suffix="1000000000000002",
        )["status"] == "admitted"

        cursor.execute(
            """
            INSERT INTO public.research_lab_source_add_reward_intents (
                intent_id, submission_id, adapter_id, miner_hotkey,
                intent_status, functional_receipt_hash,
                business_artifact_hash
            ) VALUES (%s, %s, %s, %s, 'queued', %s, %s)
            """,
            (
                "source_add_reward_intent:1000000000000002",
                alias["submission_id"],
                alias["adapter_id"],
                alias["miner_hotkey"],
                "sha256:" + "a" * 64,
                "sha256:" + "b" * 64,
            ),
        )
        _append_terminal(cursor, alias)
        cursor.execute(
            """
            SELECT reservation_status
            FROM public.research_lab_source_add_provider_origin_current
            WHERE provider_origin_hash = %s
            """,
            (alias["provider_origin_hash"],),
        )
        assert cursor.fetchone()[0] == "reserved"

        cursor.execute(
            "SELECT public.research_lab_source_add_provider_origin_contract_v1()"
        )
        contract_doc = cursor.fetchone()[0]
        assert contract_doc["coverage_complete"] is True
        assert contract_doc["collision_free"] is True
        assert contract_doc["submission_trigger_enabled"] is True
        assert contract_doc["catalog_trigger_enabled"] is True
        assert contract_doc["provision_trigger_enabled"] is True
        assert contract_doc["terminal_release_trigger_enabled"] is True
        assert contract_doc["append_only_trigger_enabled"] is True
        assert contract_doc["row_level_security_enabled"] is True
        assert contract_doc["service_role_policy_enabled"] is True
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
            ("reapply migration 169", "operator:provider-origin-postgres"),
        )
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        cursor.execute(
            "SELECT public.research_lab_source_add_provider_origin_contract_v1()"
        )
        reapplied_contract = cursor.fetchone()[0]
        assert reapplied_contract["coverage_complete"] is True
        assert reapplied_contract["collision_free"] is True
    connection.close()


def test_concurrent_same_host_admission_has_exactly_one_owner(origin_database):
    psycopg2, dsn = origin_database
    setup = psycopg2.connect(**dsn)
    setup.autocommit = True
    with setup.cursor() as cursor:
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
            ("concurrent origin admission", "operator:origin-concurrency"),
        )
    setup.close()

    records = (
        _record(
            suffix="3000000000000001",
            host="api.concurrent-origin.example",
            path="/v1",
            miner="5ConcurrentOriginOne",
        ),
        _record(
            suffix="3000000000000002",
            host="api.concurrent-origin.example",
            path="/v2",
            miner="5ConcurrentOriginTwo",
        ),
    )
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    errors: list[BaseException] = []
    outcome_lock = threading.Lock()

    def submit(index: int) -> None:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                barrier.wait(timeout=5)
                result = _admit(
                    cursor,
                        records[index],
                        identity_character=("a", "b")[index],
                        work_suffix=("3000000000000001", "3000000000000002")[index],
                        include_secondary_identities=False,
                )
                with outcome_lock:
                    outcomes.append(result["status"])
        except BaseException as exc:  # surfaced in the parent test thread
            with outcome_lock:
                errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(outcomes) == ["admitted", "duplicate"]

    verify = psycopg2.connect(**dsn)
    verify.autocommit = True
    with verify.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM public.research_lab_source_add_provider_origin_current
            WHERE provider_origin_hash = %s
              AND reservation_status = 'reserved'
            """,
            (records[0]["provider_origin_hash"],),
        )
        assert cursor.fetchone()[0] == 1
    verify.close()


def test_legacy_same_host_backfill_collision_preserves_fifo_owner(origin_database):
    psycopg2, dsn = origin_database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    database_name = "source_add_origin_collision"
    with admin.cursor() as cursor:
        cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
        cursor.execute(f"CREATE DATABASE {database_name}")
    admin.close()

    collision_dsn = {**dsn, "dbname": database_name}
    connection = psycopg2.connect(**collision_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA extensions")
            cursor.execute("CREATE EXTENSION pgcrypto WITH SCHEMA extensions")
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
            for migration in MIGRATIONS:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
                ("collision fixture", "operator:collision-fixture"),
            )
            first = _record(
                suffix="2000000000000001",
                host="api.legacy-collision.example",
                path="/v1",
                miner="5LegacyCollisionOne",
            )
            second = _record(
                suffix="2000000000000002",
                host="api.legacy-collision.example",
                path="/v2",
                miner="5LegacyCollisionTwo",
            )
            for index, record in enumerate((first, second), start=1):
                legacy_record = dict(record)
                legacy_record.pop("provider_origin_host")
                legacy_record.pop("provider_origin_hash")
                cursor.execute(
                    """
                    SELECT public.research_lab_source_add_admit(
                        %s::JSONB, %s, %s, %s, %s, 10, 20, 30
                    )
                    """,
                    (
                        _json(legacy_record),
                        "sha256:" + str(index) * 64,
                        "sha256:" + str(index + 2) * 64,
                        "sha256:" + str(index + 4) * 64,
                        f"source_add_work:200000000000000{index}",
                    ),
                )
                assert cursor.fetchone()[0]["status"] == "admitted"
            with pytest.raises(psycopg2.Error, match="must be paused"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
                ("collision migration", "operator:collision-fixture"),
            )
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                """
                SELECT submission_id, stage, precheck_status
                FROM public.research_lab_source_add_submission_current
                WHERE submission_id IN (%s, %s)
                ORDER BY submission_id
                """,
                (first["submission_id"], second["submission_id"]),
            )
            assert cursor.fetchall() == [
                (first["submission_id"], "provenance_queued", ""),
                (second["submission_id"], "rejected_precheck", "rejected_precheck"),
            ]
            cursor.execute(
                """
                SELECT submission_id, work_status,
                       result_doc->>'reason_code'
                FROM public.research_lab_source_add_work_items
                WHERE submission_id IN (%s, %s)
                ORDER BY submission_id
                """,
                (first["submission_id"], second["submission_id"]),
            )
            assert cursor.fetchall() == [
                (first["submission_id"], "queued", None),
                (
                    second["submission_id"],
                    "cancelled",
                    "duplicate_provider_origin_existing_owner",
                ),
            ]
            cursor.execute(
                """
                SELECT submission_id, array_agg(DISTINCT reservation_status)
                FROM public.research_lab_source_add_identity_current
                WHERE submission_id IN (%s, %s)
                GROUP BY submission_id
                ORDER BY submission_id
                """,
                (first["submission_id"], second["submission_id"]),
            )
            assert cursor.fetchall() == [
                (first["submission_id"], ["reserved"]),
                (second["submission_id"], ["released"]),
            ]
            cursor.execute(
                """
                SELECT submission_id, reservation_status
                FROM public.research_lab_source_add_provider_origin_current
                """
            )
            assert cursor.fetchall() == [(first["submission_id"], "reserved")]
            cursor.execute(
                "SELECT public.research_lab_source_add_provider_origin_contract_v1()"
            )
            contract = cursor.fetchone()[0]
            assert contract["coverage_complete"] is True
            assert contract["collision_free"] is True

            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_submissions
                WHERE submission_id = %s AND stage = 'rejected_precheck'
                """,
                (second["submission_id"],),
            )
            assert cursor.fetchone()[0] == 1
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
            cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
        admin.close()


def test_legacy_same_host_backfill_preserves_permanent_owner(origin_database):
    psycopg2, dsn = origin_database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    database_name = "source_add_origin_permanent"
    with admin.cursor() as cursor:
        cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
        cursor.execute(f"CREATE DATABASE {database_name}")
    admin.close()

    permanent_dsn = {**dsn, "dbname": database_name}
    connection = psycopg2.connect(**permanent_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA extensions")
            cursor.execute("CREATE EXTENSION pgcrypto WITH SCHEMA extensions")
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
            for migration in MIGRATIONS[:-1]:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
                ("permanent fixture", "operator:permanent-fixture"),
            )
            first = _record(
                suffix="2100000000000001",
                host="api.permanent-collision.example",
                path="/v1",
                miner="5PermanentCollisionOne",
            )
            second = _record(
                suffix="2100000000000002",
                host="api.permanent-collision.example",
                path="/v2",
                miner="5PermanentCollisionTwo",
            )
            for index, record in enumerate((first, second), start=1):
                legacy_record = dict(record)
                legacy_record.pop("provider_origin_host")
                legacy_record.pop("provider_origin_hash")
                cursor.execute(
                    """
                    SELECT public.research_lab_source_add_admit(
                        %s::JSONB, %s, %s, %s, %s, 10, 20, 30
                    )
                    """,
                    (
                        _json(legacy_record),
                        "sha256:" + str(index) * 64,
                        "",
                        "",
                        f"source_add_work:210000000000000{index}",
                    ),
                )
                assert cursor.fetchone()[0]["status"] == "admitted"
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_catalog (
                    catalog_id, adapter_id, miner_ref, source_name, source_kind,
                    declared_base_domains, registry_provider_id,
                    measured_trial_yield, catalog_doc, source_identity_hash
                ) VALUES (%s, %s, %s, %s, %s, %s::JSONB, %s, 0, '{}'::JSONB, %s)
                """,
                (
                    "source_catalog:2100000000000002",
                    second["adapter_id"],
                    second["miner_hotkey"],
                    second["manifest"]["source_name"],
                    second["manifest"]["source_kind"],
                    _json(second["manifest"]["declared_base_domains"]),
                    "legacy_permanent_collision",
                    "sha256:" + "2" * 64,
                ),
            )
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
                ("permanent migration", "operator:permanent-fixture"),
            )
            cursor.execute(
                (SCRIPTS / "169-research-lab-source-add-post-accept-leg1.sql")
                .read_text(encoding="utf-8")
            )
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                """
                SELECT submission_id, stage
                FROM public.research_lab_source_add_submission_current
                WHERE submission_id IN (%s, %s)
                ORDER BY submission_id
                """,
                (first["submission_id"], second["submission_id"]),
            )
            assert cursor.fetchall() == [
                (first["submission_id"], "rejected_precheck"),
                (second["submission_id"], "provenance_queued"),
            ]
            cursor.execute(
                """
                SELECT submission_id, reservation_status
                FROM public.research_lab_source_add_provider_origin_current
                """
            )
            assert cursor.fetchall() == [(second["submission_id"], "reserved")]
            cursor.execute(
                """
                SELECT submission_id, work_status
                FROM public.research_lab_source_add_work_items
                WHERE submission_id IN (%s, %s)
                ORDER BY submission_id
                """,
                (first["submission_id"], second["submission_id"]),
            )
            assert cursor.fetchall() == [
                (first["submission_id"], "cancelled"),
                (second["submission_id"], "queued"),
            ]
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
            cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
        admin.close()


def test_legacy_backfill_reconciles_live_three_group_shape(origin_database):
    psycopg2, dsn = origin_database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    database_name = "source_add_origin_live_shape"
    with admin.cursor() as cursor:
        cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
        cursor.execute(f"CREATE DATABASE {database_name}")
    admin.close()

    live_shape_dsn = {**dsn, "dbname": database_name}
    connection = psycopg2.connect(**live_shape_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA extensions")
            cursor.execute("CREATE EXTENSION pgcrypto WITH SCHEMA extensions")
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
            for migration in MIGRATIONS[:-1]:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
                ("live-shape fixture", "operator:live-shape-fixture"),
            )
            records = []
            for group, host in enumerate(
                (
                    "api.catalog-owner.example",
                    "api.reward-owner.example",
                    "api.fifo-owner.example",
                ),
                start=1,
            ):
                for member in (1, 2):
                    suffix = f"22{group:02d}{member:02d}0000000000"
                    record = _record(
                        suffix=suffix,
                        host=host,
                        path=f"/v{member}",
                        miner=f"5LiveShape{group}{member}",
                    )
                    legacy_record = dict(record)
                    legacy_record.pop("provider_origin_host")
                    legacy_record.pop("provider_origin_hash")
                    cursor.execute(
                        """
                        SELECT public.research_lab_source_add_admit(
                            %s::JSONB, %s, %s, %s, %s, 10, 20, 30
                        )
                        """,
                        (
                            _json(legacy_record),
                            "sha256:" + str((group - 1) * 2 + member) * 64,
                            "",
                            "",
                            f"source_add_work:{suffix}",
                        ),
                    )
                    assert cursor.fetchone()[0]["status"] == "admitted"
                    records.append(record)

            catalog_owner = records[1]
            reward_owner = records[3]
            fifo_owner = records[4]
            losers = (records[0], records[2], records[5])
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_catalog (
                    catalog_id, adapter_id, miner_ref, source_name, source_kind,
                    declared_base_domains, registry_provider_id,
                    measured_trial_yield, catalog_doc, source_identity_hash
                ) VALUES (%s, %s, %s, %s, %s, %s::JSONB, %s, 0, '{}'::JSONB, %s)
                """,
                (
                    "source_catalog:2201020000000000",
                    catalog_owner["adapter_id"],
                    catalog_owner["miner_hotkey"],
                    catalog_owner["manifest"]["source_name"],
                    catalog_owner["manifest"]["source_kind"],
                    _json(catalog_owner["manifest"]["declared_base_domains"]),
                    "legacy_live_shape_catalog",
                    "sha256:" + "2" * 64,
                ),
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_reward_obligations (
                    reward_ref, adapter_id, catalog_id, miner_hotkey, leg,
                    reward_kind, alpha_percent, reward_epochs, start_epoch,
                    trigger_evidence_doc
                ) VALUES (
                    'source_add_reward:2202020000000000', %s, NULL, %s, 1,
                    'source_acceptance', 1, 20, 100, '{}'::JSONB
                )
                """,
                (reward_owner["adapter_id"], reward_owner["miner_hotkey"]),
            )
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_reward_events (
                    reward_ref, seq, reward_status, reason
                ) VALUES
                    ('source_add_reward:2202020000000000', 0, 'active',
                     'legacy_pre_accept_reward'),
                    ('source_add_reward:2202020000000000', 1,
                     'stopped_forward', 'legacy_reward_retired')
                """
            )
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
                ("live-shape migration", "operator:live-shape-fixture"),
            )
            cursor.execute(
                (SCRIPTS / "169-research-lab-source-add-post-accept-leg1.sql")
                .read_text(encoding="utf-8")
            )
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))

            cursor.execute(
                """
                SELECT submission_id
                FROM public.research_lab_source_add_provider_origin_current
                WHERE reservation_status = 'reserved'
                ORDER BY submission_id
                """
            )
            assert [row[0] for row in cursor.fetchall()] == sorted(
                record["submission_id"]
                for record in (catalog_owner, reward_owner, fifo_owner)
            )
            cursor.execute(
                """
                SELECT submission_id, stage
                FROM public.research_lab_source_add_submission_current
                WHERE submission_id = ANY(%s)
                ORDER BY submission_id
                """,
                ([record["submission_id"] for record in records],),
            )
            observed_stages = dict(cursor.fetchall())
            for winner in (catalog_owner, reward_owner, fifo_owner):
                assert observed_stages[winner["submission_id"]] == "provenance_queued"
            for loser in losers:
                assert observed_stages[loser["submission_id"]] == "rejected_precheck"
            cursor.execute(
                """
                SELECT submission_id, work_status
                FROM public.research_lab_source_add_work_items
                WHERE submission_id = ANY(%s)
                ORDER BY submission_id
                """,
                ([record["submission_id"] for record in records],),
            )
            observed_work = dict(cursor.fetchall())
            for winner in (catalog_owner, reward_owner, fifo_owner):
                assert observed_work[winner["submission_id"]] == "queued"
            for loser in losers:
                assert observed_work[loser["submission_id"]] == "cancelled"
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.research_lab_source_add_submissions
                WHERE submission_id = ANY(%s)
                  AND stage = 'rejected_precheck'
                """,
                ([record["submission_id"] for record in losers],),
            )
            assert cursor.fetchone()[0] == 3
            cursor.execute(
                "SELECT public.research_lab_source_add_provider_origin_contract_v1()"
            )
            contract = cursor.fetchone()[0]
            assert contract["coverage_complete"] is True
            assert contract["collision_free"] is True
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
            cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
        admin.close()


def test_catalog_insert_without_reserved_origin_owner_fails_closed(origin_database):
    psycopg2, dsn = origin_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        with pytest.raises(psycopg2.Error, match="provider-origin owner"):
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_catalog (
                    catalog_id, adapter_id, miner_ref, source_name, source_kind,
                    declared_base_domains, registry_provider_id,
                    measured_trial_yield, catalog_doc, source_identity_hash
                ) VALUES (
                    'source_catalog:9999999999999999', 'adapter:no-origin-owner',
                    '5NoOriginOwner', 'No Origin', 'registry', '[]'::JSONB,
                    'no_origin_owner', 0.0, '{}'::JSONB, %s
                )
                """,
                ("sha256:" + "f" * 64,),
            )
    connection.close()
