"""Production-order PostgreSQL SOURCE_ADD workflow contract."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest

from research_lab.source_add_identity import (
    normalize_source_add_provider_origin,
    source_provider_origin_hash,
)
from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2
from gateway.research_lab.source_add_workflow import source_add_probe_attempt_ref
from leadpoet_canonical.attested_v2 import sha256_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PRE_ORIGIN_MIGRATIONS = (
    "72-research-lab-source-experiments.sql",
    "74-research-lab-source-add-provenance-precheck.sql",
    "78-research-lab-source-add-catalog-provisioning.sql",
    "79-research-lab-source-add-llm-leg2-evidence.sql",
    "82-research-lab-source-add-llm-only-leg2.sql",
    "84-expand-source-add-source-kinds.sql",
    "86-research-lab-attested-v2-authority.sql",
    "96-research-lab-source-add-functional-workflow.sql",
    "145-research-lab-source-add-admission-control.sql",
    "169-research-lab-source-add-post-accept-leg1.sql",
)
MIGRATIONS = PRE_ORIGIN_MIGRATIONS + (
    "170-research-lab-source-add-provider-origin-uniqueness.sql",
)
DOCKER = shutil.which("docker")
pytestmark = pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")

SUBMISSION_ID = "source_add_submission:0123456789abcdef"
ADAPTER_ID = "adapter:postgres-e2e"
MINER_HOTKEY = "5PostgresSourceAddMinerHotkey"
IDENTITY_HASH = "sha256:" + "1" * 64
CONFIG_REF = "source_add_probe_config:1111111111111111"
PROVENANCE_WORK = "source_add_work:1111111111111111"
FUNCTIONAL_WORK = "source_add_work:2222222222222222"
FUNCTIONAL_ATTEMPT = "source_add_probe_attempt:1111111111111111"
REWARD_INTENT = "source_add_reward_intent:1111111111111111"
REWARD_WORK = "source_add_work:3333333333333333"
REWARD_REF = "source_add_reward:1111111111111111"
SMOKE_WORK = "source_add_work:4444444444444444"
SMOKE_ATTEMPT = source_add_probe_attempt_ref(SUBMISSION_ID, SMOKE_WORK, 3)
FAILED_SMOKE_ATTEMPT = "source_add_probe_attempt:3333333333333333"
CATALOG_UNAVAILABLE_SMOKE_ATTEMPT = "source_add_probe_attempt:4444444444444444"
CATALOG_ID = "source_catalog:1111111111111111"
REGISTRY_PROVIDER_ID = "sourceadd_postgres_e2e"
ROUTE_HASH = "sha256:" + "2" * 64
FUNCTIONAL_RECEIPT = "sha256:" + "3" * 64
FUNCTIONAL_ARTIFACT = "sha256:" + "4" * 64
DECISION_RECEIPT = "sha256:" + "5" * 64
STALE_DECISION_RECEIPT = "sha256:" + "6" * 64
SMOKE_RECEIPT = "sha256:" + "7" * 64
SMOKE_ARTIFACT = "sha256:" + "8" * 64
FAILED_SMOKE_RECEIPT = "sha256:" + "b" * 64
FAILED_SMOKE_ARTIFACT = "sha256:" + "c" * 64
CATALOG_UNAVAILABLE_SMOKE_RECEIPT = "sha256:" + "d" * 64
CATALOG_UNAVAILABLE_SMOKE_ARTIFACT = "sha256:" + "e" * 64


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _database_with_migrations(migrations, *, setup_sql=""):
    psycopg2 = pytest.importorskip("psycopg2")
    port = _free_port()
    container = "source-add-e2e-%s" % uuid4().hex[:12]
    started = False
    try:
        result = subprocess.run(
            [
                str(DOCKER),
                "run",
                "--rm",
                "--detach",
                "--name",
                container,
                "--env",
                "POSTGRES_PASSWORD=postgres",
                "--publish",
                "127.0.0.1:%d:5432" % port,
                "postgres:15",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip("PostgreSQL container could not start: %s" % result.stderr[-300:])
        started = True
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            ready = subprocess.run(
                [str(DOCKER), "exec", container, "pg_isready", "-U", "postgres"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            pytest.fail("PostgreSQL container did not become ready")

        dsn = {
            "host": "127.0.0.1",
            "port": port,
            "user": "postgres",
            "password": "postgres",
            "dbname": "postgres",
        }
        connect_deadline = time.monotonic() + 15
        while True:
            try:
                connection = psycopg2.connect(**dsn)
                break
            except psycopg2.OperationalError:
                if time.monotonic() >= connect_deadline:
                    raise
                time.sleep(0.25)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE SCHEMA IF NOT EXISTS extensions;
                CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
                CREATE ROLE anon NOLOGIN;
                CREATE ROLE authenticated NOLOGIN;
                CREATE ROLE service_role NOLOGIN;
                CREATE TABLE public.research_lab_auto_research_loop_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type TEXT NOT NULL,
                    CONSTRAINT research_lab_auto_research_loop_events_event_type_check
                        CHECK (event_type = 'loop_started')
                );
                """
            )
            if setup_sql:
                cursor.execute(setup_sql)
            for migration in migrations:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
        connection.close()
        yield psycopg2, dsn
    finally:
        if started:
            subprocess.run(
                [str(DOCKER), "rm", "--force", container],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )


@pytest.fixture(scope="module")
def database():
    yield from _database_with_migrations(MIGRATIONS)


@pytest.fixture(scope="module")
def pre_origin_database():
    yield from _database_with_migrations(PRE_ORIGIN_MIGRATIONS)


def _json(value):
    from psycopg2.extras import Json

    return Json(value, dumps=lambda item: json.dumps(item, sort_keys=True))


def _scalar(cursor, statement: str, parameters=()):
    cursor.execute(statement, parameters)
    return cursor.fetchone()[0]


def _record_doc() -> dict:
    api_base_url = "https://api.source-add.test/v1"
    return {
        "submission_id": SUBMISSION_ID,
        "adapter_id": ADAPTER_ID,
        "miner_hotkey": MINER_HOTKEY,
        "credential_envelope": {},
        "provider_origin_host": normalize_source_add_provider_origin(
            api_base_url
        ),
        "provider_origin_hash": source_provider_origin_hash(api_base_url),
        "manifest": {
            "credential_policy": "no_credentials",
            "credential_ref": "",
            "source_name": "PostgreSQL E2E Registry",
            "source_kind": "registry",
            "declared_base_domains": ["api.source-add.test"],
        },
        "source_metadata": {
            "api_base_url": api_base_url,
            "documentation_url": "https://source-add.test/docs",
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


def _probe_doc() -> dict:
    return {
        "schema_version": "leadpoet.source_add_probe_config.v2",
        "provider_id": REGISTRY_PROVIDER_ID,
        "base_url": "https://api.source-add.test/v1",
        "auth_kind": "none",
        "auth_name": "",
        "request_headers": {},
        "probes": [
            {
                "method": "GET",
                "path": "/records",
                "query": {"limit": "1"},
                "body_json": None,
            }
        ],
    }


def _seed_boot_identity(cursor) -> None:
    cursor.execute(
        """
        INSERT INTO public.research_lab_attested_boot_identities_v2 (
            boot_identity_hash, schema_version, role, physical_role, commit_sha,
            pcr0, build_manifest_hash, dependency_lock_hash, config_hash,
            signing_pubkey, transport_pubkey, transport_certificate_hash,
            boot_nonce, attestation_user_data_hash, attestation_document_ref,
            attestation_document_hash, identity_doc, issued_at
        ) VALUES (
            %s, 'leadpoet.attested_boot_identity.v2', 'gateway_coordinator',
            'gateway_coordinator', %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, 'artifact:test-attestation', %s, '{}'::JSONB, NOW()
        ) ON CONFLICT (boot_identity_hash) DO NOTHING
        """,
        (
            "sha256:" + "9" * 64,
            "a" * 40,
            "b" * 96,
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            "sha256:" + "c" * 64,
            "c" * 64,
            "d" * 64,
            "sha256:" + "d" * 64,
            "e" * 32,
            "sha256:" + "e" * 64,
            "sha256:" + "f" * 64,
        ),
    )


def _seed_receipt(
    cursor,
    *,
    receipt_hash: str,
    purpose: str,
    job_id: str,
    output_root: str,
    sequence: int,
    parent_receipt_hashes: tuple[str, ...] = (),
) -> None:
    marker = "%064x" % (sequence + 20)
    cursor.execute(
        """
        INSERT INTO public.research_lab_attested_execution_receipts_v2 (
            receipt_hash, schema_version, role, purpose, job_id, epoch_id,
            sequence, commit_sha, pcr0, build_manifest_hash,
            dependency_lock_hash, config_hash, boot_identity_hash, input_root,
            output_root, transport_root, host_operation_root, artifact_root,
            receipt_status, failure_code, enclave_pubkey, enclave_signature,
            receipt_doc, issued_at
        ) VALUES (
            %s, 'leadpoet.attested_execution_receipt.v2',
            'gateway_coordinator', %s, %s, 700, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, 'succeeded', NULL, %s, %s,
            %s::JSONB, NOW()
        )
        """,
        (
            receipt_hash,
            purpose,
            job_id,
            sequence,
            "a" * 40,
            "b" * 96,
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            "sha256:" + "c" * 64,
            "sha256:" + "9" * 64,
            "sha256:" + marker,
            output_root,
            "sha256:" + "0" * 64,
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            "c" * 64,
            "d" * 128,
            _json(
                {
                    "parent_receipt_hashes": sorted(parent_receipt_hashes),
                }
            ),
        ),
    )
    for parent_receipt_hash in parent_receipt_hashes:
        cursor.execute(
            """
            INSERT INTO public.research_lab_attested_receipt_edges_v2 (
                child_receipt_hash, parent_receipt_hash
            ) VALUES (%s, %s)
            """,
            (receipt_hash, parent_receipt_hash),
        )


def _seed_business_link(
    cursor,
    *,
    receipt_hash: str,
    artifact_kind: str,
    artifact_ref: str,
    artifact_hash: str,
) -> None:
    cursor.execute(
        """
        INSERT INTO public.research_lab_attested_business_artifact_links_v2 (
            receipt_hash, artifact_kind, artifact_ref, artifact_hash
        ) VALUES (%s, %s, %s, %s)
        """,
        (receipt_hash, artifact_kind, artifact_ref, artifact_hash),
    )


def _finish_work(
    cursor,
    *,
    work: dict,
    stage: str,
    submission_doc: dict,
    precheck_status: str,
    result_doc: dict | None = None,
    functional_attempt: dict | None = None,
    probe_config: dict | None = None,
    next_work: dict | None = None,
    reward_intent: dict | None = None,
) -> dict:
    return _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_finish_work(
            %s, %s::UUID, 'complete', %s, %s::JSONB, %s, '{}'::JSONB,
            %s::JSONB, %s::JSONB, %s::JSONB, %s::JSONB, %s::JSONB,
            NULL, FALSE
        )
        """,
        (
            work["work_id"],
            work["lease_token"],
            stage,
            _json(submission_doc),
            precheck_status,
            _json(result_doc or {}),
            _json(functional_attempt or {}),
            _json(probe_config or {}),
            _json(next_work or {}),
            _json(reward_intent or {}),
        ),
    )


def test_source_add_leg1_contract_rejects_privilege_and_trigger_drift(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = False
    try:
        with connection.cursor() as cursor:
            contract_sql = (
                "SELECT "
                "public.research_lab_source_add_post_accept_leg1_contract_v1()"
            )
            assert _scalar(cursor, contract_sql)["permissions"][
                "v2_callable"
            ] is True
            cursor.execute(
                """
                REVOKE EXECUTE ON FUNCTION
                public.research_lab_source_add_finalize_provision_smoke_v2(
                    TEXT, UUID, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB
                ) FROM service_role
                """
            )
            assert _scalar(cursor, contract_sql)["permissions"][
                "v2_callable"
            ] is False
        connection.rollback()

        with connection.cursor() as cursor:
            expected_authority = _scalar(cursor, contract_sql)[
                "function_authority_sha256"
            ]
            cursor.execute(
                """
                CREATE OR REPLACE FUNCTION
                public.research_lab_source_add_reserve_leg1_slot_v2(
                    p_intent_id TEXT,
                    p_work_id TEXT,
                    p_work_lease_token UUID,
                    p_daily_cap INTEGER,
                    p_slot_lease_seconds INTEGER
                ) RETURNS JSONB
                LANGUAGE sql
                SECURITY DEFINER
                SET search_path = pg_catalog, public
                AS $$ SELECT '{}'::JSONB $$
                """
            )
            assert _scalar(cursor, contract_sql)[
                "function_authority_sha256"
            ] != expected_authority
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                DROP TRIGGER trg_source_add_acceptance_v2
                    ON public.research_lab_source_add_submissions;
                CREATE TRIGGER trg_source_add_acceptance_v2
                    AFTER INSERT ON public.research_lab_source_add_submissions
                    FOR EACH ROW EXECUTE FUNCTION
                    public.enforce_research_lab_source_add_acceptance_v2()
                """
            )
            assert _scalar(cursor, contract_sql)["triggers"][
                "acceptance"
            ] is False
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE OR REPLACE FUNCTION public.source_add_test_noop_trigger()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$ BEGIN RETURN NEW; END; $$;
                DROP TRIGGER trg_source_add_acceptance_v2
                    ON public.research_lab_source_add_submissions;
                CREATE TRIGGER trg_source_add_acceptance_v2
                    BEFORE INSERT ON public.research_lab_source_add_submissions
                    FOR EACH ROW EXECUTE FUNCTION
                    public.source_add_test_noop_trigger()
                """
            )
            assert _scalar(cursor, contract_sql)["triggers"][
                "acceptance"
            ] is False
        connection.rollback()
    finally:
        connection.rollback()
        connection.close()


def _seed_leased_smoke_case(cursor, *, seed: int, base_url: str) -> dict:
    """Drive one unique source through the real DB path to a leased smoke."""

    def token(offset: int) -> str:
        return f"{seed + offset:016x}"

    submission_id = "source_add_submission:" + token(0)
    adapter_id = "adapter:postgres-e2e-" + token(0)
    miner_hotkey = "5PostgresE2E" + token(0)
    identity_hash = sha256_json(
        {"kind": "source_add_test_identity", "submission_id": submission_id}
    )
    config_ref = "source_add_probe_config:" + token(1)
    provenance_work_id = "source_add_work:" + token(2)
    functional_work_id = "source_add_work:" + token(3)
    smoke_work_id = "source_add_work:" + token(4)
    catalog_id = "source_catalog:" + token(5)
    registry_provider_id = "sourceadd_e2e_" + token(0)
    host_hash = sha256_json({"host": base_url})
    record_doc = {
        "submission_id": submission_id,
        "adapter_id": adapter_id,
        "miner_hotkey": miner_hotkey,
        "credential_envelope": {},
        "provider_origin_host": normalize_source_add_provider_origin(base_url),
        "provider_origin_hash": source_provider_origin_hash(base_url),
        "manifest": {
            "credential_policy": "no_credentials",
            "credential_ref": "",
            "source_name": "PostgreSQL E2E " + token(0),
            "source_kind": "registry",
            "declared_base_domains": [normalize_source_add_provider_origin(base_url)],
        },
        "source_metadata": {
            "api_base_url": base_url,
            "documentation_url": "https://docs.example.test/" + token(0),
            "auth_type": "none",
            "endpoint_examples": [
                {
                    "method": "GET",
                    "path": "/records",
                    "purpose": "Return bounded records",
                    "example_query": "limit=1",
                }
            ],
            "rate_limit_notes": "bounded",
        },
    }
    probe_doc = {
        "schema_version": "leadpoet.source_add_probe_config.v2",
        "provider_id": registry_provider_id,
        "base_url": base_url,
        "auth_kind": "none",
        "auth_name": "",
        "request_headers": {},
        "probes": [
            {
                "method": "GET",
                "path": "/records",
                "query": {"limit": "1"},
                "body_json": None,
            }
        ],
    }
    _scalar(
        cursor,
        "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
        ("postgres helper", "operator:postgres-helper"),
    )
    admitted = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_admit_v2(
            %s::JSONB, %s, %s, %s, %s, %s, 3, 5, 10
        )
        """,
        (
            _json(record_doc),
            identity_hash,
            sha256_json({"docs": token(0)}),
            sha256_json({"identity": token(0)}),
            record_doc["provider_origin_hash"],
            provenance_work_id,
        ),
    )
    assert admitted["status"] == "admitted"
    provenance = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s, 180)",
        ("postgres-seed-" + token(0),),
    )["work"]
    assert provenance["work_id"] == provenance_work_id
    assert _finish_work(
        cursor,
        work=provenance,
        stage="functional_probe_queued",
        submission_doc=record_doc,
        precheck_status="provenance_precheck_passed",
        probe_config={
            "config_ref": config_ref,
            "probe_doc": probe_doc,
            "credential_envelope": {},
            "actor_ref": "system:postgres-seed",
        },
        next_work={
            "work_id": functional_work_id,
            "work_kind": "functional_probe",
            "priority": 20,
            "job_doc": {"config_ref": config_ref, "host_hash": host_hash},
        },
    )["status"] == "completed"
    functional = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s, 180)",
        ("postgres-seed-" + token(0),),
    )["work"]
    assert functional["work_id"] == functional_work_id
    functional_result = {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "evaluator_version": (
            "leadpoet.source_add_functional_probe_evaluator.v2.1"
        ),
        "submission_id": submission_id,
        "adapter_id": adapter_id,
        "config_ref": config_ref,
        "evaluation_mode": "functional_probe",
        "result_status": "passed",
        "route_hash": sha256_json({"route": token(0)}),
    }
    functional_receipt = sha256_json(
        {"receipt": "functional", "submission_id": submission_id}
    )
    functional_artifact = sha256_json(functional_result)
    functional_attempt_ref = source_add_probe_attempt_ref(
        submission_id,
        functional_work_id,
        int(functional["attempt_count"]),
    )
    functional_attempt = {
        "attempt_ref": functional_attempt_ref,
        "evaluation_mode": "functional_probe",
        "config_ref": config_ref,
        "result_status": "passed",
        "route_hash": functional_result["route_hash"],
        "response_hash": sha256_json({"response": token(0)}),
        "status_class": "2xx",
        "content_type": "application/json",
        "byte_count": 64,
        "duration_ms": 5,
        "retry_after_seconds": 0,
        "reason_codes": ["bounded_json_data_response"],
        "receipt_hash": functional_receipt,
        "business_artifact_hash": functional_artifact,
        "result_doc": functional_result,
    }
    assert _finish_work(
        cursor,
        work=functional,
        stage="functional_probe_passed",
        submission_doc=record_doc,
        precheck_status="provenance_precheck_passed",
        functional_attempt=functional_attempt,
    )["status"] == "completed"
    _seed_boot_identity(cursor)
    _seed_receipt(
        cursor,
        receipt_hash=functional_receipt,
        purpose="research_lab.source_add_functional_probe.v2",
        job_id="seed-functional-" + token(0),
        output_root=functional_artifact,
        sequence=1000 + (seed & 0xFFFF),
    )
    _seed_business_link(
        cursor,
        receipt_hash=functional_receipt,
        artifact_kind="source_add_functional_probe",
        artifact_ref=functional_attempt_ref,
        artifact_hash=functional_artifact,
    )
    catalog_row = {
        "catalog_id": catalog_id,
        "adapter_id": adapter_id,
        "miner_ref": miner_hotkey,
        "source_name": record_doc["manifest"]["source_name"],
        "source_kind": "registry",
        "declared_base_domains": record_doc["manifest"][
            "declared_base_domains"
        ],
        "registry_provider_id": registry_provider_id,
        "catalog_doc": {"source": "postgres-seed"},
        "source_identity_hash": identity_hash,
    }
    provision_doc = {
        "provider_registry_entry": {
            "provider_id": registry_provider_id,
            "base_url": base_url,
            "auth_kind": "none",
            "auth_name": "",
            "active": True,
        },
        "request_headers": {},
        "probe_endpoints": [{"method": "GET", "path": "/records"}],
    }

    def provision_row(reference_offset: int, status: str) -> dict:
        return {
            "provision_ref": "source_add_provision:" + token(reference_offset),
            "submission_id": submission_id,
            "adapter_id": adapter_id,
            "miner_hotkey": miner_hotkey,
            "source_identity_hash": identity_hash,
            "registry_provider_id": registry_provider_id,
            "provision_status": status,
            "provision_doc": provision_doc,
            "credential_envelope": {},
        }

    pending = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_finalize_provision_v2(
            %s, %s::JSONB, %s::JSONB, '{}'::JSONB
        )
        """,
        (
            submission_id,
            _json(catalog_row),
            _json(provision_row(6, "approved_pending_provision")),
        ),
    )
    assert pending["status"] == "provisioned"
    eligible_row = provision_row(7, "provisioned_autoresearch_eligible")
    queued = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_enqueue_provision_smoke(
            %s, %s, %s, %s, %s::JSONB, %s::JSONB
        )
        """,
        (
            smoke_work_id,
            submission_id,
            config_ref,
            host_hash,
            _json(catalog_row),
            _json(eligible_row),
        ),
    )
    assert queued["status"] == "queued"
    smoke_work = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s, 180)",
        ("postgres-seed-" + token(0),),
    )["work"]
    assert smoke_work["work_id"] == smoke_work_id
    smoke_result = {
        **functional_result,
        "evaluation_mode": "provisioning_smoke",
    }
    smoke_receipt = sha256_json(
        {"receipt": "smoke", "submission_id": submission_id}
    )
    smoke_artifact = sha256_json(smoke_result)
    smoke_attempt_ref = source_add_probe_attempt_ref(
        submission_id,
        smoke_work_id,
        int(smoke_work["attempt_count"]),
    )
    _seed_receipt(
        cursor,
        receipt_hash=smoke_receipt,
        purpose="research_lab.source_add_functional_probe.v2",
        job_id="seed-smoke-" + token(0),
        output_root=smoke_artifact,
        sequence=2000 + (seed & 0xFFFF),
    )
    _seed_business_link(
        cursor,
        receipt_hash=smoke_receipt,
        artifact_kind="source_add_provisioning_smoke",
        artifact_ref=smoke_attempt_ref,
        artifact_hash=smoke_artifact,
    )
    smoke_attempt = {
        "attempt_ref": smoke_attempt_ref,
        "work_id": smoke_work_id,
        "attempt_number": int(smoke_work["attempt_count"]),
        "evaluation_mode": "provisioning_smoke",
        "config_ref": config_ref,
        "result_status": "passed",
        "route_hash": smoke_result["route_hash"],
        "response_hash": sha256_json({"smoke_response": token(0)}),
        "status_class": "2xx",
        "content_type": "application/json",
        "byte_count": 64,
        "duration_ms": 5,
        "retry_after_seconds": 0,
        "reason_codes": ["bounded_json_data_response"],
        "receipt_hash": smoke_receipt,
        "business_artifact_hash": smoke_artifact,
        "result_doc": smoke_result,
    }
    return {
        "submission_id": submission_id,
        "adapter_id": adapter_id,
        "miner_hotkey": miner_hotkey,
        "record_doc": record_doc,
        "catalog_row": catalog_row,
        "eligible_row": eligible_row,
        "provision_doc": provision_doc,
        "smoke_work": smoke_work,
        "smoke_attempt": smoke_attempt,
        "smoke_receipt": smoke_receipt,
        "smoke_artifact": smoke_artifact,
        "functional_receipt": functional_receipt,
        "functional_artifact": functional_artifact,
        "functional_attempt_ref": functional_attempt_ref,
    }


def _finalize_seed_smoke_to_leg1(cursor, *, case: dict, seed: int) -> dict:
    token = lambda offset: f"{seed + offset:016x}"
    intent_id = "source_add_reward_intent:" + token(8)
    reward_work_id = "source_add_work:" + token(9)
    result = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_finalize_provision_smoke_v2(
            %s, %s::UUID, %s, %s::JSONB, %s::JSONB, %s::JSONB,
            %s::JSONB, %s::JSONB
        )
        """,
        (
            case["smoke_work"]["work_id"],
            case["smoke_work"]["lease_token"],
            case["submission_id"],
            _json(case["catalog_row"]),
            _json(case["eligible_row"]),
            _json(case["smoke_attempt"]),
            _json(
                {
                    "intent_id": intent_id,
                    "miner_hotkey": case["miner_hotkey"],
                    "functional_receipt_hash": case["functional_receipt"],
                    "business_artifact_hash": case["functional_artifact"],
                }
            ),
            _json(
                {
                    "work_id": reward_work_id,
                    "work_kind": "leg1_reward",
                    "priority": 30,
                    "job_doc": {
                        "intent_id": intent_id,
                        "attempt_ref": case["functional_attempt_ref"],
                    },
                }
            ),
        ),
    )
    assert result["status"] == "provisioned"
    return {
        **case,
        "intent_id": intent_id,
        "reward_work_id": reward_work_id,
    }


def _create_seed_leg1_reward(
    cursor,
    *,
    case: dict,
    seed: int,
    alpha_percent: float = 1.0,
    economics_rpc_version: int = 2,
) -> dict:
    if economics_rpc_version not in (2, 3):
        raise ValueError("unsupported SOURCE_ADD economics RPC version")
    case = _finalize_seed_smoke_to_leg1(cursor, case=case, seed=seed)
    claimed = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s, 180)",
        ("postgres-reward-%016x" % seed,),
    )["work"]
    assert claimed["work_id"] == case["reward_work_id"]
    slot = _scalar(
        cursor,
        f"""
        SELECT public.research_lab_source_add_reserve_leg1_slot_v{economics_rpc_version}(
            %s, %s, %s::UUID, 100, 300
        )
        """,
        (case["intent_id"], case["reward_work_id"], claimed["lease_token"]),
    )
    assert slot["status"] == "reserved"
    trigger = {
        "functional_probe_passed": True,
        "attempt_ref": case["functional_attempt_ref"],
        "functional_probe_receipt_hash": case["functional_receipt"],
        "business_artifact_hash": case["functional_artifact"],
        "functional_probe_result_hash": case["functional_artifact"],
        "evaluator_version": (
            "leadpoet.source_add_functional_probe_evaluator.v2.1"
        ),
        "route_hash": case["smoke_attempt"]["route_hash"],
        "provisioning_smoke_passed": True,
        "provisioning_smoke_attempt_ref": case["smoke_attempt"]["attempt_ref"],
        "provisioning_smoke_receipt_hash": case["smoke_receipt"],
        "provisioning_smoke_business_artifact_hash": case["smoke_artifact"],
        "provisioning_smoke_result_hash": case["smoke_artifact"],
        "submission_id": case["submission_id"],
        "final_acceptance_stage": "accepted",
        "provision_ref": case["eligible_row"]["provision_ref"],
        "catalog_id": case["catalog_row"]["catalog_id"],
        "registry_provider_id": case["catalog_row"]["registry_provider_id"],
        "provision_status": "provisioned_autoresearch_eligible",
    }
    reward_ref = "source_add_reward:%016x" % (seed + 10)
    decision_receipt = sha256_json(
        {"receipt": "reward", "submission_id": case["submission_id"]}
    )
    reward_payload = {
        "reward_ref": reward_ref,
        "reward_kind": "source_acceptance",
        "alpha_percent": alpha_percent,
        "reward_epochs": 20,
        "start_epoch": 10_000 + (seed & 0xFFFF),
        "state": "active",
        "trigger_evidence_doc": trigger,
        "public_label": "SOURCE_ADD Leg 1",
        "decision_receipt_hash": decision_receipt,
    }
    decision_artifact = sha256_json(
        source_add_reward_row_projection_v2(
            "source_add_leg1",
            {
                **reward_payload,
                "adapter_id": case["adapter_id"],
                "miner_hotkey": case["miner_hotkey"],
                "leg": 1,
                "initial_reward_status": "active",
            },
        )
    )
    reward_payload["decision_artifact_hash"] = decision_artifact
    _seed_receipt(
        cursor,
        receipt_hash=decision_receipt,
        purpose="research_lab.reward_decision.v2",
        job_id="seed-reward-%016x" % seed,
        output_root=decision_artifact,
        sequence=3000 + (seed & 0xFFFF),
        parent_receipt_hashes=(
            case["functional_receipt"],
            case["smoke_receipt"],
        ),
    )
    _seed_business_link(
        cursor,
        receipt_hash=decision_receipt,
        artifact_kind="source_add_reward_decision",
        artifact_ref=reward_ref,
        artifact_hash=decision_artifact,
    )
    result = _scalar(
        cursor,
        f"""
        SELECT public.research_lab_source_add_finalize_leg1_v{economics_rpc_version}(
            %s, %s, %s::UUID, %s::UUID, 100, %s::JSONB, %s::JSONB
        )
        """,
        (
            case["intent_id"],
            case["reward_work_id"],
            claimed["lease_token"],
            slot["slot_lease_token"],
            _json(reward_payload),
            _json(case["record_doc"]),
        ),
    )
    assert result == {"status": "created", "reward_ref": reward_ref}
    return {**case, "reward_ref": reward_ref}


def test_source_add_complete_database_workflow(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    record_doc = _record_doc()
    probe_doc = _probe_doc()
    host_hash = "sha256:" + "a" * 64
    with connection.cursor() as cursor:
        contract = _scalar(
            cursor,
            "SELECT public.research_lab_source_add_admission_control_contract_v1()",
        )
        assert contract["control_row_present"] is True
        assert contract["trigger_enabled"] is True
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_post_accept_leg1_contract_v1()",
        ) == {
            "schema_version": (
                "leadpoet.source_add_post_accept_leg1_contract.v1"
            ),
            "daily_cap": 10,
            "leg1_alpha_percent": 1.0,
            "leg1_reward_epochs": 20,
            "function_authority_sha256": (
                "sha256:035b4dc17bc8e8b63524df2c123892aa"
                "3ddaf0a01d08c69fc2d756921e8e96be"
            ),
            "functions": {
                "configure_probe_v2": True,
                "finalize_provision_v2": True,
                "reject_current_builtin_v2": True,
                "reserve_leg1_slot_v2": True,
                "finalize_leg1_v2": True,
                "finalize_provision_smoke_v2": True,
            },
            "triggers": {
                "acceptance": True,
                "eligible": True,
                "leg1_work": True,
                "leg1_slot": True,
                "leg1_obligation": True,
                "leg1_initial_event": True,
            },
            "permissions": {
                "service_role_exists": True,
                "v2_callable": True,
                "legacy_not_callable": True,
            },
        }
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work('paused-worker', 180)",
        )["status"] == "paused"
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
            ("postgres e2e", "operator:postgres-e2e"),
        )["paused"] is False

        admitted = _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_admit_v2(
                %s::JSONB, %s, %s, %s, %s, %s, 3, 5, 10
            )
            """,
            (
                _json(record_doc),
                IDENTITY_HASH,
                "sha256:" + "b" * 64,
                "sha256:" + "c" * 64,
                record_doc["provider_origin_hash"],
                PROVENANCE_WORK,
            ),
        )
        assert admitted["status"] == "admitted"
        duplicate = _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_admit_v2(
                %s::JSONB, %s, %s, %s, %s, %s, 3, 5, 10
            )
            """,
            (
                _json(record_doc),
                IDENTITY_HASH,
                "sha256:" + "b" * 64,
                "sha256:" + "c" * 64,
                record_doc["provider_origin_hash"],
                "source_add_work:9999999999999999",
            ),
        )
        assert duplicate["status"] == "duplicate"
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'owner_count', COUNT(*),
                'reservation_status', MIN(reservation_status)
            )
            FROM public.research_lab_source_add_provider_origin_current
            WHERE provider_origin_hash = %s
              AND submission_id = %s
              AND adapter_id = %s
            """,
            (
                record_doc["provider_origin_hash"],
                SUBMISSION_ID,
                ADAPTER_ID,
            ),
        ) == {"owner_count": 1, "reservation_status": "reserved"}

        provenance = _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work('postgres-e2e', 180)",
        )["work"]
        assert provenance["work_kind"] == "provenance"
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_begin_provider_execution(%s, %s::UUID)",
            (provenance["work_id"], provenance["lease_token"]),
        )["status"] == "started"
        assert _finish_work(
            cursor,
            work=provenance,
            stage="functional_probe_queued",
            submission_doc=record_doc,
            precheck_status="provenance_precheck_passed",
            probe_config={
                "config_ref": CONFIG_REF,
                "probe_doc": probe_doc,
                "credential_envelope": {},
                "actor_ref": "system:postgres-e2e",
            },
            next_work={
                "work_id": FUNCTIONAL_WORK,
                "work_kind": "functional_probe",
                "priority": 20,
                "job_doc": {"config_ref": CONFIG_REF, "host_hash": host_hash},
            },
        )["status"] == "completed"

        functional = _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work('postgres-e2e', 180)",
        )["work"]
        assert functional["work_kind"] == "functional_probe"
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_begin_provider_execution(%s, %s::UUID)",
            (functional["work_id"], functional["lease_token"]),
        )["status"] == "started"
        result_doc = {
            "schema_version": "leadpoet.source_add_functional_probe_result.v2",
            "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
            "submission_id": SUBMISSION_ID,
            "adapter_id": ADAPTER_ID,
            "config_ref": CONFIG_REF,
            "evaluation_mode": "functional_probe",
            "result_status": "passed",
            "route_hash": ROUTE_HASH,
        }
        functional_attempt = {
            "attempt_ref": FUNCTIONAL_ATTEMPT,
            "evaluation_mode": "functional_probe",
            "config_ref": CONFIG_REF,
            "result_status": "passed",
            "route_hash": ROUTE_HASH,
            "response_hash": "sha256:" + "d" * 64,
            "status_class": "2xx",
            "content_type": "application/json",
            "byte_count": 128,
            "duration_ms": 25,
            "retry_after_seconds": 0,
            "reason_codes": ["bounded_json_data_response"],
            "receipt_hash": FUNCTIONAL_RECEIPT,
            "business_artifact_hash": FUNCTIONAL_ARTIFACT,
            "result_doc": result_doc,
        }
        assert _finish_work(
            cursor,
            work=functional,
            stage="functional_probe_passed",
            submission_doc=record_doc,
            precheck_status="provenance_precheck_passed",
            functional_attempt=functional_attempt,
        )["status"] == "completed"

        _seed_boot_identity(cursor)
        _seed_receipt(
            cursor,
            receipt_hash=FUNCTIONAL_RECEIPT,
            purpose="research_lab.source_add_functional_probe.v2",
            job_id="source-add-functional-postgres-e2e",
            output_root=FUNCTIONAL_ARTIFACT,
            sequence=1,
        )
        _seed_business_link(
            cursor,
            receipt_hash=FUNCTIONAL_RECEIPT,
            artifact_kind="source_add_functional_probe",
            artifact_ref=FUNCTIONAL_ATTEMPT,
            artifact_hash=FUNCTIONAL_ARTIFACT,
        )

        catalog_row = {
            "catalog_id": CATALOG_ID,
            "adapter_id": ADAPTER_ID,
            "miner_ref": MINER_HOTKEY,
            "source_name": record_doc["manifest"]["source_name"],
            "source_kind": record_doc["manifest"]["source_kind"],
            "declared_base_domains": record_doc["manifest"]["declared_base_domains"],
            "registry_provider_id": REGISTRY_PROVIDER_ID,
            "catalog_doc": {"source": "postgres-e2e"},
            "source_identity_hash": IDENTITY_HASH,
        }
        provision_doc = {
            "provider_registry_entry": {
                "provider_id": REGISTRY_PROVIDER_ID,
                "base_url": probe_doc["base_url"],
                "auth_kind": "none",
                "auth_name": "",
            },
            "request_headers": {},
            "probe_endpoints": [{"method": "GET", "path": "/records"}],
        }

        def provision_row(reference: str, status: str) -> dict:
            return {
                "provision_ref": reference,
                "submission_id": SUBMISSION_ID,
                "adapter_id": ADAPTER_ID,
                "miner_hotkey": MINER_HOTKEY,
                "source_identity_hash": IDENTITY_HASH,
                "registry_provider_id": REGISTRY_PROVIDER_ID,
                "provision_status": status,
                "provision_doc": provision_doc,
                "credential_envelope": {},
            }

        pending = _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_finalize_provision_v2(
                %s, %s::JSONB, %s::JSONB, '{}'::JSONB
            )
            """,
            (
                SUBMISSION_ID,
                _json(catalog_row),
                _json(
                    provision_row(
                        "source_add_provision:1111111111111111",
                        "approved_pending_provision",
                    )
                ),
            ),
        )
        assert pending["status"] == "provisioned"
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'stage', (
                    SELECT stage
                    FROM public.research_lab_source_add_submission_current
                    WHERE submission_id = %s
                ),
                'intent_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_intents
                    WHERE submission_id = %s
                ),
                'reward_work_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_work_items
                    WHERE submission_id = %s AND work_kind = 'leg1_reward'
                )
            )
            """,
            (SUBMISSION_ID, SUBMISSION_ID, SUBMISSION_ID),
        ) == {
            "stage": "functional_probe_passed",
            "intent_count": 0,
            "reward_work_count": 0,
        }
        eligible_row = provision_row(
            "source_add_provision:2222222222222222",
            "provisioned_autoresearch_eligible",
        )
        queued_smoke = _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_enqueue_provision_smoke(
                %s, %s, %s, %s, %s::JSONB, %s::JSONB
            )
            """,
            (
                SMOKE_WORK,
                SUBMISSION_ID,
                CONFIG_REF,
                host_hash,
                _json(catalog_row),
                _json(eligible_row),
            ),
        )
        assert queued_smoke["status"] == "queued"
        smoke_work = _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work('postgres-e2e', 180)",
        )["work"]
        assert smoke_work["work_kind"] == "provisioning_smoke"
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_begin_provider_execution(%s, %s::UUID)",
            (smoke_work["work_id"], smoke_work["lease_token"]),
        )["status"] == "started"
        first_smoke_attempt_number = smoke_work["attempt_count"]
        failed_smoke_result = {
            "schema_version": "leadpoet.source_add_functional_probe_result.v2",
            "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
            "submission_id": SUBMISSION_ID,
            "adapter_id": ADAPTER_ID,
            "config_ref": CONFIG_REF,
            "evaluation_mode": "provisioning_smoke",
            "result_status": "failed",
            "route_hash": ROUTE_HASH,
        }
        _seed_receipt(
            cursor,
            receipt_hash=FAILED_SMOKE_RECEIPT,
            purpose="research_lab.source_add_functional_probe.v2",
            job_id="source-add-failed-smoke-postgres-e2e",
            output_root=FAILED_SMOKE_ARTIFACT,
            sequence=30,
        )
        _seed_business_link(
            cursor,
            receipt_hash=FAILED_SMOKE_RECEIPT,
            artifact_kind="source_add_provisioning_smoke",
            artifact_ref=FAILED_SMOKE_ATTEMPT,
            artifact_hash=FAILED_SMOKE_ARTIFACT,
        )
        failed_smoke_attempt = {
            "attempt_ref": FAILED_SMOKE_ATTEMPT,
            "work_id": SMOKE_WORK,
            "attempt_number": first_smoke_attempt_number,
            "evaluation_mode": "provisioning_smoke",
            "config_ref": CONFIG_REF,
            "result_status": "failed",
            "route_hash": ROUTE_HASH,
            "response_hash": "sha256:" + "f" * 64,
            "status_class": "4xx",
            "content_type": "application/json",
            "byte_count": 32,
            "duration_ms": 15,
            "retry_after_seconds": 0,
            "reason_codes": ["endpoint_not_found"],
            "receipt_hash": FAILED_SMOKE_RECEIPT,
            "business_artifact_hash": FAILED_SMOKE_ARTIFACT,
            "result_doc": failed_smoke_result,
        }
        assert _finish_work(
            cursor,
            work=smoke_work,
            stage="",
            submission_doc=record_doc,
            precheck_status="provenance_precheck_passed",
            result_doc=failed_smoke_result,
            functional_attempt=failed_smoke_attempt,
        )["status"] == "completed"
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'acceptance_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_submissions
                    WHERE submission_id = %s AND stage = 'accepted'
                ),
                'intent_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_intents
                    WHERE submission_id = %s
                ),
                'reward_work_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_work_items
                    WHERE submission_id = %s AND work_kind = 'leg1_reward'
                ),
                'obligation_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_obligations
                    WHERE adapter_id = %s AND leg = 1
                )
            )
            """,
            (SUBMISSION_ID, SUBMISSION_ID, SUBMISSION_ID, ADAPTER_ID),
        ) == {
            "acceptance_count": 0,
            "intent_count": 0,
            "reward_work_count": 0,
            "obligation_count": 0,
        }

        def enqueue_same_smoke() -> dict:
            retry_connection = psycopg2.connect(**dsn)
            retry_connection.autocommit = True
            try:
                with retry_connection.cursor() as retry_cursor:
                    return _scalar(
                        retry_cursor,
                        """
                        SELECT public.research_lab_source_add_enqueue_provision_smoke(
                            %s, %s, %s, %s, %s::JSONB, %s::JSONB
                        )
                        """,
                        (
                            SMOKE_WORK,
                            SUBMISSION_ID,
                            CONFIG_REF,
                            host_hash,
                            _json(catalog_row),
                            _json(eligible_row),
                        ),
                    )
            finally:
                retry_connection.close()

        cursor.execute(
            """
            UPDATE public.research_lab_source_add_work_items
            SET result_doc = jsonb_build_object(
                'status', 'provider_execution_outcome_unknown_after_worker_loss'
            )
            WHERE work_id = %s
            """,
            (SMOKE_WORK,),
        )
        assert enqueue_same_smoke()["status"] == "terminal_retry_not_allowed"
        cursor.execute(
            """
            UPDATE public.research_lab_source_add_work_items
            SET result_doc = %s::JSONB, attempt_count = 20
            WHERE work_id = %s
            """,
            (_json(failed_smoke_result), SMOKE_WORK),
        )
        assert enqueue_same_smoke()["status"] == "terminal_retry_not_allowed"
        cursor.execute(
            """
            UPDATE public.research_lab_source_add_work_items
            SET attempt_count = %s
            WHERE work_id = %s
            """,
            (first_smoke_attempt_number, SMOKE_WORK),
        )

        retry_barrier = threading.Barrier(3)
        retry_results: list[dict] = []
        retry_errors: list[BaseException] = []

        def concurrent_retry() -> None:
            try:
                retry_barrier.wait(timeout=5)
                retry_results.append(enqueue_same_smoke())
            except BaseException as exc:  # surfaced in the parent test thread
                retry_errors.append(exc)

        retry_threads = [
            threading.Thread(target=concurrent_retry),
            threading.Thread(target=concurrent_retry),
        ]
        for retry_thread in retry_threads:
            retry_thread.start()
        retry_barrier.wait(timeout=5)
        for retry_thread in retry_threads:
            retry_thread.join(timeout=5)
            assert not retry_thread.is_alive()
        assert retry_errors == []
        assert sorted(result["status"] for result in retry_results) == [
            "already_queued",
            "queued",
        ]
        assert sum(bool(result.get("requeued")) for result in retry_results) == 1
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'work_status', work_status,
                'attempt_count', attempt_count,
                'completed_at_cleared', completed_at IS NULL,
                'provider_markers_cleared', NOT (
                    job_doc ?| ARRAY[
                        'provider_execution_state',
                        'provider_execution_attempt',
                        'provider_execution_started_at',
                        'provider_execution_recovery'
                    ]
                )
            )
            FROM public.research_lab_source_add_work_items
            WHERE work_id = %s
            """,
            (SMOKE_WORK,),
        ) == {
            "work_status": "queued",
            "attempt_count": first_smoke_attempt_number,
            "completed_at_cleared": True,
            "provider_markers_cleared": True,
        }
        smoke_work = _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work('postgres-e2e', 180)",
        )["work"]
        assert smoke_work["work_kind"] == "provisioning_smoke"
        assert smoke_work["attempt_count"] == first_smoke_attempt_number + 1
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_begin_provider_execution(%s, %s::UUID)",
            (smoke_work["work_id"], smoke_work["lease_token"]),
        )["status"] == "started"
        catalog_unavailable_smoke_result = {
            "schema_version": "leadpoet.source_add_functional_probe_result.v2",
            "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
            "submission_id": SUBMISSION_ID,
            "adapter_id": ADAPTER_ID,
            "config_ref": CONFIG_REF,
            "evaluation_mode": "provisioning_smoke",
            "result_status": "passed",
            "route_hash": ROUTE_HASH,
        }
        _seed_receipt(
            cursor,
            receipt_hash=CATALOG_UNAVAILABLE_SMOKE_RECEIPT,
            purpose="research_lab.source_add_functional_probe.v2",
            job_id="source-add-catalog-unavailable-smoke-postgres-e2e",
            output_root=CATALOG_UNAVAILABLE_SMOKE_ARTIFACT,
            sequence=31,
        )
        _seed_business_link(
            cursor,
            receipt_hash=CATALOG_UNAVAILABLE_SMOKE_RECEIPT,
            artifact_kind="source_add_provisioning_smoke",
            artifact_ref=CATALOG_UNAVAILABLE_SMOKE_ATTEMPT,
            artifact_hash=CATALOG_UNAVAILABLE_SMOKE_ARTIFACT,
        )
        catalog_unavailable_smoke_attempt = {
            "attempt_ref": CATALOG_UNAVAILABLE_SMOKE_ATTEMPT,
            "work_id": SMOKE_WORK,
            "attempt_number": smoke_work["attempt_count"],
            "evaluation_mode": "provisioning_smoke",
            "config_ref": CONFIG_REF,
            "result_status": "passed",
            "route_hash": ROUTE_HASH,
            "response_hash": "sha256:" + "0" * 64,
            "status_class": "2xx",
            "content_type": "application/json",
            "byte_count": 96,
            "duration_ms": 18,
            "retry_after_seconds": 0,
            "reason_codes": ["bounded_json_data_response"],
            "receipt_hash": CATALOG_UNAVAILABLE_SMOKE_RECEIPT,
            "business_artifact_hash": CATALOG_UNAVAILABLE_SMOKE_ARTIFACT,
            "result_doc": catalog_unavailable_smoke_result,
        }
        assert _finish_work(
            cursor,
            work=smoke_work,
            stage="",
            submission_doc=record_doc,
            precheck_status="provenance_precheck_passed",
            result_doc={"status": "current_model_catalog_unavailable"},
            functional_attempt=catalog_unavailable_smoke_attempt,
        )["status"] == "completed"
        catalog_retry = enqueue_same_smoke()
        assert catalog_retry["status"] == "queued"
        assert catalog_retry["requeued"] is True
        smoke_work = _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work('postgres-e2e', 180)",
        )["work"]
        assert smoke_work["work_kind"] == "provisioning_smoke"
        assert smoke_work["attempt_count"] == first_smoke_attempt_number + 2
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_begin_provider_execution(%s, %s::UUID)",
            (smoke_work["work_id"], smoke_work["lease_token"]),
        )["status"] == "started"
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'acceptance_count', (
                    SELECT COUNT(*) FROM public.research_lab_source_add_submissions
                    WHERE submission_id = %s AND stage = 'accepted'
                ),
                'intent_count', (
                    SELECT COUNT(*) FROM public.research_lab_source_add_reward_intents
                    WHERE submission_id = %s
                ),
                'reward_work_count', (
                    SELECT COUNT(*) FROM public.research_lab_source_add_work_items
                    WHERE submission_id = %s AND work_kind = 'leg1_reward'
                ),
                'obligation_count', (
                    SELECT COUNT(*) FROM public.research_lab_source_add_reward_obligations
                    WHERE adapter_id = %s AND leg = 1
                )
            )
            """,
            (SUBMISSION_ID, SUBMISSION_ID, SUBMISSION_ID, ADAPTER_ID),
        ) == {
            "acceptance_count": 0,
            "intent_count": 0,
            "reward_work_count": 0,
            "obligation_count": 0,
        }
        replayed_smoke_attempt = {
            **catalog_unavailable_smoke_attempt,
            "attempt_number": smoke_work["attempt_count"],
        }
        with pytest.raises(
            psycopg2.errors.RaiseException,
            match="post-accept smoke lease binding differs",
        ):
            cursor.execute(
                """
                SELECT public.research_lab_source_add_finalize_provision_smoke_v2(
                    %s, %s::UUID, %s, %s::JSONB, %s::JSONB, %s::JSONB,
                    %s::JSONB, %s::JSONB
                )
                """,
                (
                    SMOKE_WORK,
                    smoke_work["lease_token"],
                    SUBMISSION_ID,
                    _json(catalog_row),
                    _json(eligible_row),
                    _json(replayed_smoke_attempt),
                    _json(
                        {
                            "intent_id": REWARD_INTENT,
                            "miner_hotkey": MINER_HOTKEY,
                            "functional_receipt_hash": FUNCTIONAL_RECEIPT,
                            "business_artifact_hash": FUNCTIONAL_ARTIFACT,
                        }
                    ),
                    _json(
                        {
                            "work_id": REWARD_WORK,
                            "work_kind": "leg1_reward",
                            "priority": 30,
                            "job_doc": {
                                "intent_id": REWARD_INTENT,
                                "attempt_ref": FUNCTIONAL_ATTEMPT,
                            },
                        }
                    ),
                ),
            )
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'work_status', work_status,
                'intent_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_intents
                    WHERE submission_id = %s
                ),
                'acceptance_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_submissions
                    WHERE submission_id = %s AND stage = 'accepted'
                )
            )
            FROM public.research_lab_source_add_work_items
            WHERE work_id = %s
            """,
            (SUBMISSION_ID, SUBMISSION_ID, SMOKE_WORK),
        ) == {
            "work_status": "leased",
            "intent_count": 0,
            "acceptance_count": 0,
        }
        smoke_result = {
            "schema_version": "leadpoet.source_add_functional_probe_result.v2",
            "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
            "submission_id": SUBMISSION_ID,
            "adapter_id": ADAPTER_ID,
            "config_ref": CONFIG_REF,
            "evaluation_mode": "provisioning_smoke",
            "result_status": "passed",
            "route_hash": ROUTE_HASH,
        }
        _seed_receipt(
            cursor,
            receipt_hash=SMOKE_RECEIPT,
            purpose="research_lab.source_add_functional_probe.v2",
            job_id="source-add-smoke-postgres-e2e",
            output_root=SMOKE_ARTIFACT,
            sequence=3,
        )
        _seed_business_link(
            cursor,
            receipt_hash=SMOKE_RECEIPT,
            artifact_kind="source_add_provisioning_smoke",
            artifact_ref=SMOKE_ATTEMPT,
            artifact_hash=SMOKE_ARTIFACT,
        )
        smoke_attempt = {
            "attempt_ref": SMOKE_ATTEMPT,
            "work_id": SMOKE_WORK,
            "attempt_number": smoke_work["attempt_count"],
            "evaluation_mode": "provisioning_smoke",
            "config_ref": CONFIG_REF,
            "result_status": "passed",
            "route_hash": ROUTE_HASH,
            "response_hash": "sha256:" + "e" * 64,
            "status_class": "2xx",
            "content_type": "application/json",
            "byte_count": 128,
            "duration_ms": 20,
            "retry_after_seconds": 0,
            "reason_codes": ["bounded_json_data_response"],
            "receipt_hash": SMOKE_RECEIPT,
            "business_artifact_hash": SMOKE_ARTIFACT,
            "result_doc": smoke_result,
        }
        with pytest.raises(
            psycopg2.errors.RaiseException,
            match="post-smoke Leg 1 authority",
        ):
            cursor.execute(
                """
                SELECT public.research_lab_source_add_finalize_provision_smoke(
                    %s, %s::UUID, %s, %s::JSONB, %s::JSONB, %s::JSONB
                )
                """,
                (
                    SMOKE_WORK,
                    smoke_work["lease_token"],
                    SUBMISSION_ID,
                    _json(catalog_row),
                    _json(eligible_row),
                    _json(smoke_attempt),
                ),
            )
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'smoke_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_provisioning_smoke_current
                    WHERE submission_id = %s
                ),
                'provision_status', (
                    SELECT provision_status
                    FROM public.research_lab_source_add_provisioning_current
                    WHERE adapter_id = %s
                ),
                'stage', (
                    SELECT stage
                    FROM public.research_lab_source_add_submission_current
                    WHERE submission_id = %s
                )
            )
            """,
            (SUBMISSION_ID, ADAPTER_ID, SUBMISSION_ID),
        ) == {
            "smoke_count": 1,
            "provision_status": "approved_pending_provision",
            "stage": "functional_probe_passed",
        }
        smoke_finalized = _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_finalize_provision_smoke_v2(
                %s, %s::UUID, %s, %s::JSONB, %s::JSONB, %s::JSONB,
                %s::JSONB, %s::JSONB
            )
            """,
            (
                SMOKE_WORK,
                smoke_work["lease_token"],
                SUBMISSION_ID,
                _json(catalog_row),
                _json(eligible_row),
                _json(smoke_attempt),
                _json(
                    {
                        "intent_id": REWARD_INTENT,
                        "miner_hotkey": MINER_HOTKEY,
                        "functional_receipt_hash": FUNCTIONAL_RECEIPT,
                        "business_artifact_hash": FUNCTIONAL_ARTIFACT,
                    }
                ),
                _json(
                    {
                        "work_id": REWARD_WORK,
                        "work_kind": "leg1_reward",
                        "priority": 30,
                        "job_doc": {
                            "intent_id": REWARD_INTENT,
                            "attempt_ref": FUNCTIONAL_ATTEMPT,
                        },
                    }
                ),
            ),
        )
        assert smoke_finalized["status"] == "provisioned"
        assert smoke_finalized["leg1_intent_id"] == REWARD_INTENT
        assert smoke_finalized["leg1_work_id"] == REWARD_WORK
        assert _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_configure_probe_v2(
                %s, 'source_add_probe_config:eeeeeeeeeeeeeeee',
                %s::JSONB, '{}'::JSONB, 'operator:late-config',
                'source_add_work:eeeeeeeeeeeeeeee', %s
            )
            """,
            (SUBMISSION_ID, _json(probe_doc), host_hash),
        ) == {"status": "terminal"}
        assert _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_finalize_provision_v2(
                %s, %s::JSONB, %s::JSONB, '{}'::JSONB
            )
            """,
            (
                SUBMISSION_ID,
                _json(catalog_row),
                _json(
                    provision_row(
                        "source_add_provision:eeeeeeeeeeeeeeee",
                        "disabled",
                    )
                ),
            ),
        ) == {"status": "final_approval_frozen"}
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'stage', (
                    SELECT stage
                    FROM public.research_lab_source_add_submission_current
                    WHERE submission_id = %s
                ),
                'intent_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_intents
                    WHERE submission_id = %s
                ),
                'reward_work_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_work_items
                    WHERE submission_id = %s AND work_kind = 'leg1_reward'
                )
            )
            """,
            (SUBMISSION_ID, SUBMISSION_ID, SUBMISSION_ID),
        ) == {
            "stage": "accepted",
            "intent_count": 1,
            "reward_work_count": 1,
        }

        reward_work = _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work('postgres-e2e', 180)",
        )["work"]
        assert reward_work["work_kind"] == "leg1_reward"
        slot = _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_reserve_leg1_slot_v2(
                %s, %s, %s::UUID, 100, 300
            )
            """,
            (REWARD_INTENT, REWARD_WORK, reward_work["lease_token"]),
        )
        assert slot["status"] == "reserved"

        trigger = {
            "functional_probe_passed": True,
            "attempt_ref": FUNCTIONAL_ATTEMPT,
            "functional_probe_receipt_hash": FUNCTIONAL_RECEIPT,
            "business_artifact_hash": FUNCTIONAL_ARTIFACT,
            "functional_probe_result_hash": FUNCTIONAL_ARTIFACT,
            "evaluator_version": result_doc["evaluator_version"],
            "route_hash": ROUTE_HASH,
            "provisioning_smoke_passed": True,
            "provisioning_smoke_attempt_ref": SMOKE_ATTEMPT,
            "provisioning_smoke_receipt_hash": SMOKE_RECEIPT,
            "provisioning_smoke_business_artifact_hash": SMOKE_ARTIFACT,
            "provisioning_smoke_result_hash": SMOKE_ARTIFACT,
            "submission_id": SUBMISSION_ID,
            "final_acceptance_stage": "accepted",
            "provision_ref": eligible_row["provision_ref"],
            "catalog_id": CATALOG_ID,
            "registry_provider_id": REGISTRY_PROVIDER_ID,
            "provision_status": "provisioned_autoresearch_eligible",
        }
        reward_payload = {
            "reward_ref": REWARD_REF,
            "reward_kind": "source_acceptance",
            "alpha_percent": 1.0,
            "reward_epochs": 20,
            "start_epoch": 701,
            "state": "active",
            "trigger_evidence_doc": trigger,
            "public_label": "SOURCE_ADD Leg 1",
            "decision_receipt_hash": DECISION_RECEIPT,
        }
        decision_artifact = sha256_json(
            source_add_reward_row_projection_v2(
                "source_add_leg1",
                {
                    **reward_payload,
                    "adapter_id": ADAPTER_ID,
                    "miner_hotkey": MINER_HOTKEY,
                    "leg": 1,
                    "initial_reward_status": "active",
                },
            )
        )
        reward_payload["decision_artifact_hash"] = decision_artifact
        cursor.execute("BEGIN")
        _seed_receipt(
            cursor,
            receipt_hash=DECISION_RECEIPT,
            purpose="research_lab.reward_decision.v2",
            job_id="source-add-reward-postgres-e2e",
            output_root=decision_artifact,
            sequence=2,
            parent_receipt_hashes=(FUNCTIONAL_RECEIPT,),
        )
        _seed_business_link(
            cursor,
            receipt_hash=DECISION_RECEIPT,
            artifact_kind="source_add_reward_decision",
            artifact_ref=REWARD_REF,
            artifact_hash=decision_artifact,
        )
        with pytest.raises(
            psycopg2.errors.RaiseException,
            match="reward decision ancestry differs",
        ):
            cursor.execute(
                """
                SELECT public.research_lab_source_add_finalize_leg1_v2(
                    %s, %s, %s::UUID, %s::UUID, 100, %s::JSONB, %s::JSONB
                )
                """,
                (
                    REWARD_INTENT,
                    REWARD_WORK,
                    reward_work["lease_token"],
                    slot["slot_lease_token"],
                    _json(reward_payload),
                    _json(record_doc),
                ),
            )
        cursor.execute("ROLLBACK")
        assert _scalar(
            cursor,
            """
            SELECT COUNT(*)
            FROM public.research_lab_source_add_reward_obligations
            WHERE reward_ref = %s
            """,
            (REWARD_REF,),
        ) == 0

        stale_decision_artifact = sha256_json(
            source_add_reward_row_projection_v2(
                "source_add_leg1",
                {
                    **reward_payload,
                    "adapter_id": ADAPTER_ID,
                    "miner_hotkey": MINER_HOTKEY,
                    "leg": 1,
                    "start_epoch": reward_payload["start_epoch"] - 1,
                    "initial_reward_status": "active",
                },
            )
        )
        _seed_receipt(
            cursor,
            receipt_hash=STALE_DECISION_RECEIPT,
            purpose="research_lab.reward_decision.v2",
            job_id="source-add-reward-postgres-e2e-stale-retry",
            output_root=stale_decision_artifact,
            sequence=3,
            parent_receipt_hashes=(FUNCTIONAL_RECEIPT, SMOKE_RECEIPT),
        )
        _seed_business_link(
            cursor,
            receipt_hash=STALE_DECISION_RECEIPT,
            artifact_kind="source_add_reward_decision",
            artifact_ref=REWARD_REF,
            artifact_hash=stale_decision_artifact,
        )
        _seed_receipt(
            cursor,
            receipt_hash=DECISION_RECEIPT,
            purpose="research_lab.reward_decision.v2",
            job_id="source-add-reward-postgres-e2e",
            output_root=decision_artifact,
            sequence=2,
            parent_receipt_hashes=(FUNCTIONAL_RECEIPT, SMOKE_RECEIPT),
        )
        _seed_business_link(
            cursor,
            receipt_hash=DECISION_RECEIPT,
            artifact_kind="source_add_reward_decision",
            artifact_ref=REWARD_REF,
            artifact_hash=decision_artifact,
        )
        finalized = _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_finalize_leg1_v2(
                %s, %s, %s::UUID, %s::UUID, 100, %s::JSONB, %s::JSONB
            )
            """,
            (
                REWARD_INTENT,
                REWARD_WORK,
                reward_work["lease_token"],
                slot["slot_lease_token"],
                _json(reward_payload),
                _json(record_doc),
            ),
        )
        assert finalized == {"status": "created", "reward_ref": REWARD_REF}
        assert _scalar(
            cursor,
            """
            SELECT COUNT(*)
            FROM public.research_lab_attested_business_artifact_links_v2
            WHERE artifact_kind = 'source_add_reward_decision'
              AND artifact_ref = %s
            """,
            (REWARD_REF,),
        ) == 2

        summary = _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'stage', (
                    SELECT stage FROM public.research_lab_source_add_submission_current
                    WHERE submission_id = %s
                ),
                'probe', (
                    SELECT result_status FROM public.research_lab_source_add_functional_probe_current
                    WHERE submission_id = %s
                ),
                'smoke', (
                    SELECT result_status FROM public.research_lab_source_add_provisioning_smoke_current
                    WHERE submission_id = %s
                ),
                'reward_status', (
                    SELECT current_reward_status FROM public.research_lab_source_add_reward_current
                    WHERE reward_ref = %s
                ),
                'intent_status', (
                    SELECT intent_status FROM public.research_lab_source_add_reward_intents
                    WHERE intent_id = %s
                ),
                'provision_status', (
                    SELECT provision_status FROM public.research_lab_source_add_provisioning_current
                    WHERE adapter_id = %s
                ),
                'completed_work', (
                    SELECT COUNT(*) FROM public.research_lab_source_add_work_items
                    WHERE submission_id = %s AND work_status = 'completed'
                ),
                'reward_count', (
                    SELECT COUNT(*) FROM public.research_lab_source_add_reward_obligations
                    WHERE adapter_id = %s AND leg = 1
                ),
                'acceptance_count', (
                    SELECT COUNT(*) FROM public.research_lab_source_add_submissions
                    WHERE submission_id = %s AND stage = 'accepted'
                ),
                'reward_catalog_id', (
                    SELECT catalog_id FROM public.research_lab_source_add_reward_obligations
                    WHERE adapter_id = %s AND leg = 1
                )
            )
            """,
            (
                SUBMISSION_ID,
                SUBMISSION_ID,
                SUBMISSION_ID,
                REWARD_REF,
                REWARD_INTENT,
                ADAPTER_ID,
                SUBMISSION_ID,
                ADAPTER_ID,
                SUBMISSION_ID,
                ADAPTER_ID,
            ),
        )
        assert summary == {
            "stage": "leg1_created",
            "probe": "passed",
            "smoke": "passed",
            "reward_status": "active",
            "intent_status": "finalized",
            "provision_status": "provisioned_autoresearch_eligible",
            "completed_work": 4,
            "reward_count": 1,
            "acceptance_count": 1,
            "reward_catalog_id": CATALOG_ID,
        }
        assert _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work('postgres-e2e', 180)",
        )["status"] == "empty"
    connection.close()


def test_current_builtin_rejection_is_atomic_terminal_and_unrewarded(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        case = _seed_leased_smoke_case(
            cursor,
            seed=0xB170000000000000,
            base_url="https://builtin-provider.example/v1",
        )
        disabled_row = json.loads(json.dumps(case["eligible_row"]))
        disabled_row["provision_ref"] = (
            "source_add_provision:b170000000000008"
        )
        disabled_row["provision_status"] = "disabled"
        disabled_row["provision_doc"]["provider_registry_entry"][
            "active"
        ] = False
        parameters = (
            case["smoke_work"]["work_id"],
            case["smoke_work"]["lease_token"],
            case["submission_id"],
            _json(case["record_doc"]),
            "provenance_precheck_passed",
            _json({}),
            _json(case["catalog_row"]),
            _json(disabled_row),
            _json(case["smoke_attempt"]),
        )
        statement = """
            SELECT public.research_lab_source_add_reject_current_builtin_v2(
                %s, %s::UUID, %s, %s::JSONB, %s, %s::JSONB,
                %s::JSONB, %s::JSONB, %s::JSONB
            )
        """
        unsafe_disabled_row = json.loads(json.dumps(disabled_row))
        unsafe_disabled_row["provision_doc"]["provider_registry_entry"][
            "active"
        ] = True
        unsafe_parameters = parameters[:7] + (
            _json(unsafe_disabled_row),
            parameters[8],
        )
        with pytest.raises(
            psycopg2.errors.RaiseException,
            match="current-provider rejection input is invalid",
        ):
            _scalar(cursor, statement, unsafe_parameters)
        assert _scalar(cursor, statement, parameters) == {
            "status": "not_eligible"
        }
        assert _scalar(cursor, statement, parameters) == {
            "status": "not_eligible"
        }
        changed_smoke = json.loads(json.dumps(case["smoke_attempt"]))
        changed_smoke["response_hash"] = "sha256:" + "f" * 64
        changed_parameters = parameters[:-1] + (_json(changed_smoke),)
        with pytest.raises(
            psycopg2.errors.RaiseException,
            match="current-provider persisted smoke differs",
        ):
            _scalar(cursor, statement, changed_parameters)
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'stage', (
                    SELECT stage
                    FROM public.research_lab_source_add_submission_current
                    WHERE submission_id = %s
                ),
                'work_status', (
                    SELECT work_status
                    FROM public.research_lab_source_add_work_items
                    WHERE work_id = %s
                ),
                'smoke_status', (
                    SELECT result_status
                    FROM public.research_lab_source_add_provisioning_smoke_current
                    WHERE submission_id = %s
                ),
                'provision_status', (
                    SELECT provision_status
                    FROM public.research_lab_source_add_provisioning_current
                    WHERE submission_id = %s
                ),
                'intent_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_intents
                    WHERE submission_id = %s
                ),
                'reward_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_obligations
                    WHERE adapter_id = %s
                ),
                'origin_status', (
                    SELECT reservation_status
                    FROM public.research_lab_source_add_provider_origin_current
                    WHERE submission_id = %s
                )
            )
            """,
            (
                case["submission_id"],
                case["smoke_work"]["work_id"],
                case["submission_id"],
                case["submission_id"],
                case["submission_id"],
                case["adapter_id"],
                case["submission_id"],
            ),
        ) == {
            "stage": "functional_probe_failed",
            "work_status": "completed",
            "smoke_status": "passed",
            "provision_status": "disabled",
            "intent_count": 0,
            "reward_count": 0,
            "origin_status": "reserved",
        }
    connection.close()


def test_current_builtin_rejection_cannot_relabel_rewarded_completion(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        seed = 0xB171000000000000
        case = _seed_leased_smoke_case(
            cursor,
            seed=seed,
            base_url="https://rewarded-provider.example/v1",
        )
        reward = _create_seed_leg1_reward(cursor, case=case, seed=seed)
        disabled_row = json.loads(json.dumps(case["eligible_row"]))
        disabled_row["provision_ref"] = (
            "source_add_provision:b171000000000008"
        )
        disabled_row["provision_status"] = "disabled"
        disabled_row["provision_doc"]["provider_registry_entry"][
            "active"
        ] = False
        assert _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_finalize_provision_v2(
                %s, %s::JSONB, %s::JSONB, '{}'::JSONB
            )
            """,
            (
                case["submission_id"],
                _json(case["catalog_row"]),
                _json(disabled_row),
            ),
        )["status"] == "provisioned"
        with pytest.raises(
            psycopg2.errors.RaiseException,
            match="current-provider rejection terminal state differs",
        ):
            _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_reject_current_builtin_v2(
                    %s, %s::UUID, %s, %s::JSONB, %s, %s::JSONB,
                    %s::JSONB, %s::JSONB, %s::JSONB
                )
                """,
                (
                    case["smoke_work"]["work_id"],
                    case["smoke_work"]["lease_token"],
                    case["submission_id"],
                    _json(case["record_doc"]),
                    "provenance_precheck_passed",
                    _json({}),
                    _json(case["catalog_row"]),
                    _json(disabled_row),
                    _json(case["smoke_attempt"]),
                ),
            )
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'stage', (
                    SELECT stage
                    FROM public.research_lab_source_add_submission_current
                    WHERE submission_id = %s
                ),
                'reward_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_obligations
                    WHERE adapter_id = %s AND leg = 1
                ),
                'reward_ref', (
                    SELECT reward_ref
                    FROM public.research_lab_source_add_reward_obligations
                    WHERE adapter_id = %s AND leg = 1
                )
            )
            """,
            (case["submission_id"], case["adapter_id"], case["adapter_id"]),
        ) == {
            "stage": "leg1_created",
            "reward_count": 1,
            "reward_ref": reward["reward_ref"],
        }
    connection.close()


def test_leg1_reservation_preserves_fifo_when_oldest_work_is_locked(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    blocker = psycopg2.connect(**dsn)
    blocker.autocommit = False
    try:
        with connection.cursor() as cursor:
            older = _finalize_seed_smoke_to_leg1(
                cursor,
                case=_seed_leased_smoke_case(
                    cursor,
                    seed=0xF1F0000000000000,
                    base_url="https://fifo-oldest.example/v1",
                ),
                seed=0xF1F0000000000000,
            )
            newer = _finalize_seed_smoke_to_leg1(
                cursor,
                case=_seed_leased_smoke_case(
                    cursor,
                    seed=0xF1F0000000000100,
                    base_url="https://fifo-newer.example/v1",
                ),
                seed=0xF1F0000000000100,
            )
        with blocker.cursor() as cursor:
            cursor.execute(
                """
                SELECT work_id
                FROM public.research_lab_source_add_work_items
                WHERE work_id = %s
                FOR UPDATE
                """,
                (older["reward_work_id"],),
            )
        with connection.cursor() as cursor:
            claimed = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_claim_work(%s, 180)",
                ("postgres-fifo-newer",),
            )["work"]
            assert claimed["work_id"] == newer["reward_work_id"]
            assert _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_reserve_leg1_slot_v2(
                    %s, %s, %s::UUID, 100, 300
                )
                """,
                (
                    newer["intent_id"],
                    newer["reward_work_id"],
                    claimed["lease_token"],
                ),
            )["status"] == "fifo_wait"
            assert _scalar(
                cursor,
                """
                SELECT jsonb_build_object(
                    'newer_status', (
                        SELECT work_status
                        FROM public.research_lab_source_add_work_items
                        WHERE work_id = %s
                    ),
                    'slot_count', (
                        SELECT COUNT(*)
                        FROM public.research_lab_source_add_reward_slots
                        WHERE intent_id = %s
                    )
                )
                """,
                (newer["reward_work_id"], newer["intent_id"]),
            ) == {"newer_status": "retry_wait", "slot_count": 0}
    finally:
        blocker.rollback()
        blocker.close()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_reward_intents
                SET intent_status = 'cancelled', updated_at = NOW()
                WHERE intent_id IN (%s, %s);
                UPDATE public.research_lab_source_add_work_items
                SET work_status = 'cancelled', lease_token = NULL,
                    leased_by = '', lease_expires_at = NULL, updated_at = NOW()
                WHERE work_id IN (%s, %s);
                """,
                (
                    older["intent_id"],
                    newer["intent_id"],
                    older["reward_work_id"],
                    newer["reward_work_id"],
                ),
            )
        connection.close()


def test_leg1_reservation_retry_preserves_existing_live_slot(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    older = None
    newer = None
    try:
        with connection.cursor() as cursor:
            older = _finalize_seed_smoke_to_leg1(
                cursor,
                case=_seed_leased_smoke_case(
                    cursor,
                    seed=0xF1F1000000000000,
                    base_url="https://retry-oldest.example/v1",
                ),
                seed=0xF1F1000000000000,
            )
            newer = _finalize_seed_smoke_to_leg1(
                cursor,
                case=_seed_leased_smoke_case(
                    cursor,
                    seed=0xF1F1000000000100,
                    base_url="https://retry-newer.example/v1",
                ),
                seed=0xF1F1000000000100,
            )
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_reward_intents
                SET intent_status = 'retry_wait',
                    available_at = NOW() + INTERVAL '1 hour',
                    updated_at = NOW()
                WHERE intent_id = %s;
                UPDATE public.research_lab_source_add_work_items
                SET work_status = 'retry_wait',
                    available_at = NOW() + INTERVAL '1 hour',
                    updated_at = NOW()
                WHERE work_id = %s;
                """,
                (older["intent_id"], older["reward_work_id"]),
            )
            claimed = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_claim_work(%s, 180)",
                ("postgres-retry-newer",),
            )["work"]
            assert claimed["work_id"] == newer["reward_work_id"]
            first = _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_reserve_leg1_slot_v2(
                    %s, %s, %s::UUID, 100, 300
                )
                """,
                (
                    newer["intent_id"],
                    newer["reward_work_id"],
                    claimed["lease_token"],
                ),
            )
            assert first["status"] == "reserved"
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_reward_intents
                SET intent_status = 'queued', available_at = NOW(),
                    updated_at = NOW()
                WHERE intent_id = %s;
                UPDATE public.research_lab_source_add_work_items
                SET work_status = 'queued', available_at = NOW(),
                    updated_at = NOW()
                WHERE work_id = %s;
                """,
                (older["intent_id"], older["reward_work_id"]),
            )
            retried = _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_reserve_leg1_slot_v2(
                    %s, %s, %s::UUID, 100, 300
                )
                """,
                (
                    newer["intent_id"],
                    newer["reward_work_id"],
                    claimed["lease_token"],
                ),
            )
            assert retried["status"] == "reserved"
            assert retried["slot_number"] == first["slot_number"]
            assert retried["slot_lease_token"] != first["slot_lease_token"]
            assert _scalar(
                cursor,
                """
                SELECT jsonb_build_object(
                    'live_slots', (
                        SELECT COUNT(*)
                        FROM public.research_lab_source_add_reward_slots
                        WHERE intent_id = %s
                          AND slot_status = 'reserved'
                          AND lease_expires_at > NOW()
                    ),
                    'work_status', (
                        SELECT work_status
                        FROM public.research_lab_source_add_work_items
                        WHERE work_id = %s
                    ),
                    'intent_status', (
                        SELECT intent_status
                        FROM public.research_lab_source_add_reward_intents
                        WHERE intent_id = %s
                    )
                )
                """,
                (
                    newer["intent_id"],
                    newer["reward_work_id"],
                    newer["intent_id"],
                ),
            ) == {
                "live_slots": 1,
                "work_status": "leased",
                "intent_status": "leased",
            }
    finally:
        if older is not None and newer is not None:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE public.research_lab_source_add_reward_slots
                    SET slot_status = 'released', updated_at = NOW()
                    WHERE intent_id IN (%s, %s)
                      AND slot_status = 'reserved';
                    UPDATE public.research_lab_source_add_reward_intents
                    SET intent_status = 'cancelled', updated_at = NOW()
                    WHERE intent_id IN (%s, %s);
                    UPDATE public.research_lab_source_add_work_items
                    SET work_status = 'cancelled', lease_token = NULL,
                        leased_by = '', lease_expires_at = NULL,
                        updated_at = NOW()
                    WHERE work_id IN (%s, %s);
                    """,
                    (
                        older["intent_id"],
                        newer["intent_id"],
                        older["intent_id"],
                        newer["intent_id"],
                        older["reward_work_id"],
                        newer["reward_work_id"],
                    ),
                )
        connection.close()


def test_leg1_daily_cap_is_server_authoritative_at_ten(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        created_today = _scalar(
            cursor,
            """
            SELECT COUNT(*)
            FROM public.research_lab_source_add_reward_events
            WHERE reason IN (
                'leg1_provenance_precheck_passed',
                'leg1_functional_probe_passed'
            )
              AND created_at >= (
                  (NOW() AT TIME ZONE 'UTC')::DATE::TIMESTAMP
                  AT TIME ZONE 'UTC'
              )
              AND created_at < (
                  (((NOW() AT TIME ZONE 'UTC')::DATE + 1)::TIMESTAMP)
                  AT TIME ZONE 'UTC'
              )
            """,
        )
        assert 0 <= created_today <= 10
        for index in range(10 - created_today):
            seed = 0xCA00000000000000 + index * 0x100
            _create_seed_leg1_reward(
                cursor,
                case=_seed_leased_smoke_case(
                    cursor,
                    seed=seed,
                    base_url=f"https://cap-{index}.example/v1",
                ),
                seed=seed,
            )
        assert _scalar(
            cursor,
            """
            SELECT COUNT(*)
            FROM public.research_lab_source_add_reward_events
            WHERE reason IN (
                'leg1_provenance_precheck_passed',
                'leg1_functional_probe_passed'
            )
              AND created_at >= (
                  (NOW() AT TIME ZONE 'UTC')::DATE::TIMESTAMP
                  AT TIME ZONE 'UTC'
              )
            """,
        ) == 10
        blocked_seed = 0xCA00000000010000
        blocked = _finalize_seed_smoke_to_leg1(
            cursor,
            case=_seed_leased_smoke_case(
                cursor,
                seed=blocked_seed,
                base_url="https://cap-eleventh.example/v1",
            ),
            seed=blocked_seed,
        )
        claimed = _scalar(
            cursor,
            "SELECT public.research_lab_source_add_claim_work(%s, 180)",
            ("postgres-cap-eleventh",),
        )["work"]
        assert claimed["work_id"] == blocked["reward_work_id"]
        assert _scalar(
            cursor,
            """
            SELECT public.research_lab_source_add_reserve_leg1_slot_v2(
                %s, %s, %s::UUID, 100, 300
            )
            """,
            (
                blocked["intent_id"],
                blocked["reward_work_id"],
                claimed["lease_token"],
            ),
        )["status"] == "daily_cap_fifo"
        assert _scalar(
            cursor,
            """
            SELECT jsonb_build_object(
                'reward_count', (
                    SELECT COUNT(*)
                    FROM public.research_lab_source_add_reward_obligations
                    WHERE adapter_id = %s AND leg = 1
                ),
                'intent_status', (
                    SELECT intent_status
                    FROM public.research_lab_source_add_reward_intents
                    WHERE intent_id = %s
                ),
                'work_status', (
                    SELECT work_status
                    FROM public.research_lab_source_add_work_items
                    WHERE work_id = %s
                )
            )
            """,
            (
                blocked["adapter_id"],
                blocked["intent_id"],
                blocked["reward_work_id"],
            ),
        ) == {
            "reward_count": 0,
            "intent_status": "retry_wait",
            "work_status": "retry_wait",
        }
    connection.close()


def test_provisioning_smoke_enqueue_does_not_hold_submission_while_waiting_on_work(
    database,
):
    psycopg2, dsn = database
    work_id = "source_add_work:16710c0000000001"
    submission_id = "source_add_submission:16710c0000000001"
    blocker = psycopg2.connect(**dsn)
    blocker.autocommit = False
    observer = psycopg2.connect(**dsn)
    observer.autocommit = True
    enqueue_started = threading.Event()
    enqueue_pid: list[int] = []
    enqueue_results: list[dict] = []
    enqueue_errors: list[BaseException] = []

    with blocker.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO public.research_lab_source_add_work_items (
                work_id, submission_id, adapter_id, work_kind,
                work_status, priority, job_doc
            ) VALUES (%s, %s, 'adapter:lock-order-167',
                      'provisioning_smoke', 'queued', 25, '{}'::JSONB)
            """,
            (work_id, submission_id),
        )
    blocker.commit()
    with blocker.cursor() as cursor:
        cursor.execute(
            """
            SELECT work_id
            FROM public.research_lab_source_add_work_items
            WHERE work_id = %s
            FOR UPDATE
            """,
            (work_id,),
        )

    def enqueue() -> None:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET statement_timeout = '5s'")
                cursor.execute("SELECT pg_backend_pid()")
                enqueue_pid.append(int(cursor.fetchone()[0]))
                enqueue_started.set()
                enqueue_results.append(
                    _scalar(
                        cursor,
                        """
                        SELECT public.research_lab_source_add_enqueue_provision_smoke(
                            %s, %s, 'source_add_probe_config:16710c0000000001',
                            %s, %s::JSONB, %s::JSONB
                        )
                        """,
                        (
                            work_id,
                            submission_id,
                            "sha256:" + "1" * 64,
                            _json({"adapter_id": "adapter:lock-order-167"}),
                            _json(
                                {
                                    "adapter_id": "adapter:lock-order-167",
                                    "miner_hotkey": "5LockOrder167Miner",
                                    "provision_status": (
                                        "provisioned_autoresearch_eligible"
                                    ),
                                }
                            ),
                        ),
                    )
                )
        except BaseException as exc:  # surfaced in the parent test thread
            enqueue_errors.append(exc)
        finally:
            connection.close()

    thread = threading.Thread(target=enqueue)
    thread.start()
    assert enqueue_started.wait(timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with observer.cursor() as cursor:
            cursor.execute(
                """
                SELECT wait_event_type
                FROM pg_stat_activity
                WHERE pid = %s
                """,
                (enqueue_pid[0],),
            )
            state = cursor.fetchone()
        if state and state[0] == "Lock":
            break
        time.sleep(0.02)
    else:
        pytest.fail("enqueue did not wait on the locked work row")

    with blocker.cursor() as cursor:
        cursor.execute("SET LOCAL lock_timeout = '1s'")
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("source-add-submission:" + submission_id,),
        )
    blocker.commit()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert enqueue_errors == []
    assert enqueue_results == [{"status": "missing"}]
    observer.close()
    blocker.close()
