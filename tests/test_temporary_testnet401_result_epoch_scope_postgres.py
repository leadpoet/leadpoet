from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "scripts/192-temporary-testnet401-result-epoch-scope.sql").read_text(
    encoding="utf-8"
)
TEST_MAPPING = "sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328"
FINNEY_MAPPING = "sha256:" + "7" * 64
GENESIS = "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
HOTKEY = "5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz"


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _allocation_result(mapping: str = TEST_MAPPING) -> dict:
    epoch = 22061
    return {
        "allocation": {"schema_version": "leadpoet.research_lab_allocation.v2", "epoch": epoch},
        "allocation_inputs": {"epoch": epoch},
        "source_state": {
            "epoch": epoch,
            "netuid": 401,
            "settlement_frontier": {"netuid": 401, "allocation_epoch": epoch},
            "fresh_network_origin": {"cutover_mapping_hash": mapping},
        },
        "source_state_hash": _hash("8"),
    }


def _observation(mapping: str = TEST_MAPPING) -> dict:
    return {
        "schema_version": "leadpoet.chain_realized_weight_observation.v2",
        "netuid": 401,
        "epoch_id": 22061,
        "cutover_mapping_hash": mapping,
        "validator_hotkey": HOTKEY,
        "validator_uid": 9,
        "chain_signing_profile": {"genesis_hash": GENESIS},
    }


def _result_row(*, receipt_hash: str, operation: str, purpose: str, result: dict) -> dict:
    return {
        "receipt_hash": receipt_hash,
        "schema_version": "leadpoet.attested_execution_result.v2",
        "role": "gateway_coordinator",
        "operation": operation,
        "purpose": purpose,
        "job_id": "job:" + receipt_hash,
        "epoch_id": 22061,
        "sequence": 0,
        "release_hash": _hash("9"),
        "input_root": _hash("a"),
        "output_root": _hash("b"),
        "artifact_root": _hash("c"),
        "result_hash": _hash("d"),
        "artifact_hashes": [],
        "result_doc": result,
    }


def _receipt(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "receipt_hash",
            "role",
            "purpose",
            "job_id",
            "epoch_id",
            "sequence",
            "input_root",
            "output_root",
            "artifact_root",
        )
    } | {"receipt_status": "succeeded"}


def _signed_weight_event(mapping: str = TEST_MAPPING) -> dict:
    payload = {
        "actor_hotkey": HOTKEY,
        "netuid": 401,
        "epoch_id": 22061,
        "block": 7_941_960,
        "weights_hash": "1" * 64,
        "bundle_hash": _hash("2"),
        "root_receipt_hash": _hash("3"),
        "epoch_authority": {
            "mode": "stateful_v1",
            "workflow_epoch_id": 22061,
            "official_subnet_epoch_id": 22061,
            "cutover_mapping_hash": mapping,
        },
    }
    signed_event = {
        "event_type": "WEIGHT_SUBMISSION_V2",
        "timestamp": "2026-09-08T18:45:00Z",
        "boot_id": "boot-testnet401",
        "monotonic_seq": 7,
        "prev_event_hash": "4" * 64,
        "payload": payload,
    }
    signed_log_entry = {
        "signed_event": signed_event,
        "event_hash": "5" * 64,
        "enclave_pubkey": "6" * 64,
        "enclave_signature": "7" * 128,
    }
    return {
        "event_type": "WEIGHT_SUBMISSION_V2",
        "epoch_id": None,
        "actor_hotkey": HOTKEY,
        "payload_hash": "8" * 64,
        "signature": "7" * 128,
        "payload": payload,
        "signed_log_entry": signed_log_entry,
        "event_hash": "5" * 64,
        "enclave_pubkey": "6" * 64,
        "boot_id": "boot-testnet401",
        "monotonic_seq": 7,
        "prev_event_hash": "4" * 64,
    }


def _json(value: dict) -> str:
    return "'%s'::JSONB" % json.dumps(value, sort_keys=True).replace("'", "''")


def test_postgres15_temporary_testnet401_scope_is_exact_and_finney_unchanged():
    if os.environ.get("RUN_POSTGRES_15_INTEGRATION") != "1":
        pytest.skip("set RUN_POSTGRES_15_INTEGRATION=1 for PostgreSQL 15")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for PostgreSQL 15")

    container = "leadpoet-testnet401-result-pg15-" + uuid.uuid4().hex[:10]

    def psql(statement: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                "docker", "exec", "-i", container, "psql", "-X", "-A", "-t",
                "-U", "postgres", "-d", "leadpoet", "-v", "ON_ERROR_STOP=1",
            ],
            input=statement,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if check and result.returncode != 0:
            raise AssertionError(result.stderr)
        return result

    def rejected(statement: str, message: str) -> None:
        result = psql(statement, check=False)
        assert result.returncode != 0, result.stdout
        assert message in result.stderr, result.stderr

    def insert_result(row: dict, *, check: bool = True):
        psql(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2 "
            "SELECT * FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_receipts_v2, {_json(_receipt(row))});"
        )
        return psql(
            "INSERT INTO public.research_lab_attested_execution_results_v2 "
            "SELECT * FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_results_v2, {_json(row)});",
            check=check,
        )

    def insert_event(event: dict, *, check: bool = True):
        fields = (
            "event_type", "epoch_id", "actor_hotkey", "payload_hash", "signature",
            "payload", "signed_log_entry", "event_hash", "enclave_pubkey",
            "boot_id", "monotonic_seq", "prev_event_hash",
        )
        columns = ",".join(fields)
        return psql(
            f"INSERT INTO public.transparency_log ({columns}) SELECT {columns} "
            "FROM pg_catalog.jsonb_populate_record(NULL::public.transparency_log, "
            f"{_json(event)});",
            check=check,
        )

    setup = f"""
CREATE ROLE anon; CREATE ROLE authenticated; CREATE ROLE service_role;
CREATE TABLE public.research_lab_stateful_subnet_epoch_cutover_state_v1 (
    singleton BOOLEAN PRIMARY KEY, mapping_hash TEXT NOT NULL
);
INSERT INTO public.research_lab_stateful_subnet_epoch_cutover_state_v1
VALUES (TRUE, '{FINNEY_MAPPING}');
CREATE TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 (
    schema_version TEXT, previous_epoch_scheme TEXT, network_genesis_hash TEXT,
    netuid INTEGER, mapping_hash TEXT, cutover_authority_hash TEXT,
    cutover_receipt_hash TEXT, first_settlement_epoch_id INTEGER,
    UNIQUE(network_genesis_hash, netuid)
);
INSERT INTO public.research_lab_stateful_subnet_epoch_cutovers_v1 VALUES (
    'leadpoet.subnet_epoch_cutover_authority.v3', 'fresh_network_v1',
    '{GENESIS}', 401, '{TEST_MAPPING}',
    'sha256:2eb438e142f4c27d5ce28b2e2c5148cc030724a38ab0bf479ce4fc0fade51bc7',
    'sha256:4db3b2649bbd2182488b3f17cbff87abf04f96c9511055cb51efe743710b6aaa',
    22042
);
CREATE TABLE public.research_lab_attested_execution_receipts_v2 (
    receipt_hash TEXT PRIMARY KEY, role TEXT, purpose TEXT, job_id TEXT,
    epoch_id BIGINT, sequence INTEGER, input_root TEXT, output_root TEXT,
    artifact_root TEXT, receipt_status TEXT
);
CREATE TABLE public.research_lab_attested_execution_results_v2 (
    receipt_hash TEXT PRIMARY KEY REFERENCES public.research_lab_attested_execution_receipts_v2,
    schema_version TEXT, role TEXT, operation TEXT, purpose TEXT, job_id TEXT,
    epoch_id BIGINT, sequence INTEGER, release_hash TEXT, input_root TEXT,
    output_root TEXT, artifact_root TEXT, result_hash TEXT,
    artifact_hashes JSONB, result_doc JSONB
);
CREATE TABLE public.transparency_log (
    id BIGSERIAL PRIMARY KEY, event_type TEXT, epoch_id BIGINT,
    actor_hotkey TEXT, payload_hash TEXT, signature TEXT, payload JSONB,
    signed_log_entry JSONB, event_hash TEXT, enclave_pubkey TEXT, boot_id TEXT,
    monotonic_seq BIGINT, prev_event_hash TEXT
);
CREATE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE mapping_value JSONB; state_mapping TEXT;
BEGIN
    SELECT mapping_hash INTO state_mapping
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton;
    FOR mapping_value IN SELECT value FROM pg_catalog.jsonb_path_query(
        pg_catalog.to_jsonb(NEW), 'strict $.**.cutover_mapping_hash'
    ) AS item(value) LOOP
        IF pg_catalog.jsonb_typeof(mapping_value) <> 'string'
           OR mapping_value #>> '{{}}' <> state_mapping THEN
            RAISE EXCEPTION 'stateful epoch active mapping authority differs on %', TG_TABLE_NAME;
        END IF;
    END LOOP;
    RETURN NEW;
END; $$;
CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
BEFORE INSERT OR UPDATE ON public.research_lab_attested_execution_results_v2
FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();
CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
BEFORE INSERT OR UPDATE ON public.transparency_log
FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();
"""

    try:
        subprocess.run(
            [
                "docker", "run", "--detach", "--rm", "--name", container,
                "--env", "POSTGRES_PASSWORD=postgres", "--env",
                "POSTGRES_DB=leadpoet", "postgres:15",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        for _ in range(80):
            if subprocess.run(
                [
                    "docker", "exec", container, "psql", "-X", "-U",
                    "postgres", "-d", "leadpoet", "-c", "SELECT 1",
                ],
                capture_output=True,
            ).returncode == 0:
                break
            time.sleep(0.25)
        else:
            raise AssertionError("PostgreSQL 15 did not become ready")
        psql(setup)
        original_definition = psql(
            "SELECT pg_catalog.pg_get_functiondef("
            "'public.enforce_research_lab_stateful_epoch_fence_v1()'::REGPROCEDURE);"
        ).stdout
        psql(SQL)
        psql(SQL)
        assert psql(
            "SELECT pg_catalog.pg_get_functiondef("
            "'public.enforce_research_lab_stateful_epoch_fence_v1()'::REGPROCEDURE);"
        ).stdout == original_definition

        allocation = _result_row(
            receipt_hash=_hash("1"), operation="research_lab_allocation",
            purpose="research_lab.allocation.v2", result=_allocation_result(),
        )
        assert insert_result(allocation).returncode == 0

        successor_result = _allocation_result()
        successor_result["source_state"].pop("fresh_network_origin")
        successor = _result_row(
            receipt_hash=_hash("a"), operation="research_lab_allocation",
            purpose="research_lab.allocation.v2", result=successor_result,
        )
        assert insert_result(successor).returncode == 0

        mixed = _result_row(
            receipt_hash=_hash("2"), operation="research_lab_allocation",
            purpose="research_lab.allocation.v2", result=_allocation_result(),
        )
        mixed["result_doc"]["allocation_inputs"]["cutover_mapping_hash"] = FINNEY_MAPPING
        rejected(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_receipts_v2, {_json(_receipt(mixed))});"
            "INSERT INTO public.research_lab_attested_execution_results_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_results_v2, {_json(mixed)});",
            "mixed epoch authority",
        )

        wrong_netuid = _result_row(
            receipt_hash=_hash("0"), operation="research_lab_allocation",
            purpose="research_lab.allocation.v2", result=_allocation_result(),
        )
        wrong_netuid["result_doc"]["source_state"]["netuid"] = 400
        wrong_netuid["result_doc"]["source_state"]["settlement_frontier"]["netuid"] = 400
        psql(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_receipts_v2, {_json(_receipt(wrong_netuid))});"
        )
        rejected(
            "INSERT INTO public.research_lab_attested_execution_results_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_results_v2, {_json(wrong_netuid)});",
            "allocation result is invalid",
        )

        observation = _result_row(
            receipt_hash=_hash("3"), operation="observe_chain_realized_weights_v1",
            purpose="research_lab.chain_weight_observation.v1", result=_observation(),
        )
        assert insert_result(observation).returncode == 0

        wrong_validator = _result_row(
            receipt_hash=_hash("e"), operation="observe_chain_realized_weights_v1",
            purpose="research_lab.chain_weight_observation.v1", result=_observation(),
        )
        wrong_validator["result_doc"]["validator_hotkey"] = "wrong-hotkey"
        psql(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_receipts_v2, {_json(_receipt(wrong_validator))});"
        )
        rejected(
            "INSERT INTO public.research_lab_attested_execution_results_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_results_v2, {_json(wrong_validator)});",
            "chain observation is invalid",
        )

        receipt_mismatch = _result_row(
            receipt_hash=_hash("4"), operation="observe_chain_realized_weights_v1",
            purpose="research_lab.chain_weight_observation.v1", result=_observation(),
        )
        bad_receipt = _receipt(receipt_mismatch)
        bad_receipt["output_root"] = _hash("e")
        psql(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_receipts_v2, {_json(bad_receipt)});"
        )
        rejected(
            "INSERT INTO public.research_lab_attested_execution_results_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_results_v2, {_json(receipt_mismatch)});",
            "result receipt differs",
        )

        weight_input = _result_row(
            receipt_hash=_hash("5"), operation="attest_weight_input",
            purpose="research_lab.allocation.v2",
            result={
                "schema_version": "leadpoet.weight_input_value.v2",
                "category": "research_lab_allocation", "netuid": 401,
                "epoch_id": 22061, "value": {},
            },
        )
        assert insert_result(weight_input).returncode == 0

        finney = _result_row(
            receipt_hash=_hash("6"), operation="research_lab_allocation",
            purpose="research_lab.allocation.v2", result=_allocation_result(FINNEY_MAPPING),
        )
        finney["result_doc"]["source_state"]["netuid"] = 71
        finney["result_doc"]["source_state"]["settlement_frontier"]["netuid"] = 71
        assert insert_result(finney).returncode == 0

        wrong_finney = _result_row(
            receipt_hash=_hash("7"), operation="research_lab_allocation",
            purpose="research_lab.allocation.v2", result=_allocation_result(_hash("f")),
        )
        wrong_finney["result_doc"]["source_state"]["netuid"] = 71
        wrong_finney["result_doc"]["source_state"]["settlement_frontier"]["netuid"] = 71
        rejected(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_receipts_v2, {_json(_receipt(wrong_finney))});"
            "INSERT INTO public.research_lab_attested_execution_results_v2 SELECT * "
            "FROM pg_catalog.jsonb_populate_record(NULL::public."
            f"research_lab_attested_execution_results_v2, {_json(wrong_finney)});",
            "stateful epoch active mapping authority differs",
        )

        signed_event = _signed_weight_event()
        assert insert_event(signed_event).returncode == 0
        unsigned = _signed_weight_event()
        for field in ("signature", "event_hash", "enclave_pubkey", "monotonic_seq"):
            unsigned[field] = None
        unsigned["signed_log_entry"]["enclave_signature"] = None
        unsigned["signed_log_entry"]["event_hash"] = None
        unsigned["signed_log_entry"]["enclave_pubkey"] = None
        unsigned["signed_log_entry"]["signed_event"]["monotonic_seq"] = None
        assert "weight submission event is invalid" in insert_event(
            unsigned, check=False
        ).stderr
        null_epoch = _signed_weight_event()
        null_epoch["payload"]["epoch_id"] = None
        null_epoch["payload"]["epoch_authority"]["workflow_epoch_id"] = None
        assert "weight submission event is invalid" in insert_event(
            null_epoch, check=False
        ).stderr
        mixed_event = _signed_weight_event()
        mixed_event["signed_log_entry"]["signed_event"]["payload"][
            "epoch_authority"
        ]["cutover_mapping_hash"] = FINNEY_MAPPING
        assert "weight submission event is invalid" in insert_event(
            mixed_event, check=False
        ).stderr
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
