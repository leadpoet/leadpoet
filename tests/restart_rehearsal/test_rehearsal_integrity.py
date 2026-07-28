from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Optional
import urllib.request

import pytest

from scripts import run_local_restart_rehearsal as rehearsal
from tests.restart_rehearsal import postgres_v2_contract_probe as postgres_probe
from tests.restart_rehearsal.artifact_identity import (
    ALL_ROLES,
    docker_save_archive,
    eif_hash,
    normalized_image_id,
    pcr0,
)
from tests.restart_rehearsal import join_evidence
from tests.restart_rehearsal import sitecustomize as rehearsal_sitecustomize
from tests.restart_rehearsal.gateway_boundary_service import (
    LocalPostgRESTState,
    RUNTIME_TABLES,
    _apply_table_query,
    _attested_store_tables,
    _direct_provider_store_tables,
    _measured_query_tables,
    _migration_schema_contract,
    _schema_contract,
)
from tests.restart_rehearsal.postgres_v2_contract_probe import (
    DisposablePostgres,
    EXPECTED_FINALIZED_VIEW_COLUMNS,
    MIGRATIONS_BEFORE_TRANSPORT_FIX,
    TRANSPORT_FIX_MIGRATION,
    _json_insert_sql,
    _validate_required_migration_declarations,
)
from tests.restart_rehearsal.local_services import LocalBoundaryServices
from tests.restart_rehearsal.verify_evidence import (
    EXPECTED_GATEWAY_PRIVATE_MODEL_ENV,
    selected_weight_storage_preflight_capability,
    verify_gateway_private_model_environment,
    verify_migration_backed_database_contract,
    verify_gateway_weight_readiness_invocations,
    verify_rehearsal_integrity,
    verify_restart_epoch_transient_recovery,
)


COMMIT = "1" * 40
VALIDATOR_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


def test_chain_settlement_activation_fixture_tracks_current_epoch(
    tmp_path,
    monkeypatch,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "SOURCE_ROOT",
        source_root,
    )
    fixture = json.loads(
        (
            source_root
            / "tests/restart_rehearsal/fixtures/production_shaped_v2.json"
        ).read_text(encoding="utf-8")
    )
    cutover = json.loads(
        (
            source_root / "config/stateful-epoch-cutover-sn71.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["network"]["current_block"] == (
        rehearsal_sitecustomize.CURRENT_BLOCK
    )
    assert fixture["network"]["subnet_epoch_index"] == (
        rehearsal_sitecustomize.SUBNET_EPOCH_INDEX
    )
    expected_epoch = int(cutover["first_settlement_epoch_id"]) + (
        int(fixture["network"]["subnet_epoch_index"])
        - int(cutover["first_subnet_epoch_index"])
    )
    state = LocalPostgRESTState(
        state_root=tmp_path,
        fixture=fixture,
        source_root=source_root,
        tables={
            "research_lab_chain_realized_settlement_activation_v1",
            "research_lab_stateful_subnet_epoch_cutover_state_v1",
            "research_lab_stateful_subnet_epoch_cutovers_v1",
        },
        rpcs=set(),
    )
    assert state.rows[
        "research_lab_chain_realized_settlement_activation_v1"
    ] == [
        {
            "netuid": 71,
            "schema_version": (
                "leadpoet.research_lab_chain_realized_settlement_activation.v1"
            ),
            "first_epoch_id": expected_epoch,
            "source_bundle_hash": "sha256:" + "a" * 64,
            "source_bundle_epoch_id": expected_epoch,
            "source_finalized_block": (
                int(fixture["network"]["current_block"]) - 1
            ),
        }
    ]
    assert rehearsal_sitecustomize._current_settlement_epoch_id() == (
        expected_epoch
    )


def test_gateway_rehearsal_discovers_candidate_direct_provider_tables() -> None:
    source_root = Path(__file__).resolve().parents[2]
    assert _direct_provider_store_tables(source_root) >= {
        "research_lab_provider_evidence_cache_v2",
        "research_lab_provider_outcome_checkpoints_v2",
    }


def test_gateway_rehearsal_applies_production_postgrest_query_semantics() -> None:
    rows = [
        {
            "receipt_hash": "sha256:" + "1" * 64,
            "start_epoch": 24217,
            "status": "active",
            "extra": "first",
        },
        {
            "receipt_hash": "sha256:" + "2" * 64,
            "start_epoch": 24219,
            "status": "queued",
            "extra": "second",
        },
        {
            "receipt_hash": "sha256:" + "3" * 64,
            "start_epoch": 24220,
            "status": "paid",
            "extra": "third",
        },
    ]
    assert _apply_table_query(
        rows,
        "select=receipt_hash,start_epoch"
        "&status=in.(active,queued)"
        "&start_epoch=gte.24217"
        "&start_epoch=lte.24219"
        "&order=start_epoch.desc"
        "&offset=0"
        "&limit=1",
    ) == [
        {
            "receipt_hash": "sha256:" + "2" * 64,
            "start_epoch": 24219,
        }
    ]
    assert _apply_table_query(
        rows,
        'receipt_hash=in.("sha256:%s")' % ("1" * 64),
    ) == [rows[0]]


def test_gateway_rehearsal_rejects_columns_absent_from_migration_schema() -> None:
    columns = frozenset({"bundle_hash", "finalization_doc"})
    with pytest.raises(ValueError, match="selection references unknown"):
        _apply_table_query(
            [],
            "select=bundle_hash,weight_receipt_hash",
            allowed_columns=columns,
        )
    with pytest.raises(ValueError, match="filter references unknown"):
        _apply_table_query(
            [],
            "weight_receipt_hash=eq.sha256:test",
            allowed_columns=columns,
        )


def test_migration_backed_contract_is_candidate_bound_and_complete(
    tmp_path,
) -> None:
    relations = {
        name: {"kind": "r", "columns": ["schema_version"]}
        for name in {
            "research_lab_attested_transport_attempts_v2",
            "research_lab_attested_execution_receipts_v2",
            "research_lab_attested_weight_bundles_v2",
            "research_lab_attested_publication_events_v2",
            "research_lab_attested_weight_finalizations_v2",
            "research_lab_finalized_allocation_epochs_v2",
            "research_lab_chain_realized_epoch_settlements_v1",
            "research_lab_chain_realized_settlement_activation_v1",
            "research_lab_chain_realized_obligation_credits_v1",
        }
    }
    relations["research_lab_finalized_allocation_epochs_v2"] = {
        "kind": "v",
        "columns": list(EXPECTED_FINALIZED_VIEW_COLUMNS),
    }
    contract = {
        "schema_version": "leadpoet.restart_rehearsal.postgres_contract.v1",
        "candidate_sha": COMMIT,
        "applied_migrations": [
            *MIGRATIONS_BEFORE_TRANSPORT_FIX,
            TRANSPORT_FIX_MIGRATION,
        ],
        "relations": relations,
        "rpcs": [
            "research_lab_attested_transport_purpose_contract_v2",
        ],
        "checks": {
            "pre_128_transport_rejected": True,
            "post_128_transport_persisted": True,
        },
    }
    path = tmp_path / "postgres-contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    relation_columns, rpcs = _migration_schema_contract(
        path,
        candidate_sha=COMMIT,
    )

    assert relation_columns[
        "research_lab_finalized_allocation_epochs_v2"
    ] == frozenset(EXPECTED_FINALIZED_VIEW_COLUMNS)
    assert "research_lab_attested_transport_purpose_contract_v2" in rpcs
    with pytest.raises(RuntimeError, match="differs from candidate"):
        _migration_schema_contract(path, candidate_sha="2" * 40)


def test_rehearsal_evidence_requires_all_postgres_contract_checks(
    tmp_path,
    monkeypatch,
) -> None:
    state_root = tmp_path / "rehearsal-state"
    state_root.mkdir()
    contract_path = state_root / "postgres-v2-schema-contract.json"
    contract = {
        "schema_version": "leadpoet.restart_rehearsal.postgres_contract.v1",
        "candidate_sha": COMMIT,
        "relations": {
            "research_lab_finalized_allocation_epochs_v2": {
                "kind": "v",
                "columns": list(EXPECTED_FINALIZED_VIEW_COLUMNS),
            }
        },
        "checks": {
            "pre_128_transport_rejected": True,
            "post_128_transport_persisted": True,
            "transport_contract_valid": True,
            "finalized_view_projection_exact": True,
            "settlement_authority_parsed": True,
            "tampered_weight_receipt_rejected": True,
            "required_schema_migrations_declared": True,
        },
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: (
            original_is_file(contract_path)
            if str(self)
            == "/rehearsal-state/postgres-v2-schema-contract.json"
            else original_is_file(self)
        ),
    )

    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def read_text(path: Path, *args, **kwargs):
        if str(path) == "/rehearsal-state/postgres-v2-schema-contract.json":
            return original_read_text(contract_path, *args, **kwargs)
        return original_read_text(path, *args, **kwargs)

    def read_bytes(path: Path):
        if str(path) == "/rehearsal-state/postgres-v2-schema-contract.json":
            return original_read_bytes(contract_path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    assert verify_migration_backed_database_contract(COMMIT) == (
        hashlib.sha256(contract_path.read_bytes()).hexdigest()
    )
    contract["checks"]["settlement_authority_parsed"] = False
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(SystemExit, match="evidence is incomplete"):
        verify_migration_backed_database_contract(COMMIT)


def test_postgres_fixture_insert_names_explicit_columns() -> None:
    statement = _json_insert_sql(
        "research_lab_attested_transport_attempts_v2",
        {"attempt_hash": "sha256:" + "1" * 64, "purpose": "example.v2"},
    )
    assert (
        "research_lab_attested_transport_attempts_v2 "
        "(attempt_hash,purpose)" in statement
    )
    assert "json_populate_record" in statement


def test_postgres_probe_resolves_system_binary_outside_launcher_path(
    tmp_path,
    monkeypatch,
) -> None:
    system_dir = tmp_path / "usr" / "sbin"
    system_dir.mkdir(parents=True)
    runuser = system_dir / "runuser"
    runuser.write_text("#!/bin/sh\n", encoding="utf-8")
    runuser.chmod(0o755)
    monkeypatch.setattr(postgres_probe, "SYSTEM_BINARY_DIRS", (system_dir,))
    monkeypatch.setattr(postgres_probe.shutil, "which", lambda _name: None)

    assert DisposablePostgres._binary("runuser") == str(runuser)
    with pytest.raises(
        postgres_probe.PostgresContractProbeError,
        match="postgres binary is unavailable",
    ):
        DisposablePostgres._binary("missing")


def test_required_schema_preflight_is_backed_by_candidate_migrations() -> None:
    source_root = Path(__file__).resolve().parents[2]
    counts = _validate_required_migration_declarations(source_root)
    assert counts["relation_probe_count"] > 20
    assert counts["rpc_probe_count"] > 10
    assert counts["migration_count"] > 10


def test_gateway_rehearsal_signing_identity_uses_its_real_private_key() -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    role = "gateway_coordinator"
    payload = b"candidate coordinator receipt digest"
    signature = rehearsal_sitecustomize._local_signing_private_key(role).sign(
        payload
    )
    Ed25519PublicKey.from_public_bytes(
        bytes.fromhex(
            rehearsal_sitecustomize._local_signing_public_key(role)
        )
    ).verify(signature, payload)


def test_gateway_rehearsal_chain_adapter_enforces_exact_cutover_reads(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize, "EVENT_PATH", tmp_path / "events.jsonl"
    )
    request = {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "chain_getBlockHash",
        "params": [rehearsal_sitecustomize.CUTOVER_BLOCK],
    }
    response = json.loads(
        rehearsal_sitecustomize._local_chain_rpc(
            json.dumps(request).encode(),
            archive=True,
        )
    )
    assert response["result"] == rehearsal_sitecustomize.CUTOVER_BLOCK_HASH

    request["method"] = "state_getRuntimeVersion"
    request["params"] = [
        rehearsal_sitecustomize._block_hash(
            rehearsal_sitecustomize.CURRENT_BLOCK
        )
    ]
    response = json.loads(
        rehearsal_sitecustomize._local_chain_rpc(
            json.dumps(request).encode(),
            archive=False,
        )
    )
    assert response["result"]["specVersion"] == 440
    assert response["result"]["transactionVersion"] == 1

    request["params"] = [
        rehearsal_sitecustomize._block_hash(
            rehearsal_sitecustomize.CURRENT_BLOCK - 1
        )
    ]
    with pytest.raises(ValueError, match="unknown RPC"):
        rehearsal_sitecustomize._local_chain_rpc(
            json.dumps(request).encode(),
            archive=False,
        )

    request["method"] = "author_submitExtrinsic"
    request["params"] = ["0x00"]
    with pytest.raises(ValueError, match="unknown RPC"):
        rehearsal_sitecustomize._local_chain_rpc(
            json.dumps(request).encode(),
            archive=False,
        )


def test_restart_rehearsal_injects_and_proves_transient_epoch_read_recovery(
    tmp_path,
    monkeypatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(rehearsal_sitecustomize, "EVENT_PATH", event_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_RESTART_EPOCH_TRANSIENT_HEAD_CALLS",
        0,
    )
    monkeypatch.setenv(
        "LEADPOET_REHEARSAL_RESTART_EPOCH_TRANSIENT_FAILURES",
        "1",
    )

    first = rehearsal_sitecustomize._LocalSubstrate().get_chain_head()
    second = rehearsal_sitecustomize._LocalSubstrate().get_chain_head()

    assert first == "malformed-transient-head"
    assert second == rehearsal_sitecustomize._block_hash(
        rehearsal_sitecustomize.CURRENT_BLOCK
    )
    rows = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    verify_restart_epoch_transient_recovery(rows)


def test_chain_adapter_signs_bittensor_10_serve_axon_runtime_shape(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize, "EVENT_PATH", tmp_path / "events.jsonl"
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "SOURCE_ROOT",
        Path(__file__).resolve().parents[2],
    )
    substrate = rehearsal_sitecustomize._LocalSubstrate()
    call = substrate.compose_call(
        call_module="SubtensorModule",
        call_function="serve_axon",
        call_params={
            "version": 901,
            "ip": 2_130_706_433,
            "port": 8091,
            "ip_type": 4,
            "netuid": 71,
            "protocol": 4,
            "placeholder1": 0,
            "placeholder2": 0,
        },
    )
    payload_kwargs = {
        "call": call,
        "era": {
            "period": 8,
            "current": rehearsal_sitecustomize.CURRENT_BLOCK,
        },
        "nonce": 7,
        "tip": 0,
        "tip_asset_id": None,
    }

    with pytest.raises(ValueError, match="no SDK signer identity"):
        substrate.generate_signature_payload(**payload_kwargs)

    assert substrate.get_account_nonce(VALIDATOR_HOTKEY) == 7
    payload = substrate.generate_signature_payload(**payload_kwargs)

    assert isinstance(payload.data, bytes)
    assert payload.data
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert any(
        event.get("operation") == "submit_extrinsic"
        and event.get("method") == "generate_signature_payload"
        for event in events
    )


def test_validator_enclave_chain_tls_boundary_runs_real_signing_reads(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sitecustomize", rehearsal_sitecustomize)
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )

    from tests.restart_rehearsal import validator_enclave_service
    from validator_tee.enclave.chain_source_v2 import (
        EnclaveChainRpcTransportV2,
        ValidatorChainSourceV2,
    )

    monkeypatch.setattr(
        EnclaveChainRpcTransportV2,
        "_http_post",
        validator_enclave_service._local_chain_http_post,
    )
    transport = EnclaveChainRpcTransportV2(sleep=lambda _seconds: None)
    source = ValidatorChainSourceV2(
        rpc_call=transport.call,
        epoch_authority_supplier=lambda: None,
    )
    result = source.read_chain_signing_runtime(
        runtime_block_hash=rehearsal_sitecustomize._block_hash(
            rehearsal_sitecustomize.CURRENT_BLOCK
        ),
        max_block_drift=64,
    )

    assert result["runtime_block"] == rehearsal_sitecustomize.CURRENT_BLOCK
    assert result["finalized_block"] == rehearsal_sitecustomize.CURRENT_BLOCK
    assert result["spec_version"] == 440
    assert result["transaction_version"] == 1
    assert result["genesis_hash"] == (
        rehearsal_sitecustomize.GENESIS_HASH.removeprefix("0x")
    )
    assert len(result["attempts"]) == 6
    assert all(
        attempt["terminal_status"] == "authenticated_response"
        for attempt in result["attempts"]
    )


def test_gateway_rehearsal_artifact_store_is_object_locked_and_immutable(
    tmp_path,
    monkeypatch,
) -> None:
    from gateway.tee.artifact_persistence_v2 import (
        ArtifactPersistenceV2Error,
        _validate_presigned_url,
    )

    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize, "EVENT_PATH", tmp_path / "events.jsonl"
    )
    client = rehearsal_sitecustomize._LocalS3()
    retain_until = datetime.now(timezone.utc) + timedelta(days=365)
    body = b'{"artifact_id":"sha256:local"}'
    response = client.put_object(
        Bucket="restart-rehearsal",
        Key="encrypted-artifacts/sha256:local.json",
        Body=body,
        ContentType="application/json",
        ObjectLockMode="COMPLIANCE",
        ObjectLockRetainUntilDate=retain_until,
    )
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert (
        client.get_object(
            Bucket="restart-rehearsal",
            Key="encrypted-artifacts/sha256:local.json",
        )["Body"].read()
        == body
    )
    assert client.head_object(
        Bucket="restart-rehearsal",
        Key="encrypted-artifacts/sha256:local.json",
    )["ObjectLockMode"] == "COMPLIANCE"
    presigned_url = client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": "restart-rehearsal",
            "Key": "encrypted-artifacts/sha256:local.json",
        },
        ExpiresIn=300,
        HttpMethod="GET",
    )
    policy = {
        "bucket_host": "restart-rehearsal.s3.us-east-1.amazonaws.com",
        "key_prefix": "/encrypted-artifacts/",
    }
    assert _validate_presigned_url(presigned_url, policy=policy) == presigned_url
    with pytest.raises(
        ArtifactPersistenceV2Error,
        match="not SigV4 signed",
    ):
        _validate_presigned_url(
            "https://restart-rehearsal.s3.us-east-1.amazonaws.com/"
            "encrypted-artifacts/sha256:local.json",
            policy=policy,
        )
    with pytest.raises(ValueError, match="immutable artifact differs"):
        client.put_object(
            Bucket="restart-rehearsal",
            Key="encrypted-artifacts/sha256:local.json",
            Body=base64.b64encode(body),
            ContentType="application/json",
            ObjectLockMode="COMPLIANCE",
            ObjectLockRetainUntilDate=retain_until,
        )


def test_gateway_rehearsal_provider_boundary_rejects_unknown_hosts() -> None:
    with pytest.raises(ValueError, match="unknown host"):
        rehearsal_sitecustomize._local_provider_transport(
            method="GET",
            url="https://example.invalid/not-production",
            headers={},
            body=b"",
            timeout_ms=1000,
        )


def test_gateway_readiness_views_are_strictly_registered() -> None:
    assert {
        "research_lab_champion_reward_current",
        "research_lab_source_add_reward_current",
    } <= RUNTIME_TABLES


def test_gateway_boundary_registers_every_candidate_measured_query() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    from gateway.tee.supabase_source_v2 import QUERY_POLICIES

    expected = {policy.table for policy in QUERY_POLICIES.values()}
    assert _measured_query_tables(repository_root) == expected
    assert {
        "research_reimbursement_schedules",
        "research_reimbursement_award_current",
    } <= expected


def test_gateway_boundary_registers_every_candidate_attested_store_table() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    from gateway.research_lab import attested_v2_store

    expected = {
        value
        for name, value in vars(attested_v2_store).items()
        if name.endswith("_TABLE") and isinstance(value, str)
    }
    assert _attested_store_tables(repository_root) == expected
    assert {
        "research_lab_attested_execution_results_v2",
        "research_lab_attested_receipt_edges_v2",
        "research_lab_attested_weight_bundles_v2",
        "research_lab_attested_publication_events_v2",
        "research_lab_attested_weight_finalizations_v2",
    } <= expected


def test_gateway_boundary_registers_background_startup_schema_contracts() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    tables, rpcs = _schema_contract(repository_root)
    assert {
        "research_lab_gateway_control_current",
        "research_lab_public_benchmark_report_current",
    } <= tables
    assert "research_lab_source_add_claim_work" in rpcs
    assert "research_lab_gateway_control_current" in (
        repository_root / "scripts/44-research-lab-maintenance-pause.sql"
    ).read_text(encoding="utf-8")
    assert "research_lab_public_benchmark_report_current" in (
        repository_root
        / "scripts/53-research-lab-benchmark-quality-current-views.sql"
    ).read_text(encoding="utf-8")
    assert "research_lab_source_add_claim_work" in (
        repository_root
        / "scripts/96-research-lab-source-add-functional-workflow.sql"
    ).read_text(encoding="utf-8")


def test_local_urlopen_routes_authenticated_weight_handoff_to_real_gateway(
    tmp_path,
    monkeypatch,
) -> None:
    sentinel = object()
    observed: dict[str, Any] = {}

    def real_urlopen(request: Any, *, timeout: Optional[float] = None) -> Any:
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return sentinel

    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_real_urlopen",
        real_urlopen,
        raising=False,
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    request = urllib.request.Request(
        "http://localhost:8000/research-lab/allocations/attested/24219",
        headers={"x-leadpoet-internal-key": "rehearsal-internal"},
        method="GET",
    )
    result = rehearsal_sitecustomize._local_urlopen(request, timeout=359.5)
    assert result is sentinel
    assert observed == {
        "url": request.full_url,
        "timeout": 359.5,
    }

    unauthenticated = urllib.request.Request(
        request.full_url,
        method="GET",
    )
    with pytest.raises(ValueError, match="handoff contract differs"):
        rehearsal_sitecustomize._local_urlopen(
            unauthenticated,
            timeout=359.5,
        )
    wrong_path = urllib.request.Request(
        "http://localhost:8000/health",
        headers={"x-leadpoet-internal-key": "rehearsal-internal"},
        method="GET",
    )
    with pytest.raises(ValueError, match="handoff contract differs"):
        rehearsal_sitecustomize._local_urlopen(
            wrong_path,
            timeout=359.5,
        )


def test_gateway_secret_enables_the_production_weight_authority(
    monkeypatch,
) -> None:
    adapter_path = (
        Path(__file__).resolve().parent / "contract_adapter.py"
    )
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    spec = importlib.util.spec_from_file_location(
        "_rehearsal_gateway_secret_contract",
        adapter_path,
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    values = adapter._gateway_secret()
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    from gateway.research_lab.config import ResearchLabGatewayConfig

    config = ResearchLabGatewayConfig.from_env()
    assert values["BITTENSOR_NETWORK"] == "finney"
    assert values["BITTENSOR_NETUID"] == "71"
    assert config.api_enabled is True
    assert config.production_writes_enabled is True
    assert config.receipts_enabled is True
    assert config.evaluation_bundles_enabled is True
    assert config.weight_mutation_enabled is True
    assert config.internal_api_key == "rehearsal-internal"

    from gateway.research_lab.capture_health import (
        capture_health_violations,
        collect_capture_health,
    )

    capture_health = collect_capture_health(config)
    assert capture_health["production_writes_enabled"] is True
    assert capture_health_violations(capture_health) == []
    assert all(
        channel["status"] == "ok"
        for channel in capture_health["channels"].values()
    )


def test_deferred_gateway_rehearsal_keeps_legacy_and_selects_v2_tls_proxies(
    monkeypatch,
) -> None:
    adapter_path = Path(__file__).resolve().parent / "contract_adapter.py"
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    monkeypatch.setenv(
        "REHEARSAL_GATEWAY_WORKER_FLEET_MODE",
        "deferred",
    )
    spec = importlib.util.spec_from_file_location(
        "_rehearsal_gateway_deferred_proxy_contract",
        adapter_path,
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    values = adapter._gateway_secret()
    assert values["RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1"].startswith(
        "http://"
    )
    assert values[
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1"
    ].startswith("http://")
    assert values[
        "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1"
    ].startswith("https://")
    assert values["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1"].startswith(
        "https://"
    )


def test_gateway_secret_exercises_tls_proxy_success_and_plaintext_failure(
    monkeypatch,
) -> None:
    adapter_path = Path(__file__).resolve().parent / "contract_adapter.py"
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    spec = importlib.util.spec_from_file_location(
        "_rehearsal_gateway_proxy_contract",
        adapter_path,
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    success = adapter._gateway_secret()
    assert success["RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1"].startswith(
        "http://"
    )
    assert success["RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1"].startswith(
        "http://"
    )
    assert success["RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1"].startswith(
        "https://"
    )
    assert success["RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1"].endswith(
        ":443"
    )
    assert success["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1"].startswith(
        "https://"
    )
    assert success["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1"].endswith(":443")

    monkeypatch.setenv(
        "REHEARSAL_WEIGHT_READINESS_SCENARIO",
        "plaintext_proxy_rejected",
    )
    failure = adapter._gateway_secret()
    assert "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1" not in failure
    assert "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1" not in failure
    assert failure["RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1"].startswith(
        "http://"
    )
    assert failure["RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1"].startswith(
        "http://"
    )


def test_contract_adapter_loads_under_production_safe_path() -> None:
    adapter = Path(__file__).resolve().parent / "contract_adapter.py"
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTHONSAFEPATH": "1",
        "REHEARSAL_CANDIDATE_SHA": COMMIT,
    }
    safe_path_flag = ["-P"] if sys.version_info >= (3, 11) else []
    result = subprocess.run(
        [sys.executable, *safe_path_flag, str(adapter)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "adapter command is missing" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_docker_image_inspect_parser_accepts_production_option_order(
    monkeypatch,
) -> None:
    adapter_path = (
        Path(__file__).resolve().parent / "contract_adapter.py"
    )
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    spec = importlib.util.spec_from_file_location(
        "_rehearsal_docker_inspect_contract",
        adapter_path,
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    production_argv = [
        "image",
        "inspect",
        "-f",
        "{{.Id}}",
        "tee-enclave:gateway_coordinator",
    ]
    assert adapter._docker_image_inspect_contract(production_argv) == (
        "tee-enclave:gateway_coordinator",
        "{{.Id}}",
    )
    assert adapter._docker_image_inspect_contract(
        [
            "image",
            "inspect",
            "tee-enclave:gateway_coordinator",
            "--format",
            "{{.Id}}",
        ]
    ) == ("tee-enclave:gateway_coordinator", "{{.Id}}")

    with pytest.raises(ValueError, match="target is missing"):
        adapter._docker_image_inspect_contract(
            ["image", "inspect", "-f", "{{.Id}}"]
        )
    with pytest.raises(ValueError, match="exactly one target"):
        adapter._docker_image_inspect_contract(
            ["image", "inspect", "first", "second"]
        )
    with pytest.raises(ValueError, match="format is invalid"):
        adapter._docker_image_inspect_contract(
            [
                "image",
                "inspect",
                "-f",
                "{{.Id}}",
                "--format",
                "{{.Id}}",
                "image",
            ]
        )


def test_sitecustomize_installs_vsock_adapter_under_production_safe_path(
    tmp_path,
) -> None:
    harness_root = Path(__file__).resolve().parent
    repository_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": f"{harness_root}:{repository_root}",
        "PYTHONSAFEPATH": "1",
        "REHEARSAL_CANDIDATE_SHA": COMMIT,
        "REHEARSAL_SCOPE": "exact",
        "REHEARSAL_STATE_ROOT": str(tmp_path),
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    safe_path_flag = ["-P"] if sys.version_info >= (3, 11) else []
    stdout_path = tmp_path / "safe-path.stdout"
    stderr_path = tmp_path / "safe-path.stderr"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                *safe_path_flag,
                "-c",
                (
                    "import os, socket, sys; "
                    "assert 'leadpoet_canonical' not in sys.modules; "
                    "assert isinstance(socket.socket, type); "
                    "ordinary = socket.socket(socket.AF_INET, "
                    "socket.SOCK_STREAM); "
                    "assert isinstance(ordinary, socket.socket); "
                    "ordinary.close(); "
                    "print(type(socket.socket(40, socket.SOCK_STREAM)).__name__, "
                    "flush=True); "
                    "os._exit(0)"
                ),
            ],
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=env,
            cwd=tmp_path,
        )
        try:
            returncode = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
    observed_stdout = stdout_path.read_text(encoding="utf-8")
    observed_stderr = stderr_path.read_text(encoding="utf-8")
    assert returncode == 0, observed_stderr
    assert observed_stdout.strip() == "_LocalVsock"
    assert "Error in sitecustomize" not in observed_stderr


def test_local_kms_recipient_uses_boot_authorized_references() -> None:
    artifact_ref = "sha256:" + "a" * 64
    provider_ref = "sha256:" + "b" * 64
    role_state = {
        "configuration": {
            "artifact_master_key_ref_hash": artifact_ref,
            "provider_ref_hashes": {"openrouter": provider_ref},
        }
    }

    assert (
        rehearsal_sitecustomize._configured_credential_ref(
            role_state,
            "artifact_master_key",
        )
        == artifact_ref
    )
    assert (
        rehearsal_sitecustomize._configured_credential_ref(
            role_state,
            "openrouter",
        )
        == provider_ref
    )
    with pytest.raises(ValueError, match="not boot-authorized"):
        rehearsal_sitecustomize._configured_credential_ref(
            role_state,
            "benchmark_openrouter",
        )


def test_gateway_enclave_service_uses_the_image_import_layout(
    monkeypatch,
) -> None:
    gateway_root = Path(__file__).resolve().parents[2] / "gateway"
    monkeypatch.setenv("GATEWAY_ROOT", str(gateway_root))

    nsm_lib, tee_service = (
        rehearsal_sitecustomize._gateway_enclave_runtime_modules()
    )

    assert Path(nsm_lib.__file__).resolve() == (
        gateway_root / "tee/nsm_lib.py"
    ).resolve()
    assert sys.modules["nsm_lib"] is nsm_lib
    assert Path(tee_service.__file__).resolve() == (
        gateway_root / "tee/tee_service.py"
    ).resolve()
    assert Path(sys.modules["merkle"].__file__).resolve() == (
        gateway_root / "tee/merkle.py"
    ).resolve()
    assert Path(
        importlib.import_module("gateway.tee.nsm_lib").__file__
    ).resolve() == (gateway_root / "tee/nsm_lib.py").resolve()


def test_gateway_event_signer_uses_local_nitro_boundary(
    monkeypatch,
    tmp_path,
) -> None:
    from gateway.tee import enclave_signer

    gateway_root = Path(__file__).resolve().parents[2] / "gateway"
    role = "gateway_coordinator"
    (tmp_path / "release-build-input.json").write_text(
        json.dumps(
            {
                "commit_sha": COMMIT,
                "gateway_roles": {
                    role: {
                        "build_identity_hash": "sha256:" + "1" * 64,
                        "commit_sha": COMMIT,
                        "pcr0": pcr0(COMMIT),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    monkeypatch.setenv("GATEWAY_ROOT", str(gateway_root))
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)

    _, tee_service = (
        rehearsal_sitecustomize._gateway_enclave_runtime_modules()
    )
    enclave_signer._reset_for_testing()
    tee_service.event_signer_initialization = None
    tee_service.event_buffer.clear()
    try:
        response = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "initialize_event_signer",
            {"prev_log_tip_hash": None},
        )
        document = json.loads(
            base64.b64decode(
                response["identity"]["attestation_document_b64"],
                validate=True,
            )
        )
        assert document["schema_version"] == (
            "leadpoet.local_nitro_document.v1"
        )
        assert document["pcr0"] == pcr0(COMMIT)
        assert response["restart_log_entry"]["signed_event"]["event_type"] == (
            "ENCLAVE_RESTART"
        )
    finally:
        enclave_signer._reset_for_testing()
        tee_service.event_signer_initialization = None
        tee_service.event_buffer.clear()


def test_gateway_runtime_identity_uses_the_installed_local_nsm_boundary(
    monkeypatch,
) -> None:
    gateway_root = Path(__file__).resolve().parents[2] / "gateway"
    monkeypatch.setenv("GATEWAY_ROOT", str(gateway_root))
    expected_document = b"strict-local-nitro-document"

    def supplier(*, user_data, public_key, nonce=b""):
        assert user_data == b"job-claim"
        assert public_key == b"recipient-key"
        assert nonce == b""
        return {"Attestation": {"document": expected_document}}

    rehearsal_sitecustomize._install_gateway_nsm_attestation(supplier)
    runtime_identity = importlib.import_module(
        "gateway.tee.runtime_identity_v2"
    )
    assert runtime_identity.nsm_attestation_document(
        user_data=b"job-claim",
        signing_pubkey=b"recipient-key",
    ) == expected_document


def test_external_build_identities_match_the_production_image_normalizer(
    tmp_path,
) -> None:
    from validator_tee.host.docker_image_normalizer_v2 import (
        normalize_saved_image,
    )

    observed = set()
    for role in sorted(ALL_ROLES):
        archive = tmp_path / f"{role}-raw.tar"
        normalized = tmp_path / f"{role}-normalized.tar"
        archive.write_bytes(
            docker_save_archive(COMMIT, role, f"rehearsal:{role}-raw")
        )
        image_id = normalize_saved_image(
            archive_path=archive,
            output_path=normalized,
            normalized_image=f"rehearsal:{role}",
            temporary_parent=tmp_path,
        )
        assert image_id == normalized_image_id(COMMIT, role)
        assert image_id not in observed
        observed.add(image_id)


def test_release_channel_uses_the_same_commit_bound_external_artifacts(
    monkeypatch,
    tmp_path,
) -> None:
    from gateway.tee.topology import ROLE_SPECS, topology_hash

    roles = {}
    for role in ROLE_SPECS:
        roles[role] = {
            "build_identity_hash": "sha256:" + "1" * 64,
            "commit_sha": COMMIT,
            "dependency_lock_hash": "sha256:" + "2" * 64,
            "dockerfile_hash": "sha256:" + "3" * 64,
            "eif_hash": eif_hash(COMMIT, role),
            "execution_manifest_hash": "sha256:" + "4" * 64,
            "normalized_image_hash": normalized_image_id(COMMIT, role),
            "pcr0": pcr0(COMMIT),
            "source_manifest_hash": "sha256:" + "6" * 64,
            "topology_hash": topology_hash(),
        }
    (tmp_path / "release-build-input.json").write_text(
        json.dumps(
            {
                "commit_sha": COMMIT,
                "gateway_roles": roles,
                "validator_app_manifest_hash": "sha256:" + "7" * 64,
                "validator_dependency_lock_hash": "sha256:" + "8" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "SOURCE_ROOT",
        Path(__file__).resolve().parents[2],
    )
    channel = rehearsal_sitecustomize._release_channel(COMMIT)
    gateway = channel["gateway_release_manifest"]
    for role, release in gateway["roles"].items():
        assert release["normalized_image_hash"] == normalized_image_id(
            COMMIT,
            role,
        )
        assert eif_hash(COMMIT, role) in release["eif_hashes"]
    validator = channel["validator_release_manifest"]["release"]
    assert validator["normalized_image_hash"] == normalized_image_id(
        COMMIT,
        "validator_weights",
    )
    assert validator["eif_hash"] == eif_hash(COMMIT, "validator_weights")


def test_local_vsock_runs_real_framing_and_rejects_unknown_rpc(
    monkeypatch,
    tmp_path,
) -> None:
    from gateway.tee.build_identity import build_identity
    from gateway.tee.topology import ROLE_SPECS, topology_hash
    from leadpoet_canonical.attested_v2 import (
        BOOT_ATTESTATION_PURPOSE,
        sha256_json,
    )

    role = "gateway_scoring"
    identity = build_identity(
        role=role,
        service_role=ROLE_SPECS[role]["service_role"],
        commit_sha=COMMIT,
        execution_manifest_hash="sha256:" + "3" * 64,
        dependency_lock_hash="sha256:" + "2" * 64,
        protected_manifest_hash="sha256:" + "4" * 64,
        topology_hash=topology_hash(),
    )
    release_role = {
        "build_identity_hash": identity["identity_hash"],
        "commit_sha": COMMIT,
        "dependency_lock_hash": "sha256:" + "2" * 64,
        "execution_manifest_hash": "sha256:" + "3" * 64,
        "pcr0": pcr0(COMMIT),
        "topology_hash": topology_hash(),
    }
    (tmp_path / "release-build-input.json").write_text(
        json.dumps(
            {
                "commit_sha": COMMIT,
                "gateway_roles": {role: release_role},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    identity_root = tmp_path / "gateway-enclave-build-identities"
    identity_root.mkdir()
    (identity_root / f"{role}.json").write_text(
        json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    socket_root = Path("/tmp") / (
        "lp-rh-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    )
    socket_root.mkdir(mode=0o700)
    monkeypatch.setenv(
        "REHEARSAL_GATEWAY_ENCLAVE_SOCKET_ROOT",
        str(socket_root),
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    configuration = {"configured_worker_count": 1}
    config_hash = sha256_json(
        {
            "schema_version": "leadpoet.enclave_runtime_config.v2",
            "physical_role": role,
            "service_role": ROLE_SPECS[role]["service_role"],
            "configuration": configuration,
        }
    )
    configured = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
        role,
        "v2_configure_runtime",
        {
            "schema_version": "leadpoet.enclave_runtime_config.v2",
            "configuration": configuration,
            "configuration_hash": config_hash,
        },
    )
    assert configured["status"] == "ready"

    source_root = Path(__file__).resolve().parents[2]
    harness_root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(harness_root), str(source_root))
            ),
            "REHEARSAL_CANDIDATE_SHA": COMMIT,
            "REHEARSAL_COMPONENT": "gateway",
            "REHEARSAL_GATEWAY_CANDIDATE_ROOT": str(
                source_root / "gateway"
            ),
            "REHEARSAL_GATEWAY_ENCLAVE_ROLE": role,
            "REHEARSAL_GATEWAY_ENCLAVE_SOCKET_ROOT": str(socket_root),
            "REHEARSAL_SCOPE": "targeted",
            "REHEARSAL_STATE_ROOT": str(tmp_path),
        }
    )
    service = subprocess.Popen(
        [sys.executable, str(harness_root / "gateway_enclave_service.py")],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    socket_path = socket_root / ("gateway-enclave-%s.sock" % role)
    try:
        for _attempt in range(100):
            if socket_path.is_socket():
                break
            if service.poll() is not None:
                stdout, stderr = service.communicate(timeout=5)
                raise AssertionError(
                    "persistent gateway enclave exited: %s %s"
                    % (stdout, stderr)
                )
            time.sleep(0.02)
        assert socket_path.is_socket()

        local_socket = rehearsal_sitecustomize._LocalVsock(
            40, socket.SOCK_STREAM
        )
        local_socket.settimeout(30.0)
        local_socket.connect((int(ROLE_SPECS[role]["cid"]), 5000))
        request = json.dumps(
            {"method": "v2_get_boot_identity", "params": {}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        local_socket.sendall(len(request).to_bytes(4, "big") + request)
        response_size = int.from_bytes(local_socket.recv(4), "big")
        response = json.loads(local_socket.recv(response_size))
        assert response["status"] == "success"
        boot = response["result"]
        verified, extracted = (
            rehearsal_sitecustomize._local_verify_nitro_attestation_full(
                attestation_b64=boot["attestation_document_b64"],
                expected_pcr0=boot["pcr0"],
                expected_pubkey=boot["signing_pubkey"],
                expected_purpose=BOOT_ATTESTATION_PURPOSE,
                role="gateway",
            )
        )
        assert verified is True
        assert extracted["pcr0"] == boot["pcr0"]

        unknown = json.dumps(
            {"method": "unknown_rpc", "params": {}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        local_socket.sendall(len(unknown).to_bytes(4, "big") + unknown)
        unknown_size = int.from_bytes(local_socket.recv(4), "big")
        rejected = json.loads(local_socket.recv(unknown_size))
        assert rejected["status"] == "error"
        assert "rejected unknown method" in rejected["error"]
    finally:
        service.terminate()
        try:
            service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait(timeout=5)
        socket_path.unlink(missing_ok=True)
        socket_path.with_suffix(".ready").unlink(missing_ok=True)
        socket_root.rmdir()


def test_local_gateway_kms_boundary_unwraps_job_credential_in_candidate(
    monkeypatch,
    tmp_path,
) -> None:
    from gateway.tee.kms_recipient_v2 import KMSRecipientV2
    from leadpoet_canonical.attested_v2 import canonical_json

    role = "gateway_coordinator"
    release_role = {
        "build_identity_hash": "sha256:" + "1" * 64,
        "commit_sha": COMMIT,
        "pcr0": pcr0(COMMIT),
    }
    (tmp_path / "release-build-input.json").write_text(
        json.dumps(
            {
                "commit_sha": COMMIT,
                "gateway_roles": {role: release_role},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    monkeypatch.setenv("REHEARSAL_COMPONENT", "gateway")

    def attestation_supplier(
        *,
        user_data: bytes,
        signing_pubkey: bytes,
    ) -> bytes:
        return canonical_json(
            {
                "schema_version": "leadpoet.local_nitro_document.v1",
                "pcr0": release_role["pcr0"],
                "public_key_b64": base64.b64encode(signing_pubkey).decode(
                    "ascii"
                ),
                "user_data_b64": base64.b64encode(user_data).decode("ascii"),
                "nonce_b64": "",
            }
        ).encode("ascii")

    slot = "openrouter_job"
    key_ref_hash = "sha256:" + "2" * 64
    credential = "sanitized-job-credential"
    credential_hash = "sha256:" + hashlib.sha256(
        credential.encode("utf-8")
    ).hexdigest()
    recipient_authority = KMSRecipientV2(
        boot_identity_supplier=lambda: {
            "boot_identity_hash": "sha256:" + "3" * 64,
        },
        expected_credential_ref_hashes={
            "supabase_service_role": "sha256:" + "4" * 64,
        },
        expected_job_slot_ref_hashes={slot: key_ref_hash},
        attestation_supplier=attestation_supplier,
    )
    recipient = recipient_authority.job_recipient_request(
        job_id="rehearsal-job",
        slot=slot,
        credential_value_hash_expected=credential_hash,
        key_ref_hash=key_ref_hash,
    )
    kms = rehearsal_sitecustomize._LocalKMS()
    key_id = "arn:aws:kms:us-east-1:111122223333:key/rehearsal"
    encrypted = kms.encrypt(
        KeyId=key_id,
        Plaintext=credential.encode("utf-8"),
        EncryptionContext={
            "schema_version": "leadpoet.job_provider_envelope.v2",
            "job_id": "rehearsal-job",
        },
    )
    decrypted = kms.decrypt(
        CiphertextBlob=encrypted["CiphertextBlob"],
        EncryptionContext={
            "schema_version": "leadpoet.job_provider_envelope.v2",
            "job_id": "rehearsal-job",
        },
        Recipient={
            "KeyEncryptionAlgorithm": "RSAES_OAEP_SHA_256",
            "AttestationDocument": base64.b64decode(
                recipient["attestation_document_b64"],
                validate=True,
            ),
        },
    )
    assert decrypted["KeyId"] == key_id
    lease = recipient_authority.unwrap_job_credential(
        request_id=recipient["request_id"],
        ciphertext_for_recipient_b64=base64.b64encode(
            decrypted["CiphertextForRecipient"]
        ).decode("ascii"),
    )
    assert lease == {
        "credential": credential,
        "credential_slot": slot,
        "credential_value_hash": credential_hash,
        "job_id": "rehearsal-job",
        "key_ref_hash": key_ref_hash,
    }


def test_local_vsock_listener_enforces_the_production_contract(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    invalid = rehearsal_sitecustomize._LocalVsock(40, socket.SOCK_STREAM)
    with pytest.raises(ValueError, match="listener address"):
        invalid.bind((0xFFFFFFFF, 4999))

    listener = rehearsal_sitecustomize._LocalVsock(40, socket.SOCK_STREAM)
    listener.bind((0xFFFFFFFF, 5001))
    with pytest.raises(ValueError, match="listener backlog"):
        listener.listen(63)
    listener.listen(64)
    listener.close()
    with pytest.raises(OSError, match="listener closed"):
        listener.accept()

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(events) == 1
    assert events[0] == {
        "at_ns": events[0]["at_ns"],
        "boundary": "nitro_enclaves",
        "fixture_authenticity": "production_shaped_sanitized",
        "implementation": "external_boundary",
        "kind": "local-chain-boundary",
        "operation": "enclave_listener",
        "port": 5001,
        "reject_unknown": True,
        "status": "bound",
    }


def test_local_gateway_attestation_boundary_matches_public_rpc_contract(
    tmp_path,
    monkeypatch,
) -> None:
    role = "gateway_coordinator"
    release_role = {
        "build_identity_hash": "sha256:" + "1" * 64,
        "commit_sha": COMMIT,
        "pcr0": pcr0(COMMIT),
    }
    (tmp_path / "release-build-input.json").write_text(
        json.dumps(
            {
                "commit_sha": COMMIT,
                "gateway_roles": {
                    role: release_role,
                    "gateway_scoring": release_role,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )

    result = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
        role,
        "get_attestation",
        {},
    )
    document = json.loads(bytes.fromhex(result["attestation_document"]))
    assert document == {
        "schema_version": "leadpoet.local_gateway_attestation.v1",
        "commit_sha": COMMIT,
        "physical_role": role,
        "build_identity_hash": release_role["build_identity_hash"],
        "pcr0": release_role["pcr0"],
    }
    assert result["pcr0"] == release_role["pcr0"]
    assert len(result["public_key"]) == 64
    assert len(result["code_hash"]) == 64
    assert len(result["pcr1"]) == 96
    assert len(result["pcr2"]) == 96

    with pytest.raises(ValueError, match="attestation RPC contract"):
        rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "get_attestation",
            {"unexpected": True},
        )
    with pytest.raises(ValueError, match="attestation RPC contract"):
        rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            "gateway_scoring",
            "get_attestation",
            {},
        )


def test_profiles_are_fixed_and_fit_the_developer_docker_budget() -> None:
    assert rehearsal.PROFILE_LIMITS == {
        "prepush": {
            "cpus": "4",
            "memory": "7g",
            "epochs": 1,
            "fault_matrix": False,
        },
        "release": {
            "cpus": "6",
            "memory": "7g",
            "epochs": 100,
            "fault_matrix": True,
        },
    }


def test_rehearsal_base_images_are_immutable_platform_children() -> None:
    expected = {
        "linux/amd64": (
            "public.ecr.aws/amazonlinux/amazonlinux@sha256:"
            "7dfb72e165c7b2f5fd2ee050c202160ee0cced24991f14736b831221f2004eee"
        ),
        "linux/arm64": (
            "public.ecr.aws/amazonlinux/amazonlinux@sha256:"
            "d23b77c815875a32165bc160248a6fcaf932dbbcdb7adc157680c39e4d254b38"
        ),
    }
    assert rehearsal.REHEARSAL_BASE_IMAGES == expected
    for docker_platform, image in expected.items():
        assert rehearsal._rehearsal_base_image(docker_platform) == image

    with pytest.raises(
        SystemExit,
        match="unsupported rehearsal Docker platform",
    ):
        rehearsal._rehearsal_base_image("linux/unknown")


def test_rehearsal_dockerfile_has_valid_immutable_default_image() -> None:
    dockerfile = (
        Path(__file__).resolve().parent / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert dockerfile.startswith(
        "ARG REHEARSAL_BASE_IMAGE="
        + rehearsal.REHEARSAL_BASE_IMAGES["linux/amd64"]
        + "\nFROM ${REHEARSAL_BASE_IMAGE}\n"
    )
    assert (
        "7942e2a958a238057cdf3304cba7e75f4056d15f75112b8d8e7c1d21a17f2d6c"
        not in dockerfile
    )


def test_rehearsal_build_binds_the_platform_specific_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def temporary_directory(*, prefix: str):
        assert prefix == "leadpoet-restart-image-"
        yield str(tmp_path)

    commands: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(
        rehearsal.tempfile,
        "TemporaryDirectory",
        temporary_directory,
    )
    monkeypatch.setattr(
        rehearsal,
        "_git_file",
        lambda _sha, path: (
            (Path(__file__).resolve().parent / "Dockerfile").read_bytes()
            if path == "tests/restart_rehearsal/Dockerfile"
            else b""
        ),
    )
    monkeypatch.setattr(
        rehearsal,
        "_run",
        lambda argv, *, cwd=None, **_kwargs: commands.append(
            (list(argv), cwd)
        ),
    )

    rehearsal._build_image(
        "leadpoet-test:platform",
        harness_sha="a" * 40,
        docker_platform="linux/amd64",
    )

    assert commands == [
        (
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--build-arg",
                "REHEARSAL_BASE_IMAGE="
                + rehearsal.REHEARSAL_BASE_IMAGES["linux/amd64"],
                "--tag",
                "leadpoet-test:platform",
                ".",
            ],
            tmp_path,
        )
    ]


def test_exact_harness_keeps_persistent_role_isolated_enclave_processes() -> None:
    harness_root = Path(__file__).resolve().parent
    run_inside = (harness_root / "run_inside.sh").read_text(encoding="utf-8")
    sitecustomize = (harness_root / "sitecustomize.py").read_text(
        encoding="utf-8"
    )
    service = (harness_root / "validator_enclave_service.py").read_text(
        encoding="utf-8"
    )
    gateway_service = (
        harness_root / "gateway_enclave_service.py"
    ).read_text(encoding="utf-8")
    tls_proxy_service = (
        harness_root / "tls_connect_proxy_service.py"
    ).read_text(encoding="utf-8")
    dockerfile = (harness_root / "Dockerfile").read_text(encoding="utf-8")
    contract = json.loads(
        (harness_root / "boundary_contract.json").read_text(encoding="utf-8")
    )

    assert "/harness/validator_enclave_service.py &" in run_inside
    assert "/harness/gateway_enclave_service.py &" in run_inside
    assert "/harness/tls_connect_proxy_service.py &" in run_inside
    assert "/harness/postgres_v2_contract_probe.py \\" in run_inside
    assert "/harness/gateway_enclave_service.py \\" in dockerfile
    assert "/harness/tls_connect_proxy_service.py \\" in dockerfile
    assert "postgresql15-server" in dockerfile
    assert "/harness/postgres_v2_contract_probe.py \\" in dockerfile
    assert (
        "tests/restart_rehearsal/gateway_enclave_service.py"
        in rehearsal.COMMITTED_HARNESS_PATHS
    )
    assert (
        "tests/restart_rehearsal/postgres_v2_contract_probe.py"
        in rehearsal.COMMITTED_HARNESS_PATHS
    )
    assert (
        "for role in gateway_coordinator gateway_scoring gateway_autoresearch"
        in run_inside
    )
    assert "for _attempt in $(seq 1 600)" in run_inside
    assert (
        'REHEARSAL_GATEWAY_CANDIDATE_ROOT="/source/gateway" \\'
        in run_inside
    )
    assert "gateway-enclave-build-identities" in run_inside
    assert "gateway-attested-runtime" in run_inside
    assert (
        "Materializing the exact measured gateway-enclave filesystem"
        in run_inside
    )
    assert "--python-version 3.9.24" in run_inside
    assert "_prepare_candidate_role_root(role)" in gateway_service
    assert (
        "_install_measured_runtime_boundary(gateway_root)"
        in gateway_service
    )
    assert "verify_runsc_artifact(" in gateway_service
    assert '"measured_runtime_surface"' in gateway_service
    assert (
        'os.environ["GATEWAY_ROOT"] = str(gateway_root)'
        in gateway_service
    )
    assert (
        'gateway_root / "tee/merkle.py"'
        in gateway_service
    )
    assert "REHEARSAL_VALIDATOR_ENCLAVE_SOCKET" in sitecustomize
    assert "REHEARSAL_GATEWAY_ENCLAVE_SOCKET_ROOT" in sitecustomize
    assert "tee_service.handle_request(request)" in service
    assert "_install_measured_drand_boundary()" in service
    assert "_install_local_chain_tls_boundary()" in service
    assert "_local_chain_http_post" in service
    assert "measured_drand_cabi" in service
    assert "measured_drand_commit" in service
    assert "trap finalize_rehearsal EXIT" in run_inside
    assert "preserve_rehearsal_evidence" in run_inside
    assert "tls-connect-proxy-ca.pem" in run_inside
    assert "authenticated_http_or_https_connect.v2" in (
        Path(__file__).resolve().parents[2]
        / "gateway/tee/prepare_gateway_envelopes_v2.py"
    ).read_text(encoding="utf-8")
    assert {
        "proxy_connect",
        "proxy_dns",
        "proxy_tls_connect",
    } <= set(contract["boundaries"]["http_service"]["allowed_operations"])
    assert "proxy-authorization:" in tls_proxy_service
    assert "ssl.PROTOCOL_TLS_SERVER" in tls_proxy_service
    assert '"openrouter.ai:443"' in tls_proxy_service
    assert '"api.exa.ai:443"' in tls_proxy_service
    assert '"api.scrapingdog.com:443"' in tls_proxy_service
    assert '"code.deepline.com:443"' in tls_proxy_service
    drand_install = run_inside.index(
        "/app/validator_tee/enclave/libbittensor_drand_v2.so"
    )
    validator_service_start = run_inside.index(
        "/harness/validator_enclave_service.py &"
    )
    assert drand_install < validator_service_start
    assert "_handle_gateway_enclave_rpc(" in gateway_service
    assert "while True:" in service
    assert "while True:" in gateway_service


def test_gateway_restart_records_proxy_preflight_as_a_pre_shutdown_stage() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    restart = (repository_root / "gw_restart.sh").read_text(encoding="utf-8")
    stage = restart.index(
        'GATEWAY_DEPLOY_STAGE="v2_credential_envelope_preparation"'
    )
    preparation = restart.index(
        "gateway.tee.prepare_gateway_envelopes_v2",
        stage,
    )
    shutdown = restart.index(
        "Stopping existing gateway and Research Lab worker processes",
        preparation,
    )
    assert stage < preparation < shutdown

    run_inside = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")
    scenario = run_inside.index(
        'if [ "$WEIGHT_READINESS_SCENARIO" = "plaintext_proxy_rejected" ]'
    )
    assert '"v2_credential_envelope_preparation"' in run_inside[scenario:]
    assert "TARGETED_RESTART_REGRESSION_SUCCESS" in run_inside[scenario:]




def test_measured_drand_boundary_preserves_candidate_c_abi_contract(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sitecustomize",
        rehearsal_sitecustomize,
    )
    from tests.restart_rehearsal import validator_enclave_service
    from validator_tee.enclave.drand_v2 import (
        CtypesDrandCommitBackendV2,
        _CRByteBuffer,
    )

    library_path = tmp_path / "libbittensor_drand_v2.so"
    library_path.write_bytes(b"exact-measured-drand-library")
    library_hash = hashlib.sha256(library_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        validator_enclave_service,
        "MEASURED_DRAND_PATH",
        library_path,
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    measured_library = validator_enclave_service._MeasuredDrandLibrary(
        library_path=str(library_path),
        expected_sha256=library_hash,
        buffer_type=_CRByteBuffer,
    )
    backend = CtypesDrandCommitBackendV2(
        library_path=library_path,
        expected_sha256=library_hash,
        library_loader=lambda _path: measured_library,
    )
    commitment, reveal_round = backend.generate_commit(
        uids=[0, 14],
        weights_u16=[65535, 3210],
        version_key=901,
        last_epoch_block=8_596_445,
        pending_epoch_at=0,
        subnet_epoch_index=23_859,
        tempo=360,
        blocks_since_last_step=263,
        current_block=8_596_708,
        subnet_reveal_period_epochs=1,
        block_time=12.0,
        hotkey_public_key=b"h" * 32,
    )

    assert len(commitment) == 64
    assert reveal_round == 23_860
    assert measured_library._buffers == {}
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [event["operation"] for event in events] == [
        "measured_drand_cabi",
        "measured_drand_commit",
    ]
    assert events[1]["destination_count"] == 2
    assert events[1]["reveal_round"] == 23_860


def test_evidence_order_uses_the_boundary_contract_nitro_operations() -> None:
    harness_root = Path(__file__).resolve().parent
    verifier = (harness_root / "verify_evidence.py").read_text(
        encoding="utf-8"
    )
    contract = json.loads(
        (harness_root / "boundary_contract.json").read_text(encoding="utf-8")
    )
    allowed = set(
        contract["boundaries"]["nitro_enclaves"]["allowed_operations"]
    )

    assert {
        "build_enclave",
        "measured_drand_cabi",
        "measured_drand_commit",
        "measured_runtime_surface",
        "run_enclave",
    } <= allowed
    assert verifier.count('"nitro:build_enclave"') == 2
    assert verifier.count('"nitro:run_enclave"') == 2


def test_workflow_runs_before_command_adapters_are_installed() -> None:
    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")
    workflow = script.index('if [ "$COMPONENT" = "workflow" ]; then')
    adapters = script.index("make_adapter()")
    assert workflow < adapters
    assert "/harness/production_workflow_runner.py" in script[workflow:adapters]


def test_local_chain_rejects_unknown_and_joins_finalized_reveal(
    tmp_path,
) -> None:
    fixture = {
        "schema_version": "fixture",
        "sanitization": {"contains_production_credentials": False},
    }
    with LocalBoundaryServices(root=tmp_path, fixture=fixture) as services:
        rejected = services.request(
            "POST",
            "/unknown",
            {"unexpected": True},
            expected_status=400,
        )
        assert rejected["status"] == "rejected"
        extrinsic = bytes.fromhex("01" * 96)
        extrinsic_hash = "0x" + hashlib.blake2b(
            extrinsic, digest_size=32
        ).hexdigest()
        services.request(
            "POST",
            "/chain/submit_extrinsic",
            {
                "epoch_id": 100,
                "extrinsic_hash": extrinsic_hash,
                "extrinsic_hex": extrinsic.hex(),
                "bundle_hash": "sha256:" + "2" * 64,
                "weights_hash": "3" * 64,
                "uids": [0, 2],
                "weights_u16": [1, 2],
            },
        )
        services.request(
            "POST",
            "/chain/finalize",
            {
                "epoch_id": 100,
                "extrinsic_hash": extrinsic_hash,
                "finalized_block": 1001,
            },
        )
        reveal = services.request(
            "POST",
            "/chain/reveal",
            {"epoch_id": 100, "uids": [0, 2], "weights_u16": [1, 2]},
        )
        assert reveal["status"] == "revealed"
        assert services.request(
            "GET", "/chain/epoch/100/last_update"
        )["last_update"] == 1001
    assert not services.thread.is_alive()


def test_joined_manifest_requires_every_authority_field(
    monkeypatch,
    tmp_path,
) -> None:
    for component in ("gateway", "validator"):
        (tmp_path / f"1-{component}-forward-{COMMIT}.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "scope": "exact",
                    "component": component,
                    "candidate_sha": COMMIT,
                    "from_sha": "2" * 40,
                    "event_count": 10,
                    "pcr0": "3" * 96,
                    "postgres_contract_sha256": "e" * 64,
                }
            ),
            encoding="utf-8",
        )
    epoch = {
        "pcr0": "3" * 96,
        "bundle_hash": "sha256:" + "4" * 64,
        "root_receipt_hash": "sha256:" + "5" * 64,
        "publication_receipt_hash": "sha256:" + "6" * 64,
        "finalization_receipt_hash": "sha256:" + "7" * 64,
        "receipt_ancestry_verified": True,
        "canonical_vector_hash": "sha256:" + "8" * 64,
        "canonical_vector_equal": True,
        "extrinsic_authorization_hash": "sha256:" + "9" * 64,
        "signed_extrinsic_hash": "0x" + "a" * 64,
        "sdk_bridge_verified": True,
        "sdk_commit_request_hash": "sha256:" + "c" * 64,
        "sdk_extrinsic_request_hash": "sha256:" + "d" * 64,
        "finalized_block": 1000,
        "last_update": 1000,
        "reveal_vector_hash": "sha256:" + "b" * 64,
        "auditor_verified": True,
        "auditor_runtime_verified": True,
    }
    workflow_path = tmp_path / "workflow.json"
    workflow = {
        "status": "passed",
        "profile": "prepush",
        "release_sha": COMMIT,
        "epoch_count": 1,
        "epochs": [epoch],
        "fault_matrix": [],
        "concurrent_write_count": 0,
        "cleanup": {
            "pending_faults": 0,
            "boundary_thread_alive_after_close": False,
            "local_chain_epochs": 1,
        },
    }
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    output = tmp_path / "joined.json"
    assert (
        join_evidence.main(
            [
                "--evidence-root",
                str(tmp_path),
                "--from-sha",
                "2" * 40,
                "--candidate-sha",
                COMMIT,
                "--profile",
                "prepush",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    joined = json.loads(output.read_text())
    assert joined["status"] == "passed"
    assert joined["postgres_contract_sha256"] == "e" * 64

    validator_path = (
        tmp_path / f"1-validator-forward-{COMMIT}.json"
    )
    validator = json.loads(validator_path.read_text(encoding="utf-8"))
    validator["postgres_contract_sha256"] = "f" * 64
    validator_path.write_text(json.dumps(validator), encoding="utf-8")
    with pytest.raises(SystemExit, match="migration-backed contracts differ"):
        join_evidence.main(
            [
                "--evidence-root",
                str(tmp_path),
                "--from-sha",
                "2" * 40,
                "--candidate-sha",
                COMMIT,
                "--profile",
                "prepush",
                "--output",
                str(output),
            ]
        )
    validator["postgres_contract_sha256"] = "e" * 64
    validator_path.write_text(json.dumps(validator), encoding="utf-8")

    workflow["epochs"][0]["canonical_vector_equal"] = False
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    with pytest.raises(SystemExit, match="complete authority"):
        join_evidence.main(
            [
                "--evidence-root",
                str(tmp_path),
                "--from-sha",
                "2" * 40,
                "--candidate-sha",
                COMMIT,
                "--profile",
                "prepush",
                "--output",
                str(output),
            ]
        )


def test_weight_storage_preflight_capability_tracks_selected_release(
    tmp_path,
) -> None:
    source = tmp_path / "gateway/tee/verify_weight_submission_ready_v2.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--repair', action='store_true')\n",
        encoding="utf-8",
    )
    assert selected_weight_storage_preflight_capability((tmp_path,)) is False

    source.write_text(
        source.read_text(encoding="utf-8")
        + "parser.add_argument('--storage-read-preflight', action='store_true')\n",
        encoding="utf-8",
    )
    assert selected_weight_storage_preflight_capability((tmp_path,)) is True


def test_gateway_rehearsal_requires_canonical_private_model_environment() -> None:
    row = {
        "kind": "process",
        "process": "gateway.main",
        "status": "started",
        "environment_contract": dict(EXPECTED_GATEWAY_PRIVATE_MODEL_ENV),
    }
    verify_gateway_private_model_environment([row])

    row["environment_contract"][
        "RESEARCH_LAB_PRIVATE_REPO_BRANCH"
    ] = "main"
    with pytest.raises(SystemExit, match="private-model source environment"):
        verify_gateway_private_model_environment([row])

    with pytest.raises(SystemExit, match="exactly one gateway.main"):
        verify_gateway_private_model_environment([])


def test_rehearsal_driver_must_match_frozen_harness_commit(
    monkeypatch,
    tmp_path,
) -> None:
    driver = tmp_path / "run_local_restart_rehearsal.py"
    driver.write_bytes(b"candidate driver\n")
    monkeypatch.setattr(rehearsal, "__file__", str(driver))
    monkeypatch.setattr(
        rehearsal,
        "_git_file",
        lambda _sha, _path: b"candidate driver\n",
    )

    rehearsal._verify_driver_identity(COMMIT)

    driver.write_bytes(b"dirty driver\n")
    with pytest.raises(SystemExit, match="differs from the frozen harness"):
        rehearsal._verify_driver_identity(COMMIT)


def test_rehearsal_preserves_exact_failure_evidence(tmp_path) -> None:
    evidence = tmp_path / "ephemeral"
    evidence.mkdir()
    (evidence / "validator-main.log").write_text(
        "coordinator failed before authority publication\n",
        encoding="utf-8",
    )

    durable = rehearsal._preserve_failure_evidence(
        evidence_root=evidence,
        candidate_sha=COMMIT,
        stage="validator-forward-1",
        command=["docker", "run", "candidate"],
    )

    assert (
        durable / "evidence" / "validator-main.log"
    ).read_text(encoding="utf-8").startswith("coordinator failed")
    assert json.loads(
        (durable / "failure.json").read_text(encoding="utf-8")
    ) == {
        "candidate_sha": COMMIT,
        "command": ["docker", "run", "candidate"],
        "stage": "validator-forward-1",
    }


def test_rehearsal_preserves_all_batched_release_failures(tmp_path) -> None:
    evidence = tmp_path / "ephemeral"
    evidence.mkdir()
    (evidence / "validator-main.log").write_text(
        "rollback failed before coordinator readiness\n",
        encoding="utf-8",
    )
    failures = [
        {
            "command": ["docker", "run", "rollback"],
            "returncode": 1,
            "stage": "validator-rollback-2",
        },
        {
            "command": ["docker", "run", "workflow"],
            "returncode": 2,
            "stage": "workflow-release",
        },
    ]

    durable = rehearsal._preserve_batched_failure_evidence(
        evidence_root=evidence,
        candidate_sha=COMMIT,
        failures=failures,
    )

    assert (
        durable / "evidence" / "validator-main.log"
    ).read_text(encoding="utf-8").startswith("rollback failed")
    assert json.loads(
        (durable / "failure-summary.json").read_text(encoding="utf-8")
    ) == {
        "candidate_sha": COMMIT,
        "failure_count": 2,
        "failures": failures,
        "status": "failed",
    }


def test_rehearsal_serializes_subprocess_stage_failure() -> None:
    exc = subprocess.CalledProcessError(
        returncode=17,
        cmd=("docker", "run", "candidate"),
    )

    assert rehearsal._stage_failure(stage="gateway-forward-1", exc=exc) == {
        "command": ["docker", "run", "candidate"],
        "returncode": 17,
        "stage": "gateway-forward-1",
    }


def test_rehearsal_source_snapshot_is_independent_and_complete(
    monkeypatch,
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Restart Test"],
        check=True,
    )
    source_file = repo / "source.txt"
    source_file.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "frozen source"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(rehearsal, "REPO_ROOT", repo)

    with rehearsal._isolated_source_snapshot(
        harness_sha=commit,
        required_shas=(commit,),
    ) as snapshot:
        assert snapshot != repo
        assert (snapshot / "source.txt").read_text(encoding="utf-8") == "frozen\n"
        assert subprocess.run(
            ["git", "-C", str(snapshot), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
        ).returncode == 0
        assert not (snapshot / ".git" / "objects" / "info" / "alternates").exists()

    assert not snapshot.exists()


def test_rehearsal_resolves_forward_and_rollback_transitions(monkeypatch) -> None:
    relationships = {
        ("from", "target"): True,
        ("target", "from"): False,
        ("newer", "older"): False,
        ("older", "newer"): True,
    }
    monkeypatch.setattr(
        rehearsal,
        "_is_ancestor",
        lambda ancestor, descendant: relationships.get(
            (ancestor, descendant),
            False,
        ),
    )

    assert rehearsal._resolve_transition("from", "target", "auto") == "forward"
    assert rehearsal._resolve_transition("newer", "older", "auto") == "rollback"
    assert (
        rehearsal._resolve_transition("newer", "older", "rollback")
        == "rollback"
    )
    with pytest.raises(SystemExit, match="does not descend"):
        rehearsal._resolve_transition("newer", "older", "forward")


def test_rehearsal_rejects_unrelated_transition(monkeypatch) -> None:
    monkeypatch.setattr(rehearsal, "_is_ancestor", lambda *_args: False)

    with pytest.raises(SystemExit, match="unrelated"):
        rehearsal._resolve_transition("from", "target", "auto")


def test_rollback_rehearsal_keeps_newer_commit_on_origin_main() -> None:
    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")
    rollback_refs = script.index(
        'if [ "$TRANSITION" = "rollback" ]; then'
    )
    component_setup = script.index(
        'if [ "$COMPONENT" = "gateway" ]; then'
    )
    section = script[rollback_refs:component_setup]
    rollback_section, forward_section = section.split("\nelse\n", 1)

    assert '"$FROM_SHA:refs/heads/main"' in rollback_section
    assert '"$CANDIDATE_SHA:refs/heads/rehearsal-target"' in rollback_section
    assert '"$CANDIDATE_SHA:refs/heads/main"' not in rollback_section
    assert '"$CANDIDATE_SHA:refs/heads/main"' in forward_section


def test_rehearsal_inherits_the_installed_cutover_manifest() -> None:
    from Leadpoet.utils.subnet_epoch import SubnetEpochCutover

    repository_root = Path(__file__).resolve().parents[2]
    manifest_path = repository_root / "config/stateful-epoch-cutover-sn71.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert SubnetEpochCutover.from_mapping(document).to_dict() == document

    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")
    assert (
        '"$FROM_SHA:config/stateful-epoch-cutover-sn71.json"'
        in script
    )
    assert '"status": "stateful_active"' not in script


def test_forward_rehearsal_uses_the_normal_unpinned_operator_paths() -> None:
    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")

    assert 'GATEWAY_DEPLOY_COMMIT="$CANDIDATE_SHA"' not in script
    assert 'VALIDATOR_DEPLOY_COMMIT="$CANDIDATE_SHA"' not in script
    assert script.count('--commit "$CANDIDATE_SHA"') == 2


def test_gateway_readiness_requires_exact_production_launcher_invocations() -> None:
    module = "gateway.tee.verify_weight_submission_ready_v2"

    def row(argv: list[str], source_kind: str) -> dict:
        return {
            "kind": "python-module",
            "module": module,
            "status": "started",
            "implementation": "production_module",
            "candidate_sha": COMMIT,
            "source_commit": COMMIT,
            "source_git_path": (
                "gateway/tee/verify_weight_submission_ready_v2.py"
            ),
            "source_kind": source_kind,
            "argv": argv,
        }

    rows = [
        row(["-m", module, "--storage-read-preflight"], "candidate_archive"),
        row(["-m", module, "--repair"], "candidate_checkout"),
        row(
            [
                "-m",
                module,
                "--gateway-url",
                "http://localhost:8000",
                "--http-timeout-seconds",
                "360",
            ],
            "candidate_checkout",
        ),
    ]
    verify_gateway_weight_readiness_invocations(
        rows,
        candidate_sha=COMMIT,
    )

    rows[2]["argv"][-1] = "30"
    with pytest.raises(SystemExit, match="launcher contract"):
        verify_gateway_weight_readiness_invocations(
            rows,
            candidate_sha=COMMIT,
        )


def test_gateway_readiness_accepts_missing_optional_preflight_only_for_rollback() -> None:
    module = "gateway.tee.verify_weight_submission_ready_v2"

    def row(argv: list[str]) -> dict:
        return {
            "kind": "python-module",
            "module": module,
            "status": "started",
            "implementation": "production_module",
            "candidate_sha": COMMIT,
            "source_commit": COMMIT,
            "source_git_path": (
                "gateway/tee/verify_weight_submission_ready_v2.py"
            ),
            "source_kind": "candidate_checkout",
            "argv": argv,
        }

    rows = [
        row(["-m", module, "--repair"]),
        row(
            [
                "-m",
                module,
                "--gateway-url",
                "http://localhost:8000",
                "--http-timeout-seconds",
                "360",
            ]
        ),
    ]
    verify_gateway_weight_readiness_invocations(
        rows,
        candidate_sha=COMMIT,
        transition="rollback",
        storage_preflight_supported=False,
    )

    with pytest.raises(SystemExit, match="current release"):
        verify_gateway_weight_readiness_invocations(
            rows,
            candidate_sha=COMMIT,
            transition="forward",
            storage_preflight_supported=False,
        )


def test_module_provenance_follows_prepared_archive_python_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_root = Path(__file__).resolve().parent
    archive_root = tmp_path / "gateway-v2-preflight.Ab12"
    module_path = (
        archive_root
        / "gateway/tee/verify_weight_submission_ready_v2.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.write_text("VALUE = 'candidate archive'\n", encoding="utf-8")
    installed_root = tmp_path / "installed"
    installed_path = (
        installed_root
        / "gateway/tee/verify_weight_submission_ready_v2.py"
    )
    installed_path.parent.mkdir(parents=True)
    installed_path.write_text("VALUE = 'installed N-1'\n", encoding="utf-8")

    monkeypatch.chdir(archive_root)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(archive_root), str(installed_root))),
    )
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    monkeypatch.setenv("REHEARSAL_FROM_SHA", "2" * 40)
    monkeypatch.setenv("REHEARSAL_STATE_ROOT", str(tmp_path / "state"))
    specification = importlib.util.spec_from_file_location(
        "rehearsal_contract_adapter_test",
        harness_root / "contract_adapter.py",
    )
    assert specification is not None
    assert specification.loader is not None
    rehearsal_contract_adapter = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(rehearsal_contract_adapter)
    monkeypatch.setattr(
        rehearsal_contract_adapter,
        "_candidate_root",
        lambda: installed_root,
    )

    assert rehearsal_contract_adapter._module_source(
        "gateway.tee.verify_weight_submission_ready_v2"
    ) == module_path


def test_gateway_readiness_rejects_substituted_or_missing_invocation() -> None:
    module = "gateway.tee.verify_weight_submission_ready_v2"
    rows = [
        {
            "kind": "python-module",
            "module": module,
            "status": "started",
            "implementation": "internal_substitution",
            "candidate_sha": COMMIT,
            "source_commit": COMMIT,
            "source_git_path": (
                "gateway/tee/verify_weight_submission_ready_v2.py"
            ),
            "source_kind": "candidate_archive",
            "argv": ["-m", module, "--storage-read-preflight"],
        }
    ]

    with pytest.raises(SystemExit, match="exact production"):
        verify_gateway_weight_readiness_invocations(
            rows,
            candidate_sha=COMMIT,
        )


def test_exact_rehearsal_rejects_repository_module_substitution() -> None:
    rows = [
        {
            "kind": "python-module",
            "module": "gateway.tee.restart_preflight_v2",
            "implementation": "internal_substitution",
        }
    ]

    with pytest.raises(SystemExit, match="repository-code substitutions"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_targeted_regression_is_distinct_and_rejects_unknown_substitution() -> None:
    known = [
        {
            "kind": "python-module",
            "module": "gateway.tee.restart_preflight_v2",
            "implementation": "internal_substitution",
        }
    ]
    verify_rehearsal_integrity(
        known,
        candidate_sha=COMMIT,
        scope="weight_readiness_regression",
    )

    unknown = [
        {
            "kind": "python-module",
            "module": "gateway.future_unexercised_stage",
            "implementation": "internal_substitution",
        }
    ]
    with pytest.raises(SystemExit, match="unclassified"):
        verify_rehearsal_integrity(
            unknown,
            candidate_sha=COMMIT,
            scope="weight_readiness_regression",
        )


def test_targeted_regression_classifies_dependency_bootstrap() -> None:
    rows = [
        {
            "kind": "python-script",
            "script": "get-pip.py",
            "substitution": "python_dependencies.bootstrap",
            "implementation": "internal_substitution",
        }
    ]

    verify_rehearsal_integrity(
        rows,
        candidate_sha=COMMIT,
        scope="weight_readiness_regression",
    )

    with pytest.raises(SystemExit, match="repository-code substitutions"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_exact_rehearsal_rejects_synthetic_external_fixture() -> None:
    rows = [
        {
            "kind": "aws",
            "operation": "secretsmanager",
            "fixture_authenticity": "synthetic",
        }
    ]

    with pytest.raises(SystemExit, match="synthetic external fixtures"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_production_stage_requires_exact_candidate_source_identity(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Restart Test"],
        check=True,
    )
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "module.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "test source"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    row = {
        "kind": "python-module",
        "module": "gateway.real_stage",
        "implementation": "production_module",
        "candidate_sha": commit,
        "source_commit": commit,
        "source_path": str(source),
        "source_git_path": "module.py",
        "source_kind": "candidate_checkout",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    verify_rehearsal_integrity(
        [row],
        candidate_sha=commit,
        scope="exact",
        candidate_roots=(tmp_path,),
    )

    row["source_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="Git identity"):
        verify_rehearsal_integrity(
            [row],
            candidate_sha=commit,
            scope="exact",
            candidate_roots=(tmp_path,),
        )

    row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed after execution"):
        verify_rehearsal_integrity(
            [row],
            candidate_sha=commit,
            scope="exact",
            candidate_roots=(tmp_path,),
        )


def test_rollback_accepts_installed_launcher_source_bound_to_from_sha(
    tmp_path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Restart Test"],
        check=True,
    )
    source = tmp_path / "compatibility.py"
    source.write_text("VALUE = 'installed'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "compatibility.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "rollback target"],
        check=True,
    )
    candidate = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("VALUE = 'installed launcher'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "compatibility.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "installed launcher"],
        check=True,
    )
    installed = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    row = {
        "kind": "python-script",
        "script": "compatibility.py",
        "implementation": "production_script",
        "candidate_sha": candidate,
        "source_commit": installed,
        "source_path": str(source),
        "source_git_path": "compatibility.py",
        "source_kind": "installed_checkout",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }

    verify_rehearsal_integrity(
        [row],
        from_sha=installed,
        candidate_sha=candidate,
        scope="exact",
        candidate_roots=(tmp_path,),
    )

    with pytest.raises(SystemExit, match="source identity is invalid"):
        verify_rehearsal_integrity(
            [row],
            from_sha="2" * 40,
            candidate_sha=candidate,
            scope="exact",
            candidate_roots=(tmp_path,),
        )


def test_installed_launcher_identity_allows_exact_candidate_checkout_handoff(
    tmp_path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Restart Test"],
        check=True,
    )
    source = tmp_path / "deploy.py"
    source.write_text("VERSION = 'installed'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "deploy.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "installed"],
        check=True,
    )
    installed = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    installed_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    source.write_text("VERSION = 'candidate'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "deploy.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "candidate"],
        check=True,
    )
    candidate = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    row = {
        "kind": "python-script",
        "script": "deploy.py",
        "implementation": "production_script",
        "candidate_sha": candidate,
        "source_commit": installed,
        "source_path": str(source),
        "source_git_path": "deploy.py",
        "source_kind": "installed_checkout",
        "source_sha256": installed_hash,
    }

    verify_rehearsal_integrity(
        [row],
        from_sha=installed,
        candidate_sha=candidate,
        scope="exact",
        candidate_roots=(tmp_path,),
    )

    source.write_text("VERSION = 'mutated'\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed after execution"):
        verify_rehearsal_integrity(
            [row],
            from_sha=installed,
            candidate_sha=candidate,
            scope="exact",
            candidate_roots=(tmp_path,),
        )


def test_installed_launcher_identity_allows_candidate_deletion(
    tmp_path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Restart Test"],
        check=True,
    )
    source = tmp_path / "legacy_deploy.py"
    source.write_text("VERSION = 'installed'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "legacy_deploy.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "installed"],
        check=True,
    )
    installed = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    installed_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    source.unlink()
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "legacy_deploy.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "candidate"],
        check=True,
    )
    candidate = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    verify_rehearsal_integrity(
        [
            {
                "kind": "python-script",
                "script": "legacy_deploy.py",
                "implementation": "production_script",
                "candidate_sha": candidate,
                "source_commit": installed,
                "source_path": str(source),
                "source_git_path": "legacy_deploy.py",
                "source_kind": "installed_checkout",
                "source_sha256": installed_hash,
            }
        ],
        from_sha=installed,
        candidate_sha=candidate,
        scope="exact",
        candidate_roots=(tmp_path,),
    )
