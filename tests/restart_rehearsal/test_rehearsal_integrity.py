from __future__ import annotations

import asyncio
import base64
import copy
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Optional
import urllib.request
import uuid

import pytest

from leadpoet_canonical.attested_v2 import sha256_json
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
from tests.restart_rehearsal import production_workflow_runner
from tests.restart_rehearsal import sitecustomize as rehearsal_sitecustomize
from tests.restart_rehearsal.fixture_contract import (
    load_rehearsal_current_settlement_epoch_id,
    load_rehearsal_metagraph_account_ids,
    load_rehearsal_metagraph_hotkeys,
)
from tests.restart_rehearsal.gateway_boundary_service import (
    EXPECTED_ATOMIC_CREDIT_RESUME_EVIDENCE as GATEWAY_ATOMIC_CREDIT_RESUME_EVIDENCE,
    LocalPostgRESTServer,
    LocalPostgRESTState,
    RUNTIME_TABLES,
    _apply_table_query,
    _attested_store_tables,
    _direct_provider_store_tables,
    _measured_query_tables,
    _migration_seed_rows,
    _migration_provider_outcome_contract,
    _migration_schema_contract,
    _schema_contract,
    _source_add_claim_control_contract,
    _source_add_claim_control_contract_v2,
)
from tests.restart_rehearsal.postgres_v2_contract_probe import (
    ALLOCATION_MIGRATION_PREREQUISITES_SQL,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_MIGRATION,
    ALLOCATION_SETTLEMENT_FRONTIER_HISTORICAL_SOURCE_MIGRATION,
    ALLOCATION_SETTLEMENT_FRONTIER_MIGRATION,
    ALLOCATION_SETTLEMENT_FRONTIER_SOURCE_CONTRACT_MIGRATION,
    ANCESTRY_CHECKPOINT_MIGRATION,
    ANCESTRY_CHECKPOINT_BOOTSTRAP_PURPOSE_MIGRATION,
    CHAMPION_LIFETIME_CREDIT_MIGRATION,
    COMPACT_ANCESTRY_CHECKPOINT_MIGRATION,
    DisposablePostgres,
    EVENT_PROJECTIONS_MIGRATION,
    EXPECTED_ATOMIC_CREDIT_RESUME_EVIDENCE,
    EXPECTED_APPLIED_MIGRATIONS,
    EXPECTED_FINALIZED_VIEW_COLUMNS,
    EXPECTED_POSTGRES_CONTRACT_CHECKS,
    MIGRATIONS_BEFORE_TRANSPORT_FIX,
    HOTKEY_ACTIVE_LOOP_CAP_MIGRATION,
    MAINTENANCE_PAUSE_MIGRATION,
    PAUSED_CAPACITY_AGING_MIGRATION,
    PROVIDER_OUTCOME_APPEND_MIGRATION,
    PROVIDER_OUTCOME_BACKPRESSURE_MIGRATION,
    PROVIDER_OUTCOME_CONTENTION_STATUS_MIGRATION,
    PROVIDER_OUTCOME_HEAD_CONTENTION_MIGRATION,
    QUEUE_CAPACITY_GUARD_MIGRATION,
    RESUME_REQUEUE_HOTKEY_GUARD_MIGRATION,
    SOURCE_CATALOG_RESULT_REPLAY_MIGRATION,
    TRANSPORT_FIX_MIGRATION,
    TRANSPORT_TERMINAL_MIGRATION,
    _json_insert_sql,
    _validate_required_migration_declarations,
)
from tests.restart_rehearsal.local_services import LocalBoundaryServices
from tests.restart_rehearsal.sanitized_weight_fixture import (
    SanitizedWeightFixture,
)
from tests.restart_rehearsal.verify_evidence import (
    events,
    selected_weight_storage_preflight_capability,
    selected_weight_storage_preflight_pins_epoch,
    verify_migration_backed_database_contract,
    verify_gateway_weight_readiness_invocations,
    verify_chain_settlement_durable_readback,
    verify_rehearsal_integrity,
    verify_restart_epoch_transient_recovery,
)
from gateway.tee.rehearsal_behavior_contract_v2 import (
    RESTART_INVARIANTS,
    build_rehearsal_behavior_contract_v2,
)
@contextmanager
def _production_named_temp_directory(prefix: str):
    path = Path("/tmp") / f"{prefix}{uuid.uuid4().hex}"
    path.mkdir(mode=0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path)


COMMIT = "1" * 40
VALIDATOR_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


@pytest.fixture
def python39_import_event_loop():
    """Isolate modules that still acquire their asyncio lock at import."""

    import asyncio

    try:
        previous_loop = asyncio.get_event_loop()
    except RuntimeError:
        previous_loop = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        loop.close()
        asyncio.set_event_loop(
            previous_loop
            if previous_loop is not None and not previous_loop.is_closed()
            else None
        )


def _provider_persistence_batch_fixture() -> dict[str, Any]:
    return {
        "batch_size": 5,
        "durable_count": 5,
        "batch_replay_exact": True,
        "batch_conflict_head_exact": True,
        "cache_put_exact": True,
        "cache_replay_exact": True,
        "schema": {
            "schema_version": (
                "leadpoet.provider_persistence_batch_contract.v1"
            ),
            "cache_put": "atomic_exact_row",
            "outcome_append": "atomic_contiguous_batch",
            "outcome_batch_max": 32,
            "conflict_head_checkpoint_row": "encrypted_or_null",
        },
    }


def _compact_weight_settlement_contract_fixture() -> dict[str, Any]:
    return {
        "schema_version": (
            "leadpoet.research_lab_compact_weight_settlement_contract.v1"
        ),
        "max_authority_bytes": 8_388_608,
        "size_constraint_valid": True,
        "append_only_trigger_enabled": True,
        "identity_unique_constraint_enabled": True,
        "row_level_security_enabled": True,
        "finalized_stage_supported": True,
    }


def _source_add_provider_origin_contract_fixture() -> dict[str, Any]:
    return {
        "schema_version": "leadpoet.source_add_provider_origin_contract.v1",
        "identity_version": "v1",
        "identity_scope": "normalized_exact_host",
        "admission_rpc": "research_lab_source_add_admit_v2",
        "recheck_rpc": "research_lab_source_add_requeue_provenance_v2",
        "owner_count": 0,
        "reserved_count": 0,
        "coverage_complete": True,
        "collision_free": True,
        "submission_trigger_enabled": True,
        "catalog_trigger_enabled": True,
        "provision_trigger_enabled": True,
        "terminal_release_trigger_enabled": True,
        "append_only_trigger_enabled": True,
        "row_level_security_enabled": True,
        "service_role_policy_enabled": True,
    }


def _source_add_duplicate_privacy_contract_fixture() -> dict[str, Any]:
    return {
        "schema_version": "leadpoet.source_add_duplicate_privacy_contract.v1",
        "admission_rpc": "research_lab_source_add_admit_v3",
        "admission_signature": (
            "jsonb,text,text,text,text,text,integer,integer,integer,integer"
        ),
        "compatibility_rpc": "research_lab_source_add_admit_v2",
        "compatibility_signature": (
            "jsonb,text,text,text,text,text,integer,integer,integer"
        ),
        "compatibility_cooldown_seconds": 20,
        "cooldown_parameter_min_seconds": 1,
        "cooldown_parameter_max_seconds": 3600,
        "cooldown_clock": "clock_timestamp_after_advisory_locks",
        "cooldown_source": "durable_miner_provenance_work",
        "duplicate_precedes_cooldown": True,
        "lock_order": [
            "provider_origin_or_identity",
            "hotkey",
            "submission_or_work",
        ],
        "function_authority_sha256": (
            "sha256:26bf34c94725b855f81c2e48b6afbd72"
            "d68db36a4aeffb5642494a5da32233e0"
        ),
        "functions": {
            "admit_v1": True,
            "admit_v2_compatibility": True,
            "admit_v3": True,
            "provider_origin_hash_v1": True,
            "provider_origin_host_v1": True,
        },
        "permissions": {
            "service_role_exists": True,
            "v3_service_role_callable": True,
            "v2_service_role_callable": True,
            "contract_service_role_callable": True,
            "anon_callable": False,
            "authenticated_callable": False,
        },
    }


def _source_add_post_accept_leg1_contract_fixture() -> dict[str, Any]:
    return {
        "schema_version": "leadpoet.source_add_post_accept_leg1_contract.v4",
        "required_migration": (
            "scripts/176-research-lab-source-add-provenance-origin-repair.sql"
        ),
        "daily_cap": 50,
        "leg1_alpha_percent": 0.2,
        "leg1_reward_epochs": 20,
        "approval_boundary": "provenance_precheck_passed",
        "backfill_policy": (
            "earliest_exact_attested_provenance_per_provider_origin"
        ),
        "provider_origin_scope": "normalized_exact_host",
        "provider_origin_winner_order": [
            "provenance_created_at",
            "submission_id",
        ],
        "cancelled_intents_are_authority": False,
        "public_trigger_fields": [
            "precheck_status",
            "provenance_artifact_hash",
            "provenance_precheck_passed",
            "provenance_receipt_hash",
            "provenance_result_hash",
            "submission_id",
        ],
        "authority_view": (
            "research_lab_source_add_provenance_leg1_authority_v1"
        ),
        "function_authority_sha256": (
            rehearsal_sitecustomize._candidate_post_accept_leg1_function_authority()
        ),
        "trigger_authority_sha256": (
            rehearsal_sitecustomize._candidate_provenance_leg1_trigger_authority()
        ),
        "view_authority_sha256": (
            rehearsal_sitecustomize._candidate_provenance_leg1_view_authority()
        ),
        "repair_function_authority_sha256": (
            rehearsal_sitecustomize
            ._candidate_provenance_origin_repair_function_authority()
        ),
        "functions": {
            "configure_probe_v3": True,
            "enqueue_leg1_after_provenance_v1": True,
            "enqueue_provision_smoke_v2": True,
            "finalize_leg1_v4": True,
            "finalize_provision_smoke_v3": True,
            "finalize_provision_v3": True,
            "reject_current_builtin_v3": True,
            "reconcile_provenance_leg1_v1": True,
            "reserve_leg1_slot_v4": True,
        },
        "triggers": {
            "automatic_enqueue": True,
            "eligible_v2": True,
            "eligible_v3": True,
            "leg1_initial_event_v3": True,
            "leg1_obligation_v3": True,
            "leg1_slot_v3": True,
            "leg1_work_v3": True,
        },
        "columns": {
            "intent_approval_kind": True,
            "intent_provenance_artifact_hash": True,
            "intent_provenance_receipt_hash": True,
            "slot_approval_kind": True,
        },
        "permissions": {
            "service_role_exists": True,
            "candidate_callable": True,
            "internal_not_callable": True,
            "rollback_v2_callable": True,
        },
    }


def _atomic_credit_resume_fixture() -> dict[str, Any]:
    return json.loads(json.dumps(EXPECTED_ATOMIC_CREDIT_RESUME_EVIDENCE))


def _source_add_miner_status_contract_fixture() -> dict[str, Any]:
    return rehearsal_sitecustomize._source_add_miner_status_contract()


def test_local_schema_adapter_returns_full_source_add_origin_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    request = urllib.request.Request(
        (
            "https://example.invalid/rest/v1/rpc/"
            "research_lab_source_add_provider_origin_contract_v1"
        ),
        data=b"{}",
        headers={
            "apikey": "rehearsal-secret",
            "Authorization": "Bearer rehearsal-secret",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with rehearsal_sitecustomize._local_urlopen(
        request,
        timeout=10.0,
    ) as response:
        contract = json.loads(response.read().decode("utf-8"))

    assert contract == _source_add_provider_origin_contract_fixture()


def test_local_schema_adapter_returns_duplicate_privacy_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    request = urllib.request.Request(
        (
            "https://example.invalid/rest/v1/rpc/"
            "research_lab_source_add_duplicate_privacy_contract_v1"
        ),
        data=b"{}",
        headers={
            "apikey": "rehearsal-secret",
            "Authorization": "Bearer rehearsal-secret",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with rehearsal_sitecustomize._local_urlopen(
        request,
        timeout=10.0,
    ) as response:
        contract = json.loads(response.read().decode("utf-8"))

    assert contract == _source_add_duplicate_privacy_contract_fixture()


def test_local_schema_adapter_returns_full_source_add_leg1_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    request = urllib.request.Request(
        (
            "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/rpc/"
            "research_lab_source_add_post_accept_leg1_contract_v4"
        ),
        data=b"{}",
        headers={
            "apikey": "rehearsal-secret",
            "Authorization": "Bearer rehearsal-secret",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with rehearsal_sitecustomize._local_urlopen(
        request,
        timeout=10.0,
    ) as response:
        contract = json.loads(response.read().decode("utf-8"))

    assert contract == _source_add_post_accept_leg1_contract_fixture()


def test_local_schema_adapter_returns_source_add_claim_control_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    request = urllib.request.Request(
        (
            "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/rpc/"
            "research_lab_source_add_claim_control_contract_v1"
        ),
        data=b"{}",
        headers={
            "apikey": "rehearsal-secret",
            "Authorization": "Bearer rehearsal-secret",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with rehearsal_sitecustomize._local_urlopen(
        request,
        timeout=10.0,
    ) as response:
        contract = json.loads(response.read().decode("utf-8"))

    assert contract == _source_add_claim_control_contract()


def test_local_schema_adapter_returns_source_add_miner_status_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    request = urllib.request.Request(
        (
            "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/rpc/"
            "research_lab_source_add_miner_status_contract_v1"
        ),
        data=b"{}",
        headers={
            "apikey": "rehearsal-secret",
            "Authorization": "Bearer rehearsal-secret",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with rehearsal_sitecustomize._local_urlopen(
        request,
        timeout=10.0,
    ) as response:
        contract = json.loads(response.read().decode("utf-8"))

    assert contract == _source_add_miner_status_contract_fixture()


def test_gateway_cli_secret_matches_initial_durable_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    monkeypatch.setenv(
        "REHEARSAL_FROM_SHA",
        "7ac1553e32d85d9babda3b3836f4c93cf92e6d60",
    )
    monkeypatch.setenv("REHEARSAL_TRANSITION", "forward")
    from tests.restart_rehearsal import contract_adapter

    monkeypatch.setattr(
        contract_adapter,
        "GATEWAY_SECRET_STATE_PATH",
        tmp_path / "missing-gateway-secret-state.json",
    )
    monkeypatch.setattr(
        contract_adapter,
        "_initial_gateway_miner_submissions_state",
        lambda: "false",
    )

    current = contract_adapter._current_gateway_secret()
    assert current == json.loads(
        rehearsal_sitecustomize._initial_gateway_secret_string()
    )
    assert current["SUPABASE_URL"] == (
        "https://qplwoislplkcegvdmbim.supabase.co"
    )


def _receipt_graph_seed_contract() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    boot_hash = "sha256:" + "b" * 64
    receipt_hash = "sha256:" + "c" * 64
    attempt_hash = "sha256:" + "d" * 64
    job_id = "rehearsal-finalized-allocation"
    purpose = "research_lab.legacy_finalized_allocation.v2"
    return (
        {
            "research_lab_attested_boot_identities_v2": {
                "kind": "r",
                "columns": ["boot_identity_hash"],
            },
            "research_lab_attested_execution_receipts_v2": {
                "kind": "r",
                "columns": [
                    "receipt_hash",
                    "boot_identity_hash",
                    "receipt_doc",
                ],
            },
            "research_lab_attested_receipt_edges_v2": {
                "kind": "r",
                "columns": ["child_receipt_hash", "parent_receipt_hash"],
            },
            "research_lab_attested_receipt_transport_v2": {
                "kind": "r",
                "columns": ["receipt_hash", "attempt_hash"],
            },
            "research_lab_attested_transport_attempts_v2": {
                "kind": "r",
                "columns": ["attempt_hash", "attempt_doc"],
            },
        },
        {
            "research_lab_attested_boot_identities_v2": [
                {"boot_identity_hash": boot_hash},
            ],
            "research_lab_attested_execution_receipts_v2": [
                {
                    "receipt_hash": receipt_hash,
                    "boot_identity_hash": boot_hash,
                    "receipt_doc": {
                        "receipt_hash": receipt_hash,
                        "boot_identity_hash": boot_hash,
                        "parent_receipt_hashes": [],
                        "job_id": job_id,
                        "purpose": purpose,
                    },
                },
            ],
            "research_lab_attested_receipt_edges_v2": [],
            "research_lab_attested_receipt_transport_v2": [
                {
                    "receipt_hash": receipt_hash,
                    "attempt_hash": attempt_hash,
                },
            ],
            "research_lab_attested_transport_attempts_v2": [
                {
                    "attempt_hash": attempt_hash,
                    "attempt_doc": {
                        "attempt_hash": attempt_hash,
                        "job_id": job_id,
                        "purpose": purpose,
                    },
                },
            ],
        },
    )


def test_historical_receipt_fixture_binds_candidate_release_identity(
    tmp_path,
    monkeypatch,
) -> None:
    from leadpoet_canonical.attested_v2 import (
        BOOT_ATTESTATION_PURPOSE,
        build_boot_attestation_user_data,
    )

    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    release_identity = {
        "commit_sha": COMMIT,
        "dependency_lock_hash": "sha256:" + "2" * 64,
        "execution_manifest_hash": "sha256:" + "3" * 64,
        "pcr0": "4" * 96,
    }
    release_path = tmp_path / "release-build-input.json"
    release_path.write_text(
        json.dumps(
            {
                "commit_sha": COMMIT,
                "gateway_roles": {
                    "gateway_coordinator": release_identity,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    loaded = postgres_probe._load_coordinator_release_identity(
        release_path,
        candidate_sha=COMMIT,
    )
    fixture = SanitizedWeightFixture(candidate_sha=COMMIT, epoch_id=24207)
    config_hash = "sha256:" + "5" * 64
    boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash=config_hash,
        release_identity=loaded,
    )
    same_epoch_boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash="sha256:" + "6" * 64,
        release_identity=loaded,
        boot_nonce_context="historical-compute-settlement",
    )
    assert same_epoch_boot["boot_nonce"] != boot["boot_nonce"]
    assert same_epoch_boot["boot_identity_hash"] != boot["boot_identity_hash"]
    receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.legacy_finalized_allocation.v2",
        job_id="historical-release-binding",
        key=fixture.coordinator_key,
        boot=boot,
        config_hash=config_hash,
    )
    verified, extracted = (
        rehearsal_sitecustomize._local_verify_nitro_attestation_full(
            attestation_b64=boot["attestation_document_b64"],
            expected_pcr0=release_identity["pcr0"],
            expected_pubkey=boot["signing_pubkey"],
            expected_purpose=BOOT_ATTESTATION_PURPOSE,
            role="gateway",
            certificate_validity_at_attestation_time=True,
        )
    )
    assert verified is True
    assert extracted == {
        "pcr0": release_identity["pcr0"],
        "enclave_pubkey": boot["signing_pubkey"],
        "user_data": build_boot_attestation_user_data(boot),
    }
    assert {
        field: boot[field]
        for field in (
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
        )
    } == {
        "commit_sha": COMMIT,
        "pcr0": release_identity["pcr0"],
        "build_manifest_hash": release_identity["execution_manifest_hash"],
        "dependency_lock_hash": release_identity["dependency_lock_hash"],
    }
    assert {
        field: receipt[field]
        for field in (
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
        )
    } == {
        "commit_sha": COMMIT,
        "pcr0": release_identity["pcr0"],
        "build_manifest_hash": release_identity["execution_manifest_hash"],
        "dependency_lock_hash": release_identity["dependency_lock_hash"],
    }

    release_path.write_text(
        json.dumps(
            {
                "commit_sha": "6" * 40,
                "gateway_roles": {
                    "gateway_coordinator": release_identity,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        postgres_probe.PostgresContractProbeError,
        match="commit differs",
    ):
        postgres_probe._load_coordinator_release_identity(
            release_path,
            candidate_sha=COMMIT,
        )


def test_historical_compute_seed_precedes_native_finalizations() -> None:
    assert postgres_probe._historical_compute_source_epoch(
        (24219, 24218)
    ) == 24217
    with pytest.raises(
        postgres_probe.PostgresContractProbeError,
        match="historical compute source epoch is unavailable",
    ):
        postgres_probe._historical_compute_source_epoch(())
    with pytest.raises(
        postgres_probe.PostgresContractProbeError,
        match="historical compute source epoch is unavailable",
    ):
        postgres_probe._historical_compute_source_epoch((0, 1))


def test_finalized_settlement_fixture_exports_complete_receipt_graphs() -> None:
    prior_rows, _, _ = postgres_probe._settlement_fixture(
        candidate_sha=COMMIT,
        epoch_id=24206,
    )
    current_rows, _, _ = postgres_probe._settlement_fixture(
        candidate_sha=COMMIT,
        epoch_id=24207,
    )
    rows = postgres_probe._deduplicate_settlement_fixture_rows(
        (*prior_rows, *current_rows)
    )
    seed_rows = postgres_probe._settlement_graph_seed_rows(rows)

    receipts = {
        row["receipt_hash"]: row["receipt_doc"]
        for row in seed_rows["research_lab_attested_execution_receipts_v2"]
    }
    boots = {
        row["boot_identity_hash"]
        for row in seed_rows["research_lab_attested_boot_identities_v2"]
    }
    attempts = {
        row["attempt_hash"]: row["attempt_doc"]
        for row in seed_rows["research_lab_attested_transport_attempts_v2"]
    }
    edges_by_child: dict[str, set[str]] = {}
    for row in seed_rows["research_lab_attested_receipt_edges_v2"]:
        edges_by_child.setdefault(row["child_receipt_hash"], set()).add(
            row["parent_receipt_hash"]
        )
    for row in seed_rows["research_lab_attested_receipt_transport_v2"]:
        assert row["receipt_hash"] in receipts
        assert row["attempt_hash"] in attempts

    finalization_rows = [
        row
        for table, row in rows
        if table == "research_lab_attested_weight_finalizations_v2"
    ]
    expected_finalization_fields = {
        "schema_version",
        "validator_hotkey",
        "netuid",
        "epoch_id",
        "weights_hash",
        "weight_receipt_hash",
        "weight_submission_event_hash",
        "extrinsic_authorization",
        "extrinsic_authorization_hash",
        "extrinsic_signature",
        "extrinsic_receipt_hash",
        "extrinsic_hash",
        "finalized_block",
        "finalized_block_hash",
        "state_transition_hash",
    }
    for row in finalization_rows:
        finalization = row["finalization_doc"]
        assert set(finalization) == expected_finalization_fields
        root = receipts[row["finalization_receipt_hash"]]
        assert root["purpose"] == "validator.weights.finalized.v2"
        assert root["parent_receipt_hashes"] == [
            finalization["extrinsic_receipt_hash"]
        ]
        scoped_attempts = [
            attempt
            for attempt in attempts.values()
            if attempt["job_id"] == root["job_id"]
            and attempt["purpose"] == root["purpose"]
        ]
        assert {
            (attempt["provider_id"], attempt["destination_host"])
            for attempt in scoped_attempts
        } == {
            ("bittensor_chain", "entrypoint-finney.opentensor.ai"),
            ("bittensor_archive", "archive.chain.opentensor.ai"),
        }

    finalization_roots = {
        row["finalization_receipt_hash"]
        for row in finalization_rows
    }
    assert len(finalization_roots) == 2
    pending = set(finalization_roots)
    visited: set[str] = set()
    while pending:
        receipt_hash = pending.pop()
        if receipt_hash in visited:
            continue
        receipt = receipts[receipt_hash]
        assert receipt["boot_identity_hash"] in boots
        parents = set(receipt["parent_receipt_hashes"])
        assert edges_by_child.get(receipt_hash, set()) == parents
        pending.update(parents)
        visited.add(receipt_hash)
    assert finalization_roots.issubset(visited)


def test_settlement_fixture_transport_identity_is_unique_across_epochs() -> None:
    prior_rows, _, _ = postgres_probe._settlement_fixture(
        candidate_sha=COMMIT,
        epoch_id=24206,
    )
    current_rows, _, _ = postgres_probe._settlement_fixture(
        candidate_sha=COMMIT,
        epoch_id=24207,
    )
    rows = postgres_probe._deduplicate_settlement_fixture_rows(
        (*prior_rows, *current_rows)
    )
    attempts = [
        row
        for table, row in rows
        if table == "research_lab_attested_transport_attempts_v2"
    ]

    request_ids = [row["request_id"] for row in attempts]
    logical_attempts = [
        (row["logical_operation_id"], row["attempt_number"])
        for row in attempts
    ]
    assert len(request_ids) == len(set(request_ids))
    assert len(logical_attempts) == len(set(logical_attempts))


def test_weight_fixture_binds_allocation_input_to_attested_authority() -> None:
    from gateway.research_lab.champion_settlement_v2 import (
        _allocation_authority_receipt_hash_v2,
    )

    fixture = SanitizedWeightFixture(candidate_sha=COMMIT, epoch_id=24207)
    bundle = fixture.bundle()
    snapshot = bundle["weight_snapshot"]
    allocation = snapshot["calculation_snapshot"][
        "research_lab_allocation_doc"
    ]
    allocation_input_hash = snapshot["input_receipt_hashes"][
        "research_lab_allocation"
    ]
    receipts = {
        receipt["receipt_hash"]: receipt
        for receipt in bundle["receipt_graph"]["receipts"]
    }
    allocation_input = receipts[allocation_input_hash]

    assert len(allocation_input["parent_receipt_hashes"]) == 1
    authority_hash = allocation_input["parent_receipt_hashes"][0]
    assert _allocation_authority_receipt_hash_v2(
        bundle_doc=bundle,
        allocation_input_receipt_hash=allocation_input_hash,
        allocation=allocation,
        epoch_id=24207,
    ) == authority_hash


def test_settlement_fixture_passes_full_allocation_authority_validation() -> None:
    from gateway.research_lab.champion_settlement_v2 import (
        validate_finalized_allocation_authorities_v2,
    )
    from leadpoet_canonical.attested_v2 import build_receipt_graph

    rows, _, _ = postgres_probe._settlement_fixture(
        candidate_sha=COMMIT,
        epoch_id=24207,
    )
    by_table: dict[str, list[dict[str, Any]]] = {}
    for table, row in rows:
        by_table.setdefault(table, []).append(row)
    bundle_row = by_table[
        "research_lab_attested_weight_bundles_v2"
    ][0]
    publication_row = by_table[
        "research_lab_attested_publication_events_v2"
    ][0]
    finalization_row = by_table[
        "research_lab_attested_weight_finalizations_v2"
    ][0]
    receipts = {
        row["receipt_hash"]: row["receipt_doc"]
        for row in by_table[
            "research_lab_attested_execution_receipts_v2"
        ]
    }
    finalization_doc = finalization_row["finalization_doc"]
    finalization_receipt = receipts[
        finalization_row["finalization_receipt_hash"]
    ]
    extrinsic_receipt = receipts[
        finalization_doc["extrinsic_receipt_hash"]
    ]
    bundle_graph = bundle_row["bundle_doc"]["receipt_graph"]
    finalization_attempts = [
        row["attempt_doc"]
        for row in by_table[
            "research_lab_attested_transport_attempts_v2"
        ]
        if row["attempt_doc"]["job_id"]
        == finalization_receipt["job_id"]
        and row["attempt_doc"]["purpose"]
        == finalization_receipt["purpose"]
    ]
    finalization_graph = build_receipt_graph(
        root_receipt_hash=finalization_receipt["receipt_hash"],
        boot_identities=bundle_graph["boot_identities"],
        receipts=[
            *[
                receipt
                for receipt in bundle_graph["receipts"]
                if receipt["purpose"]
                != "validator.hotkey_signature.v2"
            ],
            extrinsic_receipt,
            finalization_receipt,
        ],
        transport_attempts=[
            *bundle_graph["transport_attempts"],
            *finalization_attempts,
        ],
    )
    authority_rows = validate_finalized_allocation_authorities_v2(
        [{**bundle_row, **publication_row, **finalization_row}],
        finalization_graphs={
            finalization_receipt["receipt_hash"]: finalization_graph
        },
    )

    assert len(authority_rows) == 1
    assert authority_rows[0]["epoch"] == 24207
    assert authority_rows[0]["allocation_authority_receipt_hash"] in receipts


def test_settlement_fixture_uses_explicit_exact_container_source_root(
    monkeypatch,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        postgres_probe,
        "__file__",
        "/harness/postgres_v2_contract_probe.py",
    )

    with pytest.raises(
        postgres_probe.PostgresContractProbeError,
        match="candidate source root is required",
    ):
        postgres_probe._settlement_fixture(
            candidate_sha=COMMIT,
            epoch_id=24207,
        )

    rows, verified, _fixture = postgres_probe._settlement_fixture(
        candidate_sha=COMMIT,
        epoch_id=24207,
        source_root=source_root,
    )
    assert rows
    assert verified["epoch_id"] == 24207


def test_chain_settlement_activation_fixture_creates_one_epoch_backlog(
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
    current_epoch = int(cutover["first_settlement_epoch_id"]) + (
        int(fixture["network"]["subnet_epoch_index"])
        - int(cutover["first_subnet_epoch_index"])
    )
    activation_epoch = current_epoch - 1
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
            "first_epoch_id": activation_epoch,
            "source_bundle_hash": "sha256:" + "a" * 64,
            "source_bundle_epoch_id": activation_epoch,
            "source_finalized_block": (
                int(fixture["network"]["current_block"]) - 1
            ),
        }
    ]
    assert rehearsal_sitecustomize._current_settlement_epoch_id() == (
        current_epoch
    )
    assert load_rehearsal_current_settlement_epoch_id(source_root) == (
        current_epoch
    )


def test_historical_compute_seed_matches_exact_metagraph_recipients(
    monkeypatch,
) -> None:
    from leadpoet_canonical.chain_source_v2 import (
        CHAIN_SELECTIVE_RESULT_LAST_FIELDS,
        decode_selective_metagraph_result,
        ss58_encode_account_id,
    )

    source_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "SOURCE_ROOT",
        source_root,
    )
    fixture_hotkeys = load_rehearsal_metagraph_hotkeys(source_root)
    account_ids = load_rehearsal_metagraph_account_ids(source_root)
    chain_metagraphs = [
        decode_selective_metagraph_result(
            rehearsal_sitecustomize._selective_metagraph_fixture(
                8_700_040,
                last_field=last_field,
            )
        )
        for last_field in CHAIN_SELECTIVE_RESULT_LAST_FIELDS
    ]
    exact_hotkeys = rehearsal_sitecustomize._local_metagraph_hotkeys()
    reimbursements = postgres_probe._historical_compute_reimbursements(
        source_root=source_root,
        source_epoch=24207,
    )

    assert exact_hotkeys[2:] == fixture_hotkeys[2:]
    assert {
        tuple(chain_metagraph["hotkeys"])
        for chain_metagraph in chain_metagraphs
    } == {fixture_hotkeys}
    assert tuple(
        ss58_encode_account_id(account_id) for account_id in account_ids
    ) == fixture_hotkeys
    assert tuple(
        row["miner_hotkey"] for row in reimbursements
    ) == exact_hotkeys[2:]
    assert tuple(row["uid"] for row in reimbursements) == (2, 3)


def test_sitecustomize_loads_from_staged_harness_without_candidate_package(
    tmp_path,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    harness_root = Path(__file__).resolve().parent
    expected_hotkeys = load_rehearsal_metagraph_hotkeys(source_root)[2:]
    environment = {
        **os.environ,
        "PYTHONPATH": str(harness_root),
        "REHEARSAL_SOURCE_ROOT": str(source_root),
        "REHEARSAL_STATE_ROOT": str(tmp_path / "state"),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sitecustomize;"
                "assert sitecustomize._local_metagraph_hotkeys()[2:] == "
                + repr(expected_hotkeys)
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Error in sitecustomize" not in result.stderr


def test_sitecustomize_accepts_only_exact_s3_client_options(monkeypatch) -> None:
    from botocore.config import Config

    regional = {
        "region_name": "us-east-1",
        "endpoint_url": "https://s3.us-east-1.amazonaws.com",
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
        ),
    }
    rehearsal_sitecustomize._validate_local_boto3_client_options(
        "s3", regional
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "rehearsal-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "rehearsal-secret")
    upload = {
        "aws_access_key_id": "rehearsal-access",
        "aws_secret_access_key": "rehearsal-secret",
        "region_name": "us-east-1",
        "config": Config(signature_version="s3v4"),
    }
    rehearsal_sitecustomize._validate_local_boto3_client_options("s3", upload)
    for options in (
        {**upload, "aws_access_key_id": "different"},
        {**upload, "aws_secret_access_key": "different"},
        {**upload, "endpoint_url": "https://s3.us-east-1.amazonaws.com"},
    ):
        with pytest.raises(ValueError, match="client options differ"):
            rehearsal_sitecustomize._validate_local_boto3_client_options(
                "s3", options
            )

    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
    monkeypatch.setenv("LEADPOET_AWS_INSTANCE_ROLE_ONLY", "true")
    instance_role_upload = {
        **upload,
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
    }
    rehearsal_sitecustomize._validate_local_boto3_client_options(
        "s3", instance_role_upload
    )

    class ConfigImpostor:
        _user_provided_options = {
            "signature_version": "s3v4",
            "s3": {"addressing_style": "virtual"},
        }

    invalid = (
        {**regional, "endpoint_url": "https://example.invalid"},
        {**regional, "region_name": " us-east-1"},
        {
            **regional,
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        },
        {**regional, "config": Config(signature_version="s3")},
        {**regional, "config": ConfigImpostor()},
        {**regional, "use_ssl": True},
        {**instance_role_upload, "aws_access_key_id": ""},
        {**instance_role_upload, "aws_secret_access_key": ""},
        {**instance_role_upload, "aws_access_key_id": "mixed"},
        {**instance_role_upload, "aws_secret_access_key": "mixed"},
    )
    for options in invalid:
        with pytest.raises(ValueError, match="client options differ"):
            rehearsal_sitecustomize._validate_local_boto3_client_options(
                "s3", options
            )
    with pytest.raises(ValueError, match="client options differ"):
        rehearsal_sitecustomize._validate_local_boto3_client_options(
            "kms", regional
        )


def test_sitecustomize_rejects_unbound_instance_role_s3_options(
    monkeypatch,
) -> None:
    from botocore.config import Config

    options = {
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "region_name": "us-east-1",
        "config": Config(signature_version="s3v4"),
    }
    monkeypatch.delenv("LEADPOET_AWS_INSTANCE_ROLE_ONLY", raising=False)
    with pytest.raises(ValueError, match="client options differ"):
        rehearsal_sitecustomize._validate_local_boto3_client_options(
            "s3", options
        )

    monkeypatch.setenv("LEADPOET_AWS_INSTANCE_ROLE_ONLY", "false")
    with pytest.raises(ValueError, match="client options differ"):
        rehearsal_sitecustomize._validate_local_boto3_client_options(
            "s3", options
        )

    monkeypatch.setenv("LEADPOET_AWS_INSTANCE_ROLE_ONLY", "true")
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
    ):
        monkeypatch.setenv(name, "unexpected")
        with pytest.raises(ValueError, match="client options differ"):
            rehearsal_sitecustomize._validate_local_boto3_client_options(
                "s3", options
            )
        monkeypatch.delenv(name)


def test_chain_settlement_boundary_persists_zero_credit_readback(
    tmp_path,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (
            source_root
            / "tests/restart_rehearsal/fixtures/production_shaped_v2.json"
        ).read_text(encoding="utf-8")
    )
    tables = {
        "research_lab_chain_realized_settlement_activation_v1",
        "research_lab_chain_realized_epoch_settlements_v1",
        "research_lab_chain_realized_obligation_credits_v1",
        "research_lab_stateful_subnet_epoch_cutover_state_v1",
        "research_lab_stateful_subnet_epoch_cutovers_v1",
    }
    state = LocalPostgRESTState(
        state_root=tmp_path,
        fixture=fixture,
        source_root=source_root,
        tables=tables,
        rpcs={
            "persist_research_lab_chain_realized_unattributed_v2",
        },
    )
    activation_epoch = state.rows[
        "research_lab_chain_realized_settlement_activation_v1"
    ][0]["first_epoch_id"]
    settlement_hash = "sha256:" + "1" * 64
    receipt_hash = "sha256:" + "2" * 64
    settlement = {
        "netuid": 71,
        "epoch_id": activation_epoch,
        "schema_version": (
            "leadpoet.research_lab_chain_realized_epoch_settlement.v2"
        ),
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": receipt_hash,
        "settlement_doc": {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_epoch_settlement.v2"
            ),
            "netuid": 71,
            "epoch_id": activation_epoch,
            "credit_hashes": [],
        },
    }
    body = {
        "requested_settlement": settlement,
        "requested_credits": [],
    }

    first = state.persist_chain_realized_settlement(
        rpc_name="persist_research_lab_chain_realized_unattributed_v2",
        body=body,
    )
    second = state.persist_chain_realized_settlement(
        rpc_name="persist_research_lab_chain_realized_unattributed_v2",
        body=body,
    )

    assert first == second
    assert first["credit_count"] == 0
    assert len(
        state.rows[
            "research_lab_chain_realized_epoch_settlements_v1"
        ]
    ) == 1
    assert state.rows[
        "research_lab_chain_realized_obligation_credits_v1"
    ] == []


def test_chain_settlement_evidence_requires_persistence_then_readback() -> None:
    persisted = {
        "kind": "local-postgrest",
        "operation": "chain_settlement_persisted",
        "status": "ok",
        "target": "persist_research_lab_chain_realized_unattributed_v2",
    }
    settlement_read = {
        "kind": "local-postgrest",
        "operation": "select",
        "status": "ok",
        "target": "research_lab_chain_realized_epoch_settlements_v1",
        "row_count": 1,
    }
    credit_read = {
        "kind": "local-postgrest",
        "operation": "select",
        "status": "ok",
        "target": "research_lab_chain_realized_obligation_credits_v1",
        "row_count": 0,
    }
    verify_chain_settlement_durable_readback(
        [settlement_read, credit_read]
    )
    with pytest.raises(SystemExit, match="neither persisted nor read"):
        verify_chain_settlement_durable_readback(
            [{**settlement_read, "row_count": 0}, credit_read]
        )
    with pytest.raises(SystemExit, match="durable settlement$"):
        verify_chain_settlement_durable_readback(
            [settlement_read, persisted, credit_read]
        )
    with pytest.raises(SystemExit, match="settlement credits"):
        verify_chain_settlement_durable_readback(
            [persisted, settlement_read]
        )
    verify_chain_settlement_durable_readback(
        [persisted, settlement_read, credit_read]
    )


def test_rehearsal_evidence_merges_postgrest_events_in_time_order(
    tmp_path,
) -> None:
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"at_ns": 10, "kind": "host-command"}) + "\n"
        + json.dumps({"at_ns": 30, "kind": "gateway-http"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "local-postgrest-events.jsonl").write_text(
        json.dumps(
            {
                "at_ns": 20,
                "kind": "local-postgrest",
                "operation": "chain_settlement_persisted",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert [row["at_ns"] for row in events(tmp_path)] == [10, 20, 30]


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


def test_gateway_rehearsal_applies_nested_postgrest_json_filters() -> None:
    allocation_path = (
        "bundle_doc->weight_snapshot->calculation_snapshot"
        "->research_lab_allocation_doc"
    )
    rows = [
        {
            "epoch_id": 24218,
            "bundle_doc": {
                "weight_snapshot": {
                    "calculation_snapshot": {
                        "research_lab_allocation_doc": {
                            "reimbursement_allocations": [
                                {"miner_hotkey": "5eligible"}
                            ]
                        }
                    }
                }
            },
        },
        {
            "epoch_id": 24217,
            "bundle_doc": {
                "weight_snapshot": {
                    "calculation_snapshot": {
                        "research_lab_allocation_doc": {
                            "historical_compute_fallback_source_epoch": 24216,
                            "reimbursement_allocations": [
                                {"miner_hotkey": "5recursive"}
                            ],
                        }
                    }
                }
            },
        },
        {
            "epoch_id": 24216,
            "bundle_doc": {
                "weight_snapshot": {
                    "calculation_snapshot": {
                        "research_lab_allocation_doc": {
                            "reimbursement_allocations": []
                        }
                    }
                }
            },
        },
    ]

    assert _apply_table_query(
        rows,
        "select=epoch_id"
        f"&{allocation_path}->reimbursement_allocations=not.eq.[]"
        f"&{allocation_path}"
        "->>historical_compute_fallback_source_epoch=is.null"
        "&order=epoch_id.desc"
        "&limit=1",
        allowed_columns=frozenset({"bundle_doc", "epoch_id"}),
    ) == [{"epoch_id": 24218}]
    with pytest.raises(ValueError, match="filter column is invalid"):
        _apply_table_query(
            rows,
            "bundle_doc->>weight_snapshot->calculation_snapshot=eq.invalid",
            allowed_columns=frozenset({"bundle_doc", "epoch_id"}),
        )


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


def test_gateway_rehearsal_icp_schema_covers_runtime_contract() -> None:
    match = re.search(
        r"CREATE TABLE public\.qualification_private_icp_sets \((.*?)\n\);",
        ALLOCATION_MIGRATION_PREREQUISITES_SQL,
        flags=re.DOTALL,
    )
    assert match is not None
    columns = {
        line.strip().split()[0].rstrip(",")
        for line in match.group(1).splitlines()
        if line.strip()
    }
    assert columns == {
        "set_id",
        "icps",
        "icp_set_hash",
        "industry_distribution",
        "active_from",
        "active_until",
        "generation_seed",
        "is_active",
    }
    assert _apply_table_query(
        [],
        "select=set_id,icps,icp_set_hash,active_from,active_until"
        "&is_active=eq.True&limit=1",
        allowed_columns=frozenset(columns),
    ) == []


def test_migration_backed_contract_is_candidate_bound_and_complete(
    tmp_path,
) -> None:
    graph_relations, graph_seed_rows = _receipt_graph_seed_contract()
    relations = {
        name: {"kind": "r", "columns": ["schema_version"]}
        for name in {
            "research_lab_maintenance_lease",
            "research_lab_attested_transport_attempts_v2",
            "research_lab_attested_boot_identities_v2",
            "research_lab_attested_execution_receipts_v2",
            "research_lab_attested_weight_bundles_v2",
            "research_lab_attested_publication_events_v2",
            "research_lab_attested_weight_finalizations_v2",
            "research_lab_finalized_allocation_epochs_v2",
            "research_lab_emission_allocation_current",
            "research_lab_legacy_finalized_allocation_migrations_v2",
            "research_lab_chain_realized_epoch_settlements_v1",
            "research_lab_chain_realized_settlement_activation_v1",
            "research_lab_chain_realized_obligation_credits_v1",
            "research_lab_provider_outcome_checkpoints_v2",
            "research_lab_attested_ancestry_checkpoints_v2",
            "research_lab_attested_ancestry_activations_v2",
            "research_lab_allocation_settlement_frontiers_v2",
            "research_lab_allocation_settlement_frontier_activation_v2",
            "research_lab_compact_weight_authorities_v2",
            "research_lab_candidate_model_unit_terminals",
            "research_lab_candidate_waterfall_receipts",
            "research_lab_candidate_waterfall_metrics",
                "research_lab_source_add_provenance_leg1_authority_v1",
                "research_lab_source_add_miner_status_v1",
                "lab_arena_rounds",
                "lab_arena_submissions",
                "lab_arena_runs",
                "lab_arena_ledger",
                "lab_arena_reward_basis_v1",
        }
    }
    relations["research_lab_finalized_allocation_epochs_v2"] = {
        "kind": "v",
        "columns": list(EXPECTED_FINALIZED_VIEW_COLUMNS),
    }
    relations["research_lab_legacy_finalized_allocation_migrations_v2"] = {
        "kind": "r",
        "columns": ["schema_version", "netuid", "epoch_id"],
    }
    relations["research_lab_maintenance_lease"] = {
        "kind": "r",
        "columns": [
            "lease_name",
            "holder_ref",
            "acquired_at",
            "expires_at",
            "updated_at",
        ],
    }
    relations.update(graph_relations)
    contract = {
        "schema_version": "leadpoet.restart_rehearsal.postgres_contract.v1",
        "candidate_sha": COMMIT,
        "applied_migrations": list(EXPECTED_APPLIED_MIGRATIONS),
        "relations": relations,
        "rpcs": [
            "research_lab_acquire_maintenance_lease",
            "research_lab_attested_transport_purpose_contract_v2",
            "research_lab_attested_transport_terminal_contract_v2",
            "append_research_lab_provider_outcome_checkpoint_v2",
            "research_lab_provider_outcome_contention_contract_v2",
            "research_lab_provider_outcome_contention_contract_v3",
            "put_research_lab_provider_evidence_cache_v2",
            "append_research_lab_provider_outcome_checkpoints_v2",
            "research_lab_provider_persistence_batch_contract_v1",
            "persist_research_lab_chain_realized_lifetime_settlement_v2",
            "research_lab_champion_lifetime_credit_contract_v1",
            "research_lab_active_model_replay_contract_v2",
            "persist_research_lab_ancestry_checkpoint_v2",
            "research_lab_ancestry_disclosure_lookup_contract_v1",
            "leadpoet_production_parity_reader_contract_v1",
            "persist_research_lab_allocation_settlement_frontier_v2",
            "persist_research_lab_allocation_frontier_bootstrap_v2",
            "research_lab_ancestry_checkpoint_bootstrap_contract_v2",
            "research_lab_allocation_frontier_bootstrap_contract_v2",
            "research_lab_allocation_frontier_historical_source_contract_v1",
            "research_lab_source_catalog_replay_contract_v2",
            "research_lab_compact_checkpoint_graph_contract_v1",
            "resume_research_lab_credit_blocked_run_v1",
            "research_lab_compact_weight_settlement_contract_v1",
            "research_lab_candidate_hybrid_purpose_contract_v1",
            "research_lab_source_add_provider_origin_contract_v1",
            "research_lab_source_add_duplicate_privacy_contract_v1",
            "research_lab_source_add_post_accept_leg1_contract_v1",
            "research_lab_source_add_post_accept_leg1_contract_v2",
            "research_lab_source_add_post_accept_leg1_contract_v3",
            "research_lab_source_add_post_accept_leg1_contract_v4",
            "research_lab_source_add_miner_status_contract_v1",
            "research_lab_source_add_miner_status_page_v1",
            "research_lab_source_add_configure_probe_v3",
            "research_lab_source_add_enqueue_leg1_after_provenance_v1",
            "research_lab_source_add_enqueue_provision_smoke_v2",
            "research_lab_source_add_finalize_leg1_v4",
            "research_lab_source_add_finalize_provision_smoke_v3",
            "research_lab_source_add_finalize_provision_v3",
            "research_lab_source_add_reject_current_builtin_v3",
            "research_lab_source_add_reconcile_provenance_leg1_v1",
            "research_lab_source_add_reserve_leg1_slot_v4",
            "research_lab_source_add_reserve_leg1_slot_v3",
                "research_lab_source_add_finalize_leg1_v3",
                "lab_arena_current_daily_icp_set",
                "lab_arena_register_submission",
                "lab_arena_update_submission",
                "lab_arena_claim_assignment",
                "lab_arena_activate_reward",
                "lab_arena_schema_version_v1",
                "research_lab_routing_exact_model_transition_contract_v1",
            "research_lab_routing_exact_model_transition_contract_v2",
            "research_lab_routing_load_model_transition_v2",
            "research_lab_candidate_append_model_unit_terminal_v1",
            "research_lab_candidate_append_waterfall_receipt_v1",
            "research_lab_candidate_append_waterfall_metric_v1",
        ],
        "atomic_credit_resume": _atomic_credit_resume_fixture(),
        "compact_weight_settlement_contract": (
            _compact_weight_settlement_contract_fixture()
        ),
        "maintenance_lease": {
            "schema_version": "leadpoet.maintenance_lease_contract.v1",
            "atomic_acquire": True,
            "live_contention_rejected": True,
            "same_holder_renewed": True,
            "expired_holder_replaced": True,
            "invalid_ttl_rejected": True,
        },
        "checks": {
            name: True for name in EXPECTED_POSTGRES_CONTRACT_CHECKS
        },
        "provider_outcome_contention_contract": {
            "schema_version": (
                "leadpoet.provider_outcome_contention_contract.v3"
            ),
            "lock_contention_status": "busy",
            "stale_lineage_status": "conflict",
            "candidate_checkpoint_hash": True,
            "conflict_head_checkpoint_row": "encrypted_or_null",
        },
        "maintenance_lease": {
            "schema_version": "leadpoet.maintenance_lease_contract.v1",
            "atomic_acquire": True,
            "live_contention_rejected": True,
            "same_holder_renewed": True,
            "expired_holder_replaced": True,
            "invalid_ttl_rejected": True,
        },
        "provider_outcome_append": {
            "accepted_count": 1,
            "rejected_count": 1,
            "row_count": 3,
            "contention_rollback_delta": 0,
            "durable_head_conflict_verified": True,
            "empty_head_conflict_verified": True,
        },
        "provider_persistence_batch": _provider_persistence_batch_fixture(),
        "seed_rows": {
            "research_lab_finalized_allocation_epochs_v2": [
                {
                    **{
                        column: None
                        for column in EXPECTED_FINALIZED_VIEW_COLUMNS
                    },
                    "netuid": 71,
                    "epoch_id": 24_218,
                },
                {
                    **{
                        column: None
                        for column in EXPECTED_FINALIZED_VIEW_COLUMNS
                    },
                    "netuid": 71,
                    "epoch_id": 24_219,
                },
            ],
            "research_lab_emission_allocation_current": [
                {"schema_version": None},
            ],
            "research_lab_legacy_finalized_allocation_migrations_v2": [
                {
                    "schema_version": None,
                    "netuid": 71,
                    "epoch_id": 24_217,
                },
            ],
            **graph_seed_rows,
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
    assert "research_lab_attested_transport_terminal_contract_v2" in rpcs
    assert "append_research_lab_provider_outcome_checkpoint_v2" in rpcs
    assert "research_lab_provider_outcome_contention_contract_v2" in rpcs
    assert "research_lab_provider_outcome_contention_contract_v3" in rpcs
    assert (
        "persist_research_lab_chain_realized_lifetime_settlement_v2"
        in rpcs
    )
    assert "research_lab_champion_lifetime_credit_contract_v1" in rpcs
    assert "research_lab_active_model_replay_contract_v2" in rpcs
    assert "resume_research_lab_credit_blocked_run_v1" in rpcs
    assert "persist_research_lab_ancestry_checkpoint_v2" in rpcs
    assert "research_lab_ancestry_disclosure_lookup_contract_v1" in rpcs
    assert "leadpoet_production_parity_reader_contract_v1" in rpcs
    assert "persist_research_lab_allocation_settlement_frontier_v2" in rpcs
    assert "persist_research_lab_allocation_frontier_bootstrap_v2" in rpcs
    assert "research_lab_ancestry_checkpoint_bootstrap_contract_v2" in rpcs
    assert "research_lab_allocation_frontier_bootstrap_contract_v2" in rpcs
    assert _migration_seed_rows(
        path,
        candidate_sha=COMMIT,
        relation_columns=relation_columns,
    ) == contract["seed_rows"]
    assert _migration_provider_outcome_contract(
        path,
        candidate_sha=COMMIT,
    ) == contract["provider_outcome_contention_contract"]
    with pytest.raises(RuntimeError, match="differs from candidate"):
        _migration_schema_contract(path, candidate_sha="2" * 40)

    stale_migrations = json.loads(json.dumps(contract))
    stale_migrations["applied_migrations"] = stale_migrations[
        "applied_migrations"
    ][:-1]
    path.write_text(json.dumps(stale_migrations), encoding="utf-8")
    with pytest.raises(RuntimeError, match="final migration order"):
        _migration_schema_contract(path, candidate_sha=COMMIT)

    missing_atomic_resume = json.loads(json.dumps(contract))
    missing_atomic_resume.pop("atomic_credit_resume")
    path.write_text(json.dumps(missing_atomic_resume), encoding="utf-8")
    with pytest.raises(RuntimeError, match="atomic credit resume evidence"):
        _migration_schema_contract(path, candidate_sha=COMMIT)

    missing_atomic_resume_rpc = json.loads(json.dumps(contract))
    missing_atomic_resume_rpc["rpcs"].remove(
        "resume_research_lab_credit_blocked_run_v1"
    )
    path.write_text(json.dumps(missing_atomic_resume_rpc), encoding="utf-8")
    with pytest.raises(RuntimeError, match="RPCs are unavailable"):
        _migration_schema_contract(path, candidate_sha=COMMIT)

    incomplete = json.loads(json.dumps(contract))
    incomplete["seed_rows"][
        "research_lab_attested_execution_receipts_v2"
    ][0]["receipt_doc"]["parent_receipt_hashes"] = [
        "sha256:" + "e" * 64
    ]
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt graph is incomplete"):
        _migration_seed_rows(
            path,
            candidate_sha=COMMIT,
            relation_columns=relation_columns,
        )


def test_rehearsal_evidence_requires_all_postgres_contract_checks(
    tmp_path,
    monkeypatch,
) -> None:
    graph_relations, graph_seed_rows = _receipt_graph_seed_contract()
    state_root = tmp_path / "rehearsal-state"
    state_root.mkdir()
    contract_path = state_root / "postgres-v2-schema-contract.json"
    contract = {
        "schema_version": "leadpoet.restart_rehearsal.postgres_contract.v1",
        "candidate_sha": COMMIT,
        "applied_migrations": list(EXPECTED_APPLIED_MIGRATIONS),
        "relations": {
            "research_lab_maintenance_lease": {
                "kind": "r",
                "columns": [
                    "lease_name",
                    "holder_ref",
                    "acquired_at",
                    "expires_at",
                    "updated_at",
                ],
            },
            "research_lab_finalized_allocation_epochs_v2": {
                "kind": "v",
                "columns": list(EXPECTED_FINALIZED_VIEW_COLUMNS),
            },
            "research_lab_emission_allocation_current": {
                "kind": "v",
                "columns": ["epoch"],
            },
            "research_lab_legacy_finalized_allocation_migrations_v2": {
                "kind": "r",
                "columns": ["netuid", "epoch_id"],
            },
            **graph_relations,
            "research_lab_attested_ancestry_checkpoints_v2": {
                "kind": "r",
                "columns": ["root_receipt_hash"],
            },
            "research_lab_attested_ancestry_activations_v2": {
                "kind": "r",
                "columns": ["activation_root_receipt_hash"],
            },
            "research_lab_allocation_settlement_frontiers_v2": {
                "kind": "r",
                "columns": ["frontier_hash"],
            },
            "research_lab_allocation_settlement_frontier_activation_v2": {
                "kind": "r",
                "columns": ["netuid"],
            },
            "research_lab_compact_weight_authorities_v2": {
                "kind": "r",
                "columns": ["authority_hash"],
            },
            "research_lab_candidate_model_unit_terminals": {
                "kind": "r",
                "columns": ["receipt_id", "experiment_hash"],
            },
            "research_lab_candidate_waterfall_receipts": {
                "kind": "r",
                "columns": ["receipt_id", "experiment_hash"],
            },
            "research_lab_candidate_waterfall_metrics": {
                "kind": "r",
                "columns": ["metric_id", "experiment_hash"],
            },
        },
        "rpcs": [
            "research_lab_acquire_maintenance_lease",
            "persist_research_lab_ancestry_checkpoint_v2",
            "persist_research_lab_allocation_settlement_frontier_v2",
            "persist_research_lab_allocation_frontier_bootstrap_v2",
            "research_lab_ancestry_checkpoint_bootstrap_contract_v2",
            "research_lab_allocation_frontier_bootstrap_contract_v2",
            "research_lab_allocation_frontier_historical_source_contract_v1",
            "research_lab_source_catalog_replay_contract_v2",
            "research_lab_compact_checkpoint_graph_contract_v1",
            "put_research_lab_provider_evidence_cache_v2",
            "append_research_lab_provider_outcome_checkpoints_v2",
            "research_lab_provider_persistence_batch_contract_v1",
            "resume_research_lab_credit_blocked_run_v1",
            "research_lab_compact_weight_settlement_contract_v1",
            "research_lab_candidate_hybrid_purpose_contract_v1",
            "research_lab_source_add_provider_origin_contract_v1",
            "research_lab_source_add_duplicate_privacy_contract_v1",
            "research_lab_source_add_post_accept_leg1_contract_v1",
            "research_lab_source_add_post_accept_leg1_contract_v2",
            "research_lab_source_add_post_accept_leg1_contract_v3",
            "research_lab_source_add_post_accept_leg1_contract_v4",
            "research_lab_source_add_miner_status_contract_v1",
            "research_lab_source_add_miner_status_page_v1",
            "research_lab_source_add_configure_probe_v3",
            "research_lab_source_add_enqueue_leg1_after_provenance_v1",
            "research_lab_source_add_enqueue_provision_smoke_v2",
            "research_lab_source_add_finalize_leg1_v4",
            "research_lab_source_add_finalize_provision_smoke_v3",
            "research_lab_source_add_finalize_provision_v3",
            "research_lab_source_add_reject_current_builtin_v3",
            "research_lab_source_add_reconcile_provenance_leg1_v1",
            "research_lab_source_add_reserve_leg1_slot_v4",
            "research_lab_source_add_reserve_leg1_slot_v3",
            "research_lab_source_add_finalize_leg1_v3",
        ],
        "atomic_credit_resume": _atomic_credit_resume_fixture(),
        "compact_weight_settlement_contract": (
            _compact_weight_settlement_contract_fixture()
        ),
        "maintenance_lease": {
            "schema_version": "leadpoet.maintenance_lease_contract.v1",
            "atomic_acquire": True,
            "live_contention_rejected": True,
            "same_holder_renewed": True,
            "expired_holder_replaced": True,
            "invalid_ttl_rejected": True,
        },
        "checks": {
            name: True for name in EXPECTED_POSTGRES_CONTRACT_CHECKS
        },
        "provider_outcome_contention_contract": {
            "schema_version": (
                "leadpoet.provider_outcome_contention_contract.v3"
            ),
            "lock_contention_status": "busy",
            "stale_lineage_status": "conflict",
            "candidate_checkpoint_hash": True,
            "conflict_head_checkpoint_row": "encrypted_or_null",
        },
        "maintenance_lease": {
            "schema_version": "leadpoet.maintenance_lease_contract.v1",
            "atomic_acquire": True,
            "live_contention_rejected": True,
            "same_holder_renewed": True,
            "expired_holder_replaced": True,
            "invalid_ttl_rejected": True,
        },
        "provider_outcome_append": {
            "accepted_count": 1,
            "rejected_count": 1,
            "row_count": 3,
            "contention_rollback_delta": 0,
            "durable_head_conflict_verified": True,
            "empty_head_conflict_verified": True,
        },
        "provider_persistence_batch": _provider_persistence_batch_fixture(),
        "allocation_settlement_frontier": {
            "frontier_hash": "sha256:" + "a" * 64,
            "source_receipt_hash": "sha256:" + "b" * 64,
            "idempotent_replay": True,
            "frontier_count": 1,
            "activation_count": 1,
        },
        "allocation_settlement_frontier_bootstrap": {
            "frontier_hash": "sha256:" + "c" * 64,
            "allocation_source_receipt_hash": "sha256:" + "d" * 64,
            "bootstrap_receipt_hash": "sha256:" + "e" * 64,
            "idempotent_replay": True,
            "unmeasured_source_rejected": True,
            "frontier_count": 1,
            "activation_count": 1,
        },
        "seed_rows": {
            "research_lab_finalized_allocation_epochs_v2": [
                {
                    **{
                        column: None
                        for column in EXPECTED_FINALIZED_VIEW_COLUMNS
                    },
                    "netuid": 71,
                    "epoch_id": 24_218,
                },
                {
                    **{
                        column: None
                        for column in EXPECTED_FINALIZED_VIEW_COLUMNS
                    },
                    "netuid": 71,
                    "epoch_id": 24_219,
                },
            ],
            "research_lab_emission_allocation_current": [
                {"epoch": None},
            ],
            "research_lab_legacy_finalized_allocation_migrations_v2": [
                {"netuid": 71, "epoch_id": 24_217},
            ],
            **graph_seed_rows,
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
    invalid_contracts = []
    missing_check = json.loads(json.dumps(contract))
    missing_check["checks"][
        "measured_settlement_receipt_projection_exact"
    ] = False
    invalid_contracts.append(missing_check)
    missing_migration = json.loads(json.dumps(contract))
    missing_migration["applied_migrations"].remove(
        EXPECTED_APPLIED_MIGRATIONS[0]
    )
    invalid_contracts.append(missing_migration)
    reordered_migrations = json.loads(json.dumps(contract))
    reordered_migrations["applied_migrations"][0:2] = reversed(
        reordered_migrations["applied_migrations"][0:2]
    )
    invalid_contracts.append(reordered_migrations)
    missing_atomic_credit_resume = json.loads(json.dumps(contract))
    missing_atomic_credit_resume.pop("atomic_credit_resume")
    invalid_contracts.append(missing_atomic_credit_resume)
    altered_atomic_credit_resume = json.loads(json.dumps(contract))
    altered_atomic_credit_resume["atomic_credit_resume"]["row_counts"][
        "resumed_run"
    ] = 3
    invalid_contracts.append(altered_atomic_credit_resume)
    for invalid in invalid_contracts:
        contract_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(
            SystemExit,
            match="evidence is incomplete|evidence is missing|evidence differs",
        ):
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


def test_postgres_fixture_seed_rows_pin_created_at() -> None:
    row = postgres_probe._deterministic_seed_row(
        {"value": "fixture", "created_at": "runtime-default"}
    )

    assert row == {
        "value": "fixture",
        "created_at": postgres_probe.NOW,
    }


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


def test_source_add_rehearsal_migrations_preserve_prerequisite_order() -> None:
    applied = list(EXPECTED_APPLIED_MIGRATIONS)
    prerequisite_positions = [
        applied.index(name)
        for name in postgres_probe.SOURCE_ADD_PRE_V2_MIGRATIONS
    ]

    assert prerequisite_positions == sorted(prerequisite_positions)
    assert prerequisite_positions[-1] < applied.index(
        "86-research-lab-attested-v2-authority.sql"
    )
    assert applied.index(postgres_probe.GIT_TREE_AUTORESEARCH_MIGRATION) < (
        applied.index(postgres_probe.SOURCE_ADD_FUNCTIONAL_WORKFLOW_MIGRATION)
    )
    assert applied.index(postgres_probe.SOURCE_ADD_FUNCTIONAL_WORKFLOW_MIGRATION) < (
        applied.index("99-research-lab-v2-champion-settlement.sql")
    )
    assert applied.index(postgres_probe.SOURCE_ADD_FUNCTIONAL_WORKFLOW_MIGRATION) < (
        applied.index(postgres_probe.SOURCE_ADD_ADMISSION_CONTROL_MIGRATION)
    )


def test_credit_resume_rehearsal_uses_final_production_queue_guard() -> None:
    applied = list(EXPECTED_APPLIED_MIGRATIONS)
    ordered = (
        EVENT_PROJECTIONS_MIGRATION,
        QUEUE_CAPACITY_GUARD_MIGRATION,
        MAINTENANCE_PAUSE_MIGRATION,
        PAUSED_CAPACITY_AGING_MIGRATION,
        RESUME_REQUEUE_HOTKEY_GUARD_MIGRATION,
        HOTKEY_ACTIVE_LOOP_CAP_MIGRATION,
        postgres_probe.ATOMIC_CREDIT_RESUME_MIGRATION,
    )
    positions = [applied.index(name) for name in ordered]

    assert positions == sorted(positions)
    assert "CREATE SCHEMA extensions;" in ALLOCATION_MIGRATION_PREREQUISITES_SQL
    assert (
        "CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;"
        in ALLOCATION_MIGRATION_PREREQUISITES_SQL
    )
    assert "miner_hotkey TEXT NOT NULL" in ALLOCATION_MIGRATION_PREREQUISITES_SQL
    assert (
        "CREATE TABLE public.research_loop_run_queue_events"
        not in ALLOCATION_MIGRATION_PREREQUISITES_SQL
    )

    source_root = Path(__file__).resolve().parents[2]
    projections = (source_root / "scripts" / EVENT_PROJECTIONS_MIGRATION).read_text(
        encoding="utf-8"
    )
    final_guard = (
        source_root / "scripts" / HOTKEY_ACTIVE_LOOP_CAP_MIGRATION
    ).read_text(encoding="utf-8")
    assert (
        "CREATE TABLE IF NOT EXISTS public.research_loop_run_queue_events"
        in projections
    )
    assert (
        "CREATE OR REPLACE VIEW public.research_loop_run_queue_current"
        in projections
    )
    assert "hotkey_capacity_text" in final_guard
    assert "same_hotkey_count >= hotkey_capacity" in final_guard
    assert EXPECTED_ATOMIC_CREDIT_RESUME_EVIDENCE[
        "hotkey_capacity_guard_exercised"
    ] is True
    assert EXPECTED_ATOMIC_CREDIT_RESUME_EVIDENCE[
        "rpc_security_contract_valid"
    ] is True
    assert (
        GATEWAY_ATOMIC_CREDIT_RESUME_EVIDENCE
        == EXPECTED_ATOMIC_CREDIT_RESUME_EVIDENCE
    )


def test_event_projection_prerequisites_cover_migration_28_base_relations() -> None:
    source_root = Path(__file__).resolve().parents[2]
    persistence = (
        source_root / "scripts" / "28-research-lab-persistence-state.sql"
    ).read_text(encoding="utf-8")
    projections = (source_root / "scripts" / EVENT_PROJECTIONS_MIGRATION).read_text(
        encoding="utf-8"
    )
    prerequisite_fragments = (
        "CREATE TABLE public.research_loop_balance_ledger",
        "ledger_entry_id UUID PRIMARY KEY",
        "miner_hotkey TEXT NOT NULL",
        "ticket_id UUID REFERENCES public.research_loop_tickets(ticket_id)",
        "amount_microusd BIGINT NOT NULL",
        "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "CREATE TABLE public.research_weight_input_snapshots",
        "weight_input_snapshot_id UUID PRIMARY KEY",
        "snapshot_status TEXT NOT NULL CHECK",
    )
    normalized_prerequisites = " ".join(
        ALLOCATION_MIGRATION_PREREQUISITES_SQL.split()
    )
    normalized_persistence = " ".join(persistence.split())

    for fragment in prerequisite_fragments:
        normalized_fragment = " ".join(fragment.split())
        assert normalized_fragment in normalized_prerequisites
        assert normalized_fragment.replace(
            "CREATE TABLE public.", "CREATE TABLE IF NOT EXISTS public."
        ) in normalized_persistence
    assert "FROM public.research_loop_balance_ledger" in projections
    assert "FROM public.research_weight_input_snapshots" in projections


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
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "SOURCE_ROOT",
        Path(__file__).resolve().parents[2],
    )
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
    assert (
        response["result"]["specVersion"]
        == rehearsal_sitecustomize.RUNTIME_SPEC_VERSION
    )
    assert (
        rehearsal_sitecustomize._local_chain_signing_profile()["spec_version"]
        == response["result"]["specVersion"]
    )
    assert response["result"]["transactionVersion"] == 1

    request["params"] = [
        rehearsal_sitecustomize._block_hash(
            rehearsal_sitecustomize.CURRENT_BLOCK - 1
        )
    ]
    response = json.loads(
        rehearsal_sitecustomize._local_chain_rpc(
            json.dumps(request).encode(),
            archive=True,
        )
    )
    assert (
        response["result"]["specVersion"]
        == rehearsal_sitecustomize.RUNTIME_SPEC_VERSION
    )

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


def test_gateway_rehearsal_subtensor_stubs_return_exact_metagraph(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "SOURCE_ROOT",
        Path(__file__).resolve().parents[2],
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )

    sync_subtensor = rehearsal_sitecustomize._LocalSubtensor(network="finney")
    sync_metagraph = sync_subtensor.metagraph(71)
    async_subtensor = rehearsal_sitecustomize._LocalAsyncSubtensor(
        network="finney"
    )
    async_metagraph = asyncio.run(async_subtensor.metagraph(netuid=71))

    assert isinstance(sync_metagraph, rehearsal_sitecustomize._LocalMetagraph)
    assert isinstance(async_metagraph, rehearsal_sitecustomize._LocalMetagraph)
    assert sync_metagraph.netuid == async_metagraph.netuid == 71
    assert sync_metagraph.hotkeys == async_metagraph.hotkeys
    assert len(sync_metagraph.hotkeys) == sync_metagraph.n
    assert len(async_metagraph.hotkeys) == async_metagraph.n

    with pytest.raises(ValueError, match="local metagraph contract differs"):
        sync_subtensor.metagraph(72)
    with pytest.raises(ValueError, match="local metagraph contract differs"):
        asyncio.run(async_subtensor.metagraph(72))


def test_gateway_rehearsal_chain_adapter_supports_exact_epoch_close_search(
    tmp_path,
    monkeypatch,
) -> None:
    from gateway.tee.coordinator_chain_source_v2 import CoordinatorChainSourceV2
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
    from leadpoet_canonical.attested_v2 import (
        build_transport_attempt,
        sha256_bytes,
        sha256_json,
    )

    source_root = Path(__file__).resolve().parents[2]
    cutover = json.loads(
        (
            source_root / "config/stateful-epoch-cutover-sn71.json"
        ).read_text(encoding="utf-8")
    )
    chain_signing_profile = json.loads(
        (
            source_root
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize, "EVENT_PATH", tmp_path / "events.jsonl"
    )
    rehearsal_sitecustomize._BLOCK_NUMBERS_BY_HASH.clear()
    attempt_sequence = 0

    def execute(request):
        nonlocal attempt_sequence
        attempt_sequence += 1
        request_body = base64.b64decode(request["body_b64"], validate=True)
        body = rehearsal_sitecustomize._local_chain_rpc(
            request_body,
            archive=request["provider_id"] == "bittensor_archive",
        )
        response_hash = sha256_bytes(body)
        attempt = build_transport_attempt(
            request_id=f"{attempt_sequence:032x}",
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host=(
                "archive.chain.opentensor.ai"
                if request["provider_id"] == "bittensor_archive"
                else "entrypoint-finney.opentensor.ai"
            ),
            destination_port=443,
            path_hash="sha256:" + "4" * 64,
            nonsecret_headers_hash="sha256:" + "5" * 64,
            body_hash=sha256_bytes(request_body),
            credential_ref_hash="sha256:" + "6" * 64,
            retry_policy_hash=request["retry_policy_hash"],
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-25T00:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=response_hash,
            request_artifact_hash=sha256_json(
                {"request": attempt_sequence}
            ),
            response_artifact_hash=response_hash,
            tls_peer_chain_hash="sha256:" + "7" * 64,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-25T00:00:01Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "transport_attempt": attempt,
        }

    source = CoordinatorChainSourceV2(
        execute_provider=execute,
        retry_policy_hashes={
            "bittensor_chain": "sha256:" + "1" * 64,
            "bittensor_archive": "sha256:" + "2" * 64,
            "coingecko": "sha256:" + "3" * 64,
        },
        epoch_authority={
            "mode": "stateful_v1",
            "cutover": cutover,
            "chain_signing_profile": chain_signing_profile,
        },
        sleep=lambda _seconds: None,
    )
    settlement_epoch = rehearsal_sitecustomize._current_settlement_epoch_id() - 1
    result = source.read_stateful_epoch_close_weights(
        netuid=71,
        epoch_id=settlement_epoch,
        validator_hotkey=VALIDATOR_HOTKEY,
        context=ExecutionContextV2(
            job_id="rehearsal-stateful-close",
            purpose="research_lab.chain_weight_observation.v1",
            epoch_id=settlement_epoch,
        ),
    )

    assert result["epoch_id"] == settlement_epoch
    assert result["official_subnet_epoch_id"] == (
        rehearsal_sitecustomize.SUBNET_EPOCH_INDEX - 1
    )
    assert result["next_epoch_block"] == rehearsal_sitecustomize.LAST_EPOCH_BLOCK
    assert result["close_block"] == rehearsal_sitecustomize.LAST_EPOCH_BLOCK - 1
    assert result["validator_uid"] == 0
    assert result["latest_commit_source_epoch_id"] == settlement_epoch
    assert result["scheduled_reveal_source_epoch_id"] == settlement_epoch - 1
    assert result["scheduled_reveal_subnet_epoch_id"] == (
        rehearsal_sitecustomize.SUBNET_EPOCH_INDEX - 2
    )
    assert result["epoch_start_block"] == (
        rehearsal_sitecustomize._subnet_epoch_transition_block(
            rehearsal_sitecustomize.SUBNET_EPOCH_INDEX - 1
        )
    )
    assert result["reveal_window_start_block"] == (
        rehearsal_sitecustomize._subnet_epoch_transition_block(
            rehearsal_sitecustomize.SUBNET_EPOCH_INDEX - 1
        )
    )
    assert result["subnet_reveal_period_epochs"] == 1
    assert result["reveal_period_storage_override"] == 1
    assert result["reveal_period_runtime_spec_version"] == 452
    assert result["weights"] == [[0, 65_535], [1, 16_384]]

    legacy_epoch = int(cutover["last_legacy_epoch_id"])
    historical = source.read_historical_finalized_weights(
        netuid=71,
        epoch_id=legacy_epoch,
        validator_hotkey=VALIDATOR_HOTKEY,
        context=ExecutionContextV2(
            job_id="rehearsal-historical-layout",
            purpose="research_lab.legacy_finalized_allocation.v2",
            epoch_id=legacy_epoch,
        ),
    )
    assert historical["epoch_id"] == legacy_epoch
    assert historical["target_block"] == (legacy_epoch + 1) * 360 - 1
    assert historical["validator_uid"] == 0
    assert historical["weights"] == [[0, 65_535], [1, 16_384]]


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
    assert result["spec_version"] == (
        rehearsal_sitecustomize.RUNTIME_SPEC_VERSION
    )
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


@pytest.mark.parametrize(
    ("from_sha", "transition", "ancestor", "state", "mode"),
    [
        (
            "0dd3a385a23a3af0fa17210bfe02a39cc4023952",
            "forward",
            False,
            "true",
            "legacy_first_rollout",
        ),
        (
            "7ac1553e32d85d9babda3b3836f4c93cf92e6d60",
            "forward",
            True,
            "false",
            "post_rollout",
        ),
        ("8" * 40, "forward", True, "false", "post_rollout"),
    ],
)
def test_gateway_rehearsal_miner_state_is_derived_from_release_lineage(
    monkeypatch: pytest.MonkeyPatch,
    from_sha: str,
    transition: str,
    ancestor: bool,
    state: str,
    mode: str,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    monkeypatch.setenv("REHEARSAL_FROM_SHA", from_sha)
    monkeypatch.setenv("REHEARSAL_TRANSITION", transition)
    ancestry_calls: list[tuple[str, str]] = []

    def is_ancestor(parent: str, child: str, **_kwargs: Any) -> bool:
        ancestry_calls.append((parent, child))
        return ancestor

    monkeypatch.setattr(contract_adapter, "_git_commit_is_ancestor", is_ancestor)
    assert contract_adapter._initial_gateway_miner_submissions_state() == state
    assert contract_adapter._validate_gateway_miner_submissions_state(state) == mode
    if from_sha in {
        contract_adapter.LEGACY_GATEWAY_MINER_MAINTENANCE_FROM_SHA,
        contract_adapter.POST_GATEWAY_MINER_MAINTENANCE_FROM_SHA,
    }:
        assert ancestry_calls == []
    else:
        assert set(ancestry_calls) == {
            (
                contract_adapter.POST_GATEWAY_MINER_MAINTENANCE_FROM_SHA,
                from_sha,
            )
        }


def test_gateway_rehearsal_rollback_requires_durable_false_and_stays_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    monkeypatch.setenv("REHEARSAL_FROM_SHA", "9" * 40)
    monkeypatch.setenv("REHEARSAL_TRANSITION", "rollback")
    monkeypatch.setattr(
        contract_adapter,
        "_git_commit_is_ancestor",
        lambda *_args, **_kwargs: pytest.fail("rollback must use durable state"),
    )
    with pytest.raises(ValueError, match="rollback requires durable state"):
        contract_adapter._initial_gateway_miner_submissions_state()
    assert (
        contract_adapter._validate_gateway_miner_submissions_state("false")
        == "rollback"
    )
    with pytest.raises(ValueError, match="conflicts with release lineage"):
        contract_adapter._validate_gateway_miner_submissions_state("true")


def test_gateway_rehearsal_unknown_miner_lineage_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    monkeypatch.setenv("REHEARSAL_FROM_SHA", "a" * 40)
    monkeypatch.setenv("REHEARSAL_TRANSITION", "forward")
    monkeypatch.setattr(
        contract_adapter,
        "_git_commit_is_ancestor",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(ValueError, match="lineage is unknown"):
        contract_adapter._initial_gateway_miner_submissions_state()
    with pytest.raises(ValueError, match="lineage is unknown"):
        contract_adapter._validate_gateway_miner_submissions_state("false")


def test_gateway_rehearsal_post_rollout_state_is_false_with_zero_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.tee.disable_gateway_miner_submissions_secret import GATEWAY_SECRET_ID
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    state_root = tmp_path / "state"
    durable_root = tmp_path / "durable"
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", state_root)
    monkeypatch.setattr(
        rehearsal_sitecustomize, "EVENT_PATH", state_root / "events.jsonl"
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "DURABLE_STATE_ROOT", durable_root)
    monkeypatch.setenv(
        "REHEARSAL_FROM_SHA",
        contract_adapter.POST_GATEWAY_MINER_MAINTENANCE_FROM_SHA,
    )
    monkeypatch.setenv("REHEARSAL_TRANSITION", "forward")
    monkeypatch.setattr(
        contract_adapter,
        "_git_commit_is_ancestor",
        lambda ancestor, descendant, **_kwargs: (
            ancestor == contract_adapter.POST_GATEWAY_MINER_MAINTENANCE_FROM_SHA
            and descendant == contract_adapter.POST_GATEWAY_MINER_MAINTENANCE_FROM_SHA
        ),
    )
    client = rehearsal_sitecustomize._LocalInstanceRoleSession(
        botocore_session=object(),
        region_name="us-east-1",
    ).client("secretsmanager")
    current = client.get_secret_value(
        SecretId=GATEWAY_SECRET_ID,
        VersionStage="AWSCURRENT",
    )
    client.describe_secret(SecretId=GATEWAY_SECRET_ID)
    assert (
        json.loads(current["SecretString"])[
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"
        ]
        == "false"
    )
    operations = [
        row["operation"]
        for row in (
            json.loads(line)
            for line in (state_root / "events.jsonl").read_text().splitlines()
        )
        if row.get("boundary") == "aws_secretsmanager"
    ]
    assert set(operations) == {"describe_secret", "get_secret_value"}
    assert "put_secret_value" not in operations
    assert "update_secret_version_stage" not in operations


def test_gateway_rehearsal_instance_role_secret_transaction_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.tee.disable_gateway_miner_submissions_secret import (
        GATEWAY_SECRET_ID,
        _apply_gateway_miner_submissions_secret,
    )

    state_root = tmp_path / "state"
    durable_root = tmp_path / "durable"
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", state_root)
    monkeypatch.setattr(
        rehearsal_sitecustomize, "EVENT_PATH", state_root / "events.jsonl"
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize, "DURABLE_STATE_ROOT", durable_root
    )
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    monkeypatch.setenv(
        "REHEARSAL_FROM_SHA",
        "0dd3a385a23a3af0fa17210bfe02a39cc4023952",
    )
    monkeypatch.setenv("REHEARSAL_TRANSITION", "forward")
    session = rehearsal_sitecustomize._LocalInstanceRoleSession(
        botocore_session=object(),
        region_name="us-east-1",
    )
    assert session.get_credentials().method == "iam-role"
    assert session.client("sts").get_caller_identity()["Account"] == (
        "493765492819"
    )
    client = session.client("secretsmanager")
    initial = client.get_secret_value(
        SecretId=GATEWAY_SECRET_ID,
        VersionStage="AWSCURRENT",
    )
    assert (
        json.loads(initial["SecretString"])[
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"
        ]
        == "true"
    )
    journal = tmp_path / "private" / "transaction.json"
    journal.parent.mkdir(mode=0o700)
    result = _apply_gateway_miner_submissions_secret(
        secrets_client=client,
        expected_current_version_id=initial["VersionId"],
        recovery_journal_path=journal,
    )
    assert result["status"] == "updated"
    final = client.get_secret_value(
        SecretId=GATEWAY_SECRET_ID,
        VersionStage="AWSCURRENT",
    )
    assert (
        json.loads(final["SecretString"])[
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"
        ]
        == "false"
    )
    topology = client.describe_secret(SecretId=GATEWAY_SECRET_ID)[
        "VersionIdsToStages"
    ]
    assert topology[initial["VersionId"]] == ["AWSPREVIOUS"]
    assert topology[final["VersionId"]] == ["AWSCURRENT"]
    assert not journal.exists()
    assert not [path for path in durable_root.iterdir() if path.suffix == ".tmp"]
    json.loads(
        (durable_root / "gateway-secret-state.json").read_text(encoding="utf-8")
    )
    from tests.restart_rehearsal import contract_adapter

    monkeypatch.setattr(
        contract_adapter,
        "GATEWAY_SECRET_STATE_PATH",
        durable_root / "gateway-secret-state.json",
    )
    retry_state = contract_adapter._current_gateway_secret()[
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"
    ]
    assert retry_state == "false"
    assert (
        contract_adapter._validate_gateway_miner_submissions_state(retry_state)
        == "legacy_retry"
    )
    operations = {
        row["operation"]
        for row in (
            json.loads(line)
            for line in (state_root / "events.jsonl").read_text().splitlines()
        )
        if row.get("boundary") == "aws_secretsmanager"
    }
    assert operations == {
        "describe_secret",
        "get_secret_value",
        "put_secret_value",
        "update_secret_version_stage",
    }
    with pytest.raises(ValueError, match="version fence differs"):
        client.update_secret_version_stage(
            SecretId=GATEWAY_SECRET_ID,
            VersionStage="AWSCURRENT",
            MoveToVersionId=initial["VersionId"],
            RemoveFromVersionId=initial["VersionId"],
        )
    with pytest.raises(ValueError, match="service is unknown"):
        session.client("lambda")
    with pytest.raises(TypeError):
        client.describe_secret(SecretId=GATEWAY_SECRET_ID, Unknown=True)


def test_gateway_rehearsal_release_channel_is_singleton_and_version_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize, "EVENT_PATH", tmp_path / "events.jsonl"
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_release_channel",
        lambda commit: {"commit_sha": commit},
    )
    client = rehearsal_sitecustomize._LocalS3()
    key = f"attested-v2/releases/{COMMIT}/release-channel-v2.json"
    arguments = {
        "Bucket": "leadpoet-attested-v2-artifacts-493765492819",
        "Key": key,
    }
    history = client.list_object_versions(
        Bucket=arguments["Bucket"], Prefix=key, MaxKeys=1000
    )
    assert history["IsTruncated"] is False
    assert history["DeleteMarkers"] == []
    assert len(history["Versions"]) == 1
    version_id = history["Versions"][0]["VersionId"]
    head = client.head_object(**arguments, VersionId=version_id)
    response = client.get_object(**arguments, VersionId=version_id)
    assert response["Body"].read() == (
        b'{"commit_sha":"' + COMMIT.encode() + b'"}\n'
    )
    assert {
        (
            history["Versions"][0]["VersionId"],
            history["Versions"][0]["ETag"],
            history["Versions"][0]["Size"],
        ),
        (head["VersionId"], head["ETag"], head["ContentLength"]),
        (response["VersionId"], response["ETag"], response["ContentLength"]),
    } == {(version_id, head["ETag"], head["ContentLength"])}
    assert head["ObjectLockMode"] == "COMPLIANCE"
    assert head["ObjectLockRetainUntilDate"] > datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="version differs"):
        client.get_object(**arguments, VersionId="wrong-version")


def test_gateway_rehearsal_git_adapter_rewrites_only_github_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    fixture_remote = tmp_path / "origin.git"
    fixture_remote.mkdir()
    repository = tmp_path / "repo"
    git_directory = repository / ".git"
    git_directory.mkdir(parents=True)
    (git_directory / "config").write_text(
        '[core]\n\trepositoryformatversion = 0\n'
        '[remote "origin"]\n'
        '\turl = https://github.com/leadpoet/leadpoet.git\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        contract_adapter, "GITHUB_GIT_FIXTURE_REMOTE", fixture_remote
    )
    monkeypatch.setattr(
        contract_adapter,
        "GIT_FETCH_REPOSITORY_ROOTS",
        (tmp_path,),
    )
    observed: dict[str, Any] = {}

    def fake_boundary(**kwargs: Any) -> None:
        observed["boundary"] = kwargs

    def fake_exec(_executable: str, argv: list[str]) -> None:
        observed["argv"] = argv
        raise RuntimeError("exec captured")

    monkeypatch.setattr(contract_adapter, "_record_external_boundary", fake_boundary)
    monkeypatch.setattr(contract_adapter.os, "execv", fake_exec)
    with pytest.raises(RuntimeError, match="exec captured"):
        contract_adapter.command_git(
            [
                "-C",
                str(repository),
                "fetch",
                "--prune",
                "origin",
                "+refs/heads/main:refs/remotes/origin/main",
            ]
        )
    assert observed["argv"] == [
        contract_adapter.REAL_GIT,
        "-C",
        str(repository),
        "fetch",
        "--prune",
        str(fixture_remote),
        "+refs/heads/main:refs/remotes/origin/main",
    ]
    assert observed["boundary"]["boundary"] == "github_git_transport"
    assert observed["boundary"]["operation"] == "fetch"


def test_gateway_rehearsal_git_adapter_preserves_ref_less_origin_main_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    fixture_remote = tmp_path / "origin.git"
    fixture_remote.mkdir()
    repository = tmp_path / "repo"
    git_directory = repository / ".git"
    git_directory.mkdir(parents=True)
    (git_directory / "config").write_text(
        '[core]\n\trepositoryformatversion = 0\n'
        '[remote "origin"]\n'
        '\turl = https://github.com/leadpoet/leadpoet.git\n'
        '\tfetch = +refs/heads/*:refs/remotes/origin/*\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        contract_adapter, "GITHUB_GIT_FIXTURE_REMOTE", fixture_remote
    )
    monkeypatch.setattr(
        contract_adapter,
        "GIT_FETCH_REPOSITORY_ROOTS",
        (tmp_path,),
    )
    observed: dict[str, Any] = {}

    def fake_exec(_executable: str, argv: list[str]) -> None:
        observed["argv"] = argv
        raise RuntimeError("exec captured")

    monkeypatch.setattr(contract_adapter.os, "execv", fake_exec)
    monkeypatch.setattr(
        contract_adapter,
        "_record_external_boundary",
        lambda **kwargs: observed.setdefault("boundary", kwargs),
    )

    with pytest.raises(RuntimeError, match="exec captured"):
        contract_adapter.command_git(
            ["-C", str(repository), "fetch", "origin"]
        )

    assert observed["argv"] == [
        contract_adapter.REAL_GIT,
        "-C",
        str(repository),
        "fetch",
        str(fixture_remote),
        "+refs/heads/main:refs/remotes/origin/main",
    ]
    assert observed["boundary"]["boundary"] == "github_git_transport"


def test_gateway_rehearsal_git_adapter_keeps_candidate_fetch_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.restart_rehearsal import contract_adapter

    fixture_remote = tmp_path / "origin.git"
    fixture_remote.mkdir()
    repository = tmp_path / "repo"
    git_directory = repository / ".git"
    git_directory.mkdir(parents=True)
    (git_directory / "config").write_text(
        '[core]\n\trepositoryformatversion = 0\n'
        '[remote "origin"]\n'
        f'\turl = {fixture_remote}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        contract_adapter, "GITHUB_GIT_FIXTURE_REMOTE", fixture_remote
    )
    monkeypatch.setattr(
        contract_adapter,
        "GIT_FETCH_REPOSITORY_ROOTS",
        (tmp_path,),
    )
    observed: list[list[str]] = []

    def fake_exec(_executable: str, argv: list[str]) -> None:
        observed.append(argv)
        raise RuntimeError("exec captured")

    monkeypatch.setattr(contract_adapter.os, "execv", fake_exec)
    monkeypatch.setattr(
        contract_adapter,
        "_record_external_boundary",
        lambda **_kwargs: pytest.fail("local fetch is not an external boundary"),
    )
    with pytest.raises(RuntimeError, match="exec captured"):
        contract_adapter.command_git(
            ["-C", str(repository), "fetch", "origin", "main"]
        )
    assert observed == [
        [
            contract_adapter.REAL_GIT,
            "-C",
            str(repository),
            "fetch",
            str(fixture_remote),
            "main",
        ]
    ]


@pytest.mark.parametrize(
    ("origin_url", "fetch_tail", "reason"),
    [
        (
            "https://attacker.invalid/leadpoet.git",
            ["origin", "main"],
            "origin is not allowlisted",
        ),
        (
            "https://github.com/leadpoet/leadpoet.git",
            ["https://attacker.invalid/leadpoet.git", "main"],
            "arguments are unsafe",
        ),
        (
            "https://github.com/leadpoet/leadpoet.git",
            ["--upload-pack=/tmp/attacker", "origin", "main"],
            "arguments are unsafe",
        ),
    ],
)
def test_gateway_rehearsal_git_adapter_rejects_network_fetch_without_real_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    origin_url: str,
    fetch_tail: list[str],
    reason: str,
) -> None:
    from tests.restart_rehearsal import contract_adapter

    fixture_remote = tmp_path / "origin.git"
    fixture_remote.mkdir()
    repository = tmp_path / "repo"
    git_directory = repository / ".git"
    git_directory.mkdir(parents=True)
    (git_directory / "config").write_text(
        '[core]\n\trepositoryformatversion = 0\n'
        '[remote "origin"]\n'
        f'\turl = {origin_url}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        contract_adapter, "GITHUB_GIT_FIXTURE_REMOTE", fixture_remote
    )
    monkeypatch.setattr(
        contract_adapter,
        "GIT_FETCH_REPOSITORY_ROOTS",
        (tmp_path,),
    )
    exec_calls: list[list[str]] = []
    failures: list[str] = []

    def fake_exec(_executable: str, argv: list[str]) -> None:
        exec_calls.append(argv)
        raise AssertionError("unexpected real Git invocation")

    def fake_fail(_kind: str, _argv: list[str], message: str) -> int:
        failures.append(message)
        return 97

    monkeypatch.setattr(contract_adapter.os, "execv", fake_exec)
    monkeypatch.setattr(contract_adapter, "_fail", fake_fail)
    assert (
        contract_adapter.command_git(
            ["-C", str(repository), "fetch", *fetch_tail]
        )
        == 97
    )
    assert exec_calls == []
    assert failures == [f"candidate Git fetch {reason}"]


def test_gateway_rehearsal_git_adapter_allows_only_local_fixture_seed_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.restart_rehearsal import contract_adapter

    fixture_remote = tmp_path / "origin.git"
    fixture_source = tmp_path / "source"
    fixture_remote.mkdir()
    fixture_source.mkdir()
    monkeypatch.setattr(
        contract_adapter, "GITHUB_GIT_FIXTURE_REMOTE", fixture_remote
    )
    monkeypatch.setattr(
        contract_adapter, "LOCAL_GIT_FIXTURE_SOURCE", fixture_source
    )
    observed: list[list[str]] = []

    def fake_exec(_executable: str, argv: list[str]) -> None:
        observed.append(argv)
        raise RuntimeError("exec captured")

    monkeypatch.setattr(contract_adapter.os, "execv", fake_exec)
    argv = [
        f"--git-dir={fixture_remote}",
        "fetch",
        "-q",
        str(fixture_source),
        f"{'1' * 40}:refs/heads/main",
    ]
    with pytest.raises(RuntimeError, match="exec captured"):
        contract_adapter.command_git(argv)
    assert observed == [[contract_adapter.REAL_GIT, *argv]]


def test_gateway_rehearsal_git_adapter_keeps_non_fetch_local_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.restart_rehearsal import contract_adapter

    observed: list[list[str]] = []

    def fake_exec(_executable: str, argv: list[str]) -> None:
        observed.append(argv)
        raise RuntimeError("exec captured")

    monkeypatch.setattr(contract_adapter.os, "execv", fake_exec)
    argv = ["-C", str(tmp_path), "status", "--short"]
    with pytest.raises(RuntimeError, match="exec captured"):
        contract_adapter.command_git(argv)
    assert observed == [[contract_adapter.REAL_GIT, *argv]]


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
        "lab_arena_reward_basis_v1",
    } <= tables
    # Private competition state belongs to the standalone service, not gateway
    # startup. Keep the existing reward read boundary without coupling restarts.
    assert {
        "lab_arena_rounds",
        "lab_arena_submissions",
        "lab_arena_runs",
        "lab_arena_ledger",
    }.isdisjoint(tables)
    assert {
        "research_lab_source_add_claim_work",
    } <= rpcs
    assert {
        "lab_arena_current_daily_icp_set",
        "lab_arena_register_submission",
        "lab_arena_update_submission",
        "lab_arena_claim_assignment",
        "lab_arena_activate_reward",
        "lab_arena_schema_version_v1",
    }.isdisjoint(rpcs)
    assert {
        "research_lab_candidate_evaluation_current",
        "research_lab_candidate_promotion_events",
        "research_lab_public_benchmark_report_current",
    }.isdisjoint(tables)
    assert "research_lab_gateway_control_current" in (
        repository_root / "scripts/44-research-lab-maintenance-pause.sql"
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
    assert values[
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1"
    ].startswith("http://")
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
    assert success["RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1"].startswith(
        "http://"
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
    assert "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1" not in failure
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
        "REHEARSAL_SOURCE_ROOT": str(repository_root),
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


def test_sitecustomize_fails_closed_when_candidate_topology_is_unavailable(
    tmp_path,
) -> None:
    harness_root = Path(__file__).resolve().parent
    source_root = tmp_path / "missing-candidate-topology"
    source_root.mkdir()
    environment = {
        **os.environ,
        "PYTHONPATH": str(harness_root),
        "PYTHONSAFEPATH": "1",
        "REHEARSAL_CANDIDATE_SHA": COMMIT,
        "REHEARSAL_SCOPE": "exact",
        "REHEARSAL_SOURCE_ROOT": str(source_root),
        "REHEARSAL_STATE_ROOT": str(tmp_path / "state"),
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    safe_path_flag = ["-P"] if sys.version_info >= (3, 11) else []

    stdout_path = tmp_path / "missing-topology.stdout"
    stderr_path = tmp_path / "missing-topology.stderr"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            [
                sys.executable,
                *safe_path_flag,
                "-c",
                "print('candidate-main-reached')",
            ],
            cwd=tmp_path,
            env=environment,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=30,
        )

    observed_stdout = stdout_path.read_text(encoding="utf-8")
    observed_stderr = stderr_path.read_text(encoding="utf-8")
    assert result.returncode != 0
    assert "candidate-main-reached" not in observed_stdout
    assert (
        "SystemExit: exact rehearsal candidate topology bootstrap failed"
        in observed_stderr
    )


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
    tee_service.pending_checkpoint = None
    tee_service.sequence_counter = 0
    tee_service.checkpoint_count = 0
    tee_service.prev_checkpoint_root = None
    tee_service.checkpoint_start_time = datetime.utcnow()
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
        stats = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "get_buffer_stats",
            {},
        )
        assert stats["size"] == 1
        assert rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "get_buffer_size",
            {},
        ) == 1
        checkpoint = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "build_checkpoint",
            {},
        )
        assert checkpoint["status"] == "success"
        assert checkpoint["header"]["event_count"] == 1
        acknowledged = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "acknowledge_checkpoint",
            {
                "checkpoint_number": checkpoint["header"][
                    "checkpoint_number"
                ],
                "merkle_root": checkpoint["header"]["merkle_root"],
                "sequence_range": checkpoint["header"]["sequence_range"],
            },
        )
        assert acknowledged["status"] == "acknowledged"
        assert acknowledged["remaining_count"] == 0
        appended = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "append_event",
            {"event": {"event_type": "REHEARSAL_EVENT"}},
        )
        assert appended["status"] == "buffered"
        assert len(
            rehearsal_sitecustomize._handle_gateway_enclave_rpc(
                role,
                "get_buffer",
                {},
            )
        ) == 1
        payload = {"candidate_sha": COMMIT}
        signed = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "sign_transparency_event",
            {
                "event_type": "REHEARSAL_SIGNED_EVENT",
                "payload": payload,
                "payload_hash": hashlib.sha256(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            },
        )
        assert signed["buffer"]["status"] == "buffered"
        rejected_clear = (
            rehearsal_sitecustomize._handle_gateway_enclave_rpc(
                role,
                "clear_buffer",
                {},
            )
        )
        assert rejected_clear == {
            "status": "rejected",
            "reason": "checkpoint_acknowledgement_required",
            "cleared_count": 0,
        }
        assert rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            role,
            "get_buffer_size",
            {},
        ) == 2
        with pytest.raises(
            ValueError,
            match="local event buffer RPC params differ",
        ):
            rehearsal_sitecustomize._handle_gateway_enclave_rpc(
                role,
                "get_buffer_stats",
                {"unexpected": True},
            )
    finally:
        enclave_signer._reset_for_testing()
        tee_service.event_signer_initialization = None
        tee_service.event_buffer.clear()
        tee_service.pending_checkpoint = None
        tee_service.sequence_counter = 0
        tee_service.checkpoint_count = 0
        tee_service.prev_checkpoint_root = None
        tee_service.checkpoint_start_time = datetime.utcnow()


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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter
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
        images = {}
        assert contract_adapter._docker_load(
            normalized,
            images,
            {},
        ) == [f"rehearsal:{role}"]
        assert images[f"rehearsal:{role}"]["rootfs_layers"]
        assert all(
            re.fullmatch(r"sha256:[0-9a-f]{64}", layer)
            for layer in images[f"rehearsal:{role}"]["rootfs_layers"]
        )


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
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    channel = rehearsal_sitecustomize._release_channel(COMMIT)
    gateway = channel["gateway_release_manifest"]
    assert gateway["schema_version"] == "leadpoet.gateway_local_release.v1"
    assert gateway["verified_build_count"] == len(ROLE_SPECS)
    for role, release in gateway["roles"].items():
        assert release["verified_build_count"] == 1
        assert release["normalized_image_hash"] == normalized_image_id(
            COMMIT,
            role,
        )
    validator_manifest = channel["validator_release_manifest"]
    assert validator_manifest["schema_version"] == (
        "leadpoet.validator_local_release.v1"
    )
    assert validator_manifest["verified_build_count"] == 1
    validator = validator_manifest["release"]
    assert validator["normalized_image_hash"] == normalized_image_id(
        COMMIT,
        "validator_weights",
    )
    assert validator["eif_hash"] == eif_hash(COMMIT, "validator_weights")


def test_release_channel_preserves_the_exact_historical_role_topology(
    monkeypatch,
    tmp_path,
) -> None:
    from gateway.tee.release_manifest_v2 import (
        HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    )
    from gateway.tee.topology import ROLE_SPECS

    historical_commit = "2" * 40
    role_names = {*ROLE_SPECS, "gateway_autoresearch"}
    roles = {}
    for role in role_names:
        historical_role = role == "gateway_autoresearch"
        roles[role] = {
            "build_identity_hash": "sha256:" + "1" * 64,
            "commit_sha": historical_commit,
            "dependency_lock_hash": "sha256:" + "2" * 64,
            "dockerfile_hash": "sha256:" + "3" * 64,
            "eif_hash": (
                "sha256:" + "5" * 64
                if historical_role
                else eif_hash(historical_commit, role)
            ),
            "execution_manifest_hash": "sha256:" + "4" * 64,
            "normalized_image_hash": (
                "sha256:" + "9" * 64
                if historical_role
                else normalized_image_id(
                    historical_commit,
                    role,
                )
            ),
            "pcr0": pcr0(historical_commit),
            "service_role": role,
            "source_manifest_hash": "sha256:" + "6" * 64,
            "topology_hash": HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        }
    (tmp_path / "release-build-input.json").write_text(
        json.dumps(
            {
                "commit_sha": historical_commit,
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

    channel = rehearsal_sitecustomize._release_channel(historical_commit)

    gateway = channel["gateway_release_manifest"]
    assert gateway["topology_hash"] == HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
    assert set(gateway["roles"]) == role_names
    assert gateway["verified_build_count"] == 18

    roles["gateway_autoresearch"]["service_role"] = "gateway_scoring"
    (tmp_path / "release-build-input.json").write_text(
        json.dumps(
            {
                "commit_sha": historical_commit,
                "gateway_roles": roles,
                "validator_app_manifest_hash": "sha256:" + "7" * 64,
                "validator_dependency_lock_hash": "sha256:" + "8" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="release role service mismatch"):
        rehearsal_sitecustomize._release_channel(historical_commit)


def test_release_channel_resolves_both_transition_release_inputs(
    monkeypatch,
    tmp_path,
) -> None:
    historical_commit = "2" * 40
    state_root = tmp_path / "state"
    from_seed_root = tmp_path / "from"
    durable_seed_root = tmp_path / "durable"
    for path in (state_root, from_seed_root, durable_seed_root):
        path.mkdir()
    current = {"commit_sha": COMMIT, "gateway_roles": {"current": {}}}
    historical = {
        "commit_sha": historical_commit,
        "gateway_roles": {"historical": {}},
    }
    (state_root / "release-build-input.json").write_text(
        json.dumps(current, sort_keys=True),
        encoding="utf-8",
    )
    (durable_seed_root / "release-build-input.json").write_text(
        json.dumps(current, sort_keys=True),
        encoding="utf-8",
    )
    (from_seed_root / "release-build-input.json").write_text(
        json.dumps(historical, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", state_root)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "FROM_FIXTURE_SEED_ROOT",
        from_seed_root,
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "DURABLE_SCHEMA_SEED_ROOT",
        durable_seed_root,
    )

    assert (
        rehearsal_sitecustomize._release_build_input_for_commit(COMMIT)
        == current
    )
    assert (
        rehearsal_sitecustomize._release_build_input_for_commit(
            historical_commit
        )
        == historical
    )
    with pytest.raises(ValueError, match="commit is unavailable"):
        rehearsal_sitecustomize._release_build_input_for_commit("3" * 40)

    conflicting = {**current, "gateway_roles": {"different": {}}}
    (durable_seed_root / "release-build-input.json").write_text(
        json.dumps(conflicting, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inputs conflict"):
        rehearsal_sitecustomize._release_build_input_for_commit(COMMIT)


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

        # A worker may be terminated after sending an RPC but before reading
        # the response. The persistent role process must isolate that client
        # disconnect and remain available to the resumed worker.
        abandoned = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        abandoned.connect(str(socket_path))
        abandoned_request = json.dumps(
            {
                "method": "unknown_rpc",
                "params": {"padding": "x" * (8 * 1024 * 1024)},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        abandoned.sendall(
            len(abandoned_request).to_bytes(4, "big")
            + abandoned_request
        )
        abandoned.shutdown(socket.SHUT_RDWR)
        abandoned.close()
        for _attempt in range(100):
            if service.poll() is not None:
                stdout, stderr = service.communicate(timeout=5)
                raise AssertionError(
                    "persistent gateway enclave exited after a client "
                    "disconnect: %s %s" % (stdout, stderr)
                )
            time.sleep(0.02)

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


def test_local_gateway_kms_boundary_unwraps_boot_credential_in_candidate(
    monkeypatch,
    tmp_path,
) -> None:
    from gateway.tee.kms_recipient_v2 import KMSRecipientV2
    from gateway.tee.provider_broker_v2 import credential_reference_hash
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

    slot = "openrouter"
    credential = "sanitized-boot-credential"
    recipient_authority = KMSRecipientV2(
        boot_identity_supplier=lambda: {
            "boot_identity_hash": "sha256:" + "3" * 64,
        },
        expected_credential_ref_hashes={
            slot: credential_reference_hash(credential),
        },
        attestation_supplier=attestation_supplier,
    )
    recipient = recipient_authority.recipient_request(slot)
    kms = rehearsal_sitecustomize._LocalKMS()
    key_id = "arn:aws:kms:us-east-1:111122223333:key/rehearsal"
    context = {
        "leadpoet:purpose": "gateway-boot-credential-v2",
        "leadpoet:slot": slot,
    }
    encrypted = kms.encrypt(
        KeyId=key_id,
        Plaintext=credential.encode("utf-8"),
        EncryptionContext=context,
    )
    decrypted = kms.decrypt(
        CiphertextBlob=encrypted["CiphertextBlob"],
        EncryptionContext=context,
        Recipient={
            "KeyEncryptionAlgorithm": "RSAES_OAEP_SHA_256",
            "AttestationDocument": base64.b64decode(
                recipient["attestation_document_b64"],
                validate=True,
            ),
        },
    )

    assert decrypted["KeyId"] == key_id
    assert recipient_authority.unwrap_credential(
        slot=slot,
        ciphertext_for_recipient_b64=base64.b64encode(
            decrypted["CiphertextForRecipient"]
        ).decode("ascii"),
    ) == credential


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

    monkeypatch.setenv("REHEARSAL_COMPONENT", "validator")
    validator_listener = rehearsal_sitecustomize._LocalVsock(
        40,
        socket.SOCK_STREAM,
    )
    validator_listener.bind((0xFFFFFFFF, 5002))
    with pytest.raises(ValueError, match="listener backlog"):
        validator_listener.listen(64)
    validator_listener.listen(8)
    validator_listener.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(events) == 2
    for event, port in zip(events, (5001, 5002)):
        assert event == {
            "at_ns": event["at_ns"],
            "boundary": "nitro_enclaves",
            "fixture_authenticity": "production_shaped_sanitized",
            "implementation": "external_boundary",
            "kind": "local-chain-boundary",
            "operation": "enclave_listener",
            "port": port,
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
            "target_seconds": 600,
        },
        "release": {
            "cpus": "6",
            "memory": "7g",
            "epochs": 100,
            "fault_matrix": True,
            "target_seconds": None,
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


def test_rehearsal_local_image_inspection_is_bounded_and_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = "leadpoet-test:exact"
    calls: list[tuple[list[str], bool, float | None]] = []
    payload = {"Id": "sha256:" + "1" * 64}

    def fake_run(command, *, capture=False, timeout_seconds=None):
        calls.append((list(command), capture, timeout_seconds))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps([payload]),
            stderr="",
        )

    monkeypatch.setattr(rehearsal, "_run", fake_run)

    assert rehearsal._inspect_local_image(reference) == payload
    assert calls == [
        (
            ["docker", "image", "inspect", reference],
            True,
            rehearsal._LOCAL_IMAGE_OPERATION_TIMEOUT_SECONDS,
        )
    ]


@pytest.mark.parametrize("payload", ("{}", "[]", "[1]", "not-json"))
def test_rehearsal_local_image_inspection_rejects_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    monkeypatch.setattr(
        rehearsal,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=payload, stderr=""
        ),
    )

    with pytest.raises(SystemExit, match="inspection is"):
        rehearsal._inspect_local_image("leadpoet-test:malformed")


def test_rehearsal_local_image_inspection_distinguishes_absence_from_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = "leadpoet-test:missing"

    def missing(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            1,
            [],
            stderr=f"Error response from daemon: No such image: {reference}\n",
        )

    monkeypatch.setattr(rehearsal, "_run", missing)
    assert rehearsal._inspect_local_image(reference) is None

    def denied(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            1,
            [],
            stderr="permission denied\n",
        )

    monkeypatch.setattr(rehearsal, "_run", denied)
    with pytest.raises(SystemExit, match="unable to inspect"):
        rehearsal._inspect_local_image(reference)


def test_rehearsal_local_image_inspection_preserves_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout = subprocess.TimeoutExpired([], 1.0)
    monkeypatch.setattr(
        rehearsal,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(timeout),
    )

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        rehearsal._inspect_local_image("leadpoet-test:timeout")
    assert raised.value is timeout


def test_rehearsal_local_base_alias_binds_exact_pinned_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = rehearsal.REHEARSAL_BASE_IMAGES["linux/arm64"]
    image_id = "sha256:" + "1" * 64
    image = {
        "Architecture": "arm64",
        "Id": image_id,
        "Os": "linux",
        "RepoDigests": [pinned],
    }
    installed: dict[str, dict[str, Any]] = {pinned: image}
    commands: list[list[str]] = []

    monkeypatch.setattr(
        rehearsal,
        "_inspect_local_image",
        lambda reference: installed.get(reference),
    )

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        assert command[:3] == ["docker", "image", "tag"]
        alias = str(command[4])
        installed[alias] = dict(image)

    monkeypatch.setattr(rehearsal, "_run", fake_run)

    actual_alias = rehearsal._prepare_local_rehearsal_base_image("linux/arm64")

    alias = (
        "leadpoet-local-restart-rehearsal-base:arm64-"
        + pinned.rsplit("@sha256:", 1)[1]
    )
    assert actual_alias == alias
    assert commands == [["docker", "image", "tag", pinned, alias]]


def test_rehearsal_local_base_alias_pulls_absent_pinned_image_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = rehearsal.REHEARSAL_BASE_IMAGES["linux/arm64"]
    image = {
        "Architecture": "arm64",
        "Id": "sha256:" + "1" * 64,
        "Os": "linux",
        "RepoDigests": [pinned],
    }
    installed: dict[str, dict[str, Any]] = {}
    commands: list[tuple[list[str], float | None]] = []

    monkeypatch.setattr(
        rehearsal,
        "_inspect_local_image",
        lambda reference: installed.get(reference),
    )

    def fake_run(command, *, timeout_seconds=None, **_kwargs):
        commands.append((list(command), timeout_seconds))
        if command[1] == "pull":
            installed[pinned] = dict(image)
        else:
            alias = str(command[4])
            installed[alias] = dict(image)

    monkeypatch.setattr(rehearsal, "_run", fake_run)

    alias = rehearsal._prepare_local_rehearsal_base_image("linux/arm64")

    assert alias.startswith("leadpoet-local-restart-rehearsal-base:arm64-")
    assert commands[0] == (
        ["docker", "pull", "--platform", "linux/arm64", pinned],
        rehearsal._REHEARSAL_BASE_PULL_TIMEOUT_SECONDS,
    )
    assert commands[1][0][:3] == ["docker", "image", "tag"]
    assert commands[1][1] == rehearsal._LOCAL_IMAGE_OPERATION_TIMEOUT_SECONDS


def test_rehearsal_local_base_alias_reuses_exact_existing_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = rehearsal.REHEARSAL_BASE_IMAGES["linux/arm64"]
    digest = pinned.rsplit("@sha256:", 1)[1]
    alias = f"leadpoet-local-restart-rehearsal-base:arm64-{digest}"
    image = {
        "Architecture": "arm64",
        "Id": "sha256:" + "1" * 64,
        "Os": "linux",
        "RepoDigests": [pinned],
    }
    monkeypatch.setattr(
        rehearsal,
        "_inspect_local_image",
        lambda reference: image if reference in {pinned, alias} else None,
    )
    monkeypatch.setattr(
        rehearsal,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("exact alias must be reused"),
    )

    assert rehearsal._prepare_local_rehearsal_base_image("linux/arm64") == alias


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("Architecture", "amd64", "platform differs"),
        ("RepoDigests", [], "repository digest differs"),
        ("Id", "not-a-digest", "image ID is malformed"),
    ),
)
def test_rehearsal_local_base_alias_rejects_invalid_pinned_image(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    pinned = rehearsal.REHEARSAL_BASE_IMAGES["linux/arm64"]
    image = {
        "Architecture": "arm64",
        "Id": "sha256:" + "1" * 64,
        "Os": "linux",
        "RepoDigests": [pinned],
    }
    image[field] = value
    monkeypatch.setattr(
        rehearsal,
        "_inspect_local_image",
        lambda reference: image if reference == pinned else None,
    )
    monkeypatch.setattr(
        rehearsal,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("invalid base must not be tagged"),
    )

    with pytest.raises(SystemExit, match=message):
        rehearsal._prepare_local_rehearsal_base_image("linux/arm64")


def test_rehearsal_local_base_alias_rejects_existing_wrong_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = rehearsal.REHEARSAL_BASE_IMAGES["linux/arm64"]
    pinned_image = {
        "Architecture": "arm64",
        "Id": "sha256:" + "1" * 64,
        "Os": "linux",
        "RepoDigests": [pinned],
    }

    def inspect(reference: str):
        if reference == pinned:
            return pinned_image
        return {**pinned_image, "Id": "sha256:" + "2" * 64}

    monkeypatch.setattr(rehearsal, "_inspect_local_image", inspect)
    monkeypatch.setattr(
        rehearsal,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("wrong alias must not be replaced"),
    )

    with pytest.raises(SystemExit, match="alias differs"):
        rehearsal._prepare_local_rehearsal_base_image("linux/arm64")


def test_rehearsal_local_base_alias_rejects_wrong_post_tag_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = rehearsal.REHEARSAL_BASE_IMAGES["linux/arm64"]
    pinned_image = {
        "Architecture": "arm64",
        "Id": "sha256:" + "1" * 64,
        "Os": "linux",
        "RepoDigests": [pinned],
    }
    alias_reads = 0

    def inspect(reference: str):
        nonlocal alias_reads
        if reference == pinned:
            return pinned_image
        alias_reads += 1
        if alias_reads == 1:
            return None
        return {**pinned_image, "Id": "sha256:" + "2" * 64}

    monkeypatch.setattr(rehearsal, "_inspect_local_image", inspect)
    monkeypatch.setattr(rehearsal, "_run", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit, match="not installed exactly"):
        rehearsal._prepare_local_rehearsal_base_image("linux/arm64")


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
    local_base = "leadpoet-local-rehearsal-base:amd64-exact"
    buildx = Path("/validated/docker-buildx")

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
            else (
                b"same-scoring-lock"
                if path == "gateway/tee/requirements-scoring-py39.lock"
                else b""
            )
        ),
    )
    monkeypatch.setattr(
        rehearsal,
        "_run",
        lambda argv, *, cwd=None, **_kwargs: commands.append(
            (list(argv), cwd)
        ),
    )
    monkeypatch.setattr(
        rehearsal,
        "_prepare_local_rehearsal_base_image",
        lambda docker_platform: (
            local_base
            if docker_platform == "linux/amd64"
            else pytest.fail("unexpected Docker platform")
        ),
    )
    rehearsal._build_image(
        "leadpoet-test:platform",
        harness_sha="a" * 40,
        docker_platform="linux/amd64",
        buildx_executable=buildx,
        wheelhouse_shas=("b" * 40, "a" * 40),
    )

    assert commands == [
        (
            [
                str(buildx),
                "build",
                "--builder",
                "default",
                "--output",
                "type=docker,compression=zstd,compression-level=1,"
                "force-compression=true",
                "--progress=plain",
                "--pull=false",
                "--platform",
                "linux/amd64",
                "--build-arg",
                "REHEARSAL_BASE_IMAGE=" + local_base,
                "--build-arg",
                "REHEARSAL_SCORING_LOCK_SHA256="
                + hashlib.sha256(b"same-scoring-lock").hexdigest(),
                "--tag",
                "leadpoet-test:platform",
                ".",
            ],
            tmp_path,
        )
    ]
    assert sorted(path.name for path in (tmp_path / "scoring-locks").iterdir()) == [
        hashlib.sha256(b"same-scoring-lock").hexdigest() + ".lock",
    ]
    assert json.loads(
        (tmp_path / "scoring-lock-aliases.json").read_text(encoding="utf-8")
    ) == {
        "a" * 40: hashlib.sha256(b"same-scoring-lock").hexdigest(),
        "b" * 40: hashlib.sha256(b"same-scoring-lock").hexdigest(),
    }


def test_rehearsal_candidate_identity_does_not_invalidate_stable_dependencies():
    dockerfile = (
        Path(__file__).resolve().parent / "Dockerfile"
    ).read_text(encoding="utf-8")

    candidate_arg = dockerfile.index("ARG REHEARSAL_SCORING_LOCK_SHA256")
    assert dockerfile.index("dnf install") < candidate_arg
    assert dockerfile.index("COPY requirements.txt") < candidate_arg
    assert dockerfile.index("python3.11 -m pip install") < candidate_arg
    assert dockerfile.index("COPY scoring-locks/") < candidate_arg
    assert dockerfile.index("python3.11 -m pip download") < candidate_arg
    assert dockerfile.index(
        "python3.11 /opt/leadpoet/prepare_external_artifacts.py"
    ) < candidate_arg
    assert candidate_arg < dockerfile.index("COPY scoring-lock-aliases.json")
    assert candidate_arg < dockerfile.index("COPY harness/ /harness/")
    assert dockerfile.count("COPY harness/ /harness/") == 1
    assert "COPY harness/prepare_scoring_wheelhouse_aliases.py" not in dockerfile
    assert "cp /harness/prepare_scoring_wheelhouse_aliases.py" in dockerfile
    assert (
        "python3.11 /opt/leadpoet/prepare_scoring_wheelhouse_aliases.py"
        in dockerfile
    )
    assert candidate_arg < dockerfile.index(
        'scoring-wheelhouses/${REHEARSAL_SCORING_LOCK_SHA256}'
    )
    assert 'sha256sum "${lock}"' in dockerfile
    assert dockerfile.count("\nENV ") == 1
    for phase in (
        "system-packages",
        "python-dependencies",
        "scoring-wheelhouses",
        "external-artifacts",
        "image-finalization",
    ):
        for status in ("started", "passed", "failed"):
            assert dockerfile.count(
                f"REHEARSAL_IMAGE_BUILD_PHASE phase={phase} status={status}"
            ) == 1
    assert dockerfile.count('phase_status="$?"') == 5
    assert dockerfile.count('exit "${phase_status}"') == 5
    for name in (
        "HOME=/home/ec2-user",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "PIP_NO_INDEX=1",
    ):
        assert name in dockerfile
    assert (
        "tests/restart_rehearsal/prepare_scoring_wheelhouse_aliases.py"
        in rehearsal.COMMITTED_HARNESS_PATHS
    )


def test_rehearsal_build_maps_both_transition_commits_to_exact_lock_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_sha = "1" * 40
    candidate_sha = "2" * 40
    from_lock = b"from-lock"
    candidate_lock = b"candidate-lock"

    @contextmanager
    def temporary_directory(*, prefix: str):
        assert prefix == "leadpoet-restart-image-"
        yield str(tmp_path)

    def git_file(sha: str, path: str) -> bytes:
        if path == "tests/restart_rehearsal/Dockerfile":
            return (Path(__file__).resolve().parent / "Dockerfile").read_bytes()
        if path == "gateway/tee/requirements-scoring-py39.lock":
            return from_lock if sha == from_sha else candidate_lock
        return b""

    monkeypatch.setattr(
        rehearsal.tempfile,
        "TemporaryDirectory",
        temporary_directory,
    )
    monkeypatch.setattr(rehearsal, "_git_file", git_file)
    monkeypatch.setattr(rehearsal, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        rehearsal,
        "_prepare_local_rehearsal_base_image",
        lambda _docker_platform: "leadpoet-local-rehearsal-base:exact",
    )
    rehearsal._build_image(
        "leadpoet-test:transition-locks",
        harness_sha=candidate_sha,
        docker_platform="linux/amd64",
        buildx_executable=Path("/validated/docker-buildx"),
        wheelhouse_shas=(from_sha, candidate_sha),
    )

    aliases = json.loads(
        (tmp_path / "scoring-lock-aliases.json").read_text(encoding="utf-8")
    )
    assert aliases == {
        from_sha: hashlib.sha256(from_lock).hexdigest(),
        candidate_sha: hashlib.sha256(candidate_lock).hexdigest(),
    }
    assert {
        path.name for path in (tmp_path / "scoring-locks").iterdir()
    } == {
        hashlib.sha256(from_lock).hexdigest() + ".lock",
        hashlib.sha256(candidate_lock).hexdigest() + ".lock",
    }


def test_scoring_wheelhouse_aliases_resolve_n_minus_one_and_candidate(
    tmp_path: Path,
) -> None:
    from tests.restart_rehearsal.prepare_scoring_wheelhouse_aliases import (
        install_aliases,
    )

    from_sha = "1" * 40
    candidate_sha = "2" * 40
    from_digest = "a" * 64
    candidate_digest = "b" * 64
    root = tmp_path / "scoring-wheelhouses"
    (root / from_digest).mkdir(parents=True)
    (root / candidate_digest).mkdir()
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text(
        json.dumps(
            {
                from_sha: from_digest,
                candidate_sha: candidate_digest,
            }
        ),
        encoding="utf-8",
    )

    install_aliases(root=root, aliases_path=aliases_path)

    assert (root / from_sha).resolve() == (root / from_digest).resolve()
    assert (root / candidate_sha).resolve() == (root / candidate_digest).resolve()


def test_host_fixture_preparation_consumes_both_transition_wheelhouse_aliases(
    tmp_path: Path,
) -> None:
    from tests.restart_rehearsal.prepare_host_fixtures import (
        _prepare_offline_root,
    )
    from tests.restart_rehearsal.prepare_scoring_wheelhouse_aliases import (
        install_aliases,
    )

    from_sha = "1" * 40
    candidate_sha = "2" * 40
    from_digest = "a" * 64
    candidate_digest = "b" * 64
    wheelhouses = tmp_path / "scoring-wheelhouses"
    for digest, marker in (
        (from_digest, b"n-minus-one"),
        (candidate_digest, b"candidate"),
    ):
        target = wheelhouses / digest
        target.mkdir(parents=True)
        (target / "locked.whl").write_bytes(marker)
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text(
        json.dumps(
            {
                from_sha: from_digest,
                candidate_sha: candidate_digest,
            }
        ),
        encoding="utf-8",
    )
    install_aliases(root=wheelhouses, aliases_path=aliases_path)

    source_root = tmp_path / "source"
    (source_root / "gateway/tee").mkdir(parents=True)
    (source_root / "validator_tee").mkdir()
    (source_root / "gateway/tee/runsc-runtime.lock.json").write_text(
        json.dumps({"artifact_filename": "runsc.bin"}),
        encoding="utf-8",
    )
    (source_root / "validator_tee/runtime-artifacts-v2.lock.json").write_text(
        json.dumps({"artifacts": {"runtime": {"filename": "runtime.bin"}}}),
        encoding="utf-8",
    )
    external = tmp_path / "external-artifacts"
    external.mkdir()
    (external / "runsc.bin").write_bytes(b"runsc")
    (external / "runtime.bin").write_bytes(b"runtime")

    observed = {}
    for commit in (from_sha, candidate_sha):
        destination = tmp_path / ("offline-" + commit)
        destination.mkdir()
        _prepare_offline_root(
            source_root=source_root,
            destination=destination,
            commit=commit,
            scoring_wheelhouse_root=wheelhouses,
            external_artifact_root=external,
        )
        observed[commit] = (
            destination / "scoring-wheelhouse-py39/locked.whl"
        ).read_bytes()
        assert (destination / "runsc.bin").read_bytes() == b"runsc"
        assert (
            destination / "validator-runtime/runtime.bin"
        ).read_bytes() == b"runtime"
    assert observed == {
        from_sha: b"n-minus-one",
        candidate_sha: b"candidate",
    }


def test_host_fixture_preparation_removes_release_builder_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.restart_rehearsal import prepare_host_fixtures

    destination = tmp_path / "state"
    destination.mkdir()
    retained = destination / "release-build-input.json"

    def prepare(*, commit: str, destination: Path) -> dict:
        assert commit == "1" * 40
        scratch = destination / "release-builder/source"
        scratch.mkdir(parents=True)
        (scratch / "large-disposable-input").write_bytes(b"scratch")
        retained.write_text("{}\n", encoding="utf-8")
        return {"commit_sha": commit}

    monkeypatch.setattr(prepare_host_fixtures, "_release_build_input", prepare)

    result = prepare_host_fixtures._release_build_input_without_scratch(
        commit="1" * 40,
        destination=destination,
    )

    assert result == {"commit_sha": "1" * 40}
    assert retained.read_text(encoding="utf-8") == "{}\n"
    assert not (destination / "release-builder").exists()


def test_host_fixture_failure_removes_scratch_without_masking_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.restart_rehearsal import prepare_host_fixtures

    destination = tmp_path / "state"
    destination.mkdir()
    expected = RuntimeError("fixture generation failed")

    def fail(*, commit: str, destination: Path) -> dict:
        assert commit == "2" * 40
        scratch = destination / "release-builder/offline"
        scratch.mkdir(parents=True)
        (scratch / "partial").write_bytes(b"partial")
        raise expected

    monkeypatch.setattr(prepare_host_fixtures, "_release_build_input", fail)

    with pytest.raises(RuntimeError) as raised:
        prepare_host_fixtures._release_build_input_without_scratch(
            commit="2" * 40,
            destination=destination,
        )

    assert raised.value is expected
    assert not (destination / "release-builder").exists()


@pytest.mark.parametrize(
    ("aliases", "target_digest", "message"),
    (
        ({"../escape": "a" * 64}, "a" * 64, "commit is invalid"),
        ({"1" * 40: "not-a-digest"}, "a" * 64, "digest is invalid"),
        ({"1" * 40: "b" * 64}, "a" * 64, "target is unavailable"),
    ),
)
def test_scoring_wheelhouse_aliases_fail_closed(
    tmp_path: Path,
    aliases: dict[str, str],
    target_digest: str,
    message: str,
) -> None:
    from tests.restart_rehearsal.prepare_scoring_wheelhouse_aliases import (
        install_aliases,
    )

    root = tmp_path / "scoring-wheelhouses"
    (root / target_digest).mkdir(parents=True)
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text(json.dumps(aliases), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        install_aliases(root=root, aliases_path=aliases_path)


def test_rehearsal_build_stages_compact_weight_readiness_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]

    @contextmanager
    def temporary_directory(*, prefix: str):
        assert prefix == "leadpoet-restart-image-"
        yield str(tmp_path)

    monkeypatch.setattr(
        rehearsal.tempfile,
        "TemporaryDirectory",
        temporary_directory,
    )
    monkeypatch.setattr(
        rehearsal,
        "_git_file",
        lambda _sha, path: (repository_root / path).read_bytes(),
    )
    monkeypatch.setattr(rehearsal, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        rehearsal,
        "_prepare_local_rehearsal_base_image",
        lambda _docker_platform: "leadpoet-local-rehearsal-base:exact",
    )
    candidate_sha = "a" * 40
    rehearsal._build_image(
        "leadpoet-test:compact-weight-import",
        harness_sha=candidate_sha,
        docker_platform="linux/amd64",
        buildx_executable=Path("/validated/docker-buildx"),
        wheelhouse_shas=(candidate_sha,),
    )

    staged_harness = tmp_path / "harness"
    staged_compact_runner = staged_harness / "compact_weight_joined_runner.py"
    staged_dependency = staged_harness / "weight_readiness_runner.py"
    assert staged_compact_runner.is_file()
    assert staged_dependency.is_file()

    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    environment.update(
        {
            "REHEARSAL_CANDIDATE_SHA": candidate_sha,
            "REHEARSAL_SCOPE": "exact",
            "REHEARSAL_SOURCE_ROOT": str(repository_root),
            "REHEARSAL_STATE_ROOT": str(tmp_path / "state"),
        }
    )
    # Match /source:/harness resolution without auto-loading the unrelated
    # process-wide sitecustomize adapters during this import-only probe.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "from pathlib import Path",
                    "sys.path[:0] = "
                    f"[{str(repository_root)!r}, {str(staged_harness)!r}]",
                    "import compact_weight_joined_runner as compact",
                    "import weight_readiness_runner as readiness",
                    "assert Path(compact.__file__).resolve() == "
                    f"Path({str(staged_compact_runner)!r}).resolve()",
                    "assert 'weight_readiness_runner' in "
                    "compact._allocation_guard.__code__.co_names",
                    "assert Path(readiness.__file__).resolve() == "
                    f"Path({str(staged_dependency)!r}).resolve()",
                )
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0


def test_rehearsal_does_not_recreate_removed_acceptance_corpus() -> None:
    harness_root = Path(__file__).resolve().parent
    for name in ("contract_adapter.py", "run_inside.sh"):
        source = (harness_root / name).read_text(encoding="utf-8")
        assert "acceptance-corpus-v2" not in source
        assert "leadpoet.acceptance_corpus.v2" not in source


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
    assert "postgresql15-contrib" in dockerfile
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
        "tests/restart_rehearsal/fixture_contract.py"
        in rehearsal.COMMITTED_HARNESS_PATHS
    )
    assert "for role in gateway_coordinator gateway_scoring" in run_inside
    assert "for _attempt in $(seq 1 600)" in run_inside
    assert (
        'REHEARSAL_GATEWAY_CANDIDATE_ROOT="$SELECTED_GATEWAY_SOURCE_ROOT/gateway" \\'
        in run_inside
    )
    assert "gateway-enclave-build-identities" in run_inside
    assert "gateway-attested-runtime" in run_inside
    assert (
        "Materializing the exact measured gateway-enclave filesystem"
        in run_inside
    )
    assert "_prepare_candidate_role_root(role)" in gateway_service
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
    assert (
        'PYTHONPATH="/harness" /usr/bin/python3.11 \\\n'
        "  /harness/verify_evidence.py"
    ) in run_inside
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
    assert "verify" in set(
        contract["boundaries"]["aws_kms"]["allowed_operations"]
    )
    behavior_contract = build_rehearsal_behavior_contract_v2(
        source_root=Path(__file__).resolve().parents[2],
        candidate_sha=COMMIT,
        profile="prepush",
        epoch_count=1,
    )
    assert "restart-summary-deadline-classification" in set(
        behavior_contract["behavior_scenarios"]
    )
    assert "restart_summary_deadline_classification_exact" in set(
        behavior_contract["required_invariant_ids"]
    )
    assert "validator_role_release_identity_exact" in set(
        behavior_contract["required_restart_invariant_ids"]
    )
    assert "compact-weight-joined-path" in set(
        behavior_contract["behavior_scenarios"]
    )
    assert "compact_ancestry_unknown_commit_recovery_verified" in set(
        behavior_contract["required_invariant_ids"]
    )
    assert {
        "gateway/tee/execution_job_manager_v2.py",
        "gateway/tee/provider_broker_v2.py",
        "gateway/tee/provider_client_v2.py",
        "gateway/tee/provider_outcome_store_v2.py",
        "gateway/tee/rpc_authority.py",
        "gateway/main.py",
        "gateway/research_lab/source_add_trial_runner.py",
        "gateway/tee/code_hash.py",
        "gateway/tee/prepare_gateway_envelopes_v2.py",
        "gateway/tee/protected_workflows.py",
        "gateway/tee/stage_attested_runtime.sh",
        "gateway/tee/topology.json",
        "research_lab/docker_operation_lock_v2.py",
        "validator_tee/host/docker_operation_guard_v2.py",
        "validator_tee/scripts/docker_operation_lock_v2.sh",
        "validator_tee/scripts/reclaim_docker_storage_v2.sh",
    } <= set(behavior_contract["production_source_paths"])
    assert "leadpoet_observability/sentry_operations.py" not in set(
        behavior_contract["production_source_paths"]
    )
    assert production_workflow_runner.HOST_RESTART_SUMMARY_SOURCE_PATHS == (
        "leadpoet_observability/sentry_operations.py",
    )
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


def test_coordinator_broker_httpx_grant_evidence_is_exact_and_fail_closed() -> None:
    evidence = production_workflow_runner._exercise_coordinator_broker_owned_httpx_grant()
    assert (
        production_workflow_runner._coordinator_broker_httpx_evidence_is_complete(
            evidence
        )
        is True
    )

    for case in production_workflow_runner._BROKER_OWNED_HTTPX_FAIL_CLOSED_CASES:
        mutated = {
            **evidence,
            "fail_closed_cases": {
                **evidence["fail_closed_cases"],
                case: False,
            },
        }
        assert (
            production_workflow_runner._coordinator_broker_httpx_evidence_is_complete(
                mutated
            )
            is False
        )

    for required_field in (
        "coordinator_role_authority_bound",
        "direct_supabase_sidecar_receipt_bound",
        "real_broker_external_send_bound",
    ):
        mutated = {**evidence, required_field: False}
        assert (
            production_workflow_runner._coordinator_broker_httpx_evidence_is_complete(
                mutated
            )
            is False
        )


def test_restart_summary_deadline_action_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(
        production_workflow_runner,
        "SOURCE_ROOT",
        source_root,
    )
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", candidate_sha)
    evidence = (
        production_workflow_runner._exercise_restart_summary_deadline_classification()
    )
    assert (
        production_workflow_runner._restart_summary_deadline_evidence_is_complete(
            evidence
        )
        is True
    )
    for field in (
        production_workflow_runner._RESTART_SUMMARY_DEADLINE_EVIDENCE_FIELDS
    ):
        assert (
            production_workflow_runner._restart_summary_deadline_evidence_is_complete(
                {**evidence, field: False}
            )
            is False
        )
    identity = evidence["host_source_identities"][0]
    for field, value in (
        ("path", "leadpoet_observability/not-the-candidate.py"),
        ("commit_sha", "0" * 40),
        ("sha256", "c" * 64),
    ):
        assert (
            production_workflow_runner._restart_summary_deadline_evidence_is_complete(
                {
                    **evidence,
                    "host_source_identities": [
                        {**identity, field: value},
                    ],
                }
            )
            is False
        )
    assert (
        production_workflow_runner._restart_summary_deadline_evidence_is_complete(
            {**evidence, "host_source_identities": []}
        )
        is False
    )
    assert (
        production_workflow_runner._restart_summary_deadline_evidence_is_complete(
            {**evidence, "unexpected": True}
        )
        is False
    )


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
    release_builder = (
        harness_root.parents[1] / "gateway/tee/build_local_release_v2.sh"
    ).read_text(encoding="utf-8")
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
    assert "len(GATEWAY_ROLE_SPECS)" in verifier
    assert "exact three-enclave topology" not in verifier
    assert "python3 -m gateway.tee.local_release_v2" in release_builder
    assert verifier.count('"module:gateway.tee.local_release_v2"') == 2
    assert '"module:gateway.tee.release_channel_v2"' not in verifier


def test_evidence_verifier_keeps_stdout_json_channel_clean() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    harness_root = Path(__file__).resolve().parent
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"runpy.run_path({str(harness_root / 'verify_evidence.py')!r}); "
                "print('VERIFIER_IMPORT_OK')"
            ),
        ],
        cwd=harness_root,
        env={
            **os.environ,
            "PYTHONPATH": str(harness_root),
            "REHEARSAL_CANDIDATE_SOURCE_ROOT": str(repository_root),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "VERIFIER_IMPORT_OK\n"


def test_workflow_runs_before_command_adapters_are_installed() -> None:
    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")
    workflow = script.index('if [ "$COMPONENT" = "workflow" ]; then')
    adapters = script.index("make_adapter()")
    assert workflow < adapters
    assert "/harness/production_workflow_runner.py" in script[workflow:adapters]


def test_workflow_uses_the_strict_exact_external_boundaries(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    from_sha = "0" * 40
    candidate_sha = "1" * 40

    def run(command):
        captured["command"] = list(command)

    monkeypatch.setattr(rehearsal, "_run", run)
    rehearsal._run_workflow(
        "rehearsal-image",
        source_root=tmp_path / "source",
        evidence_root=tmp_path / "evidence",
        from_sha=from_sha,
        candidate_sha=candidate_sha,
        profile="prepush",
        docker_platform="linux/amd64",
    )

    command = captured["command"]
    environment = {
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--env"
    }
    assert "REHEARSAL_SCOPE=exact" in environment
    assert "PYTHONPATH=/source:/harness" in environment
    assert f"REHEARSAL_FROM_SHA={from_sha}" in environment
    assert f"REHEARSAL_CANDIDATE_SHA={candidate_sha}" in environment


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


def test_local_boundary_service_survives_process_wide_exact_adapters(
    tmp_path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    harness_root = repository_root / "tests" / "restart_rehearsal"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{repository_root}:{harness_root}",
            "REHEARSAL_SCOPE": "exact",
            "REHEARSAL_SOURCE_ROOT": str(repository_root),
            "REHEARSAL_STATE_ROOT": str(tmp_path / "state"),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "from pathlib import Path",
                    "from local_services import LocalBoundaryServices",
                    "fixture = {'sanitization': {'contains_production_credentials': False}}",
                    f"root = Path({str(tmp_path / 'boundary')!r})",
                    "with LocalBoundaryServices(root=root, fixture=fixture) as services:",
                    "    value = services.request('POST', '/database/insert', {'kind': 'probe', 'epoch_id': 1, 'body': {'ok': True}})",
                    "assert value['status'] == 'persisted'",
                )
            ),
        ],
        cwd=repository_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0


def test_joined_manifest_requires_every_authority_field(
    monkeypatch,
    tmp_path,
) -> None:
    initial_hash = "sha256:" + "a" * 64
    final_hash = _write_final_durable_state(tmp_path, revision=1)
    for component in ("gateway", "validator"):
        end_revision = 1 if component == "gateway" else 0
        end_hash = final_hash if component == "gateway" else initial_hash
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
                    "durable_schema_sha": COMMIT,
                    "durable_boundary_state": {
                        "schema_version": (
                            "leadpoet.restart_rehearsal."
                            "durable_boundary_state.v1"
                        ),
                        "durable_schema_sha": COMMIT,
                        "start_revision": 0,
                        "start_state_hash": initial_hash,
                        "end_revision": end_revision,
                        "end_state_hash": end_hash,
                    },
                    "restart_invariants": (
                        {invariant: True for invariant in RESTART_INVARIANTS}
                        if component == "validator"
                        else {}
                    ),
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
    source_root = Path(__file__).resolve().parents[2]
    behavior_contract = build_rehearsal_behavior_contract_v2(
        source_root=source_root,
        candidate_sha=COMMIT,
        profile="prepush",
        epoch_count=1,
    )
    workflow_path = tmp_path / "workflow.json"
    workflow = {
        "status": "passed",
        "profile": "prepush",
        "release_sha": COMMIT,
        "epoch_count": 1,
        "epochs": [epoch],
        "fault_matrix": [],
        "concurrent_write_count": 0,
        "behavior_contract": behavior_contract,
        "behavior_contract_hash": behavior_contract["contract_hash"],
        "behavior_evidence": {
            scenario: {"status": "passed"}
            for scenario in behavior_contract["behavior_scenarios"]
        },
        "behavioral_invariants": {
            invariant: True
            for invariant in behavior_contract["required_invariant_ids"]
        },
        "production_source_identities": [
            {
                "path": path,
                "commit_sha": COMMIT,
                "sha256": "f" * 64,
            }
            for path in behavior_contract["production_source_paths"]
        ],
        "cleanup": {
            "pending_faults": 0,
            "boundary_thread_alive_after_close": False,
            "local_chain_epochs": 1,
        },
        "stages": [
            {"stage": stage, "status": "passed"}
            for stage in behavior_contract["required_stage_ids"]
        ],
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
    assert joined["durable_boundary_state_continuity"] is True
    assert (
        joined["behavior_contract_hash"]
        == behavior_contract["contract_hash"]
    )

    validator_launcher_path = (
        tmp_path / f"1-validator-forward-{COMMIT}.json"
    )
    validator_launcher = json.loads(
        validator_launcher_path.read_text(encoding="utf-8")
    )
    validator_launcher["restart_invariants"] = {}
    validator_launcher_path.write_text(
        json.dumps(validator_launcher),
        encoding="utf-8",
    )
    with pytest.raises(
        SystemExit,
        match="validator restart invariant evidence is incomplete",
    ):
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
    validator_launcher["restart_invariants"] = {
        invariant: True for invariant in RESTART_INVARIANTS
    }
    validator_launcher_path.write_text(
        json.dumps(validator_launcher),
        encoding="utf-8",
    )

    missing_stage = workflow["stages"].pop(1)
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    with pytest.raises(
        SystemExit,
        match="workflow stage evidence is incomplete",
    ):
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
    workflow["stages"].insert(1, missing_stage)
    workflow["stages"].append(
        {"stage": "unexpected-future-stage", "status": "passed"}
    )
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    with pytest.raises(
        SystemExit,
        match="workflow stage evidence is incomplete",
    ):
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
    workflow["stages"].pop()

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

    validator["durable_boundary_state"]["start_state_hash"] = (
        "sha256:" + "c" * 64
    )
    validator_path.write_text(json.dumps(validator), encoding="utf-8")
    with pytest.raises(
        SystemExit,
        match="durable boundary revision has conflicting state hashes",
    ):
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
    validator["durable_boundary_state"]["start_state_hash"] = initial_hash
    validator_path.write_text(json.dumps(validator), encoding="utf-8")

    workflow["stages"][-1]["status"] = "unexercised"
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    with pytest.raises(
        SystemExit,
        match="workflow stage evidence is incomplete",
    ):
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
    workflow["stages"][-1]["status"] = "passed"

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


def test_behavior_contract_tracks_candidate_runtime_policies(
    monkeypatch,
) -> None:
    import leadpoet_canonical.chain_source_v2 as chain_source
    from leadpoet_canonical.chain_source_v2 import (
        chain_source_policy_document,
        chain_source_policy_hash,
    )

    source_root = Path(__file__).resolve().parents[2]
    baseline = build_rehearsal_behavior_contract_v2(
        source_root=source_root,
        candidate_sha=COMMIT,
        profile="prepush",
        epoch_count=1,
    )

    assert baseline["policy_commitments"]["chain_source"] == {
        "policy": chain_source_policy_document(),
        "policy_hash": chain_source_policy_hash(),
    }

    monkeypatch.setattr(
        chain_source,
        "CHAIN_SELECTIVE_RESULT_LAST_FIELDS",
        (73, 76, 80),
    )
    layout_changed = build_rehearsal_behavior_contract_v2(
        source_root=source_root,
        candidate_sha=COMMIT,
        profile="prepush",
        epoch_count=1,
    )
    assert layout_changed["policy_commitments"]["chain_source"][
        "policy"
    ]["selective_result_last_fields"] == [73, 76, 80]
    assert layout_changed["contract_hash"] != baseline["contract_hash"]


def test_candidate_behavior_scenario_follows_chain_source_policy(
    monkeypatch,
) -> None:
    from leadpoet_canonical.chain_source_v2 import (
        chain_source_policy_document,
        chain_source_policy_hash,
    )

    source_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        production_workflow_runner,
        "SOURCE_ROOT",
        source_root,
    )
    historical_layouts = (
        production_workflow_runner._exercise_historical_metagraph_layouts()
    )

    assert historical_layouts["policy_hash"] == chain_source_policy_hash()
    assert historical_layouts["accepted_layouts"] == (
        chain_source_policy_document()["selective_result_last_fields"]
    )
    assert historical_layouts["rpc_call_counts"] == {
        str(last_field): 6
        for last_field in historical_layouts["accepted_layouts"]
    }


def test_historical_layout_scenario_follows_candidate_policy(
    monkeypatch,
) -> None:
    import leadpoet_canonical.chain_source_v2 as chain_source

    source_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        production_workflow_runner,
        "SOURCE_ROOT",
        source_root,
    )
    monkeypatch.setattr(
        chain_source,
        "CHAIN_SELECTIVE_RESULT_LAST_FIELDS",
        (73, 76, 80),
    )

    evidence = (
        production_workflow_runner._exercise_historical_metagraph_layouts()
    )

    assert evidence["accepted_layouts"] == [73, 76, 80]
    assert evidence["rpc_call_counts"] == {"73": 6, "76": 6, "80": 6}


def test_receipt_graph_aggregate_pagination_scenario_uses_candidate_bounds() -> None:
    from gateway.research_lab import attested_v2_store

    evidence = (
        production_workflow_runner._exercise_receipt_graph_aggregate_pagination()
    )

    assert evidence == {
        "aggregate_rows": attested_v2_store._MAX_GRAPH_ROWS + 1,
        "aggregate_evidence_paged": True,
        "checkpoint_parent_first_persistence": True,
        "per_query_row_limit": attested_v2_store._MAX_GRAPH_ROWS,
        "query_chunk": attested_v2_store._GRAPH_QUERY_CHUNK,
        "query_count": (
            (
                attested_v2_store._MAX_GRAPH_ROWS
                + attested_v2_store._GRAPH_QUERY_CHUNK
            )
            // attested_v2_store._GRAPH_QUERY_CHUNK
        ),
        "structural_limit_enforced": True,
    }


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

    source.write_text(
        source.read_text(encoding="utf-8")
        + "parser.add_argument('--epoch', type=int)\n",
        encoding="utf-8",
    )
    assert selected_weight_storage_preflight_capability((tmp_path,)) is True


def test_weight_storage_preflight_epoch_binding_tracks_restart_wrapper(
    tmp_path,
) -> None:
    restart = tmp_path / "gw_restart.sh"
    prefix = (
        'GATEWAY_DEPLOY_STAGE="validator_weight_input_storage_preflight"\n'
        "run_prepared_gateway_module \\\n"
        "  gateway.tee.verify_weight_submission_ready_v2 \\\n"
    )
    suffix = 'GATEWAY_DEPLOY_STAGE="ancestry_precheckpoint"\n'
    restart.write_text(
        prefix + "  --storage-read-preflight\n" + suffix,
        encoding="utf-8",
    )
    assert selected_weight_storage_preflight_pins_epoch((tmp_path,)) is False

    restart.write_text(
        prefix
        + "  --storage-read-preflight \\\n"
        + '  --epoch "$GATEWAY_WEIGHT_STORAGE_PREFLIGHT_EPOCH"\n'
        + suffix,
        encoding="utf-8",
    )
    assert selected_weight_storage_preflight_pins_epoch((tmp_path,)) is True
    assert selected_weight_storage_preflight_pins_epoch(
        (Path(__file__).resolve().parents[2],)
    ) is True


def test_gateway_rehearsal_has_no_retired_provider_preflight_receipt_gate() -> None:
    verifier = (
        Path(__file__).resolve().parent / "verify_evidence.py"
    ).read_text(encoding="utf-8")

    assert "def verify_gateway_provider_preflight(" not in verifier
    assert "gateway provider preflight did not durably append" not in verifier


def test_gateway_rehearsal_serves_source_add_restart_contracts(
    tmp_path,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (
            source_root
            / "tests/restart_rehearsal/fixtures/production_shaped_v2.json"
        ).read_text(encoding="utf-8")
    )
    admission_rpc = "research_lab_source_add_admission_control_contract_v1"
    restart_rpc = "research_lab_source_add_claim_control_contract_v2"
    state = LocalPostgRESTState(
        state_root=tmp_path,
        fixture=fixture,
        source_root=source_root,
        tables=set(),
        rpcs={admission_rpc, restart_rpc},
    )
    server = LocalPostgRESTServer(("127.0.0.1", 0), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        responses = {}
        for rpc_name in (admission_rpc, restart_rpc):
            request = urllib.request.Request(
                "http://127.0.0.1:%d/rest/v1/rpc/%s"
                % (server.server_address[1], rpc_name),
                data=b"{}",
                headers={
                    "apikey": "rehearsal-secret",
                    "authorization": "Bearer rehearsal-secret",
                    "content-type": "application/json",
                },
                method="POST",
            )
            with opener.open(request, timeout=2.0) as response:
                assert response.status == 200
                responses[rpc_name] = json.loads(response.read())
        assert responses == {
            admission_rpc: {
                "schema_version": (
                    "leadpoet.source_add_admission_control_contract.v1"
                ),
                "control_row_present": True,
                "trigger_enabled": True,
                "pause_rpc": "research_lab_source_add_set_paused",
                "admission_trigger": "trg_source_add_work_admission_control",
            },
            restart_rpc: _source_add_claim_control_contract_v2(source_root),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_gateway_rehearsal_provider_checkpoint_rpc_matches_migration_134(
    tmp_path,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (
            source_root
            / "tests/restart_rehearsal/fixtures/production_shaped_v2.json"
        ).read_text(encoding="utf-8")
    )
    table = "research_lab_provider_outcome_checkpoints_v2"
    columns = frozenset(
        {
            "schema_version",
            "artifact_master_key_ref_hash",
            "utc_day",
            "sequence",
            "checkpoint_hash",
            "previous_checkpoint_hash",
            "state_document_hash",
            "checkpoint_artifact_id",
            "encrypted_checkpoint_doc",
            "created_at",
        }
    )
    contract = {
        "schema_version": "leadpoet.provider_outcome_contention_contract.v3",
        "lock_contention_status": "busy",
        "stale_lineage_status": "conflict",
        "candidate_checkpoint_hash": True,
        "conflict_head_checkpoint_row": "encrypted_or_null",
    }
    state = LocalPostgRESTState(
        state_root=tmp_path,
        fixture=fixture,
        source_root=source_root,
        tables={table},
        rpcs={"append_research_lab_provider_outcome_checkpoint_v2"},
        relation_columns={table: columns},
        provider_outcome_contract=contract,
    )

    def checkpoint(
        sequence: int,
        checkpoint_hash: str,
        previous_hash: str,
        suffix: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "leadpoet.provider_outcome_checkpoint_row.v2",
            "artifact_master_key_ref_hash": "sha256:" + "a" * 64,
            "utc_day": "2026-07-29",
            "sequence": sequence,
            "checkpoint_hash": checkpoint_hash,
            "previous_checkpoint_hash": previous_hash,
            "state_document_hash": "sha256:" + suffix * 64,
            "checkpoint_artifact_id": "sha256:" + suffix * 64,
            "encrypted_checkpoint_doc": {"fixture": suffix},
        }

    first_hash = "sha256:" + "1" * 64
    first = checkpoint(1, first_hash, "", "2")
    assert state.append_provider_outcome_checkpoint(
        {"checkpoint_row": first}
    ) == {"status": "inserted", "checkpoint_hash": first_hash}
    assert state.append_provider_outcome_checkpoint(
        {"checkpoint_row": first}
    ) == {"status": "existing", "checkpoint_hash": first_hash}

    stale_hash = "sha256:" + "3" * 64
    stale = checkpoint(1, stale_hash, "", "4")
    assert state.append_provider_outcome_checkpoint(
        {"checkpoint_row": stale}
    ) == {
        "status": "conflict",
        "checkpoint_hash": stale_hash,
        "head_checkpoint_row": first,
    }

    second_hash = "sha256:" + "5" * 64
    second = checkpoint(2, second_hash, first_hash, "6")
    lock = state._provider_outcome_lock(
        second["artifact_master_key_ref_hash"],
        second["utc_day"],
    )
    lock.acquire()
    try:
        assert state.append_provider_outcome_checkpoint(
            {"checkpoint_row": second}
        ) == {"status": "busy", "checkpoint_hash": second_hash}
    finally:
        lock.release()
    assert state.append_provider_outcome_checkpoint(
        {"checkpoint_row": second}
    ) == {"status": "inserted", "checkpoint_hash": second_hash}
    assert [
        int(row["sequence"]) for row in state.rows[table]
    ] == [1, 2]


def test_gateway_rehearsal_ancestry_checkpoint_rpc_matches_migration_135(
    tmp_path,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (
            source_root
            / "tests/restart_rehearsal/fixtures/production_shaped_v2.json"
        ).read_text(encoding="utf-8")
    )
    checkpoint_table = "research_lab_attested_ancestry_checkpoints_v2"
    activation_table = "research_lab_attested_ancestry_activations_v2"
    checkpoint_columns = frozenset(
        {
            "root_receipt_hash",
            "schema_version",
            "lineage_id",
            "certificate_hash",
            "certificate_sequence",
            "issuer_boot_identity_hash",
            "proof_hash",
            "checkpoint_graph_hash",
            "certificate_doc",
            "proof_doc",
            "checkpoint_graph_doc",
            "created_at",
        }
    )
    activation_columns = frozenset(
        {
            "lineage_id",
            "activation_root_receipt_hash",
            "activation_certificate_hash",
            "activated_at",
        }
    )
    durable_state = tmp_path / "postgrest-state.json"
    state = LocalPostgRESTState(
        state_root=tmp_path,
        fixture=fixture,
        source_root=source_root,
        tables={
            checkpoint_table,
            activation_table,
            "research_lab_attested_execution_receipts_v2",
            "research_lab_attested_boot_identities_v2",
        },
        rpcs={"persist_research_lab_ancestry_checkpoint_v2"},
        relation_columns={
            checkpoint_table: checkpoint_columns,
            activation_table: activation_columns,
        },
        durable_state_path=durable_state,
        durable_schema_sha=COMMIT,
    )

    def sha(character: str) -> str:
        return "sha256:" + character * 64

    lineage = sha("1")
    issuer = sha("2")
    legacy_root = sha("3")
    disclosed_root = sha("4")
    roots = [sha(character) for character in "abcde"]
    state.rows["research_lab_attested_boot_identities_v2"].append(
        {"boot_identity_hash": issuer}
    )
    state.rows["research_lab_attested_execution_receipts_v2"].extend(
        {"receipt_hash": root} for root in roots
    )

    def checkpoint(
        *,
        root: str,
        sequence: int,
        hash_characters: str,
        parents: list[dict[str, Any]],
        disclosed_receipts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        assert len(hash_characters) == 3
        certificate_hash = sha(hash_characters[0])
        proof_hash = sha(hash_characters[1])
        graph_hash = sha(hash_characters[2])
        claim = {
            "output_root_receipt_hash": root,
            "lineage_id": lineage,
            "certificate_sequence": sequence,
            "issuer_boot_identity_hash": issuer,
            "parent_authorities": parents,
        }
        certificate = {
            "schema_version": "leadpoet.attested_ancestry_certificate.v2",
            "certificate_hash": certificate_hash,
            "claim": claim,
        }
        proof = {
            "schema_version": (
                "leadpoet.attested_ancestry_compact_proof.v2"
            ),
            "proof_hash": proof_hash,
            "certificate": certificate,
            "disclosed_receipts": list(disclosed_receipts or []),
        }
        graph = {
            "schema_version": (
                "leadpoet.attested_checkpointed_receipt_graph.v3"
            ),
            "root_receipt_hash": root,
            "ancestry_lineage_id": lineage,
            "ancestry_proof": proof,
        }
        return {
            "root_receipt_hash": root,
            "schema_version": "leadpoet.attested_ancestry_certificate.v2",
            "lineage_id": lineage,
            "certificate_hash": certificate_hash,
            "certificate_sequence": sequence,
            "issuer_boot_identity_hash": issuer,
            "proof_hash": proof_hash,
            "checkpoint_graph_hash": graph_hash,
            "certificate_doc": certificate,
            "proof_doc": proof,
            "checkpoint_graph_doc": graph,
        }

    first = checkpoint(
        root=roots[0],
        sequence=0,
        hash_characters="567",
        parents=[
            {
                "authority_kind": "full_projection",
                "parent_receipt_hash": legacy_root,
            }
        ],
        disclosed_receipts=[{"receipt_hash": disclosed_root}],
    )
    expected_ack = {
        "status": "persisted",
        "root_receipt_hash": roots[0],
        "lineage_id": lineage,
        "certificate_hash": first["certificate_hash"],
        "certificate_sequence": 0,
        "proof_hash": first["proof_hash"],
        "checkpoint_graph_hash": first["checkpoint_graph_hash"],
        "root_activated": True,
    }
    assert state.persist_ancestry_checkpoint({"checkpoint": first}) == (
        expected_ack
    )
    first_revision = state.durable_state_identity()["revision"]
    assert state.persist_ancestry_checkpoint({"checkpoint": first}) == (
        expected_ack
    )
    assert state.durable_state_identity()["revision"] == first_revision

    certificate_child = checkpoint(
        root=roots[1],
        sequence=1,
        hash_characters="89a",
        parents=[
            {
                "authority_kind": "certificate",
                "parent_receipt_hash": roots[0],
                "authority_hash": first["certificate_hash"],
                "authority_sequence": 0,
            }
        ],
    )
    assert state.persist_ancestry_checkpoint(
        {"checkpoint": certificate_child}
    )["root_activated"] is True

    disclosure_child = checkpoint(
        root=roots[2],
        sequence=1,
        hash_characters="bcd",
        parents=[
            {
                "authority_kind": "certificate_disclosure",
                "parent_receipt_hash": disclosed_root,
                "authority_sequence": 0,
            }
        ],
    )
    assert state.persist_ancestry_checkpoint(
        {"checkpoint": disclosure_child}
    )["root_activated"] is True

    forbidden_full_parent = checkpoint(
        root=roots[3],
        sequence=0,
        hash_characters="ef0",
        parents=[
            {
                "authority_kind": "full_projection",
                "parent_receipt_hash": roots[0],
            }
        ],
    )
    with pytest.raises(
        ValueError,
        match="compacted ancestry root rejects full graph parent",
    ):
        state.persist_ancestry_checkpoint(
            {"checkpoint": forbidden_full_parent}
        )

    missing_parent = checkpoint(
        root=roots[4],
        sequence=2,
        hash_characters="123",
        parents=[
            {
                "authority_kind": "certificate",
                "parent_receipt_hash": sha("9"),
                "authority_hash": sha("0"),
                "authority_sequence": 1,
            }
        ],
    )
    with pytest.raises(ValueError, match="parent is not durable"):
        state.persist_ancestry_checkpoint({"checkpoint": missing_parent})

    conflict = json.loads(json.dumps(first))
    conflict["proof_hash"] = sha("f")
    conflict["proof_doc"]["proof_hash"] = sha("f")
    conflict["checkpoint_graph_doc"]["ancestry_proof"][
        "proof_hash"
    ] = sha("f")
    with pytest.raises(ValueError, match="durable readback conflicts"):
        state.persist_ancestry_checkpoint({"checkpoint": conflict})

    assert len(state.rows[checkpoint_table]) == 3
    assert len(state.rows[activation_table]) == 3


def test_rehearsal_provider_boundaries_require_job_credentials_and_tls_proxy(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(rehearsal_sitecustomize, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    proxy = (
        "https://rehearsal-scoring:rehearsal-scoring-password@"
        "93.184.216.34:443"
    )
    connection_scope = "sha256:" + "a" * 64
    exa = rehearsal_sitecustomize._local_provider_transport(
        method="POST",
        url="https://api.exa.ai/search",
        headers={
            "x-api-key": "rehearsal-exa",
            "content-type": "application/json",
        },
        body=b'{"numResults":1,"query":"provider preflight"}',
        timeout_ms=12_000,
        upstream_proxy_url=proxy,
        connection_scope=connection_scope,
    )
    scrapingdog = rehearsal_sitecustomize._local_provider_transport(
        method="GET",
        url=(
            "https://api.scrapingdog.com/account?"
            "api_key=rehearsal-scrapingdog"
        ),
        headers={},
        body=b"",
        timeout_ms=12_000,
        upstream_proxy_url=proxy,
        connection_scope=connection_scope,
    )

    assert exa["http_status"] == scrapingdog["http_status"] == 200
    with pytest.raises(ValueError, match="job-scoped TLS proxy"):
        rehearsal_sitecustomize._local_provider_transport(
            method="POST",
            url="https://api.exa.ai/search",
            headers={"x-api-key": "rehearsal-exa"},
            body=b'{"numResults":1,"query":"provider preflight"}',
            timeout_ms=12_000,
        )
    with pytest.raises(ValueError, match="measured connection scope"):
        rehearsal_sitecustomize._local_provider_transport(
            method="POST",
            url="https://api.exa.ai/search",
            headers={"x-api-key": "rehearsal-exa"},
            body=b'{"numResults":1,"query":"provider preflight"}',
            timeout_ms=12_000,
            upstream_proxy_url=proxy,
        )
    with pytest.raises(ValueError, match="unexpected connection scope"):
        rehearsal_sitecustomize._local_provider_transport(
            method="GET",
            url="https://api.coingecko.com/api/v3/simple/price",
            headers={},
            body=b"",
            timeout_ms=12_000,
            connection_scope=connection_scope,
        )


def test_rehearsal_scoring_provider_calls_cross_the_coordinator_process() -> None:
    source = Path(rehearsal_sitecustomize.__file__).read_text(
        encoding="utf-8"
    )
    runtime = source[
        source.index("def _gateway_runtime_objects(") :
        source.index("def _unwrap_candidate_rpc(")
    ]
    handler = source[
        source.index("def _handle_gateway_enclave_rpc(") :
        source.index("class _LocalVsock:")
    ]

    assert "rehearsal_inter_enclave_provider_execute" in runtime
    assert "seal_artifact_over_attested_tls_v2" in runtime
    assert "_PersistentInterEnclaveArtifactClient" in runtime
    assert "handle_inter_enclave_rpc(" in handler
    assert '"provider_execute"' in handler
    assert '"rehearsal_inter_enclave_artifact_call"' in handler
    assert '"artifact_seal_finish"' in handler


def test_rehearsal_artifact_client_crosses_the_coordinator_process(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def call(
        role: str,
        method: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        calls.append((role, method, dict(params)))
        return {"status": "accepted"}

    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_call_persistent_gateway_enclave",
        call,
    )
    client = rehearsal_sitecustomize._PersistentInterEnclaveArtifactClient(
        peer_role="gateway_scoring"
    )
    params = {"upload_id": "artifact_upload:" + "1" * 32}

    assert client.call(
        target_physical_role="gateway_coordinator",
        method="artifact_seal_finish",
        params=params,
        channel_id="2" * 32,
    ) == {"status": "accepted"}
    assert calls == [
        (
            "gateway_coordinator",
            "rehearsal_inter_enclave_artifact_call",
            {
                "peer_role": "gateway_scoring",
                "method": "artifact_seal_finish",
                "params": params,
                "channel_id": "2" * 32,
            },
        )
    ]

    with pytest.raises(ValueError, match="artifact channel differs"):
        client.call(
            target_physical_role="gateway_scoring",
            method="artifact_seal_finish",
            params=params,
            channel_id="2" * 32,
        )
    with pytest.raises(ValueError, match="artifact channel differs"):
        client.call(
            target_physical_role="gateway_coordinator",
            method="provider_execute",
            params=params,
            channel_id="2" * 32,
        )
    with pytest.raises(ValueError, match="artifact channel differs"):
        client.call(
            target_physical_role="gateway_coordinator",
            method="artifact_seal_finish",
            params=params,
            channel_id="not-a-channel",
        )


def test_rehearsal_coordinator_binds_artifact_calls_to_peer_boot_identity(
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    class CandidateRuntime:
        def handle_inter_enclave_rpc(
            self,
            method: str,
            params: dict[str, object],
            peer: dict[str, object],
        ) -> dict[str, object]:
            calls.append((method, dict(params), dict(peer)))
            return {"status": "candidate"}

    runtime = CandidateRuntime()
    state = {
        "roles": {
            "gateway_coordinator": {
                "config_hash": "sha256:" + "1" * 64,
            },
            "gateway_scoring": {
                "config_hash": "sha256:" + "2" * 64,
            },
        }
    }
    boot_identity = {
        "physical_role": "gateway_scoring",
        "role": "gateway_scoring",
        "boot_identity_hash": "sha256:" + "3" * 64,
    }
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_release_input",
        lambda: {
            "gateway_roles": {
                "gateway_coordinator": {},
                "gateway_scoring": {},
            }
        },
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_enclave_state",
        lambda _mutate=None: (state, None),
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_runtime_objects",
        lambda _role, _state: {"tee_service": runtime},
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_local_boot_identity",
        lambda role, config_hash: {
            **boot_identity,
            "physical_role": role,
            "config_hash": config_hash,
        },
    )
    params = {"upload_id": "artifact_upload:" + "4" * 32}

    assert rehearsal_sitecustomize._handle_gateway_enclave_rpc(
        "gateway_coordinator",
        "rehearsal_inter_enclave_artifact_call",
        {
            "peer_role": "gateway_scoring",
            "method": "artifact_seal_finish",
            "params": params,
            "channel_id": "5" * 32,
        },
    ) == {"status": "candidate"}
    assert calls == [
        (
            "artifact_seal_finish",
            params,
            {
                "physical_role": "gateway_scoring",
                "service_role": "gateway_scoring",
                "boot_identity": {
                    **boot_identity,
                    "config_hash": "sha256:" + "2" * 64,
                },
            },
        )
    ]

    with pytest.raises(ValueError, match="artifact peer differs"):
        rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            "gateway_coordinator",
            "rehearsal_inter_enclave_artifact_call",
            {
                "peer_role": "gateway_scoring",
                "method": "provider_execute",
                "params": params,
                "channel_id": "5" * 32,
            },
        )
    with pytest.raises(ValueError, match="artifact peer differs"):
        rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            "gateway_coordinator",
            "rehearsal_inter_enclave_artifact_call",
            {
                "peer_role": "gateway_scoring",
                "method": "artifact_seal_finish",
                "params": params,
                "channel_id": "not-a-channel",
            },
        )


def test_rehearsal_gateway_boot_generation_separates_restarts(
    monkeypatch,
) -> None:
    role = "gateway_coordinator"
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_release_input",
        lambda: {
            "gateway_roles": {
                role: {
                    "commit_sha": "a" * 40,
                    "pcr0": "b" * 96,
                    "execution_manifest_hash": "sha256:" + "c" * 64,
                    "dependency_lock_hash": "sha256:" + "d" * 64,
                }
            }
        },
    )
    environment_name = (
        rehearsal_sitecustomize.REHEARSAL_GATEWAY_BOOT_GENERATION_ENV
    )
    first_config = "sha256:" + "1" * 64
    second_config = "sha256:" + "2" * 64

    monkeypatch.setenv(environment_name, "3" * 32)
    first = rehearsal_sitecustomize._local_boot_identity(role, first_config)
    first_replay = rehearsal_sitecustomize._local_boot_identity(
        role, first_config
    )
    assert first_replay == first

    monkeypatch.setenv(environment_name, "4" * 32)
    same_config_restart = rehearsal_sitecustomize._local_boot_identity(
        role, first_config
    )
    changed_config_restart = rehearsal_sitecustomize._local_boot_identity(
        role, second_config
    )

    def durable_boot_key(value: Mapping[str, object]) -> tuple[object, ...]:
        return (
            value["physical_role"],
            value["commit_sha"],
            value["pcr0"],
            value["boot_nonce"],
        )

    assert same_config_restart["boot_identity_hash"] != first[
        "boot_identity_hash"
    ]
    assert durable_boot_key(same_config_restart) != durable_boot_key(first)
    assert durable_boot_key(changed_config_restart) != durable_boot_key(first)

    monkeypatch.setenv(environment_name, "not-a-generation")
    with pytest.raises(ValueError, match="boot generation is invalid"):
        rehearsal_sitecustomize._local_boot_identity(role, first_config)


def test_rehearsal_routes_credential_ingress_to_candidate_runtime(
    monkeypatch,
) -> None:
    class CandidateRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def handle_v2_runtime_rpc(
            self,
            method: str,
            params: dict[str, object],
        ) -> dict[str, object]:
            self.calls.append((method, dict(params)))
            return {"result": {"method": method, "status": "candidate"}}

    runtime = CandidateRuntime()
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_release_input",
        lambda: {"gateway_roles": {"gateway_coordinator": {}}},
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_enclave_state",
        lambda _mutate=None: (
            {
                "roles": {
                    "gateway_coordinator": {
                        "config_hash": "sha256:" + "1" * 64,
                    }
                }
            },
            None,
        ),
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_runtime_objects",
        lambda _role, _state: {"tee_service": runtime},
    )
    requests = {
        "v2_get_source_add_ingress_recipient": {
            "miner_hotkey": "miner",
            "adapter_ref": "source_add:test",
            "credential_ref": "encrypted_ref:source_add:" + "2" * 32,
        },
        "v2_seal_source_add_ingress_credential": {
            "request_id": "sha256:" + "3" * 64,
            "ciphertext_b64": "Y2lwaGVydGV4dA==",
        },
    }

    for method, params in requests.items():
        assert rehearsal_sitecustomize._handle_gateway_enclave_rpc(
            "gateway_coordinator",
            method,
            params,
        ) == {"method": method, "status": "candidate"}

    assert runtime.calls == list(requests.items())


def test_rehearsal_routes_provider_boot_credentials_to_candidate_runtime(
    monkeypatch,
) -> None:
    class CandidateRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def handle_v2_runtime_rpc(
            self,
            method: str,
            params: dict[str, object],
        ) -> dict[str, object]:
            self.calls.append((method, dict(params)))
            return {
                "result": {
                    "credential_slots": ["openrouter"],
                    "method": method,
                    "status": "provisioning",
                }
            }

    runtime = CandidateRuntime()
    state = {
        "roles": {
            "gateway_coordinator": {
                "config_hash": "sha256:" + "1" * 64,
                "configuration": {
                    "artifact_master_key_ref_hash": "sha256:" + "2" * 64,
                    "provider_ref_hashes": {
                        "openrouter": "sha256:" + "3" * 64,
                    },
                },
            }
        },
        "provisioned_slots": ["artifact_master_key"],
    }

    def enclave_state(mutate=None):
        if mutate is not None:
            mutate(state)
        return state, None

    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_release_input",
        lambda: {"gateway_roles": {"gateway_coordinator": {}}},
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_enclave_state",
        enclave_state,
    )
    monkeypatch.setattr(
        rehearsal_sitecustomize,
        "_gateway_runtime_objects",
        lambda _role, _state: {"tee_service": runtime},
    )

    recipient = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
        "gateway_coordinator",
        "v2_get_kms_recipient",
        {"credential_slot": "openrouter"},
    )
    provision = rehearsal_sitecustomize._handle_gateway_enclave_rpc(
        "gateway_coordinator",
        "v2_provision_encrypted_secret",
        {
            "credential_slot": "openrouter",
            "ciphertext_for_recipient_b64": "Y2lwaGVydGV4dA==",
        },
    )

    assert recipient["method"] == "v2_get_kms_recipient"
    assert provision["method"] == "v2_provision_encrypted_secret"
    assert state["provisioned_slots"] == ["artifact_master_key", "openrouter"]
    assert runtime.calls == [
        ("v2_get_kms_recipient", {"credential_slot": "openrouter"}),
        (
            "v2_provision_encrypted_secret",
            {
                "credential_slot": "openrouter",
                "ciphertext_for_recipient_b64": "Y2lwaGVydGV4dA==",
            },
        ),
    ]

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

    stages = [
        {
            "command": ["docker", "run", "candidate"],
            "stage": "validator-forward-1",
            "status": "failed",
        }
    ]
    durable = rehearsal._preserve_batched_failure_evidence(
        evidence_root=evidence,
        candidate_sha=COMMIT,
        stages=stages,
    )

    assert (
        durable / "evidence" / "validator-main.log"
    ).read_text(encoding="utf-8").startswith("coordinator failed")
    assert json.loads(
        (durable / "failure-summary.json").read_text(encoding="utf-8")
    ) == {
        "candidate_sha": COMMIT,
        "failure_count": 1,
        "failures": stages,
        "stage_count": 1,
        "stages": stages,
        "status": "failed",
        "unexercised_count": 0,
    }


def test_rehearsal_preserves_complete_full_path_stage_ledger(tmp_path) -> None:
    evidence = tmp_path / "ephemeral"
    evidence.mkdir()
    (evidence / "validator-main.log").write_text(
        "rollback failed before coordinator readiness\n",
        encoding="utf-8",
    )
    stages = [
        {
            "stage": "gateway-forward-1",
            "status": "passed",
        },
        {
            "command": ["docker", "run", "rollback"],
            "returncode": 1,
            "stage": "validator-rollback-2",
            "status": "failed",
        },
        {
            "command": ["docker", "run", "workflow"],
            "returncode": 2,
            "stage": "workflow-release",
            "status": "failed",
        },
        {
            "blocked_by": ["workflow-release"],
            "stage": "evidence-join-release",
            "status": "unexercised",
        },
    ]

    durable = rehearsal._preserve_batched_failure_evidence(
        evidence_root=evidence,
        candidate_sha=COMMIT,
        stages=stages,
    )

    assert (
        durable / "evidence" / "validator-main.log"
    ).read_text(encoding="utf-8").startswith("rollback failed")
    assert json.loads(
        (durable / "failure-summary.json").read_text(encoding="utf-8")
    ) == {
        "candidate_sha": COMMIT,
        "failure_count": 2,
        "failures": stages[1:3],
        "stage_count": 4,
        "stages": stages,
        "status": "failed",
        "unexercised_count": 1,
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


def test_prepush_runs_validator_and_workflow_after_gateway_failure(
    monkeypatch,
    tmp_path,
) -> None:
    from_sha = "2" * 40
    candidate_sha = "3" * 40
    calls: list[str] = []
    captured: dict[str, Any] = {}
    component_barrier = threading.Barrier(2)

    @contextmanager
    def source_snapshot(**_kwargs):
        yield tmp_path

    @contextmanager
    def fixture_seed(*_args, **_kwargs):
        yield tmp_path

    monkeypatch.setattr(
        rehearsal,
        "_git_sha",
        lambda value: from_sha if value == "FROM" else candidate_sha,
    )
    monkeypatch.setattr(
        rehearsal,
        "REHEARSAL_LOCK_PATH",
        tmp_path / "controller.lock",
    )
    monkeypatch.setattr(
        rehearsal,
        "_resolve_transition",
        lambda *_args: "forward",
    )
    monkeypatch.setattr(rehearsal, "_verify_driver_identity", lambda _sha: None)
    monkeypatch.setattr(rehearsal, "_docker_platform", lambda _profile: "linux/amd64")
    monkeypatch.setattr(rehearsal, "_image_tag", lambda *_args, **_kwargs: "image")
    monkeypatch.setattr(rehearsal, "_image_exists", lambda _tag: True)
    monkeypatch.setattr(
        rehearsal,
        "_isolated_source_snapshot",
        source_snapshot,
    )
    monkeypatch.setattr(
        rehearsal,
        "_run_python37_finalization_probe",
        lambda _root: None,
    )
    monkeypatch.setattr(
        rehearsal,
        "_prepare_drand_artifact",
        lambda **_kwargs: tmp_path,
    )
    monkeypatch.setattr(rehearsal, "_prepared_fixture_seed", fixture_seed)
    # This test exercises independent-stage continuation after failures. Ownership
    # normalization has its own coverage and would otherwise try to run the
    # fake image tag below through Docker.
    monkeypatch.setattr(
        rehearsal,
        "_normalize_evidence_ownership",
        lambda *_args, **_kwargs: None,
    )

    def run_component(_tag, *, component, **_kwargs):
        calls.append(component)
        component_barrier.wait(timeout=1)
        if component == "gateway":
            raise subprocess.CalledProcessError(17, ["gateway-restart"])

    def run_workflow(*_args, **_kwargs):
        calls.append("workflow")
        raise subprocess.CalledProcessError(23, ["workflow"])

    def preserve(*, stages, **_kwargs):
        captured["stages"] = stages
        return tmp_path

    monkeypatch.setattr(rehearsal, "_run_component", run_component)
    monkeypatch.setattr(rehearsal, "_run_workflow", run_workflow)
    monkeypatch.setattr(
        rehearsal,
        "_join_evidence",
        lambda *_args, **_kwargs: pytest.fail("join must be dependency-blocked"),
    )
    monkeypatch.setattr(
        rehearsal,
        "_preserve_batched_failure_evidence",
        preserve,
    )

    with pytest.raises(
        SystemExit,
        match="prepush rehearsal failed after completing independent stages",
    ):
        rehearsal.main(
            [
                "--from-sha",
                "FROM",
                "--candidate-sha",
                "CANDIDATE",
                "--profile",
                "prepush",
            ]
        )

    assert sorted(calls) == ["gateway", "validator", "workflow"]
    assert all(calls.count(component) == 1 for component in calls)
    by_stage = {
        item["stage"]: item for item in captured["stages"]
    }
    assert by_stage["gateway-forward-1"]["status"] == "failed"
    assert by_stage["gateway-forward-1"]["command"] == ["gateway-restart"]
    assert by_stage["gateway-forward-1"]["returncode"] == 17
    assert by_stage["validator-forward-1"]["status"] == "passed"
    assert "command" not in by_stage["validator-forward-1"]
    assert "returncode" not in by_stage["validator-forward-1"]
    assert by_stage["workflow-prepush"]["status"] == "failed"
    assert by_stage["workflow-prepush"]["command"] == ["workflow"]
    assert by_stage["workflow-prepush"]["returncode"] == 23
    assert by_stage["evidence-join-prepush"] == {
        "blocked_by": ["gateway-forward-1", "workflow-prepush"],
        "stage": "evidence-join-prepush",
        "status": "unexercised",
    }


def _durable_interval(
    start_revision: int,
    start_hash: str,
    end_revision: int,
    end_hash: str,
) -> dict[str, Any]:
    return {
        "durable_boundary_state": {
            "schema_version": (
                "leadpoet.restart_rehearsal.durable_boundary_state.v1"
            ),
            "durable_schema_sha": COMMIT,
            "start_revision": start_revision,
            "start_state_hash": start_hash,
            "end_revision": end_revision,
            "end_state_hash": end_hash,
        }
    }


def _write_final_durable_state(
    tmp_path: Path,
    *,
    revision: int,
) -> str:
    state = {
        "schema_version": "leadpoet.local_postgrest_durable_state.v1",
        "durable_schema_sha": COMMIT,
        "revision": revision,
        "rows": {"example": [{"revision": revision}]},
    }
    state["state_hash"] = join_evidence._state_hash(state)
    path = tmp_path / "durable-boundary-state/postgrest-state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    return str(state["state_hash"])


def test_prepush_durable_continuity_accepts_overlapping_launcher_intervals(
    tmp_path,
) -> None:
    initial_hash = "sha256:" + "0" * 64
    final_hash = _write_final_durable_state(tmp_path, revision=9)

    join_evidence._verify_durable_boundary_continuity(
        [
            _durable_interval(0, initial_hash, 9, final_hash),
            _durable_interval(0, initial_hash, 0, initial_hash),
        ],
        profile="prepush",
        evidence_root=tmp_path,
        candidate_sha=COMMIT,
    )


def test_prepush_durable_continuity_rejects_conflicting_revision_hashes(
    tmp_path,
) -> None:
    initial_hash = "sha256:" + "0" * 64
    conflicting_hash = "sha256:" + "1" * 64
    final_hash = _write_final_durable_state(tmp_path, revision=9)

    with pytest.raises(SystemExit, match="conflicting state hashes"):
        join_evidence._verify_durable_boundary_continuity(
            [
                _durable_interval(0, initial_hash, 9, final_hash),
                _durable_interval(0, conflicting_hash, 0, conflicting_hash),
            ],
            profile="prepush",
            evidence_root=tmp_path,
            candidate_sha=COMMIT,
        )


def test_prepush_durable_continuity_rejects_stale_final_state(
    tmp_path,
) -> None:
    initial_hash = "sha256:" + "0" * 64
    observed_hash = "sha256:" + "9" * 64
    _write_final_durable_state(tmp_path, revision=8)

    with pytest.raises(SystemExit, match="did not survive activation"):
        join_evidence._verify_durable_boundary_continuity(
            [_durable_interval(0, initial_hash, 9, observed_hash)],
            profile="prepush",
            evidence_root=tmp_path,
            candidate_sha=COMMIT,
        )


def test_release_durable_continuity_remains_strictly_sequential(
    tmp_path,
) -> None:
    hashes = ["sha256:" + str(index) * 64 for index in range(3)]
    join_evidence._verify_durable_boundary_continuity(
        [
            _durable_interval(0, hashes[0], 5, hashes[1]),
            _durable_interval(5, hashes[1], 9, hashes[2]),
        ],
        profile="release",
        evidence_root=tmp_path,
        candidate_sha=COMMIT,
    )
    with pytest.raises(SystemExit, match="did not survive activation"):
        join_evidence._verify_durable_boundary_continuity(
            [
                _durable_interval(0, hashes[0], 5, hashes[1]),
                _durable_interval(0, hashes[0], 9, hashes[2]),
            ],
            profile="release",
            evidence_root=tmp_path,
            candidate_sha=COMMIT,
        )


def test_workflow_runner_continues_across_failed_release_epochs(
    monkeypatch,
    tmp_path,
) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "sanitization": {"contains_production_credentials": False},
                "fault_matrix": [],
            }
        ),
        encoding="utf-8",
    )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "forbidden_substitutions": [
                    "gateway",
                    "validator",
                    "auditor",
                    "canonical_bundle",
                    "receipt_graph",
                    "signature",
                    "sdk_extrinsic",
                    "verification",
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "workflow.json"
    epoch_calls: list[int] = []

    class Thread:
        alive = False

        def is_alive(self):
            return self.alive

    class Services:
        def __init__(self, **_kwargs):
            self.thread = Thread()
            self.state = type(
                "State",
                (),
                {"events": [], "faults": [], "chain": {}},
            )()

        def __enter__(self):
            self.thread.alive = True
            return self

        def __exit__(self, *_args):
            self.thread.alive = False

    def run_epoch(*, epoch_id, **_kwargs):
        epoch_calls.append(epoch_id)
        if epoch_id in {30_000, 30_001}:
            raise RuntimeError(f"blocked epoch {epoch_id}")
        return {
            "epoch_id": epoch_id,
            "canonical_vector_equal": True,
            "receipt_ancestry_verified": True,
            "auditor_verified": True,
            "auditor_runtime_verified": True,
            "last_update": epoch_id + 1,
            "finalized_block": epoch_id + 1,
        }

    monkeypatch.setattr(
        production_workflow_runner,
        "_file_identity",
        lambda path, candidate_sha: {
            "path": path,
            "commit_sha": candidate_sha,
        },
    )
    monkeypatch.setattr(
        production_workflow_runner,
        "LocalBoundaryServices",
        Services,
    )
    monkeypatch.setattr(
        production_workflow_runner,
        "_exercise_concurrency",
        lambda _services: 32,
    )
    monkeypatch.setattr(
        production_workflow_runner,
        "_run_epoch",
        run_epoch,
    )
    behavior_contract = {
        "candidate_sha": COMMIT,
        "production_source_paths": [],
        "behavior_scenarios": [],
        "authority_diagnostics": [],
        "fault_ids": [],
        "required_stage_ids": [
            "input-contract",
            "concurrency",
            "boundary-start",
            *[f"epoch-{30_000 + ordinal}" for ordinal in range(100)],
            "boundary-cleanup",
            "workflow-evidence-validation",
        ],
        "required_invariant_ids": [
            "candidate_identity_exact",
            "protected_source_identity_exact",
            "chain_settlement_state_space_complete",
            "canonical_vector_primary_auditor_equal",
            "receipt_ancestry_verified",
            "sdk_signing_bridge_verified",
            "submission_finalized",
            "last_update_readback_equal",
            "boundary_cleanup_complete",
            "unknown_boundaries_rejected",
        ],
        "policy_commitments": {},
        "contract_hash": "sha256:" + "a" * 64,
    }
    monkeypatch.setattr(
        production_workflow_runner,
        "build_rehearsal_behavior_contract_v2",
        lambda **_kwargs: behavior_contract,
    )
    monkeypatch.setattr(
        production_workflow_runner,
        "validate_rehearsal_behavior_contract_v2",
        lambda value: value,
    )
    monkeypatch.setattr(
        production_workflow_runner,
        "_run_independent_epoch_diagnostics",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "production_workflow_runner.py",
            "--profile",
            "release",
            "--candidate-sha",
            COMMIT,
            "--epochs",
            "100",
            "--fixture",
            str(fixture),
            "--boundary-contract",
            str(contract),
            "--output",
            str(output),
        ],
    )

    assert production_workflow_runner.main() == 1
    assert epoch_calls == list(range(30_000, 30_100))
    manifest = json.loads(output.read_text(encoding="utf-8"))
    stages = {item["stage"]: item for item in manifest["stages"]}
    assert stages["epoch-30000"]["status"] == "failed"
    assert stages["epoch-30001"]["status"] == "failed"
    assert stages["epoch-30002"]["status"] == "passed"
    assert stages["boundary-cleanup"]["status"] == "passed"
    assert stages["workflow-evidence-validation"]["status"] == "unexercised"


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


def test_exact_rehearsal_supplies_paired_active_release_handoff() -> None:
    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")

    assert "prepare_validator_initial_active_lineage_v2(" in script
    assert "prepare_gateway_final_active_lineage_v2(" in script
    assert "load_source_add_graphs=no_active_graphs" in script
    assert "validate_active_release_requirements_v2(conflicting)" in script
    assert "fetch_prior_release_channel_v2(" in script
    assert "running_channel[\"gateway_release_manifest\"]" in script
    assert '"leadpoet-validator-main" in containers' in script
    assert 'f"VALIDATOR_V2_DEPLOY_COMMIT={running_commit}"' in script
    assert (
        'ACTIVE_RELEASE_VALIDATOR_REQUIREMENTS="/tmp/leadpoet-'
        "validator-active-release-requirements.${ACTIVE_RELEASE_FIXTURE_SUFFIX}.json\""
        in script
    )
    assert (
        'ACTIVE_RELEASE_GATEWAY_REQUIREMENTS="/tmp/leadpoet-'
        "gateway-active-release-requirements.${ACTIVE_RELEASE_FIXTURE_SUFFIX}.json\""
        in script
    )
    assert (
        'ACTIVE_RELEASE_GATEWAY_LINEAGE="/tmp/leadpoet-'
        "gateway-active-release-lineage.${ACTIVE_RELEASE_FIXTURE_SUFFIX}.json\""
        in script
    )
    assert '"GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED=1"' in script
    assert '"GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS=' in script
    assert '"GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_FILE=' in script
    assert '"VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED=1"' in script
    assert '"VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT=' in script
    assert '"VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT=' in script
    assert '"VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT=' in script
    assert '"VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE=' in script
    assert '"${GATEWAY_ACTIVE_RELEASE_ENV[@]}" \\' in script
    assert '"${VALIDATOR_ACTIVE_RELEASE_ENV[@]}" \\' in script


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


def test_forward_rehearsal_uses_canonical_first_rollout_and_keeps_direct_paths() -> None:
    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")
    adapter = (
        Path(__file__).resolve().parent / "sitecustomize.py"
    ).read_text(encoding="utf-8")
    contract = (
        Path(__file__).resolve().parent / "contract_adapter.py"
    ).read_text(encoding="utf-8")

    assert 'GATEWAY_DEPLOY_COMMIT="$CANDIDATE_SHA"' not in script
    assert 'VALIDATOR_DEPLOY_COMMIT="$CANDIDATE_SHA"' not in script
    assert "0dd3a385a23a3af0fa17210bfe02a39cc4023952" in contract
    assert "7ac1553e32d85d9babda3b3836f4c93cf92e6d60" in contract
    assert '[ "$MINER_MAINTENANCE_MODE" = "legacy_first_rollout" ]' in script
    assert '"legacy_retry", "post_rollout", "rollback"' in script
    assert '"$MINER_BOOTSTRAP_ROOT/candidate/gw_restart.sh"' in script
    assert '--miner-maintenance-bootstrap-plan "$MINER_BOOTSTRAP_ROOT/plan.json"' in script
    assert '--miner-maintenance-handoff-file "$MINER_HANDOFF_FILE"' in script
    assert (
        "Prepared exact-candidate miner maintenance under the canonical restart lock"
        in script
    )
    assert 'MINER_BOOTSTRAP_READY_EPOCH="$(date -u +%s)"' in script
    assert '"GATEWAY_RESTART_TIMING stage=controller_reexec"' in script
    assert "miner-maintenance N-1 handoff lost its cwd or timing ledger" in script
    assert "gateway restart timing ledger is unavailable" in script
    assert 'bash /home/ec2-user/gw_restart.sh --commit "$CANDIDATE_SHA"' in script
    assert 'GATEWAY_RESTART_AUTHORITY_ROOT="$FORWARD_AUTHORITY_ROOT"' in script
    assert 'GATEWAY_RESTART_AUTHORITY_COMMIT="$CANDIDATE_SHA"' in script
    assert (
        'bash "$FORWARD_AUTHORITY_ROOT/gw_restart.sh" --commit "$CANDIDATE_SHA"'
        in script
    )
    assert "direct miner-maintenance restart performed secret writes" in script
    launcher = script.split(
        '  set +e\n  if [ "$TRANSITION" = "rollback" ]; then', 1
    )[1].split("  set -e\n", 1)[0]
    rollback, remaining = launcher.split(
        '  elif [ "$MINER_FIRST_ROLLOUT" = "1" ]; then', 1
    )
    first_rollout, direct = remaining.split("  else\n", 1)
    assert "bash /home/ec2-user/gw_restart.sh" not in direct
    git_environment_overrides = (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
    )
    for restart_path in (rollback, first_rollout, direct):
        for variable in git_environment_overrides:
            assert f"-u {variable}" in restart_path
    assert "return _real_boto3_client" not in adapter
    assert 'raise ValueError("local boto3 AWS service is unknown")' in adapter


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
        row(
            [
                "-m",
                module,
                "--storage-read-preflight",
                "--epoch",
                "24304",
            ],
            "candidate_archive",
        ),
        row(
            ["-m", module, "--repair-chain-settlements"],
            "candidate_checkout",
        ),
        row(
            ["-m", module, "--repair", "--epoch", "24304"],
            "candidate_checkout",
        ),
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
        storage_preflight_pins_epoch=True,
    )

    recovered_rows = [
        rows[0],
        row(
            ["-m", module, "--storage-read-preflight"],
            "candidate_checkout",
        ),
        *rows[1:],
    ]
    verify_gateway_weight_readiness_invocations(
        recovered_rows,
        candidate_sha=COMMIT,
        storage_preflight_pins_epoch=True,
    )

    advanced_epoch_rows = copy.deepcopy(rows)
    advanced_epoch_rows[0]["argv"][-1] = "24303"
    verify_gateway_weight_readiness_invocations(
        advanced_epoch_rows,
        candidate_sha=COMMIT,
        storage_preflight_pins_epoch=True,
    )

    malformed_epoch_rows = copy.deepcopy(rows)
    malformed_epoch_rows[0]["argv"][-1] = "not-an-epoch"
    with pytest.raises(SystemExit, match="exact production"):
        verify_gateway_weight_readiness_invocations(
            malformed_epoch_rows,
            candidate_sha=COMMIT,
            storage_preflight_pins_epoch=True,
        )

    rows[3]["argv"][-1] = "30"
    with pytest.raises(SystemExit, match="launcher contract"):
        verify_gateway_weight_readiness_invocations(
            rows,
            candidate_sha=COMMIT,
            storage_preflight_pins_epoch=True,
        )

    legacy_rows = copy.deepcopy(rows)
    legacy_rows[0]["argv"] = ["-m", module, "--storage-read-preflight"]
    legacy_rows[3]["argv"][-1] = "360"
    verify_gateway_weight_readiness_invocations(
        legacy_rows,
        candidate_sha=COMMIT,
    )


def test_gateway_storage_preflight_routes_only_canonical_supabase_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    modules = (
        "gateway.main",
        "gateway.research_lab.stateful_epoch_cutover_cli_v1",
        "gateway.tee.bootstrap_active_ancestry_checkpoints_v2",
        "gateway.tee.prepare_active_release_lineage_v2",
        "gateway.tee.verify_weight_submission_ready_v2",
    )
    for module in modules:
        monkeypatch.setenv(
            "SUPABASE_URL", contract_adapter.PRODUCTION_SUPABASE_ORIGIN
        )
        contract_adapter._route_host_storage_preflight_to_local_postgrest(module)
        assert (
            os.environ["SUPABASE_URL"]
            == contract_adapter.LOCAL_POSTGREST_ORIGIN
        )

        monkeypatch.setenv("SUPABASE_URL", "https://unexpected.invalid")
        with pytest.raises(ValueError, match="Supabase origin differs"):
            contract_adapter._route_host_storage_preflight_to_local_postgrest(
                module
            )

    contract_adapter._route_host_storage_preflight_to_local_postgrest(
        "gateway.tee.restart_preflight_v2"
    )
    assert os.environ["SUPABASE_URL"] == "https://unexpected.invalid"


@pytest.mark.parametrize(
    "module",
    (
        "gateway.research_lab.stateful_epoch_cutover_cli_v1",
        "gateway.tee.bootstrap_active_ancestry_checkpoints_v2",
        "gateway.tee.prepare_active_release_lineage_v2",
    ),
)
def test_gateway_storage_helpers_route_before_exec(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
) -> None:
    from tests.restart_rehearsal import contract_adapter

    observed: dict[str, Any] = {}
    monkeypatch.setenv(
        "SUPABASE_URL", contract_adapter.PRODUCTION_SUPABASE_ORIGIN
    )
    monkeypatch.setenv("PYTHONPATH", "/source")
    monkeypatch.setattr(
        contract_adapter,
        "_record_production_module",
        lambda name, argv: observed.update(module=name, recorded_argv=argv),
    )

    def fake_exec(executable: str, argv: list[str]) -> None:
        observed.update(
            executable=executable,
            exec_argv=argv,
            supabase_url=os.environ["SUPABASE_URL"],
        )
        raise RuntimeError("exec captured")

    monkeypatch.setattr(contract_adapter.os, "execv", fake_exec)
    argv = ["-m", module, "--help"]
    with pytest.raises(RuntimeError, match="exec captured"):
        contract_adapter.command_python(argv)

    assert observed == {
        "module": module,
        "recorded_argv": argv,
        "executable": contract_adapter.REAL_PYTHON,
        "exec_argv": [contract_adapter.REAL_PYTHON, *argv],
        "supabase_url": contract_adapter.LOCAL_POSTGREST_ORIGIN,
    }


def test_gateway_main_routes_storage_before_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.restart_rehearsal import contract_adapter

    observed: dict[str, Any] = {}
    monkeypatch.setenv(
        "SUPABASE_URL", contract_adapter.PRODUCTION_SUPABASE_ORIGIN
    )
    monkeypatch.setenv("PYTHONPATH", "/source")
    monkeypatch.setattr(
        contract_adapter, "_locked_state", lambda: (object(), {})
    )
    monkeypatch.setattr(contract_adapter, "_save_state", lambda *_args: None)
    monkeypatch.setattr(contract_adapter, "_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        contract_adapter, "_module_source", lambda _module: Path("gateway/main.py")
    )
    monkeypatch.setattr(
        contract_adapter, "_source_identity", lambda _path: {}
    )

    def fake_exec(executable: str, argv: list[str]) -> None:
        observed.update(
            executable=executable,
            argv=argv,
            supabase_url=os.environ["SUPABASE_URL"],
        )
        raise RuntimeError("exec captured")

    monkeypatch.setattr(contract_adapter.os, "execv", fake_exec)
    argv = ["-m", "gateway.main"]
    with pytest.raises(RuntimeError, match="exec captured"):
        contract_adapter._exec_long_lived_production_module(
            "gateway.main", "gateway.main", argv
        )

    assert observed == {
        "executable": contract_adapter.REAL_PYTHON,
        "argv": [contract_adapter.REAL_PYTHON, *argv],
        "supabase_url": contract_adapter.LOCAL_POSTGREST_ORIGIN,
    }


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
        row(["-m", module, "--repair-chain-settlements"]),
        row(["-m", module, "--repair", "--epoch", "24304"]),
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


def test_module_provenance_accepts_only_candidate_miner_bootstrap_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    with _production_named_temp_directory(
        "gateway-miner-maintenance-bootstrap."
    ) as bootstrap_root:
        module_path = (
            bootstrap_root
            / "candidate/gateway/tee/gateway_miner_maintenance_restart_v1.py"
        )
        module_path.parent.mkdir(parents=True)
        module_path.write_text("VALUE = 'candidate archive'\n", encoding="utf-8")
        outside = bootstrap_root / "unbound.py"
        outside.write_text("VALUE = 'outside candidate'\n", encoding="utf-8")

        relative, source_kind = contract_adapter._candidate_git_path(
            module_path.resolve(),
            Path("/home/ec2-user/leadpoet_repo"),
        )

        assert relative == Path(
            "gateway/tee/gateway_miner_maintenance_restart_v1.py"
        )
        assert source_kind == "candidate_archive"
        with pytest.raises(RuntimeError, match="outside the candidate checkout"):
            contract_adapter._candidate_git_path(
                outside.resolve(),
                Path("/home/ec2-user/leadpoet_repo"),
            )


@pytest.mark.parametrize(
    "prefix",
    (
        "gateway-restart-controller-bootstrap.",
        "validator-restart-controller-bootstrap.",
    ),
)
def test_module_provenance_accepts_candidate_restart_authority_archive(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    with _production_named_temp_directory(prefix) as bootstrap_root:
        module_path = bootstrap_root / "authority/scripts/gateway_git_deploy.py"
        module_path.parent.mkdir(parents=True)
        module_path.write_text("VALUE = 'candidate archive'\n", encoding="utf-8")
        outside = bootstrap_root / "outside.py"
        outside.write_text("VALUE = 'outside authority'\n", encoding="utf-8")

        relative, source_kind = contract_adapter._candidate_git_path(
            module_path.resolve(),
            Path("/home/ec2-user/leadpoet_repo"),
        )

        assert relative == Path("scripts/gateway_git_deploy.py")
        assert source_kind == "candidate_archive"
        with pytest.raises(RuntimeError, match="recognized candidate"):
            contract_adapter._candidate_git_path(
                outside.resolve(),
                Path("/home/ec2-user/leadpoet_repo"),
            )

        escaped_link = bootstrap_root / "authority/escaped.py"
        escaped_link.symlink_to(outside)
        with pytest.raises(RuntimeError, match="recognized candidate"):
            contract_adapter._candidate_git_path(
                escaped_link.resolve(),
                Path("/home/ec2-user/leadpoet_repo"),
            )


@pytest.mark.parametrize("tampered", (False, True))
def test_frozen_local_release_archive_keeps_exact_source_verification(
    monkeypatch: pytest.MonkeyPatch,
    tampered: bool,
) -> None:
    root = Path(__file__).resolve().parents[2]
    candidate_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", candidate_sha)
    from tests.restart_rehearsal import contract_adapter

    monkeypatch.setattr(contract_adapter, "_candidate_root", lambda: root)
    relative = Path("gateway/tee/local_release_v2.py")
    with _production_named_temp_directory("leadpoet-local-release-source.") as archive:
        source = archive / relative
        source.parent.mkdir(parents=True)
        source.write_bytes((root / relative).read_bytes() + (b"\n# changed\n" if tampered else b""))
        if tampered:
            with pytest.raises(RuntimeError, match="source bytes differ"):
                contract_adapter._source_identity(source)
        else:
            identity = contract_adapter._source_identity(source)
            assert identity["source_kind"] == "candidate_archive"
            assert identity["source_commit"] == candidate_sha
            assert identity["source_git_path"] == relative.as_posix()


def test_module_provenance_accepts_only_exact_candidate_local_release_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    build_root = tmp_path / "gateway-release-build-v2"
    monkeypatch.setenv("GATEWAY_V2_BUILD_WORK_ROOT", str(build_root))
    module_path = (
        build_root
        / f"{COMMIT}-local"
        / "gateway_coordinator/source/gateway/tee/build_identity.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.write_text("VALUE = 'candidate archive'\n", encoding="utf-8")

    relative, source_kind = contract_adapter._candidate_git_path(
        module_path.resolve(),
        Path("/home/ec2-user/leadpoet_repo"),
    )

    assert relative == Path("gateway/tee/build_identity.py")
    assert source_kind == "candidate_archive"

    wrong_candidate = (
        build_root
        / f"{'2' * 40}-local"
        / "gateway_coordinator/source/gateway/tee/build_identity.py"
    )
    wrong_candidate.parent.mkdir(parents=True)
    wrong_candidate.write_text("VALUE = 'wrong candidate'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="recognized candidate"):
        contract_adapter._candidate_git_path(
            wrong_candidate.resolve(),
            Path("/home/ec2-user/leadpoet_repo"),
        )

    wrong_role = (
        build_root
        / f"{COMMIT}-local"
        / "gateway_unknown/source/gateway/tee/build_identity.py"
    )
    wrong_role.parent.mkdir(parents=True)
    wrong_role.write_text("VALUE = 'wrong role'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="recognized candidate"):
        contract_adapter._candidate_git_path(
            wrong_role.resolve(),
            Path("/home/ec2-user/leadpoet_repo"),
        )

    outside_source = (
        build_root
        / f"{COMMIT}-local"
        / "gateway_coordinator/outside/gateway/tee/build_identity.py"
    )
    outside_source.parent.mkdir(parents=True)
    outside_source.write_text("VALUE = 'outside source'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="recognized candidate"):
        contract_adapter._candidate_git_path(
            outside_source.resolve(),
            Path("/home/ec2-user/leadpoet_repo"),
        )

    wrong_root = (
        tmp_path
        / "untrusted/gateway-release-build-v2"
        / f"{COMMIT}-local"
        / "gateway_coordinator/source/gateway/tee/build_identity.py"
    )
    wrong_root.parent.mkdir(parents=True)
    wrong_root.write_text("VALUE = 'wrong root'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="recognized candidate"):
        contract_adapter._candidate_git_path(
            wrong_root.resolve(),
            Path("/home/ec2-user/leadpoet_repo"),
        )


def test_gateway_verification_image_build_is_bound_to_role_and_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    context = Path("/build/gateway_coordinator/source/gateway")
    argv = [
        "build",
        "--pull",
        "--no-cache",
        "--build-arg",
        "SOURCE_DATE_EPOCH=0",
        "--build-arg",
        "LEADPOET_ENCLAVE_ROLE=gateway_coordinator",
        "-f",
        str(context / "tee/Dockerfile.enclave"),
        "-t",
        f"leadpoet-gateway-verify:gateway_coordinator-{COMMIT[:12]}-1-raw",
        str(context),
    ]

    assert (
        contract_adapter._external_build_role(argv, argv[-2])
        == "gateway_coordinator"
    )

    for invalid_tag in (
        f"leadpoet-gateway-verify:gateway_coordinator-{'2' * 12}-1-raw",
        f"leadpoet-gateway-verify:gateway_unknown-{COMMIT[:12]}-1-raw",
        f"leadpoet-gateway-verify:gateway_coordinator-{COMMIT[:12]}-raw",
    ):
        with pytest.raises(ValueError, match="gateway verification image"):
            contract_adapter._external_build_role(argv, invalid_tag)


def test_gateway_verification_normalized_image_is_bound_to_role_and_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    from tests.restart_rehearsal import contract_adapter

    role = "gateway_coordinator"
    record = {
        "commit": COMMIT,
        "id": normalized_image_id(COMMIT, role),
        "role": role,
    }
    image = f"leadpoet-gateway-verify:{role}-{COMMIT[:12]}-1"
    assert contract_adapter._gateway_verification_image_is_bound(
        image=image,
        record=record,
    )

    invalid_cases = (
        (f"leadpoet-gateway-verify:{role}-{'2' * 12}-1", record),
        (f"leadpoet-gateway-verify:gateway_scoring-{COMMIT[:12]}-1", record),
        (f"leadpoet-gateway-verify:{role}-{COMMIT[:12]}-0", record),
        (image + "-raw", record),
        (image, {**record, "id": "sha256:" + "2" * 64}),
        (image, {**record, "provenance": "copied"}),
    )
    for invalid_image, invalid_record in invalid_cases:
        assert not contract_adapter._gateway_verification_image_is_bound(
            image=invalid_image,
            record=invalid_record,
        )

def test_docker_contract_inherits_name_only_environment_without_argv_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness_root = Path(__file__).resolve().parent
    secret = "rehearsal-secret-that-must-not-enter-argv"
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", COMMIT)
    monkeypatch.setenv("REHEARSAL_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("LEADPOET_SENTRY_DSN", secret)
    monkeypatch.delenv("LEADPOET_SENTRY_RELEASE", raising=False)
    specification = importlib.util.spec_from_file_location(
        "rehearsal_contract_adapter_env_test",
        harness_root / "contract_adapter.py",
    )
    assert specification is not None
    assert specification.loader is not None
    adapter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(adapter)
    argv = [
        "run",
        "-e",
        "LEADPOET_SENTRY_DSN",
        "--env",
        "LEADPOET_SENTRY_RELEASE",
        "leadpoet-validator:latest",
    ]

    _, environment, _, invocation = adapter._docker_run_contract(argv)

    assert environment == {"LEADPOET_SENTRY_DSN": secret}
    assert invocation == ["leadpoet-validator:latest"]
    assert secret not in argv


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
